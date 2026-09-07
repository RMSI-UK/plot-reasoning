#!/usr/bin/env python3
"""Batch runner: locate the application plot on every panel in a folder.

Concurrent by default. The model takes a median 8 s per page, so a serial run of 783 pages
would sit idle for nearly two hours; at 10 workers it is about ten minutes.

Results stream to a JSONL file one line per page, so a run can be interrupted and resumed
without re-paying for pages already done.

    python run.py --src /path/to/panels --out results/            # whole folder
    python run.py --src ... --limit 20 --stride 37                # a deterministic sample
    python run.py --src ... --no-metadata                         # image only
    python run.py --src ... --render                              # also draw overlays
    python run.py --price                                         # check live pricing, no calls
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the repo's venv carries pydantic-ai, PIL, cv2 and the OpenRouter model resolver
REPO = Path("/env/code/plot-reasoning/GeoPlanAgent")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dotenv import load_dotenv                                    # noqa: E402

load_dotenv(str(REPO / ".env"))                                   # OPENROUTER_API_KEY

from geoplanagent.utils import resolve_model                       # noqa: E402
from locate import (MODEL, USD_PER_MTOK_IN, USD_PER_MTOK_OUT,      # noqa: E402
                    build_agent, load_prompt, locate_one)
from metadata import records_for                                   # noqa: E402


def live_price(model_id: str) -> tuple[float, float] | None:
    req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                 headers={"User-Agent": "find_plot_in_plan"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=60))["data"]
    except Exception:                                              # noqa: BLE001
        return None
    for m in data:
        if m["id"] == model_id:
            p = m.get("pricing") or {}
            try:
                return float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6
            except (KeyError, TypeError, ValueError):
                return None
    return None


def pick(src: Path, limit: int | None, stride: int) -> list[Path]:
    files = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    files = files[::stride] if stride > 1 else files
    return files[:limit] if limit else files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, help="folder of panel images")
    ap.add_argument("--out", type=Path, default=HERE / "results",
                    help="output folder (default: ./results)")
    ap.add_argument("--workers", type=int, default=10,
                    help="concurrent requests (default 10)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N pages")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth file, for a deterministic sample")
    ap.add_argument("--no-metadata", action="store_true",
                    help="do not send the council's address and proposal")
    ap.add_argument("--metadata-csv", type=Path, default=None)
    ap.add_argument("--render", action="store_true",
                    help="also write overlay images next to the results")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--price", action="store_true",
                    help="print live pricing and exit without calling the model")
    args = ap.parse_args()

    price = live_price(args.model)
    if args.price:
        if price:
            pin, pout = price
            print(f"{args.model}: ${pin:.2f}/${pout:.2f} per Mtok")
            # measured token profile on this corpus, verbose schema, full-size pages
            per = 3130 * pin / 1e6 + 575 * pout / 1e6
            print(f"measured token profile 3130 in / 575 out -> ${per:.5f}/page")
            for n in (100, 783, 1000):
                print(f"  {n:5d} pages  ${per*n:7.2f}  =  GBP {per*n/1.26:7.2f}")
        else:
            print(f"{args.model}: not listed on OpenRouter")
        return 0
    if not args.src:
        ap.error("--src is required (or use --price)")
    if price and (abs(price[0] - USD_PER_MTOK_IN) > 1e-9
                  or abs(price[1] - USD_PER_MTOK_OUT) > 1e-9):
        print(f"note: live pricing is ${price[0]:.2f}/${price[1]:.2f} per Mtok, but locate.py "
              f"assumes ${USD_PER_MTOK_IN:.2f}/${USD_PER_MTOK_OUT:.2f}. Costs below use the "
              f"assumed rates.", flush=True)

    files = pick(args.src, args.limit, args.stride)
    if not files:
        print(f"no images in {args.src}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "results.jsonl"

    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["stem"])
            except Exception:                                      # noqa: BLE001
                continue
    todo = [f for f in files if f.stem not in done]
    print(f"{len(files)} pages, {len(done)} already in {jsonl.name}, {len(todo)} to do, "
          f"{args.workers} workers", flush=True)
    if not todo:
        return 0

    records = {} if args.no_metadata else records_for([f.stem for f in todo],
                                                      args.metadata_csv)
    if not args.no_metadata:
        print(f"case records matched: {len(records)}/{len(todo)}", flush=True)

    prompt = load_prompt()
    agent = build_agent(prompt)
    model = resolve_model(args.model)

    started = time.time()
    n_ok = n_err = 0
    usd = 0.0
    # one shared file handle, appended under a lock: a worker finishing must not wait on
    # anything except the write itself
    import threading
    lock = threading.Lock()
    with jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(locate_one, agent, model, f, records.get(f.stem)): f
                   for f in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            with lock:
                fh.write(res.to_json() + "\n")
                fh.flush()
            if res.error:
                n_err += 1
                print(f"[{i}/{len(todo)}] {res.stem[:34]:34s} ERROR {res.error[:60]}",
                      flush=True)
                continue
            n_ok += 1
            usd += res.usd
            state = ("not-a-plan" if not res.is_plan
                     else "declined" if not res.found
                     else f"({res.point[0]},{res.point[1]})")
            warn = "" if res.point_in_range in (True, None) else "  OUT OF RANGE"
            print(f"[{i}/{len(todo)}] {res.stem[:34]:34s} {state:>14s}{warn}  "
                  f"{(res.what_marked_it or '')[:38]}", flush=True)

    elapsed = time.time() - started
    print(f"\n{n_ok} ok, {n_err} errors in {elapsed/60:.1f} min "
          f"({elapsed/max(1,len(todo)):.1f} s/page wall, {args.workers} workers)")
    print(f"spend this run: ${usd:.4f}  = GBP {usd/1.26:.4f}")

    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    if ok:
        summary = {
            "model": args.model,
            "metadata_sent": not args.no_metadata,
            "opencv_used": False,
            "pages": len(ok),
            "errors": len(rows) - len(ok),
            "not_a_plan": sum(1 for r in ok if r.get("is_plan") is False),
            "located": sum(1 for r in ok if r.get("found")),
            "declined_though_a_plan": sum(1 for r in ok
                                          if r.get("is_plan") and not r.get("found")),
            "point_out_of_range": sum(1 for r in ok if r.get("point_in_range") is False),
            "box_out_of_range": sum(1 for r in ok if r.get("box_in_range") is False),
            "usd_total": round(sum(r.get("usd", 0) for r in ok), 4),
            "usd_per_page": round(sum(r.get("usd", 0) for r in ok) / len(ok), 5),
        }
        summary["gbp_per_1000"] = round(summary["usd_per_page"] * 1000 / 1.26, 2)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print("\n=== summary ===")
        for k, v in summary.items():
            print(f"  {k:24s} {v}")

    if args.render:
        from render import render_all
        n = render_all(args.src, jsonl, args.out / "shots")
        print(f"\nrendered {n} overlays to {args.out / 'shots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
