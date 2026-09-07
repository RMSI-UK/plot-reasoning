#!/usr/bin/env python3
"""Batch runner: locate the plot, then trace its boundary.

    python run_boundary.py --src /path/to/panels --out results/
    python run_boundary.py --src ... --limit 20 --stride 37
    python run_boundary.py --src ... --no-regrow          # skip the targeted second crop

Per page:
  1 locate  gpt-5.6-luna, whole page, with the council's record   -> point + box
  2 window  contains the box, padded, never scaled below 1.0      -> crop
  3 trace   gpt-5.6-luna-pro on the crop                          -> vertices in page coordinates
  4 regrow  if the ring ran along a crop edge that is inside the page, widen ON THOSE SIDES ONLY
            and trace once more. The second result is kept only if it EXTENDS the first.
  5 flags   which pages a human should look at. Nothing is corrected.

Results stream to JSONL, one line per page, so an interrupted run resumes without paying twice.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/env/code/plot-reasoning/GeoPlanAgent")
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402
from dotenv import load_dotenv                                    # noqa: E402

load_dotenv(str(REPO / ".env"))

from geoplanagent.utils import resolve_model                      # noqa: E402
import boundary as B                                              # noqa: E402
import locate as L                                                # noqa: E402
from metadata import records_for                                  # noqa: E402

LOCATE_MODEL = "openai/gpt-5.6-luna"     # measured the most reliable boxer, and the cheapest
LOCATE_IN, LOCATE_OUT = 0.10, 0.60


def one_page(path: Path, agents, models, case, do_regrow: bool,
             locate_twice: bool = True) -> dict:
    a_loc, a_bnd = agents
    m_loc, m_bnd = models
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"stem": path.stem, "error": "unreadable image"}
    ph, pw = gray.shape
    row: dict = {"stem": path.stem, "page_w": pw, "page_h": ph, "usd": 0.0}

    loc = L.locate_one(a_loc, m_loc, path, case)
    loc2 = L.locate_one(a_loc, m_loc, path, case) if locate_twice else None
    # locate.py prices with terra's rates; re-price for the model actually used
    loc.usd = round(loc.tokens_in * LOCATE_IN / 1e6 + loc.tokens_out * LOCATE_OUT / 1e6, 6)
    row["usd"] += loc.usd
    row.update({"is_plan": loc.is_plan, "plan_kind": loc.plan_kind, "located": loc.found,
                "box": loc.box, "point": loc.point, "locate_said": loc.what_marked_it,
                "locate_seconds": loc.seconds})
    if loc.error:
        row["error"] = f"locate: {loc.error}"
        return row
    if not (loc.found and loc.box):
        row["skipped"] = "stage 1 found no plot"
        return row

    box = loc.box
    if loc2 is not None:
        loc2.usd = round(loc2.tokens_in * LOCATE_IN / 1e6 + loc2.tokens_out * LOCATE_OUT / 1e6, 6)
        row["usd"] += loc2.usd
        row["box_2"] = loc2.box
        if loc2.found and loc2.box:
            box, agree, ok2 = B.reconcile_boxes(loc.box, loc2.box)
            row["box_agree"] = agree
            row["box_used"] = box
        else:
            row["box_agree"] = 0.0            # one call found a plot and the other did not
    win = B.window_for(box, pw, ph)
    row["window"] = [win.x, win.y, win.w, win.h]
    row["scale"] = round(win.scale, 3)
    row["window_contains_box"] = B.window_contains_box(win, box)
    block = case.as_prompt_block() if case else ""

    res = B.trace_one(a_bnd, m_bnd, gray, path.stem, win, block)
    row["usd"] = round(row["usd"] + res.usd, 6)
    row.update({"traced": res.found, "said": res.said, "confidence": res.confidence,
                "trace_seconds": res.seconds})
    if res.error:
        row["error"] = f"trace: {res.error}"
        return row
    if not res.vertices_page:
        row["declined_trace"] = True
        return row

    dt = B.ink_distance(gray)
    row["vertices_page"] = res.vertices_page
    row.update(B.ring_stats(res.vertices_page, dt))
    row["edges"] = B.edge_contact(res.vertices_page, win, pw, ph)
    row["source"] = "first crop"

    if do_regrow and row["edges"]["cut_sides"]:
        win2 = B.grow_window(win, row["edges"]["cut_sides"], pw, ph)
        res2 = B.trace_one(a_bnd, m_bnd, gray, path.stem, win2, block)
        row["usd"] = round(row["usd"] + res2.usd, 6)
        row["regrow"] = {"window": [win2.x, win2.y, win2.w, win2.h],
                         "scale": round(win2.scale, 3), "said": res2.said,
                         "error": res2.error}
        after = (B.ring_stats(res2.vertices_page, dt) if res2.vertices_page else None)
        keep, why = B.accept_regrow(row, after, res.vertices_page, res2.vertices_page, pw, ph)
        row["regrow"]["verdict"] = why
        row["regrow"]["accepted"] = keep
        if after:
            row["regrow"].update(after)
        if keep:
            row["vertices_page"] = res2.vertices_page
            row.update(after)
            row["edges"] = B.edge_contact(res2.vertices_page, win2, pw, ph)
            row["source"] = "widened crop"

    row["flags"] = B.review_flags(row, row["edges"], box_agree=row.get("box_agree"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "boundaries")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pages", type=Path, default=None,
                    help="JSON list of stems to run, instead of scanning --src")
    ap.add_argument("--no-metadata", action="store_true")
    ap.add_argument("--no-regrow", action="store_true")
    ap.add_argument("--trace-model", default=B.MODEL,
                    help=f"model for the tracing step (default {B.MODEL})")
    ap.add_argument("--locate-once", action="store_true",
                    help="skip the second locate call and its disagreement check "
                         "(saves GBP 0.61 per 1000, loses the only guard on stage 1)")
    args = ap.parse_args()

    if args.pages:
        stems = json.loads(args.pages.read_text())
        files = [args.src / f"{s}.jpg" for s in stems]
    else:
        files = sorted(args.src.glob("*.jpg")) + sorted(args.src.glob("*.png"))
        files = files[::args.stride] if args.stride > 1 else files
        files = files[:args.limit] if args.limit else files
    if not files:
        print(f"no images for {args.src}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "boundaries.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["stem"])
            except Exception:                                     # noqa: BLE001
                continue
    todo = [f for f in files if f.stem not in done]
    print(f"{len(files)} pages, {len(done)} already done, {len(todo)} to run, "
          f"{args.workers} workers", flush=True)
    if not todo:
        return 0

    records = {} if args.no_metadata else records_for([f.stem for f in todo])
    agents = (L.build_agent(L.load_prompt()), B.build_agent(B.load_prompt()))
    models = (resolve_model(LOCATE_MODEL), resolve_model(args.trace_model))
    if args.trace_model != B.MODEL:
        # boundary.py prices luna-pro; re-price so the totals mean something
        rates = {"openai/gpt-5.6-terra": (1.00, 6.00), "openai/gpt-5.6-luna": (0.10, 0.60),
                 "openai/gpt-5.6-luna-pro": (0.10, 0.60), "openai/gpt-5.4-mini": (0.25, 2.00)}
        if args.trace_model in rates:
            B.USD_PER_MTOK_IN, B.USD_PER_MTOK_OUT = rates[args.trace_model]
        else:
            print(f"note: no price on file for {args.trace_model}; costs below use luna-pro rates")

    lock = threading.Lock()
    started = time.time()
    rows = []
    with jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one_page, f, agents, models, records.get(f.stem),
                            not args.no_regrow, not args.locate_once): f
                for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if row.get("error"):
                state = f"ERROR {row['error'][:50]}"
            elif row.get("skipped") or row.get("declined_trace"):
                state = "declined"
            else:
                state = (f"{row['n_vertices']:3d}v p90 {row['ring_to_ink_p90_px']:6.2f} "
                         f"blank {row['blank_share']*100:3.0f}% "
                         f"{'[' + row['source'] + ']' if row.get('regrow') else ''}")
            print(f"[{i}/{len(todo)}] {row['stem'][:34]:34s} {state}  "
                  f"{'; '.join(row.get('flags') or []) [:60]}", flush=True)

    ok = [r for r in rows if r.get("vertices_page")]
    M = lambda k: (round(float(np.median([r[k] for r in ok if r.get(k) is not None])), 3)
                   if ok else None)
    summary = {
        "locate_model": LOCATE_MODEL, "trace_model": args.trace_model,
        "pages": len(rows), "located": sum(1 for r in rows if r.get("located")),
        "traced": len(ok),
        "window_contains_box": sum(1 for r in rows if r.get("window_contains_box")),
        "never_downscaled": all(r.get("scale", 1) >= 1.0 for r in rows),
        "locate_disagreed": sum(1 for r in rows if (r.get("box_agree") is not None
                                                    and r["box_agree"] < 0.5)),
        "box_agree_median": (round(float(np.median([r["box_agree"] for r in rows
                                                    if r.get("box_agree") is not None])), 3)
                             if any(r.get("box_agree") is not None for r in rows) else None),
        "regrow_fired": sum(1 for r in rows if r.get("regrow")),
        "regrow_accepted": sum(1 for r in rows if (r.get("regrow") or {}).get("accepted")),
        "n_vertices": M("n_vertices"), "ring_to_ink_median_px": M("ring_to_ink_median_px"),
        "ring_to_ink_p90_px": M("ring_to_ink_p90_px"), "frac_within_3px": M("frac_within_3px"),
        "blank_share": M("blank_share"), "area_pct": M("area_pct"),
        "pages_clean": sum(1 for r in ok if not r.get("flags")),
        "pages_flagged": sum(1 for r in ok if r.get("flags")),
        "usd_per_page": round(float(np.mean([r.get("usd", 0) for r in rows])), 5),
    }
    summary["gbp_per_1000"] = round(summary["usd_per_page"] * 1000 / 1.26, 2)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n=== summary ({time.time()-started:.0f}s) ===")
    for k, v in summary.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
