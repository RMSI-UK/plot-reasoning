#!/usr/bin/env python3
"""Plan scan in, plot boundary and a calibrated confidence out.

    python extract_boundary.py page.jpg                     # one page, JSON to stdout
    python extract_boundary.py page.jpg --overlay out.png   # ...and a picture to check it by
    python extract_boundary.py --src panels/ --out results/ --workers 12
    python extract_boundary.py --score-only results/        # recompute confidence, no API calls

THE PIPELINE

    1  locate     three models are asked, on the whole sheet, where the application plot is:
                  gpt-5.6-luna, gpt-5.6-luna-pro and gemini-3.7-flash. Their boxes are clustered
                  by IoU >= 0.5 and the largest cluster wins.
    2  window     a crop that CONTAINS the winning box, padded, enlarged to 1024 px and never
                  scaled below 1.0.
    3  trace      gpt-5.6-terra draws the boundary on that crop, in crop pixels, mapped back.
                  If it declines and the page's only marking is a building, one retry allows the
                  building outline instead.
    4  check      seven independent checks, none of which needs an answer key.
    5  score      a calibrated probability that the ring is the right plot to within IoU 0.7.

WHAT IT IS WORTH, and how that was established

    Measured against 159 boundaries drawn by hand, 100 of them on pages drawn BLIND -- a random
    sample of pages the pipeline had never seen, hand-drawn before it ran, with no pipeline output
    anywhere in the annotation tool.

        right plot, IoU >= 0.7      blind 60% [49-70]      combined 57% [48-65]
        median IoU where traced     blind 0.837
        AUTO lane (zero flags)      blind 91% precision over 34% of pages
        cost                        about GBP 15 per 1000 pages

    The earlier 62-page set was biased optimistic by about 8 points because it contained pages
    picked by eye for being hard; quote the blind or combined figure.

WHAT IT IS NOT

    Two of five pages are wrong at IoU 0.7, and roughly one in five is a completely different
    parcel. This is a drafting tool with a reliable high-confidence lane, not an extractor. The
    confidence is what makes it usable: it separates the 34% you can ship from the rest.

    Three attempts to raise the accuracy all failed and are documented rather than deleted:
    a geometric line-network partition (much worse), a subdivision second pass (one page), and
    asking a model to choose between the locators' own boxes (nothing, even though a correct box
    is present 84% of the time). See geometry/optimise/index.html.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/env/code/plot-reasoning/GeoPlanAgent")
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402
from dotenv import load_dotenv                                    # noqa: E402

load_dotenv(str(REPO / ".env"))

from geoplanagent.utils import resolve_model                      # noqa: E402
import boundary as B                                              # noqa: E402
import locate as L                                                # noqa: E402
import snap_boundary_v2 as V2                                     # noqa: E402
from reliability import featurise, FEATURES                       # noqa: E402
from metadata import records_for                                  # noqa: E402

VERSION = "extract_boundary_1.0"
MODEL_PATH = HERE / "reliability_model.json"

# A 402 or a 401 is not a page that has no plot on it. The last batch run lost 141 of 159 pages to
# an exhausted account and wrote them out as ordinary declines, producing a summary that read like
# a result. Billing and auth failures now stop the run.
FATAL = ("status_code: 401", "status_code: 402", "status_code: 403",
         "exceed your available credits", "Insufficient credits", "No auth credentials")


class BillingError(RuntimeError):
    pass


def fatal_reason(text: str | None) -> str | None:
    if not text:
        return None
    return next((m for m in FATAL if m in text), None)


def load_scorer():
    """The reliability model, or None if it has not been fitted yet."""
    if not MODEL_PATH.exists():
        return None
    m = json.loads(MODEL_PATH.read_text())
    if m.get("features") != FEATURES:
        print(f"note: {MODEL_PATH.name} was fitted on different features; ignoring it",
              file=sys.stderr)
        return None
    return m


def score_row(row: dict, model) -> dict:
    """Attach a calibrated probability that this ring is the right plot (IoU >= 0.7)."""
    if not row.get("vertices_page"):
        row["confidence"] = 0.0
        row["confidence_band"] = "none"
        return row
    if model is None:
        # fall back on the single strongest signal measured: the review-flag count, AUC 0.719
        # held out, which beat a fifteen-feature fit at 0.703
        row["confidence"] = 0.85 if not row.get("flags") else 0.35
        row["confidence_band"] = "high" if not row.get("flags") else "low"
        row["confidence_source"] = "flag count (no fitted model on disk)"
        return row
    f = featurise(row)
    x = np.array([f[k] for k in model["features"]], float)
    z = (x - np.array(model["mean"])) / np.array(model["scale"])
    p = 1.0 / (1.0 + np.exp(-(float(np.dot(z, model["coef"])) + model["intercept"])))
    row["confidence"] = round(float(p), 4)
    # bands come from the held-out calibration table, not from round numbers
    row["confidence_band"] = "high" if p >= 0.6 else "medium" if p >= 0.4 else "low"
    row["confidence_source"] = f"logistic fit, held-out AUC {model.get('auc_heldout')}"
    return row


def overlay(gray: np.ndarray, row: dict, path: Path) -> None:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    th = max(2, int(round(max(h, w) / 700)))
    if row.get("vertices_page"):
        cv2.polylines(vis, [np.asarray(row["vertices_page"], np.int32).reshape(-1, 1, 2)],
                      True, (0, 0, 235), th, cv2.LINE_AA)
    pt = row.get("point")
    if pt and list(pt) != [0, 0]:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), th*5, (200, 200, 0), th, cv2.LINE_AA)
    band = row.get("confidence_band", "?")
    txt = f"{row.get('confidence', 0):.2f} {band}"
    fs = max(0.6, max(h, w)/1400)
    col = {"high": (0, 150, 0), "medium": (0, 150, 220), "low": (0, 0, 220)}.get(band, (90, 90, 90))
    (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(vis, (6, 6), (16+tw, 16+tht), col, -1)
    cv2.putText(vis, txt, (11, 11+tht), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def public(row: dict) -> dict:
    """The answer, without the working."""
    return {
        "stem": row.get("stem"), "version": VERSION,
        "found": bool(row.get("vertices_page")),
        "vertices_page": row.get("vertices_page"),
        "confidence": row.get("confidence"),
        "confidence_band": row.get("confidence_band"),
        "auto_usable": bool(row.get("auto_usable")),
        "flags": row.get("flags") or [],
        "n_vertices": row.get("n_vertices"),
        "area_pct": row.get("area_pct"),
        "locators_agreed": f"{row.get('vote_cluster')}/{row.get('vote_total')}",
        "traced_by": V2.TRACE_MODEL,
        "located_by": row.get("locate_model"),
        "what_it_traced": row.get("said"),
        "why_it_was_that_plot": row.get("locate_said"),
        "usd": row.get("usd"),
        "error": row.get("error") or row.get("skipped"),
    }


def run_one(stem_path: Path, agents, models, case, scorer) -> dict:
    row = V2.one_page(stem_path.stem, agents, models, case, None)
    for c in (row.get("locate") or []):
        why = fatal_reason(c.get("error"))
        if why:
            raise BillingError(why)
    why = fatal_reason(row.get("error"))
    if why:
        raise BillingError(why)
    return score_row(row, scorer)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", type=Path, help="a single plan scan")
    ap.add_argument("--src", type=Path, help="a directory of scans")
    ap.add_argument("--out", type=Path, help="output directory (batch) or JSON file (single)")
    ap.add_argument("--overlay", type=Path, help="write a picture of the result here")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-metadata", action="store_true",
                    help="do not send the council's address and proposal")
    ap.add_argument("--score-only", type=Path,
                    help="recompute confidence over an existing boundaries.jsonl; no API calls")
    args = ap.parse_args()

    scorer = load_scorer()
    if scorer:
        print(f"confidence: logistic fit, held-out AUC {scorer['auc_heldout']}, "
              f"fitted on {scorer['n_fit']} pages, validated on {scorer['n_validate']}",
              file=sys.stderr)
    else:
        print("confidence: no fitted model on disk, falling back to the flag count",
              file=sys.stderr)

    if args.score_only:
        p = args.score_only
        p = p / "boundaries.jsonl" if p.is_dir() else p
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        out = [score_row(r, scorer) for r in rows]
        dst = p.with_name("scored.jsonl")
        dst.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in out))
        band = {}
        for r in out:
            band[r.get("confidence_band")] = band.get(r.get("confidence_band"), 0) + 1
        print(f"scored {len(out)} rows -> {dst}")
        for k in ("high", "medium", "low", "none"):
            if k in band:
                print(f"   {k:7s} {band[k]:4d}  ({band[k]/len(out)*100:.0f}%)")
        return 0

    if not args.image and not args.src:
        ap.error("give an image, or --src, or --score-only")

    files = ([args.image] if args.image
             else sorted(f for f in args.src.iterdir()
                         if f.suffix.lower() in (".jpg", ".jpeg", ".png")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("no images found")
        return 1

    cases = {} if args.no_metadata else records_for([f.stem for f in files])
    agents = (L.build_agent(L.load_prompt()), B.build_agent(B.load_prompt()))
    models = {m: resolve_model(m) for m in list(V2.LOCATORS) + [V2.TRACE_MODEL]}

    V2.PANELS = files[0].parent          # one_page globs for the stem in this directory

    if args.image:
        try:
            row = run_one(args.image, agents, models, cases.get(args.image.stem), scorer)
        except BillingError as e:
            print(f"\nSTOPPED: the API rejected the request -- {e}\n"
                  f"This is a billing or credentials problem, not a page without a plot on it.\n",
                  file=sys.stderr)
            return 2
        ans = public(row)
        if args.overlay:
            g = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
            overlay(g, row, args.overlay)
            print(f"overlay -> {args.overlay}", file=sys.stderr)
        text = json.dumps(ans, indent=1, ensure_ascii=False)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
            print(f"-> {args.out}", file=sys.stderr)
        else:
            print(text)
        return 0

    out_dir = args.out or Path("boundaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "boundaries.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["stem"])
    todo = [f for f in files if f.stem not in done]
    print(f"{len(files)} pages, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return 0

    lock = threading.Lock()
    t0 = time.time()
    rows, stop = [], []
    with jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, f, agents, models, cases.get(f.stem), scorer): f
                for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except BillingError as e:
                if not stop:
                    stop.append(str(e))
                    for g in futs:
                        g.cancel()
                continue
            except Exception as e:                                 # noqa: BLE001
                row = {"stem": futs[fut].stem, "error": f"{e!s:.200}"}
            rows.append(row)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False)+"\n")
                fh.flush()
            if not stop:
                print(f"[{i}/{len(todo)}] {time.time()-t0:5.0f}s {row['stem'][:34]:34s} "
                      f"conf {row.get('confidence', 0):.2f} {row.get('confidence_band','-'):6s}"
                      f" {'AUTO' if row.get('auto_usable') else 'review'}", flush=True)

    if stop:
        print(f"\nSTOPPED after {len(rows)} pages: the API rejected requests -- {stop[0]}\n"
              f"This is a billing or credentials problem. {len(rows)} good rows are in {jsonl};\n"
              f"add credits and re-run the same command to continue from there.\n", file=sys.stderr)
        return 2

    ok = [r for r in rows if r.get("vertices_page")]
    hi = [r for r in rows if r.get("confidence_band") == "high"]
    summary = {
        "version": VERSION, "pages": len(rows), "traced": len(ok),
        "high_confidence": len(hi), "high_confidence_share": round(len(hi)/max(1, len(rows)), 3),
        "auto_usable": sum(1 for r in rows if r.get("auto_usable")),
        "median_confidence": (round(float(np.median([r["confidence"] for r in ok])), 3)
                              if ok else None),
        "usd_per_page": round(float(np.mean([r.get("usd", 0) for r in rows])), 5),
        "wall_seconds": round(time.time()-t0, 1),
        "expected_precision_high_lane": "about 88-91%, measured blind on 97 pages",
    }
    summary["gbp_per_1000"] = round(summary["usd_per_page"]*1000/1.26, 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
