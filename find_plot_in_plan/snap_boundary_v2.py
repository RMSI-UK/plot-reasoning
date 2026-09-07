#!/usr/bin/env python3
"""v2: an ensemble locator, because stage 1 is the only stage that matters.

    python snap_boundary_v2.py --out results/ [--pages stems.json] [--workers 16]

Measured against 54 hand-drawn boundaries (labels/hand_labels.json, 2026-09-04):

  the locate point lands inside the true plot on 35/53 pages (66%)
  the finished v1.x pipeline gets the right plot on         34/53 pages (64%)

The tracer is therefore nearly lossless and stage 1 sets the ceiling. Every prompt experiment this
project ran -- traps, topology, heavier-ink clauses -- worked on the tracer, which was not the
bottleneck. So v2 changes stage 1 and leaves the tracer alone.

Three locators, one call each, chosen by measuring them against the same truth:

  gpt-5.6-luna       38/54   GBP 0.70/1000    the incumbent, and genuinely the best single model
  gpt-5.6-luna-pro   38/54   GBP 2.92/1000
  gemini-3.7-flash   28/54   GBP 2.01/1000    weakest alone, but it fails on DIFFERENT pages
  gpt-5.6-terra      34/54   GBP 5.26/1000    dropped: worse than luna at seven times the price

Their boxes are clustered by IoU >= 0.5 and the largest cluster wins, highest confidence inside it
taking the point. That is 41/54 = 76% against luna's 70%, and it costs GBP 5.63 per 1000.

The clustering earns its place twice over, because whether the vote was unanimous is itself a
correctness signal available without any ground truth:

  unanimous  30 pages   83% correct   -> usable automatically
  split      24 pages   67% correct   -> a human should look

Nothing here is corrected automatically. `needs_human` says which pages to check.
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
from metadata import records_for                                  # noqa: E402

VERSION = "snap_boundary_v2"
PANELS = Path("/data/braintree/scan-processed/scan-processed-wp7-multi-plan-panels/plan")
TRUTH = HERE / "labels" / "hand_labels.json"

# model -> (USD per Mtok in, out). terra is excluded on measured accuracy, not on price.
LOCATORS = {
    "openai/gpt-5.6-luna":     (0.10, 0.60),
    "openai/gpt-5.6-luna-pro": (0.10, 0.60),
    "google/gemini-3.7-flash": (0.375, 1.875),
}
TRACE_MODEL, TR_IN, TR_OUT = "openai/gpt-5.6-terra", 1.00, 6.00
VOTE_IOU = 0.5                 # two boxes are "the same plot" at or above this

# Appended to the tracing prompt ONLY after a first attempt has declined. The tracing prompt asks
# for the whole curtilage "not just the building footprint"; on a page whose only marking is a
# building with no boundary drawn around its land, that leaves nothing it is allowed to trace.
# v2 shipped without this and took 9 declines against v1.1's 1, losing four pages v1.1 got right.
# It stays a retry rather than a default because a human judged this fallback WRONG on 93-00352,
# where a curtilage IS drawn and the real fault was the box.
BUILDING_FALLBACK = """

If the only thing on the page that marks the application is a building or structure -- stippled,
hatched, inked solid or drawn more heavily than its neighbours -- and no boundary is drawn around
the land it stands in, then trace that marked structure's own outline and say in what_marked_it
that no curtilage was drawn. An outline of the marked building is far more use than nothing. Do not
do this when a curtilage boundary IS drawn: there, the curtilage is the answer."""


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def vote(cands: list[dict]) -> tuple[dict | None, int, int]:
    """Largest cluster of mutually-overlapping boxes; the most confident member wins.

    A plain average would be wrong here: when two locators pick the plot and one picks the
    building next door, the mean box is a region no locator proposed.
    """
    n = len(cands)
    if n == 0:
        return None, 0, 0
    adj = [[j for j in range(n) if j != i and box_iou(cands[i]["box"], cands[j]["box"]) >= VOTE_IOU]
           for i in range(n)]
    seen: set[int] = set()
    best: list[int] = []
    for i in range(n):
        if i in seen:
            continue
        comp, stack = [], [i]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            stack += adj[k]
        if len(comp) > len(best):
            best = comp
    win = max(best, key=lambda i: (cands[i].get("confidence") or 0))
    return cands[win], len(best), n


def locate_all(agent, models, path: Path, case) -> list[dict]:
    out = []
    for name, (cin, cout) in LOCATORS.items():
        r = L.locate_one(agent, models[name], path, case)
        if r.error or not r.found or not r.box or not r.point:
            out.append({"model": name, "found": False, "error": r.error,
                        "usd": round(r.tokens_in*cin/1e6 + r.tokens_out*cout/1e6, 6)})
            continue
        out.append({"model": name, "found": True, "box": r.box, "point": r.point,
                    "confidence": r.confidence, "said": r.what_marked_it,
                    "usd": round(r.tokens_in*cin/1e6 + r.tokens_out*cout/1e6, 6),
                    "seconds": r.seconds})
    return out


def one_page(stem: str, agents, models, case, truth_ring) -> dict:
    a_loc, a_bnd = agents
    path = next(iter(PANELS.glob(f"{stem}.*")), None)
    if path is None:
        return {"stem": stem, "error": "no image"}
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"stem": stem, "error": "unreadable"}
    ph, pw = gray.shape
    row = {"stem": stem, "version": VERSION, "page_w": pw, "page_h": ph, "usd": 0.0}

    cands = locate_all(a_loc, models, path, case)
    row["locate"] = cands
    row["usd"] += sum(c.get("usd", 0) for c in cands)
    usable = [c for c in cands if c.get("found")]
    win_c, size, n = vote(usable)
    row["vote_cluster"] = size
    row["vote_total"] = n
    row["unanimous"] = bool(n > 1 and size == n)
    if win_c is None:
        row["skipped"] = "no locator found a plot"
        row["needs_human"] = True
        return row
    box, point = win_c["box"], win_c["point"]
    row.update({"box": box, "point": point, "locate_model": win_c["model"],
                "locate_said": win_c.get("said"),
                "locate_confidence": win_c.get("confidence")})

    win = B.window_for(box, pw, ph)
    row["window"] = [win.x, win.y, win.w, win.h]
    row["scale"] = round(win.scale, 3)
    block = case.as_prompt_block() if case else ""
    res = B.trace_one(a_bnd, models[TRACE_MODEL], gray, stem, win, block)
    row["usd"] = round(row["usd"] + res.tokens_in*TR_IN/1e6 + res.tokens_out*TR_OUT/1e6, 6)
    # The tracer's own confidence was being dropped here. Measured on the 78 v1.x rings that did
    # record it, it scores AUC 0.674 against IoU>=0.7 -- better than every ink metric in the
    # project (ring_to_ink_p90 is 0.545, frac_within_3px 0.467) and, unlike the locators', it is
    # not saturated: median 0.78, only 18% above 0.90, and monotone against truth.
    row.update({"traced": res.found, "said": res.said, "trace_seconds": res.seconds,
                "trace_confidence": res.confidence})
    if res.error:
        row["error"] = f"trace: {res.error}"
        row["needs_human"] = True
        return row
    if not res.vertices_page:
        row["declined_first"] = True
        row["declined_said"] = res.said
        res = B.trace_one(a_bnd, models[TRACE_MODEL], gray, stem, win, block,
                          retry_note=BUILDING_FALLBACK)
        row["usd"] = round(row["usd"] + res.tokens_in*TR_IN/1e6 + res.tokens_out*TR_OUT/1e6, 6)
        row["fallback_said"] = res.said
        if not res.vertices_page:
            row["declined_trace"] = True
            row["needs_human"] = True
            return row
        row["source"] = "building fallback"
        row["said"] = res.said

    row["vertices_page"] = res.vertices_page
    dt = B.ink_distance(gray)
    row.update(B.ring_stats(res.vertices_page, dt))
    row["edges"] = B.edge_contact(res.vertices_page, win, pw, ph)
    row["point_inside_ring"] = B.point_inside_ring(point, res.vertices_page)

    if truth_ring:                                # scoring only; never fed back into the pipeline
        row["iou_truth"] = round(mask_iou(truth_ring, res.vertices_page, pw, ph), 4)
    # the flag count was the strongest correctness signal measured (AUC 0.861), so it stays
    flags = B.review_flags(row, row["edges"], box_agree=None) or []
    if not row["unanimous"]:
        flags.insert(0, f"locators split {size}/{n} on which plot this is")
    if row["point_inside_ring"] is False:
        flags.append("the ring does not contain the locator's point")
    row["flags"] = flags
    if row.get("source") == "building fallback":
        flags.append("traced the marked building; no curtilage was drawn")
    row["flags"] = flags
    row["needs_human"] = bool(flags)
    row["auto_usable"] = not flags
    return row


def mask_iou(a, b, w: int, h: int) -> float:
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.asarray(a, np.int32).reshape(-1, 1, 2)], 1)
    cv2.fillPoly(mb, [np.asarray(b, np.int32).reshape(-1, 1, 2)], 1)
    u = np.count_nonzero(ma | mb)
    return np.count_nonzero(ma & mb) / u if u else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pages", type=Path, default=None)
    args = ap.parse_args()

    truth = {}
    if TRUTH.exists():
        truth = {x["stem"]: x["vertices_page"]
                 for x in json.loads(TRUTH.read_text())["labels"]
                 if x.get("status") == "drawn" and x.get("vertices_page")}
    stems = json.loads(args.pages.read_text()) if args.pages else sorted(truth)
    # a blind evaluation set has no truth yet; iou_truth is simply absent for those pages

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "boundaries.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["stem"])
    todo = [s for s in stems if s not in done]
    print(f"{len(stems)} pages, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return 0

    cases = records_for(todo)
    agents = (L.build_agent(L.load_prompt()), B.build_agent(B.load_prompt()))
    models = {m: resolve_model(m) for m in list(LOCATORS) + [TRACE_MODEL]}
    lock = threading.Lock()
    t0 = time.time()
    rows = []
    with jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one_page, s, agents, models, cases.get(s), truth.get(s)): s
                for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            iou = row.get("iou_truth")
            print(f"[{i}/{len(todo)}] {time.time()-t0:5.0f}s {row['stem'][:34]:34s} "
                  f"vote {row.get('vote_cluster','-')}/{row.get('vote_total','-')} "
                  f"IoU {iou if iou is not None else '  -  '}  "
                  f"{'AUTO' if row.get('auto_usable') else 'review'}", flush=True)

    scored = [r for r in rows if r.get("iou_truth") is not None]
    good = [r for r in scored if r["iou_truth"] >= 0.5]
    summary = {
        "version": VERSION, "locators": list(LOCATORS), "trace_model": TRACE_MODEL,
        "pages": len(rows), "traced": sum(1 for r in rows if r.get("vertices_page")),
        "scored_against_truth": len(scored),
        "right_plot": len(good),
        "right_plot_pct": round(len(good)/len(scored)*100, 1) if scored else None,
        "median_iou": round(float(np.median([r["iou_truth"] for r in scored])), 4) if scored else None,
        "wrong_plot_iou_under_0.1": sum(1 for r in scored if r["iou_truth"] < 0.1),
        "unanimous": sum(1 for r in rows if r.get("unanimous")),
        "trace_confidence_median": (round(float(np.median(
            [r["trace_confidence"] for r in rows if r.get("trace_confidence") is not None])), 3)
            if any(r.get("trace_confidence") is not None for r in rows) else None),
        "fallback_tried": sum(1 for r in rows if r.get("declined_first")),
        "fallback_rescued": sum(1 for r in rows if r.get("source") == "building fallback"),
        "auto_usable": sum(1 for r in rows if r.get("auto_usable")),
        "needs_human": sum(1 for r in rows if r.get("needs_human")),
        "usd_per_page": round(float(np.mean([r.get("usd", 0) for r in rows])), 5),
        "wall_seconds": round(time.time()-t0, 1),
    }
    summary["gbp_per_1000"] = round(summary["usd_per_page"]*1000/1.26, 2)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
