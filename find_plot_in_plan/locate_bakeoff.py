#!/usr/bin/env python3
"""Which model actually finds the right plot? Scored against hand-drawn truth.

    python locate_bakeoff.py --out <dir> [--calls 2] [--workers 24]

Stage 1 alone, no tracing. The diagnosis that prompted this: the locate point falls inside the
true plot on 35 of 53 pages (66%) while the finished pipeline gets 34 right (64%) -- the tracer is
nearly lossless, so stage 1 sets the ceiling and is the only place worth spending on.

luna was chosen as the locator because it "measured the most reliable boxer". That measurement
used ring-to-ink metrics, which have since scored AUC 0.545 against truth, so the choice rests on
nothing. This re-runs it with a metric that means something: is the returned point inside the
polygon the project owner drew?

Results stream to JSONL, so an interrupted run resumes without paying twice.
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
import locate as L                                                # noqa: E402
from metadata import records_for                                  # noqa: E402

PANELS = Path("/data/braintree/scan-processed/scan-processed-wp7-multi-plan-panels/plan")
TRUTH = HERE / "labels" / "hand_labels.json"

# in / out USD per Mtok, as carried elsewhere in this package
MODELS = {
    "openai/gpt-5.6-luna":     (0.10, 0.60),      # today's locator
    "openai/gpt-5.6-terra":    (1.00, 6.00),      # today's tracer, never tried as locator
    "openai/gpt-5.6-luna-pro": (0.10, 0.60),
    "google/gemini-3.7-flash": (0.375, 1.875),    # the only model that ever redirected a tracer
}


def truth_rings() -> dict[str, list]:
    data = json.loads(TRUTH.read_text(encoding="utf-8"))
    return {x["stem"]: x["vertices_page"] for x in data["labels"]
            if x.get("status") == "drawn" and x.get("vertices_page")}


def point_in(ring, pt) -> bool | None:
    if not pt or list(pt) == [0, 0]:
        return None
    return bool(cv2.pointPolygonTest(np.asarray(ring, np.int32).reshape(-1, 1, 2),
                                     (float(pt[0]), float(pt[1])), False) >= 0)


def bbox(ring) -> list[int]:
    a = np.asarray(ring)
    return [int(a[:, 0].min()), int(a[:, 1].min()), int(a[:, 0].max()), int(a[:, 1].max())]


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def one(stem: str, model_name: str, call: int, agent, model, case, ring) -> dict:
    path = next(iter(PANELS.glob(f"{stem}.*")), None)
    if path is None:
        return {"stem": stem, "model": model_name, "call": call, "error": "no image"}
    r = L.locate_one(agent, model, path, case)
    cin, cout = MODELS[model_name]
    row = {"stem": stem, "model": model_name, "call": call,
           "is_plan": r.is_plan, "found": r.found, "point": r.point, "box": r.box,
           "confidence": r.confidence, "what_marked_it": r.what_marked_it,
           "seconds": r.seconds, "error": r.error,
           "usd": round(r.tokens_in * cin / 1e6 + r.tokens_out * cout / 1e6, 6)}
    if r.point and ring:
        row["point_in_truth"] = point_in(ring, r.point)
    if r.box and ring:
        row["box_iou_truth"] = round(box_iou(r.box, bbox(ring)), 4)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    rings = truth_rings()
    stems = sorted(rings)
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "locate.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["stem"], d["model"], d["call"]))

    cases = records_for(stems)
    agent = L.build_agent(L.load_prompt())
    jobs = [(s, m, c) for m in args.models for s in stems for c in range(args.calls)
            if (s, m, c) not in done]
    print(f"{len(stems)} pages with hand truth x {len(args.models)} models x {args.calls} calls "
          f"= {len(stems)*len(args.models)*args.calls}, {len(jobs)} left to run", flush=True)
    if not jobs:
        return 0

    resolved = {m: resolve_model(m) for m in args.models}
    lock = threading.Lock()
    t0 = time.time()
    with jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, s, m, c, agent, resolved[m], cases.get(s), rings[s]): (s, m, c)
                for s, m, c in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            hit = row.get("point_in_truth")
            mark = "IN " if hit else ("out" if hit is False else "-  ")
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.time()-t0:5.0f}s {row['model'][-18:]:18s} "
                      f"{mark} {row['stem'][:32]}", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
