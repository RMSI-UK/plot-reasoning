#!/usr/bin/env python3
"""Second pass: cut an over-reaching ring down to the one property it should have been.

    python subdivide.py --out <dir> [--workers 12]

The failure this attacks, measured on 54 hand-drawn pages:

    right plot, IoU >= 0.7                  28   52%
    DISEASE 1  wrong plot, locate missed    13   24%   median IoU 0.048
    DISEASE 2  right plot, wrong extent     15   28%   median IoU 0.447
               of which TOO BIG 10, TOO SMALL 3
               median over-reach 2.03x, and recall ~1.00 -- the true plot is INSIDE the ring

Disease 2 is the bigger half and it points one way: the tracer takes a terrace, a close or a pair
of semis and returns the block instead of the strip. Because the true plot is contained in what it
returned, the repair is a subdivision, not a search.

The geometric route was tried first and failed outright: seeding cand_vector's line-network
partition with the locate point moved the median IoU from 0.846 to 0.104 and cut IoU>=0.7 from 26
to 8, because the face holding the seed is one compartment of the plot -- a garden, a drive -- not
the plot. So this pass is semantic instead: it re-crops to the ring the tracer produced, says that
the region may hold more than one property, and asks for only the one containing the identity
point. A model can read a house number and a party boundary; a skeleton cannot.

It runs on every page, not only the suspect ones, because a repair that silently damages the 28
already-correct pages is not a repair.
"""
from __future__ import annotations

import argparse
import io
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

from PIL import Image                                             # noqa: E402
from pydantic_ai import Agent, BinaryContent, NativeOutput        # noqa: E402
from pydantic_ai.usage import UsageLimits                         # noqa: E402
from geoplanagent.utils import resolve_model                      # noqa: E402
import boundary as B                                              # noqa: E402
from metadata import records_for                                  # noqa: E402

GEO = Path("/data/braintree/scan-processed/wp7-panels-sam3-seg/geometry")
SRC = Path("/data/braintree/scan-processed/scan-processed-wp7-multi-plan-panels/plan")
MODEL, IN_USD, OUT_USD = "openai/gpt-5.6-terra", 1.00, 6.00
MARGIN = 0.22          # of the ring's longer side, so the dividing lines just outside are visible
TARGET = 1100          # long side sent, never below 1.0 scale

PROMPT = """You are shown a crop of a scanned UK planning drawing. A first pass has already found
the application site and drawn a boundary on it, shown as the RED outline in the second image. That
outline is known to be reliable about WHERE the site is and unreliable about HOW MUCH land it
covers. Its usual error is to take in the neighbours.

The CYAN dot marks a point inside the property the application actually concerns.

Your job: decide whether the red outline encloses ONE property or SEVERAL, and return the boundary
of the single property containing the cyan dot.

Look for the lines that divide one property from the next: the fence, wall or hedge lines running
back from the street between houses; the line between a pair of semi-detached houses; the edge of a
shared drive. In a terrace or a close, each property is a narrow strip running back from the street
frontage, and the red outline has often swallowed two or three of those strips.

If the red outline already encloses exactly one property, say so and return the same boundary,
following the drawn line closely. Do not shrink a correct answer.

Never cut along something that is not drawn. If the only division you can see is imaginary, keep the
outline as it is and say why. A house number written inside a strip belongs to that strip.

Return ordered vertices in the pixel coordinates of THIS crop, 8 to 40 of them, on the drawn line."""


def annotate(crop, ring_xy, point_xy):
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop.copy()
    th = max(2, int(round(max(vis.shape[:2]) / 500)))
    cv2.polylines(vis, [np.asarray(ring_xy, np.int32).reshape(-1, 1, 2)], True,
                  (0, 0, 235), th, cv2.LINE_AA)
    if point_xy is not None:
        cv2.circle(vis, (int(point_xy[0]), int(point_xy[1])), th * 5, (235, 235, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(point_xy[0]), int(point_xy[1])), th * 5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def one(stem, row, agent, model, case, truth_ring):
    gray = cv2.imread(str(next(iter(SRC.glob(f"{stem}.*")))), cv2.IMREAD_GRAYSCALE)
    ph, pw = gray.shape
    ring = np.asarray(row["vertices_page"], np.int32)
    x0, y0 = ring[:, 0].min(), ring[:, 1].min()
    x1, y1 = ring[:, 0].max(), ring[:, 1].max()
    m = int(MARGIN * max(x1 - x0, y1 - y0)) + 20
    cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
    cx1, cy1 = min(pw, x1 + m), min(ph, y1 + m)
    patch = gray[cy0:cy1, cx0:cx1]
    scale = max(1.0, TARGET / max(patch.shape))
    if scale > 1.0:
        patch = cv2.resize(patch, (int(patch.shape[1]*scale), int(patch.shape[0]*scale)),
                           interpolation=cv2.INTER_CUBIC)
    to_crop = lambda p: [(p[0]-cx0)*scale, (p[1]-cy0)*scale]
    ring_c = [to_crop(p) for p in row["vertices_page"]]
    pt = row.get("point")
    pt_c = to_crop(pt) if pt and list(pt) != [0, 0] else None

    bufs = []
    for img in (patch, annotate(patch, ring_c, pt_c)):
        b = io.BytesIO()
        Image.fromarray(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(b, "PNG")
        bufs.append(b.getvalue())

    out = {"stem": stem}
    t0 = time.time()
    try:
        run = agent.run_sync(
            [BinaryContent(data=bufs[0], media_type="image/png"),
             BinaryContent(data=bufs[1], media_type="image/png"),
             f"This crop is {patch.shape[1]} x {patch.shape[0]} pixels."
             + (case.as_prompt_block() if case else "")],
            model=model, usage_limits=UsageLimits(request_limit=3))
    except Exception as exc:                                       # noqa: BLE001
        out["error"] = f"{exc!s:.200}"
        return out
    o, u = run.output, run.usage()
    out["seconds"] = round(time.time()-t0, 1)
    out["usd"] = round((u.input_tokens or 0)*IN_USD/1e6 + (u.output_tokens or 0)*OUT_USD/1e6, 6)
    out["said"] = o.what_marked_it
    if o.found and len(o.vertices) >= 3:
        out["vertices_page"] = [[int(round(v.x/scale + cx0)), int(round(v.y/scale + cy0))]
                                for v in o.vertices]
    if truth_ring and out.get("vertices_page"):
        def fill(r):
            mm = np.zeros((ph, pw), np.uint8)
            cv2.fillPoly(mm, [np.asarray(r, np.int32).reshape(-1, 1, 2)], 1)
            return mm
        a, b = fill(truth_ring), fill(out["vertices_page"])
        uu = np.count_nonzero(a | b)
        out["iou_after"] = round(np.count_nonzero(a & b)/uu, 4) if uu else 0.0
        c = fill(row["vertices_page"])
        uu2 = np.count_nonzero(a | c)
        out["iou_before"] = round(np.count_nonzero(a & c)/uu2, 4) if uu2 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    lab = json.loads((HERE/"labels"/"hand_labels.json").read_text())["labels"]
    truth = {x["stem"]: x for x in lab}
    v2 = {}
    for d_ in ("snapv2", "snapv2_noboundary"):
        for line in (GEO/d_/"boundaries.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                v2[r["stem"]] = r
    stems = [s for s, t in truth.items()
             if s in v2 and v2[s].get("vertices_page")
             and t["status"] == "drawn" and "v2" not in str(t.get("ring_source", ""))]

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out/"subdivide.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["stem"])
    todo = [s for s in stems if s not in done]
    print(f"{len(stems)} pages, {len(todo)} to run", flush=True)
    if not todo:
        return 0
    cases = records_for(todo)
    agent = Agent("test", output_type=NativeOutput(B.BoundaryAnswer), retries=1, output_retries=0,
                  model_settings={"max_tokens": 6000, "timeout": 200}, instructions=PROMPT)
    model = resolve_model(MODEL)
    lock = threading.Lock()
    t0 = time.time()
    rows = []
    with jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, s, v2[s], agent, model, cases.get(s),
                            truth[s].get("vertices_page")): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False)+"\n")
                fh.flush()
            b, a = r.get("iou_before"), r.get("iou_after")
            d = (f"{b:.3f} -> {a:.3f}" if b is not None and a is not None
                 else r.get("error", "declined")[:40])
            print(f"[{i}/{len(todo)}] {time.time()-t0:5.0f}s {r['stem'][:34]:34s} {d}", flush=True)

    sc = [r for r in rows if r.get("iou_after") is not None]
    if sc:
        b = [r["iou_before"] for r in sc]
        a = [r["iou_after"] for r in sc]
        print(f"\n  n={len(sc)}   median {np.median(b):.3f} -> {np.median(a):.3f}")
        print(f"  IoU>=0.7  {sum(1 for x in b if x >= .7)} -> {sum(1 for x in a if x >= .7)}")
        print(f"  better {sum(1 for r in sc if r['iou_after']-r['iou_before'] > .05)}  "
              f"worse {sum(1 for r in sc if r['iou_before']-r['iou_after'] > .05)}")
        print(f"  usd/page {np.mean([r.get('usd',0) for r in rows]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
