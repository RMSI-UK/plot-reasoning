#!/usr/bin/env python3
"""snap_boundary_v1 -- the frozen pipeline for tracing a UK planning application's plot boundary.

    python snap_boundary_v1.py --src /path/to/panels --out results/
    python snap_boundary_v1.py --src ... --pages stems.json --workers 20
    python snap_boundary_v1.py --src ... --locate-once --no-regrow   # cheaper, fewer guards

Five steps. Each one is here because the alternative was measured and was worse; the figures below
are from 20-page batches on the Braintree wp7 panels unless stated.

  1 LOCATE   gpt-5.6-luna, whole sheet, with the council's reference/address/proposal.
             Returns the plot box, the point, and -- new in v1 -- what kind of marking identifies
             the plot and where the leader line ENDS.
             Run TWICE. Box IoU between runs had a median of 0.85 across two full runs but fell
             below 0.5 on 3 of 19 pages, and on one of those the box moved to a nine-times-larger
             field and took the boundary with it. When the two disagree the smaller box is used and
             the page is flagged. gpt-5.6-luna beat six other candidates on containment (22/26
             against terra's 20/25) at an eighth of terra's price.

  2 WINDOW   square, 1.5x the longer box side, and it must CONTAIN the box. Forcing a square and
             clamping its side to the page's short edge produced a window shorter than the box on
             portrait sheets -- 90-01394 lost 60 px off each end of an 831 px plot and the model,
             shown only the middle, declined twice. Falls back to a rectangle rather than cropping
             the box.
             Padding was swept: 1.25 / 1.5 / 2.0 gave p90 6.58 / 4.78 / 6.14, so 1.5 stays.
             The union of the box with a box round the label was tried and rejected: it does put
             the label in frame (18/18) but the window grows by a median x1.43 and by x25 on one
             page, and traced pages fall from 19/20 to 15/20.

  3 ZOOM     bicubic to a 1024 px long side, NEVER below 1.0. Windows wider than 1024 were being
             shrunk; pages at zoom < 1.15 had a p90 of 4.71 against 1.91 for the rest, worst 55.7.
             Cropping beats handing over the whole sheet mainly by making the model willing to
             answer -- 19/20 against 14/20 -- rather than by precision (p90 2.32 against 3.21).
             Upscaling the whole sheet 2x instead of cropping made both models worse, so the gain
             is from removing the rest of the drawing, not from more pixels.

  4 TRACE    gpt-5.6-terra on the crop. Against luna-pro on identical crops: p90 2.32 against 3.55,
             93% against 88% within 3 px, 19/20 against 18/20, and twice as fast, for 76% more
             money -- four pounds over the whole 783-page corpus.
             Four non-OpenAI models were tested on the same crops under three different prompts.
             All landed at p90 13-30 with 18-44% of the ring over blank paper against terra's 0%.
             Stripping the prompt back improved their scores only by collapsing the answer to a
             quadrilateral, so it is capability rather than wording.

  5 CHECKS   Nothing is corrected. Six independent tests say which pages a human should look at:
               the two locate calls disagree about where the plot is
               the ring runs along a crop edge that lies inside the page (the window cut the plot)
               the ring runs along the page's own edge (the plot leaves the paper -- not fixable)
               part of the ring lies over blank paper, so those sides were not traced from anything
               the drawn line network does not enclose the ring (OpenCV snap)
               THE RING DOES NOT CONTAIN THE POINT THE ARROW IS AIMED AT
             The last is the only one about plot IDENTITY. Every other check, and every metric in
             this project, asks whether the line follows drawn ink -- and a ring can do that
             immaculately while enclosing the neighbour's plot. On 20 pages, 5 of terra's 19 rings
             failed it, three of them scoring p90 0.00 to 0.95 and passing everything else.

  RE-CROP    if the ring ran along a crop edge inside the page, the window is widened ON THOSE SIDES
             ONLY and the trace repeated. Widening uniformly does not work -- padding 1.25 to 2.0
             moved truncation from 9/20 pages to 7/20 -- because it enlarges symmetrically around an
             off-centre box. The second result is kept only if it EXTENDS the first (keeps >=90% of
             its area and grows); "the ring no longer touches an edge" is satisfied by throwing the
             answer away, and one page passed that test while shrinking 31%.

THERE IS NO GROUND TRUTH FOR THIS CORPUS. Every distance figure measures the ring against drawn
ink. The arrow-tip check is the only evidence here about whether it is the right ink.
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

from pydantic import BaseModel, Field                             # noqa: E402
from pydantic_ai import Agent, BinaryContent, NativeOutput        # noqa: E402
from pydantic_ai.usage import UsageLimits                         # noqa: E402
from PIL import Image                                             # noqa: E402

from geoplanagent.utils import resolve_model                      # noqa: E402
import boundary as B                                              # noqa: E402
import locate as L                                                # noqa: E402
from metadata import records_for                                  # noqa: E402

VERSION = "snap_boundary_v1.1"

# Appended to the tracing prompt ONLY on a retry, after a first attempt has declined.
#
# The tracing prompt tells the model the plot is the whole curtilage "not just the building
# footprint". On a page whose only marking is a building -- stippled, hatched, inked solid -- with
# no boundary drawn around the land it stands in, that leaves nothing it is permitted to trace, and
# all three declines in the first v1 run said exactly that.
#
# It is a RETRY, not a default, because a human checked both fixes on 93-00352 and judged the
# fallback WRONG there: that page does have a drawn curtilage, and the decline was caused by a bad
# box, which fix 1 repairs. Applying the fallback up front would trace the building and look fine
# on every automatic metric. Fix the box first; fall back only when a good box still yields nothing.
BUILDING_FALLBACK = """

If the only thing on the page that marks the application is a building or structure -- stippled,
hatched, inked solid or drawn more heavily than its neighbours -- and no boundary is drawn around
the land it stands in, then trace that marked structure's own outline and say in what_marked_it
that no curtilage was drawn. An outline of the marked building is far more use than nothing. Do not
do this when a curtilage boundary IS drawn: there, the curtilage is the answer."""
LOCATE_MODEL, LOC_IN, LOC_OUT = "openai/gpt-5.6-luna", 0.10, 0.60
TRACE_MODEL, TR_IN, TR_OUT = "openai/gpt-5.6-terra", 1.00, 6.00

# appended to prompt.txt, which is checksummed and left untouched so its measured figures stand
MARKING_EXTRA = """

Separately, describe the MARKING that told you which plot it is -- the label, house number, arrow,
leader line, hatching or heavier outline you actually used.

If that marking is a label with a line or arrow running from it to the plot, report where that line
ENDS -- the point on the drawing the arrow is aimed at. Give it as marking_tip_xy. That end point is
the draughtsman's own statement of which plot the application concerns, so place it as precisely as
you can: on the feature the line touches, not in the middle of the label and not halfway along.

If the marking is not a label with a line -- hatching, a heavier outline, a number written inside
the plot -- give a point at the middle of that marking instead and say so.

If nothing on the page marks the plot at all, set marking_kind to "none" and leave the point zero."""


class _XY(BaseModel):
    x: int
    y: int


class _Box(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class LocateAnswer(BaseModel):
    is_plan: bool = Field(description="True if this page is a PLAN OF LAND seen from above.")
    plan_kind: str = Field(description="What the page actually is, in your own words.")
    found: bool = Field(description="True if you can tell which single plot this is about.")
    point_xy: _XY = Field(description="A point INSIDE that plot. Zeros if found is false.")
    box_xy: _Box = Field(description="Axis-aligned box just containing the whole plot.")
    marking_kind: str = Field(
        description='"leader" for a label with a line or arrow, "hatching", "outline", '
                    '"number", "other", or "none".')
    marking_tip_xy: _XY = Field(
        description="Where the leader line ENDS on the drawing, or the middle of the marking if "
                    "it is not a leader. Zeros if marking_kind is none.")
    what_marked_it: str = Field(description="IN YOUR OWN WORDS: what told you it was this plot.")
    confidence: float = Field(description="0.0 to 1.0")


def build_locate_agent() -> Agent:
    return Agent("test", output_type=NativeOutput(LocateAnswer), retries=1, output_retries=0,
                 model_settings={"max_tokens": 3000, "timeout": 120},
                 instructions=L.load_prompt() + MARKING_EXTRA)


def locate_once(agent: Agent, model, gray: np.ndarray, case) -> dict:
    h, w = gray.shape
    buf = io.BytesIO()
    Image.fromarray(gray).convert("RGB").save(buf, format="PNG")
    out: dict = {}
    t0 = time.time()
    try:
        r = agent.run_sync(
            [BinaryContent(data=buf.getvalue(), media_type="image/png"),
             f"This image is W={w} pixels wide and H={h} pixels high. Is it a plan of land, and "
             f"if so where is the application plot?" + (case.as_prompt_block() if case else "")],
            model=model, usage_limits=UsageLimits(request_limit=3))
    except Exception as exc:                                       # noqa: BLE001
        return {"error": f"{exc!s:.200}", "seconds": round(time.time() - t0, 1)}
    o, u = r.output, r.usage()
    out["seconds"] = round(time.time() - t0, 1)
    out["usd"] = round((u.input_tokens or 0) * LOC_IN / 1e6
                       + (u.output_tokens or 0) * LOC_OUT / 1e6, 6)
    out.update({"is_plan": o.is_plan, "plan_kind": o.plan_kind, "found": o.found,
                "said": o.what_marked_it, "confidence": o.confidence,
                "marking_kind": o.marking_kind,
                "point": [o.point_xy.x, o.point_xy.y],
                "box": [o.box_xy.x0, o.box_xy.y0, o.box_xy.x1, o.box_xy.y1],
                "tip": [o.marking_tip_xy.x, o.marking_tip_xy.y]})
    return out


TIP_TOLERANCE_PX = 6.0


def tip_inside(tip, ring, tol: float = TIP_TOLERANCE_PX) -> bool | None:
    """Is the point the arrow aims at inside the traced ring? None when there is no marking.

    A tolerance is needed because a leader very often ends ON the boundary rather than inside the
    plot -- that is how these drawings point at a parcel. A strict test called 90-01100 a failure
    when its magenta tip sits on the ring's own lower edge, which is the arrow doing exactly what
    it should. cv2 returns a signed distance, negative outside.
    """
    if not tip or tip == [0, 0] or not ring:
        return None
    d = cv2.pointPolygonTest(np.asarray(ring, np.int32).reshape(-1, 1, 2),
                             (float(tip[0]), float(tip[1])), True)
    return bool(d >= -tol)


def snap_iou(ring, gray: np.ndarray, bridge: int = 14, cover: float = 0.55) -> float | None:
    """Union of drawn-line faces that sit mostly inside the ring, against the ring.

    Dashes are why the bridge is 14 px: at 6 px the oracle ceiling over face unions was 0.265, at
    14 px it is 0.679, and individual pages step rather than drift (0.085 -> 0.924). Past 20 px real
    boundaries start merging and it reverses.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    h, w = gray.shape
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = t > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge))
    net = skeletonize(cv2.morphologyEx(skeletonize(ink).astype(np.uint8) * 255,
                                       cv2.MORPH_CLOSE, k) > 0)
    lab, n = ndi.label(~net, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    target = np.zeros((h, w), np.uint8)
    cv2.fillPoly(target, [np.asarray(ring, np.int32)], 1)
    target = target > 0
    snap = np.zeros((h, w), bool)
    for i in range(1, n + 1):
        m = lab == i
        a = m.sum()
        if not (0.001 * w * h <= a <= 0.40 * w * h):
            continue
        ys, xs = np.nonzero(m)
        if xs.min() == 0 and xs.max() == w - 1 and ys.min() == 0 and ys.max() == h - 1:
            continue
        if (m & target).sum() / a >= cover:
            snap |= m
    if not snap.any():
        return 0.0
    return round(float((snap & target).sum() / (snap | target).sum()), 3)


def one_page(path: Path, agents, models, case, locate_twice: bool, do_regrow: bool,
             do_snap: bool) -> dict:
    a_loc, a_bnd = agents
    m_loc, m_bnd = models
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"stem": path.stem, "error": "unreadable image"}
    ph, pw = gray.shape
    row: dict = {"version": VERSION, "stem": path.stem, "page_w": pw, "page_h": ph, "usd": 0.0}

    # ---- 1. locate, twice
    l1 = locate_once(a_loc, m_loc, gray, case)
    row["locate_usd"] = l1.get("usd", 0.0)
    row["usd"] += l1.get("usd", 0.0)
    row.update({k: l1.get(k) for k in ("is_plan", "plan_kind", "found", "marking_kind",
                                       "point", "box", "tip")})
    row["locate_said"] = l1.get("said")
    if l1.get("error"):
        row["error"] = f"locate: {l1['error']}"
        return row
    if not (l1.get("found") and l1.get("box")):
        row["skipped"] = "stage 1 found no plot"
        return row
    box = l1["box"]
    if locate_twice:
        l2 = locate_once(a_loc, m_loc, gray, case)
        row["usd"] += l2.get("usd", 0.0)
        row["locate_usd"] = round(row["locate_usd"] + l2.get("usd", 0.0), 6)
        row["box_2"] = l2.get("box")
        if l2.get("found") and l2.get("box"):
            agree = B.box_iou(l1["box"], l2["box"])
            if agree >= 0.5:
                box = l1["box"]
            else:
                # the UNION, not the smaller box. When one call boxes the building and the other
                # the plot, "smaller" takes the building -- 93-00352 declined for exactly that
                # reason, and its union box produced the ring a human then judged correct.
                a, b2 = l1["box"], l2["box"]
                box = [min(a[0], b2[0]), min(a[1], b2[1]), max(a[2], b2[2]), max(a[3], b2[3])]
            row["box_agree"] = agree
        else:
            row["box_agree"] = 0.0
    row["box_used"] = box

    # ---- 2/3. window and zoom
    win = B.window_for(box, pw, ph)
    row["window"] = [win.x, win.y, win.w, win.h]
    row["scale"] = round(win.scale, 3)
    row["window_contains_box"] = B.window_contains_box(win, box)
    block = case.as_prompt_block() if case else ""

    # ---- 4. trace
    res = B.trace_one(a_bnd, m_bnd, gray, path.stem, win, block)
    row["usd"] = round(row["usd"] + res.usd, 6)
    row["trace_usd"] = res.usd
    row["trace_tokens"] = [res.tokens_in, res.tokens_out]
    row.update({"traced": res.found, "said": res.said, "confidence": res.confidence,
                "trace_seconds": res.seconds})
    if res.error:
        row["error"] = f"trace: {res.error}"
        return row
    if not res.vertices_page:
        # the box has already been reconciled, so a decline here means the page really may have no
        # drawn curtilage. Offer the building outline instead, once.
        a_fb = B.build_agent(B.load_prompt() + BUILDING_FALLBACK)
        res_fb = B.trace_one(a_fb, m_bnd, gray, path.stem, win, block)
        row["usd"] = round(row["usd"] + res_fb.usd, 6)
        row["fallback_tried"] = True
        row["fallback_said"] = res_fb.said
        if not res_fb.vertices_page:
            row["declined_trace"] = True
            return row
        res = res_fb
        row["source_note"] = "building outline -- no curtilage was drawn"
        row["said"] = res_fb.said

    dt = B.ink_distance(gray)
    row["vertices_page"] = res.vertices_page
    row.update(B.ring_stats(res.vertices_page, dt))
    row["edges"] = B.edge_contact(res.vertices_page, win, pw, ph)
    row["source"] = "first crop"

    # ---- re-crop, only on sides the ring actually ran along
    if do_regrow and row["edges"]["cut_sides"]:
        win2 = B.grow_window(win, row["edges"]["cut_sides"], pw, ph)
        res2 = B.trace_one(a_bnd, m_bnd, gray, path.stem, win2, block)
        row["usd"] = round(row["usd"] + res2.usd, 6)
        after = B.ring_stats(res2.vertices_page, dt) if res2.vertices_page else None
        keep, why = B.accept_regrow(row, after, res.vertices_page, res2.vertices_page, pw, ph)
        row["regrow"] = {"window": [win2.x, win2.y, win2.w, win2.h],
                         "scale": round(win2.scale, 3), "accepted": keep, "verdict": why,
                         "said": res2.said, **(after or {})}
        if keep:
            row["vertices_page"] = res2.vertices_page
            row.update(after)
            row["edges"] = B.edge_contact(res2.vertices_page, win2, pw, ph)
            row["source"] = "widened crop"

    # ---- 5. checks
    row["snap_iou"] = snap_iou(row["vertices_page"], gray) if do_snap else None
    row["tip_inside_ring"] = tip_inside(row.get("tip"), row["vertices_page"])
    flags = B.review_flags(row, row["edges"], snap_iou=row["snap_iou"],
                           box_agree=row.get("box_agree"))
    if row["tip_inside_ring"] is False:
        flags.insert(0, f"the ring does not contain the point the {row.get('marking_kind')} "
                        f"marking aims at -- it may be round the wrong plot")
    row["flags"] = flags
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "snap_boundary")
    ap.add_argument("--pages", type=Path, default=None, help="JSON list of stems")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--locate-once", action="store_true",
                    help="one locate call instead of two: saves GBP 0.6 per 1000 and loses the "
                         "only guard on the box")
    ap.add_argument("--no-regrow", action="store_true")
    ap.add_argument("--no-snap", action="store_true", help="skip the OpenCV face check")
    ap.add_argument("--no-metadata", action="store_true")
    ap.add_argument("--trace-model", default=TRACE_MODEL)
    args = ap.parse_args()

    if args.pages:
        files = [args.src / f"{s}.jpg" for s in json.loads(args.pages.read_text())]
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
            except Exception:                                      # noqa: BLE001
                continue
    todo = [f for f in files if f.stem not in done]
    print(f"{VERSION} | {len(files)} pages, {len(done)} done, {len(todo)} to run, "
          f"{args.workers} workers\n  locate {LOCATE_MODEL} x{1 if args.locate_once else 2}"
          f" | trace {args.trace_model}", flush=True)
    if not todo:
        return 0

    records = {} if args.no_metadata else records_for([f.stem for f in todo])
    agents = (build_locate_agent(), B.build_agent(B.load_prompt()))
    models = (resolve_model(LOCATE_MODEL), resolve_model(args.trace_model))
    # boundary.py's module-level rates are luna-pro's. Setting them only when the model differs
    # from this file's default silently priced terra at a tenth of its cost -- the first v1 run
    # reported GBP 2.37 per 1000 when the true figure is near GBP 12. Always set them.
    RATES = {"openai/gpt-5.6-terra": (1.00, 6.00), "openai/gpt-5.6-luna-pro": (0.10, 0.60),
             "openai/gpt-5.6-luna": (0.10, 0.60), "openai/gpt-5.4-mini": (0.25, 2.00)}
    if args.trace_model in RATES:
        B.USD_PER_MTOK_IN, B.USD_PER_MTOK_OUT = RATES[args.trace_model]
    else:
        print(f"note: no price on file for {args.trace_model}; costs use "
              f"${B.USD_PER_MTOK_IN}/${B.USD_PER_MTOK_OUT} per Mtok")

    lock = threading.Lock()
    started = time.time()
    rows = []
    with jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one_page, f, agents, models, records.get(f.stem),
                            not args.locate_once, not args.no_regrow, not args.no_snap): f
                for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if row.get("error"):
                state = f"ERROR {row['error'][:44]}"
            elif row.get("skipped") or row.get("declined_trace"):
                state = "declined"
            else:
                tip = {True: "tip in", False: "TIP OUT", None: "no tip"}[row["tip_inside_ring"]]
                state = (f"{row['n_vertices']:3d}v p90 {row['ring_to_ink_p90_px']:6.2f} "
                         f"blank {row['blank_share']*100:3.0f}% {tip:7s}")
            print(f"[{i}/{len(todo)}] {row['stem'][:32]:32s} {state}  "
                  f"{('; '.join(row.get('flags') or []))[:52]}", flush=True)

    ok = [r for r in rows if r.get("vertices_page")]
    M = lambda k: (round(float(np.median([r[k] for r in ok if r.get(k) is not None])), 3)
                   if ok else None)
    tips = [r for r in ok if r.get("tip_inside_ring") is not None]
    summary = {
        "version": VERSION, "locate_model": LOCATE_MODEL, "trace_model": args.trace_model,
        "pages": len(rows), "located": sum(1 for r in rows if r.get("found")), "traced": len(ok),
        "window_contains_box": sum(1 for r in rows if r.get("window_contains_box")),
        "never_downscaled": all(r.get("scale", 1) >= 1.0 for r in rows),
        "locate_disagreed": sum(1 for r in rows
                                if r.get("box_agree") is not None and r["box_agree"] < 0.5),
        "fallback_tried": sum(1 for r in rows if r.get("fallback_tried")),
        "fallback_rescued": sum(1 for r in rows if r.get("fallback_tried")
                                and r.get("vertices_page")),
        "regrow_fired": sum(1 for r in rows if r.get("regrow")),
        "regrow_accepted": sum(1 for r in rows if (r.get("regrow") or {}).get("accepted")),
        "n_vertices": M("n_vertices"), "ring_to_ink_median_px": M("ring_to_ink_median_px"),
        "ring_to_ink_p90_px": M("ring_to_ink_p90_px"), "frac_within_3px": M("frac_within_3px"),
        "blank_share": M("blank_share"), "area_pct": M("area_pct"), "snap_iou": M("snap_iou"),
        "marking_kinds": {k: sum(1 for r in rows if r.get("marking_kind") == k)
                          for k in sorted({r.get("marking_kind") for r in rows if r.get("marking_kind")})},
        "tip_checkable": len(tips),
        "tip_inside_ring": sum(1 for r in tips if r["tip_inside_ring"]),
        "pages_clean": sum(1 for r in ok if not r.get("flags")),
        "pages_flagged": sum(1 for r in ok if r.get("flags")),
        "usd_per_page": round(float(np.mean([r.get("usd", 0) for r in rows])), 5),
        "wall_seconds": round(time.time() - started, 1),
    }
    summary["gbp_per_1000"] = round(summary["usd_per_page"] * 1000 / 1.26, 2)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n=== {VERSION} summary ===")
    for k, v in summary.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
