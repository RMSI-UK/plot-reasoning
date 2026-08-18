#!/usr/bin/env python3
"""Identity-guided v1.2 pipeline for tracing UK planning-application plot boundaries.

    python snap_boundary_v1.py --src /path/to/panels --out results/
    python snap_boundary_v1.py --src ... --pages stems.json --workers 20

The whole-page locator runs twice. A trace is attempted only when two calls agree (box IoU >= 0.5);
a third locator call may adjudicate an inconsistent pair. The chosen locator supplies the crop box,
an identity point, an optional spatial-marking tip, marking type, and target-geometry type.

The tracer receives both the clean crop and an annotated copy showing that identity evidence. Its
ring must contain the identity point and, for spatial markings, the marking tip. Candidate selection
prefers identity containment and an uncut crop before line-fit metrics, so a smaller correct retrace
can replace a larger wrong result. Non-spatial text, building-only geometry, location indicators,
failed locator consensus, and fallback proxies are never labelled automatically usable.

There is no human ground truth for this corpus. Ink-distance metrics only measure whether a ring
follows drawn lines; they do not prove that those lines belong to the intended legal parcel.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
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

VERSION = "snap_boundary_v1.2"

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
the plot, or a circle around the site -- give a point ON THE TARGET DRAWING at the middle of that
spatial marking and set marking_is_spatial true.

If words identify a property but have no line or arrow pointing from the words to the drawing, use
marking_kind "text", set marking_is_spatial false, and leave marking_tip_xy zero. Never put the
tip in the words themselves: that is not a point on the target plot.

Set target_geometry to "curtilage" when parcel boundaries are visible, "explicit_outline" when a
closed marked line is itself the requested site boundary, "building_only" when only a building can
be identified, "location_indicator" when a circle or label identifies only a vicinity without a
recoverable parcel boundary, or "unknown" when you cannot tell.

If nothing on the page marks the plot at all, set marking_kind to "none", marking_is_spatial false,
and leave the point zero."""


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
                    '"number", "circle", "text", "other", or "none".')
    marking_is_spatial: bool = Field(
        description="True only when marking_tip_xy is a point on the target drawing, not in text.")
    marking_tip_xy: _XY = Field(
        description="Where the leader ENDS on the target drawing, or the middle of a spatial "
                    "marking. Zeros for unpointed text or no marking.")
    target_geometry: str = Field(
        description='One of "curtilage", "explicit_outline", "building_only", '
                    '"location_indicator", or "unknown".')
    what_marked_it: str = Field(description="IN YOUR OWN WORDS: what told you it was this plot.")
    confidence: float = Field(description="0.0 to 1.0")


def build_locate_agent() -> Agent:
    return Agent("test", output_type=NativeOutput(LocateAnswer), retries=1, output_retries=0,
                 model_settings={"max_tokens": 4500, "timeout": 120},
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
                "marking_spatial": o.marking_is_spatial,
                "target_geometry": o.target_geometry,
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


def locate_consensus(locates: list[dict], agree_at: float = 0.5):
    """Require two agreeing whole-page locates; never trace from a lone unstable answer."""
    valid = [(i, loc) for i, loc in enumerate(locates)
             if loc.get("found") and loc.get("box") and not loc.get("error")]
    if len(locates) == 1:
        if not valid:
            return None, None, {"best_iou": None, "members": []}
        return dict(valid[0][1]), list(valid[0][1]["box"]), {
            "best_iou": None, "members": [valid[0][0]]}
    if len(valid) < 2:
        return None, None, {"best_iou": 0.0, "members": []}

    pairs = []
    for (ia, a), (ib, b) in combinations(valid, 2):
        pairs.append((B.box_iou(a["box"], b["box"]), ia, ib, a, b))
    iou, ia, ib, a, b = max(pairs, key=lambda item: item[0])
    if iou < agree_at:
        return None, None, {"best_iou": iou, "members": []}
    chosen = dict(max((a, b), key=lambda loc: float(loc.get("confidence") or 0.0)))
    ba, bb = a["box"], b["box"]
    box = [min(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]), max(ba[3], bb[3])]
    chosen["box"] = box
    return chosen, box, {"best_iou": iou, "members": [ia, ib]}


def trace_guidance(loc: dict, box: list[int]) -> dict:
    return {"point": loc.get("point"), "tip": loc.get("tip"), "box": box,
            "marking_kind": loc.get("marking_kind"),
            "marking_spatial": bool(loc.get("marking_spatial")),
            "target_geometry": loc.get("target_geometry"),
            "locate_said": loc.get("said")}


def make_candidate(res, win, gray, dt, page_w: int, page_h: int, guidance: dict,
                   source: str, fallback: bool = False) -> dict | None:
    if not res.vertices_page:
        return None
    stats = B.ring_stats(res.vertices_page, dt)
    edges = B.edge_contact(res.vertices_page, win, page_w, page_h)
    point_ok = B.point_inside_ring(guidance.get("point"), res.vertices_page)
    tip_ok = (B.point_inside_ring(guidance.get("tip"), res.vertices_page)
              if guidance.get("marking_spatial") else None)
    identity_ok = point_ok is True and (tip_ok is True if guidance.get("marking_spatial") else True)
    return {"vertices_page": res.vertices_page, "said": res.said,
            "confidence": res.confidence, "trace_seconds": res.seconds,
            "window": [win.x, win.y, win.w, win.h], "scale": round(win.scale, 3),
            "source": source, "fallback": fallback, "point_inside_ring": point_ok,
            "tip_inside_ring": tip_ok, "identity_ok": identity_ok, "edges": edges,
            "_win": win, **stats}


def public_candidate(candidate: dict | None) -> dict | None:
    return ({k: v for k, v in candidate.items() if k != "_win"} if candidate else None)


def apply_candidate(row: dict, candidate: dict) -> None:
    for key in ("vertices_page", "said", "confidence", "trace_seconds", "window", "scale",
                "source", "point_inside_ring", "tip_inside_ring", "identity_ok", "edges",
                "n_vertices", "ring_to_ink_median_px", "ring_to_ink_p90_px",
                "frac_within_3px", "blank_share", "area_pct"):
        row[key] = candidate.get(key)
    row["traced"] = True
    if candidate.get("fallback"):
        row["source_note"] = "building outline -- no curtilage was drawn"


def one_page(path: Path, agents, models, case, locate_twice: bool, do_regrow: bool,
             do_snap: bool) -> dict:
    a_loc, a_bnd = agents
    m_loc, m_bnd = models
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"stem": path.stem, "error": "unreadable image"}
    ph, pw = gray.shape
    row: dict = {"version": VERSION, "stem": path.stem, "page_w": pw, "page_h": ph, "usd": 0.0}

    # ---- 1. locate twice; a third call adjudicates any disagreement or lone answer
    locates = [locate_once(a_loc, m_loc, gray, case)]
    if locate_twice:
        locates.append(locate_once(a_loc, m_loc, gray, case))
    loc, box, consensus = locate_consensus(locates)
    if locate_twice and loc is None:
        locates.append(locate_once(a_loc, m_loc, gray, case))
        loc, box, consensus = locate_consensus(locates)
    row["locate_usd"] = round(sum(l.get("usd", 0.0) for l in locates), 6)
    row["usd"] = row["locate_usd"]
    row["locate_calls"] = len(locates)
    row["locate_adjudicated"] = len(locates) == 3
    row["locate_candidates"] = [
        {k: l.get(k) for k in ("found", "box", "point", "tip", "marking_kind",
                                "marking_spatial", "target_geometry", "confidence", "said", "error")}
        for l in locates]
    row["box_agree"] = consensus.get("best_iou")
    row["locate_consensus_members"] = consensus.get("members")
    if loc is None or box is None:
        row["found"] = False
        row["skipped"] = "stage 1 had no two-call consensus"
        row["needs_human"] = True
        row["flags"] = ["whole-page locators did not reach a two-call consensus"]
        return row
    row.update({k: loc.get(k) for k in ("is_plan", "plan_kind", "found", "marking_kind",
                                        "marking_spatial", "target_geometry", "point", "tip")})
    row["box"] = loc.get("box")
    row["box_used"] = box
    row["locate_said"] = loc.get("said")

    # ---- 2/3. window and zoom
    win = B.window_for(box, pw, ph)
    row["window"] = [win.x, win.y, win.w, win.h]
    row["scale"] = round(win.scale, 3)
    row["window_contains_box"] = B.window_contains_box(win, box)
    block = case.as_prompt_block() if case else ""
    guidance = trace_guidance(loc, box)

    # ---- 4. trace
    res = B.trace_one(a_bnd, m_bnd, gray, path.stem, win, block, guidance)
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
        if row.get("target_geometry") in {"building_only", "location_indicator"}:
            row["declined_trace"] = True
            row["needs_human"] = True
            row["flags"] = [f"locator says {row['target_geometry']}; no recoverable parcel boundary"]
            return row
        a_fb = B.build_agent(B.load_prompt() + BUILDING_FALLBACK)
        res_fb = B.trace_one(a_fb, m_bnd, gray, path.stem, win, block, guidance)
        row["usd"] = round(row["usd"] + res_fb.usd, 6)
        row["trace_usd"] = round(row["trace_usd"] + res_fb.usd, 6)
        row["fallback_tried"] = True
        row["fallback_said"] = res_fb.said
        if not res_fb.vertices_page:
            row["declined_trace"] = True
            return row
        res = res_fb

    dt = B.ink_distance(gray)
    chosen = make_candidate(res, win, gray, dt, pw, ph, guidance, "first crop",
                            fallback=bool(row.get("fallback_tried")))
    if chosen is None:
        row["declined_trace"] = True
        return row

    # Identity failure is not merely flagged: retry with the target constraint made explicit.
    if chosen["identity_ok"] is not True and not chosen.get("fallback"):
        retry_note = (
            "IDENTITY RETRY: the previous ring missed the cyan target or magenta spatial marking. "
            "Trace the intended plot that CONTAINS those locator marks; do not return a neighbour.")
        res_id = B.trace_one(a_bnd, m_bnd, gray, path.stem, win, block, guidance, retry_note)
        row["usd"] = round(row["usd"] + res_id.usd, 6)
        row["trace_usd"] = round(row["trace_usd"] + res_id.usd, 6)
        identity_candidate = make_candidate(
            res_id, win, gray, dt, pw, ph, guidance, "identity retry")
        row["identity_retry"] = public_candidate(identity_candidate)
        if identity_candidate:
            chosen, row["identity_retry_verdict"] = B.choose_candidate(chosen, identity_candidate)

    # ---- re-crop, only on sides the ring actually ran along
    if do_regrow and chosen["edges"]["cut_sides"]:
        win2 = B.grow_window(chosen["_win"], chosen["edges"]["cut_sides"], pw, ph)
        res2 = B.trace_one(a_bnd, m_bnd, gray, path.stem, win2, block, guidance)
        row["usd"] = round(row["usd"] + res2.usd, 6)
        row["trace_usd"] = round(row["trace_usd"] + res2.usd, 6)
        regrown = make_candidate(res2, win2, gray, dt, pw, ph, guidance, "widened crop")
        if regrown:
            selected, verdict = B.choose_candidate(chosen, regrown)
            accepted = selected is regrown
            chosen = selected
        else:
            accepted, verdict = False, "widened crop produced no polygon"
        row["regrow"] = {**(public_candidate(regrown) or {}),
                         "accepted": accepted, "verdict": verdict,
                         "error": res2.error}

    apply_candidate(row, chosen)

    # ---- 5. checks
    row["snap_iou"] = snap_iou(row["vertices_page"], gray) if do_snap else None
    flags = B.review_flags(row, row["edges"], snap_iou=row["snap_iou"],
                           box_agree=row.get("box_agree"))
    if row["point_inside_ring"] is False:
        flags.insert(0, "the ring does not contain the locator's point inside the application plot")
    if row["tip_inside_ring"] is False:
        flags.insert(0, f"the ring does not contain the point the {row.get('marking_kind')} "
                        f"marking aims at -- it may be round the wrong plot")
    if chosen.get("fallback"):
        flags.insert(0, "building fallback produced a proxy outline, not a verified parcel boundary")
    if row.get("target_geometry") in {"building_only", "location_indicator", "unknown"}:
        flags.insert(0, f"locator geometry is {row['target_geometry']}, not a verified parcel")
    if not row.get("marking_spatial") and row.get("marking_kind") in {"text", "other"}:
        flags.insert(0, "site is identified only by non-spatial text; polygon identity needs review")
    row["flags"] = flags
    row["needs_human"] = bool(flags)
    row["auto_usable"] = bool(row["identity_ok"] and not flags and not chosen.get("fallback"))
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
                         f"blank {row['blank_share']*100:3.0f}% {tip:7s} "
                         f"{'AUTO' if row.get('auto_usable') else 'REVIEW'}")
            print(f"[{i}/{len(todo)}] {row['stem'][:32]:32s} {state}  "
                  f"{('; '.join(row.get('flags') or []))[:52]}", flush=True)

    ok = [r for r in rows if r.get("vertices_page")]
    M = lambda k: (round(float(np.median([r[k] for r in ok if r.get(k) is not None])), 3)
                   if ok else None)
    tips = [r for r in ok if r.get("tip_inside_ring") is not None]
    points = [r for r in ok if r.get("point_inside_ring") is not None]
    summary = {
        "version": VERSION, "locate_model": LOCATE_MODEL, "trace_model": args.trace_model,
        "pages": len(rows), "located": sum(1 for r in rows if r.get("found")), "traced": len(ok),
        "window_contains_box": sum(1 for r in rows if r.get("window_contains_box")),
        "never_downscaled": all(r.get("scale", 1) >= 1.0 for r in rows),
        "locate_disagreed": sum(1 for r in rows
                                if r.get("box_agree") is not None and r["box_agree"] < 0.5),
        "locate_adjudicated": sum(1 for r in rows if r.get("locate_adjudicated")),
        "locate_consensus_failed": sum(1 for r in rows
                                        if r.get("skipped") == "stage 1 had no two-call consensus"),
        "fallback_tried": sum(1 for r in rows if r.get("fallback_tried")),
        "fallback_rescued": sum(1 for r in rows if r.get("fallback_tried")
                                and r.get("vertices_page")),
        "regrow_fired": sum(1 for r in rows if r.get("regrow")),
        "regrow_accepted": sum(1 for r in rows if (r.get("regrow") or {}).get("accepted")),
        "identity_retry_fired": sum(1 for r in rows if r.get("identity_retry") is not None),
        "identity_retry_selected": sum(1 for r in rows
                                        if r.get("source") == "identity retry"),
        "n_vertices": M("n_vertices"), "ring_to_ink_median_px": M("ring_to_ink_median_px"),
        "ring_to_ink_p90_px": M("ring_to_ink_p90_px"), "frac_within_3px": M("frac_within_3px"),
        "blank_share": M("blank_share"), "area_pct": M("area_pct"), "snap_iou": M("snap_iou"),
        "marking_kinds": {k: sum(1 for r in rows if r.get("marking_kind") == k)
                          for k in sorted({r.get("marking_kind") for r in rows if r.get("marking_kind")})},
        "tip_checkable": len(tips),
        "tip_inside_ring": sum(1 for r in tips if r["tip_inside_ring"]),
        "point_checkable": len(points),
        "point_inside_ring": sum(1 for r in points if r["point_inside_ring"]),
        "pages_clean": sum(1 for r in ok if not r.get("flags")),
        "pages_flagged": sum(1 for r in ok if r.get("flags")),
        "pages_auto_usable": sum(1 for r in rows if r.get("auto_usable")),
        "pages_needs_human": sum(1 for r in rows if r.get("needs_human")),
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
