"""Trace the application plot's boundary, given the box that `locate.py` produced.

The chain is locate -> window -> enlarge -> trace, and each of the three geometric steps here
exists because measuring the alternative showed it was worse:

  cropping        beats handing over the whole page (p90 5.68 -> 3.69 for terra, 8.06 -> 5.73 for
                  luna-pro), and it is the *removal of the rest of the drawing* that does the work,
                  not the extra pixels: upscaling the whole page 2x made both models worse and
                  tripled terra's declines.
  never shrinking that crop is why `enlarge` refuses to scale below 1.0. Pages whose window came
                  out at zoom < 1.15 had a p90 of 4.71 against 1.91 for the rest, with one at 55.7.
  a window that   contains the box is not automatic. The old code forced a square and clamped its
                  side to the page's short edge, so a tall box on a portrait page produced a window
                  shorter than the box itself -- 90-01394 lost 60 px off each end of an 831 px plot
                  and the model, shown only the middle, declined twice.

The checks at the bottom never correct anything. They say which pages a human should look at, and
each is independent of the others and of any ground truth, because this corpus has none.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.usage import UsageLimits

MODEL = "openai/gpt-5.6-luna-pro"
PROMPT_PATH = Path(__file__).with_name("boundary_prompt.txt")
PROMPT_SHA16 = None          # set by pinning below; None means "do not check"

USD_PER_MTOK_IN = 0.10
USD_PER_MTOK_OUT = 0.60

PAD = 1.5                    # window side, as a multiple of the box's longer side
TARGET_LONG = 1024           # enlarge the window's long side to this, but never shrink it
MIN_SIDE = 48                # a degenerate box must not produce a degenerate crop
GROW = 0.6                   # re-crop: extend a touched side by this fraction of the window

# a ring point further than this from any ink is over blank paper, not on a drawn line
BLANK_PX = 10


class _V(BaseModel):
    x: int
    y: int


class BoundaryAnswer(BaseModel):
    found: bool = Field(description="True if you can trace the application plot's boundary here.")
    vertices: list[_V] = Field(
        description="The plot boundary as ordered vertices in THIS cropped image's pixels, "
                    "tracing the boundary line as actually drawn. 8 to 40 points for a typical "
                    "curtilage. Empty if found is false.")
    what_marked_it: str = Field(description="In your own words, what you traced and why.")
    confidence: float = Field(description="0.0 to 1.0")


@dataclass
class Window:
    """A crop of the page, in page pixels, plus the scale it is sent at."""

    x: int
    y: int
    w: int
    h: int
    scale: float

    @property
    def sent_size(self) -> tuple[int, int]:
        return int(round(self.w * self.scale)), int(round(self.h * self.scale))

    def to_page(self, pts) -> np.ndarray:
        """Crop-image coordinates -> page coordinates."""
        p = np.asarray(pts, float) / self.scale
        p[:, 0] += self.x
        p[:, 1] += self.y
        return p

    def to_crop(self, pts) -> np.ndarray:
        p = np.asarray(pts, float) - [self.x, self.y]
        return p * self.scale


def window_for(box, page_w: int, page_h: int, pad: float = PAD) -> Window:
    """A window that CONTAINS the box, padded, inside the page.

    Square when a square that contains the padded box fits on the page, because a square keeps the
    x and y quantisation of the returned integer coordinates identical. When it does not fit --
    a long plot on a page whose short edge is shorter than the plot -- the window becomes a
    rectangle rather than silently cropping the box, which is the bug that made 90-01394 fail.
    """
    x0, y0, x1, y1 = (float(c) for c in box)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    side = max(MIN_SIDE, pad * max(bw, bh))
    if side <= min(page_w, page_h):
        w = h = side
    else:
        w = min(float(page_w), max(MIN_SIDE, pad * bw, bw))
        h = min(float(page_h), max(MIN_SIDE, pad * bh, bh))
    w, h = min(w, float(page_w)), min(h, float(page_h))

    gx = float(np.clip(cx - w / 2, 0, page_w - w))
    gy = float(np.clip(cy - h / 2, 0, page_h - h))
    w, h = int(round(w)), int(round(h))
    gx, gy = int(round(gx)), int(round(gy))
    # never scale down: a shrunk crop measured worse than no crop at all
    scale = max(1.0, TARGET_LONG / max(w, h))
    return Window(gx, gy, w, h, scale)


def window_contains_box(win: Window, box) -> bool:
    x0, y0, x1, y1 = box
    return (win.x <= x0 and win.y <= y0
            and win.x + win.w >= x1 and win.y + win.h >= y1)


def crop_of(gray: np.ndarray, win: Window) -> np.ndarray:
    patch = gray[win.y:win.y + win.h, win.x:win.x + win.w]
    if win.scale == 1.0:
        return patch
    return cv2.resize(patch, win.sent_size, interpolation=cv2.INTER_CUBIC)


def valid_point(point) -> bool:
    return bool(point and len(point) == 2 and point != [0, 0])


def point_inside_ring(point, ring, tol: float = 6.0) -> bool | None:
    """Whether a locator point is inside/on a candidate ring, with boundary tolerance."""
    if not valid_point(point) or not ring:
        return None
    distance = cv2.pointPolygonTest(
        np.asarray(ring, np.int32).reshape(-1, 1, 2),
        (float(point[0]), float(point[1])), True)
    return bool(distance >= -tol)


def guidance_message(win: Window, guidance: dict | None, retry_note: str = "") -> str:
    """Describe whole-page locator evidence in the crop coordinate system."""
    if not guidance:
        return retry_note

    def crop_xy(point):
        if not valid_point(point):
            return None
        p = win.to_crop([point])[0]
        return [int(round(p[0])), int(round(p[1]))]

    target = crop_xy(guidance.get("point"))
    tip = crop_xy(guidance.get("tip")) if guidance.get("marking_spatial") else None
    lines = [
        "The second image is an annotated copy of the first. Its coloured marks are artificial "
        "guidance from a whole-page locator; NEVER trace the coloured marks themselves.",
    ]
    if target:
        lines.append(
            f"The cyan TARGET circle is at crop coordinate {target}. Your closed ring MUST "
            "contain this point because it lies inside the intended application plot.")
    if tip:
        lines.append(
            f"The magenta MARKING cross is at crop coordinate {tip}. It is the spatial marking "
            "that identifies the target, so the ring must contain or touch it.")
    if guidance.get("locate_said"):
        lines.append(f"The whole-page locator said: {guidance['locate_said']}")
    if guidance.get("marking_kind"):
        lines.append(f"Marking type: {guidance['marking_kind']}.")
    if guidance.get("target_geometry"):
        lines.append(f"Locator geometry assessment: {guidance['target_geometry']}.")
    if retry_note:
        lines.append(retry_note)
    return "\n".join(lines)


def annotated_crop(crop: np.ndarray, win: Window, guidance: dict | None) -> np.ndarray | None:
    """A second, visibly annotated image that carries locator evidence into tracing."""
    if not guidance:
        return None
    image = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    height, width = crop.shape

    def crop_xy(point):
        if not valid_point(point):
            return None
        p = win.to_crop([point])[0]
        x, y = int(round(p[0])), int(round(p[1]))
        return (x, y) if 0 <= x < width and 0 <= y < height else None

    box = guidance.get("box")
    if box and len(box) == 4:
        p0, p1 = win.to_crop([[box[0], box[1]], [box[2], box[3]]])
        x0, y0 = np.round(p0).astype(int)
        x1, y1 = np.round(p1).astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 155, 0), 5, cv2.LINE_AA)
        cv2.putText(image, "LOCATOR BOX", (max(5, x0 + 8), max(24, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 155, 0), 2, cv2.LINE_AA)
    target = crop_xy(guidance.get("point"))
    if target:
        cv2.circle(image, target, 13, (0, 190, 255), 5, cv2.LINE_AA)
        cv2.putText(image, "TARGET", (target[0] + 16, max(24, target[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 190, 255), 2, cv2.LINE_AA)
    if guidance.get("marking_spatial"):
        tip = crop_xy(guidance.get("tip"))
        if tip:
            cv2.drawMarker(image, tip, (255, 0, 210), cv2.MARKER_CROSS, 30, 5, cv2.LINE_AA)
            cv2.putText(image, "MARKING", (tip[0] + 16, min(height - 8, tip[1] + 28)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 210), 2, cv2.LINE_AA)
    return image


def load_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if PROMPT_SHA16:
        got = hashlib.sha256(text.encode()).hexdigest()[:16]
        if got != PROMPT_SHA16:
            raise RuntimeError(f"boundary_prompt.txt changed (sha256 {got}, "
                               f"expected {PROMPT_SHA16}); re-measure or restore it.")
    return text


def build_agent(prompt: str, max_tokens: int = 6000, retries: int = 1,
                timeout_s: float = 180.0) -> Agent:
    """One agent, reused across pages.

    Retries are capped at 1. The default of 3 (plus an output retry) let a single page issue a
    dozen calls when the model kept returning something the schema rejected -- one trace step
    measured at 652 s against a 46 s median, which at 20 workers meant the whole batch waited on
    one drawing. A page that will not validate twice is better recorded as an error and re-run
    later than retried until it blocks the queue.
    """
    return Agent("test", output_type=NativeOutput(BoundaryAnswer), retries=retries,
                 output_retries=0,
                 model_settings={"max_tokens": max_tokens, "timeout": timeout_s},
                 instructions=prompt)


@dataclass
class Boundary:
    stem: str
    page_w: int = 0
    page_h: int = 0
    window: list[int] = field(default_factory=list)
    scale: float = 1.0
    found: bool | None = None
    vertices_page: list[list[int]] | None = None
    said: str | None = None
    confidence: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def trace_one(agent: Agent, model, gray: np.ndarray, stem: str, win: Window,
              case_block: str = "", guidance: dict | None = None,
              retry_note: str = "") -> Boundary:
    """One tracing call. Never raises: a failure comes back as Boundary.error."""
    import time

    h, w = gray.shape
    res = Boundary(stem=stem, page_w=w, page_h=h,
                   window=[win.x, win.y, win.w, win.h], scale=round(win.scale, 3))
    crop = crop_of(gray, win)
    sw, sh = crop.shape[1], crop.shape[0]
    buf = io.BytesIO()
    Image.fromarray(crop).convert("RGB").save(buf, format="PNG")
    messages = [BinaryContent(data=buf.getvalue(), media_type="image/png")]
    marked = annotated_crop(crop, win, guidance)
    if marked is not None:
        marked_buf = io.BytesIO()
        Image.fromarray(marked).save(marked_buf, format="PNG")
        messages.append(BinaryContent(data=marked_buf.getvalue(), media_type="image/png"))
    guide = guidance_message(win, guidance, retry_note)

    started = time.time()
    try:
        run = agent.run_sync(
            messages + [f"This crop is {sw} x {sh} pixels. Trace the application plot boundary."
                        + case_block + (f"\n\n{guide}" if guide else "")],
            model=model, usage_limits=UsageLimits(request_limit=4))
    except Exception as exc:                      # noqa: BLE001 - one page must not stop a batch
        res.error = f"{exc!s:.300}"
        res.seconds = round(time.time() - started, 2)
        return res

    out, usage = run.output, run.usage()
    res.seconds = round(time.time() - started, 2)
    res.tokens_in = usage.input_tokens or 0
    res.tokens_out = usage.output_tokens or 0
    res.usd = round(res.tokens_in * USD_PER_MTOK_IN / 1e6
                    + res.tokens_out * USD_PER_MTOK_OUT / 1e6, 6)
    res.found = out.found
    res.said = out.what_marked_it
    res.confidence = out.confidence
    if out.found and len(out.vertices) >= 3:
        pts = [[v.x, v.y] for v in out.vertices]
        res.vertices_page = np.round(win.to_page(pts)).astype(int).tolist()
    return res


def candidate_rank(candidate: dict) -> tuple:
    """Rank trace candidates by identity first, then completeness, then line quality.

    The first v1.1 rule only accepted a larger regrown polygon. That kept a known-bad clipped
    answer on 92-00392 even though the wider crop produced a clean closed outline. Identity is a
    hard requirement; a tidy ring around a neighbour must never beat the intended plot.
    """
    identity_ok = candidate.get("identity_ok") is True
    no_cut = not (candidate.get("edges") or {}).get("cut_sides")
    not_fallback = not candidate.get("fallback")
    blank = float(candidate.get("blank_share", 1.0))
    within = float(candidate.get("frac_within_3px", 0.0))
    p90 = float(candidate.get("ring_to_ink_p90_px", 1e9))
    return (identity_ok, no_cut, not_fallback, -blank, within, -p90)


def choose_candidate(first: dict, second: dict) -> tuple[dict, str]:
    """Choose between independently traced candidates without anchoring to the first mistake."""
    if candidate_rank(second) > candidate_rank(first):
        return second, f"selected {second.get('source', 'second')} by identity/completeness/ink rank"
    return first, f"kept {first.get('source', 'first')} by identity/completeness/ink rank"


# --------------------------------------------------------------------------- checks


def ink_distance(gray: np.ndarray) -> np.ndarray:
    """Distance in px from every pixel to the nearest ink pixel."""
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return cv2.distanceTransform((t == 0).astype(np.uint8), cv2.DIST_L2, 3)


def ring_stats(vertices_page, dt: np.ndarray) -> dict:
    """How much of the ring lies on drawn ink, by LENGTH.

    The median alone is misleading: a ring half on ink and half over nothing still has a low
    median. `blank_share` -- the fraction of ring length further than BLANK_PX from any ink --
    is what separates a traced boundary from an invented one, and it correlates with p90 at 0.93.
    """
    h, w = dt.shape
    p = np.asarray(vertices_page, float)
    n = len(p)
    lengths, dists = [], []
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        seg = float(np.hypot(*(b - a)))
        k = max(2, int(seg / 2))
        q = a + (b - a) * np.linspace(0, 1, k, endpoint=False)[:, None]
        d = dt[np.clip(np.round(q[:, 1]).astype(int), 0, h - 1),
               np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)]
        lengths.append(seg)
        dists.append(d)
    total = sum(lengths) or 1.0
    alld = np.concatenate(dists)
    blank = sum(L * float((d > BLANK_PX).mean()) for L, d in zip(lengths, dists)) / total
    ring = np.round(p).astype(np.int32).reshape(-1, 1, 2)
    return {"n_vertices": n,
            "ring_to_ink_median_px": round(float(np.median(alld)), 2),
            "ring_to_ink_p90_px": round(float(np.percentile(alld, 90)), 2),
            "frac_within_3px": round(float((alld <= 3).mean()), 3),
            "blank_share": round(float(blank), 3),
            "area_pct": round(100.0 * abs(cv2.contourArea(ring)) / (w * h), 3)}


def edge_contact(vertices_page, win: Window, page_w: int, page_h: int) -> dict:
    """Which crop edges the ring runs along, split by whether they are ours or the paper's.

    A ring on a crop edge that sits INSIDE the page means our window cut the plot, and re-cropping
    can recover it. A ring on a crop edge that IS the page edge means the plot runs off the sheet;
    nothing is recoverable there and widening is pointless. Conflating the two made 89-01771 look
    like a fixable crop failure when it is not.
    """
    p = win.to_crop(vertices_page)
    sw, sh = win.sent_size
    hit = {"L": bool((p[:, 0] < 3).any()), "R": bool((p[:, 0] > sw - 4).any()),
           "T": bool((p[:, 1] < 3).any()), "B": bool((p[:, 1] > sh - 4).any())}
    at_page = {"L": win.x <= 1, "R": win.x + win.w >= page_w - 1,
               "T": win.y <= 1, "B": win.y + win.h >= page_h - 1}
    on = ((p[:, 0] < 3) | (p[:, 0] > sw - 4) | (p[:, 1] < 3) | (p[:, 1] > sh - 4))
    return {"cut_sides": [k for k in "LRTB" if hit[k] and not at_page[k]],
            "off_page_sides": [k for k in "LRTB" if hit[k] and at_page[k]],
            "frac_vertices_on_edge": round(float(on.mean()), 3)}


def grow_window(win: Window, sides, page_w: int, page_h: int, grow: float = GROW) -> Window:
    """Extend only the sides the ring actually ran along.

    Growing uniformly does not work: padding 1.25 -> 2.0 moved truncation from 9/20 pages to 7/20,
    because a symmetric enlargement around an off-centre box leaves the short side short. Reading
    the sides off the first ring took 5 flagged pages to 1.
    """
    g = grow * max(win.w, win.h)
    x0 = win.x - (g if "L" in sides else 0)
    x1 = win.x + win.w + (g if "R" in sides else 0)
    y0 = win.y - (g if "T" in sides else 0)
    y1 = win.y + win.h + (g if "B" in sides else 0)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(page_w), x1), min(float(page_h), y1)
    w, h = int(round(x1 - x0)), int(round(y1 - y0))
    scale = max(1.0, TARGET_LONG / max(w, h))
    return Window(int(round(x0)), int(round(y0)), w, h, scale)


def accept_regrow(before: dict, after: dict, before_pts, after_pts,
                  page_w: int, page_h: int) -> tuple[bool, str]:
    """Take the second trace only if it EXTENDS the first rather than replacing it.

    "The ring no longer touches an edge" is not a repair criterion -- it is satisfied by throwing
    the answer away. 90-01770 passed that test while shrinking 31% and keeping only 57% of the
    original area, and its p90 improved purely because the edge-following segments were gone.
    A genuine repair grows into the newly visible area and keeps what it had.
    """
    if after is None or after_pts is None:
        return False, "second trace produced nothing"

    def mask(pts):
        m = np.zeros((page_h, page_w), np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(pts, float)).astype(np.int32)], 1)
        return m > 0

    a, b = mask(before_pts), mask(after_pts)
    kept = float((a & b).sum() / max(1, a.sum()))
    if kept < 0.90:
        return False, f"second ring keeps only {kept*100:.0f}% of the first -- a different answer"
    if after["area_pct"] <= before["area_pct"]:
        return False, "second ring is no larger, so nothing was recovered"
    return True, f"extends the first ring (keeps {kept*100:.0f}%, area up)"


def review_flags(stats: dict, edges: dict, snap_iou: float | None = None,
                 self_iou: float | None = None, box_agree: float | None = None) -> list[str]:
    """What a reviewer should look at. Not corrections -- none of these can say which side is wrong."""
    out = []
    if box_agree is not None and box_agree < 0.5:
        out.append(f"the two locate calls disagree about where the plot is (box IoU "
                   f"{box_agree:.2f}) -- the boundary may be round the wrong thing")
    if edges.get("cut_sides"):
        out.append(f"the window cuts the plot on the {'/'.join(edges['cut_sides'])} side")
    if edges.get("off_page_sides"):
        out.append(f"the plot runs off the sheet on the {'/'.join(edges['off_page_sides'])} side "
                   f"(not a crop problem)")
    if stats.get("blank_share", 0) > 0.10:
        out.append(f"{stats['blank_share']*100:.0f}% of the ring lies over blank paper, so those "
                   f"sides were not traced from anything drawn")
    if self_iou is not None and self_iou < 0.5:
        out.append(f"two runs disagree with each other (IoU {self_iou:.2f})")
    if snap_iou is not None and snap_iou < 0.3:
        out.append(f"the drawn line network does not enclose it (snap IoU {snap_iou:.2f})")
    return out

def box_iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return round(float(inter / ua), 3) if ua > 0 else 0.0


def reconcile_boxes(a, b, agree_at: float = 0.5) -> tuple[list, float, bool]:
    """Two independent locate calls -> the box to use, their IoU, and whether they agreed.

    Stage 1 is the pipeline's dominant risk and nothing downstream can detect it: a box round the
    wrong region produces a perfectly tidy ring round the wrong plot, with every ink-based metric
    looking healthy. Measured over two full runs of the same 20 pages, the box IoU between runs had
    a median of 0.85 but fell below 0.5 on 3 -- and one of those, 89-00135, moved from the small
    bold rectangle that "THE SITE" points at to a nine-times-larger field, taking the ring with it.

    When they disagree the SMALLER box is used. That is a weak preference from 3 disagreements --
    the smaller box was the better one twice and neutral once -- and it also preserves enlargement,
    so it is the safer default rather than a measured optimum. Either way the page is flagged.
    """
    iou = box_iou(a, b)
    if iou >= agree_at:
        return list(a), iou, True
    area = lambda q: (q[2] - q[0]) * (q[3] - q[1])
    return (list(a) if area(a) <= area(b) else list(b)), iou, False
