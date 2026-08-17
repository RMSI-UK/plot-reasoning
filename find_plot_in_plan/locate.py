"""Locate the application plot on one scanned planning drawing.

One model call per page. No OpenCV, no post-processing: the point and box are returned exactly
as the model gave them, with the coordinate range checked but not silently corrected.

The prompt lives in prompt.txt and is deliberately open-ended -- it names no cues to look for.
Every earlier attempt that enumerated cues ("a label with a leader line, hatching, a house
number") hit that list as a ceiling and declined on pages marked some other way; one such list
declined on 42.5% of pages. Removing it entirely, and keeping only the gate "if this is not a
plan of land, decline", is what made this work.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.usage import UsageLimits

MODEL = "openai/gpt-5.6-terra"
PROMPT_PATH = Path(__file__).with_name("prompt.txt")
PROMPT_SHA16 = "4d47e5df8d26de16"   # guards against silent prompt drift

# measured on OpenRouter 2026-08; re-fetch with `python run.py --price` to confirm
USD_PER_MTOK_IN = 1.00
USD_PER_MTOK_OUT = 6.00


class _XY(BaseModel):
    x: int
    y: int


class _Box(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class PlotAnswer(BaseModel):
    """What the model returns. Field descriptions are part of the contract -- editing them
    changes the model's behaviour as much as editing the prompt does."""

    is_plan: bool = Field(description="True if this page is a PLAN OF LAND seen from above.")
    plan_kind: str = Field(description="What the page actually is, in your own words.")
    found: bool = Field(
        description="True if you can tell which single plot this application is about.")
    point_xy: _XY = Field(description="A point INSIDE that plot. Zeros if found is false.")
    box_xy: _Box = Field(
        description="Axis-aligned box just containing the whole plot. Zeros if found is false.")
    what_marked_it: str = Field(
        description="IN YOUR OWN WORDS: what on the page told you it was this plot.")
    confidence: float = Field(description="0.0 to 1.0")
    evidence: str = Field(description="Quote the marks.")


@dataclass
class CaseRecord:
    """The council's own record for the application, if available. Optional.

    Measured effect on 20 unseen pages: two declines became answers, the remaining points moved
    a median of 33 px, and the model's stated reasons started citing house numbers. It reads as
    confirmation rather than redirection. One caveat worth carrying: on one page the
    plan-or-not verdict flipped once the record was supplied, and whether a page is a plan is a
    property of the image that text about the application should not be able to change.
    """

    reference: str = ""
    address: str = ""
    proposal: str = ""

    def as_prompt_block(self) -> str:
        if not (self.reference or self.address or self.proposal):
            return ""
        return (
            f"\n\nCase file details for this application, from the council's records:\n"
            f"  reference: {self.reference}\n"
            f"  address: {self.address}\n"
            f"  proposal: {self.proposal}\n"
            f"Use these to identify the plot. A house number or property name from the address "
            f"is often written on the drawing, and the proposal names what is being built, "
            f"which is often the part that is hatched or newly outlined."
        )


@dataclass
class Result:
    stem: str
    width: int
    height: int
    is_plan: bool | None = None
    plan_kind: str | None = None
    found: bool | None = None
    point: list[int] | None = None
    box: list[int] | None = None
    point_in_range: bool | None = None
    box_in_range: bool | None = None
    what_marked_it: str | None = None
    confidence: float | None = None
    evidence: str | None = None
    case_reference: str | None = None
    case_address: str | None = None
    case_proposal: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_prompt() -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    import hashlib
    got = hashlib.sha256(text.encode()).hexdigest()[:16]
    if got != PROMPT_SHA16:
        raise RuntimeError(
            f"prompt.txt has changed (sha256 {got}, expected {PROMPT_SHA16}). Every measured "
            f"figure in README.md was produced with the original text. Update PROMPT_SHA16 "
            f"deliberately and re-measure, or restore the file."
        )
    return text


def build_agent(prompt: str, max_tokens: int = 4096) -> Agent:
    """One agent, reused across pages.

    No temperature is set: this model runs with reasoning enabled and rejects sampling
    parameters, warning and ignoring them. Its answers therefore vary slightly between runs --
    measured at a 33 px median point shift over 18 pages, with the is_plan and found verdicts
    stable on all 20.
    """
    return Agent(
        "test",                                  # overridden per call
        output_type=NativeOutput(PlotAnswer),
        retries=3,
        output_retries=0,
        model_settings={"max_tokens": max_tokens},
        instructions=prompt,
    )


def locate_one(agent: Agent, model, image_path: Path,
               case: CaseRecord | None = None) -> Result:
    """Run one page. Never raises: a failure comes back as Result.error."""
    import time

    with Image.open(image_path) as im:
        width, height = im.size
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        png = buf.getvalue()

    res = Result(stem=image_path.stem, width=width, height=height)
    if case:
        res.case_reference, res.case_address, res.case_proposal = (
            case.reference, case.address, case.proposal)

    started = time.time()
    try:
        run = agent.run_sync(
            [BinaryContent(data=png, media_type="image/png"),
             f"This image is W={width} pixels wide and H={height} pixels high. "
             f"Is it a plan of land, and if so where is the application plot?"
             f"{case.as_prompt_block() if case else ''}"],
            model=model, usage_limits=UsageLimits(request_limit=4))
    except Exception as exc:                      # noqa: BLE001 - one page must not stop a batch
        res.error = f"{exc!s:.300}"
        res.seconds = round(time.time() - started, 2)
        return res

    out = run.output
    usage = run.usage()
    res.seconds = round(time.time() - started, 2)
    res.tokens_in = usage.input_tokens or 0
    res.tokens_out = usage.output_tokens or 0
    res.usd = round(res.tokens_in * USD_PER_MTOK_IN / 1e6
                    + res.tokens_out * USD_PER_MTOK_OUT / 1e6, 6)
    res.is_plan = out.is_plan
    res.plan_kind = out.plan_kind
    res.found = out.found
    res.what_marked_it = out.what_marked_it
    res.confidence = out.confidence
    res.evidence = out.evidence
    if out.found:
        px, py = out.point_xy.x, out.point_xy.y
        b = [out.box_xy.x0, out.box_xy.y0, out.box_xy.x1, out.box_xy.y1]
        res.point = [px, py]
        res.box = b
        # Flagged, never silently clamped: an out-of-range coordinate means the model
        # misread the stated dimensions, and hiding that would hide the misread. Measured
        # at 0 of 40 pages for this model, but it does happen with others.
        res.point_in_range = bool(0 <= px < width and 0 <= py < height)
        res.box_in_range = bool(
            b[2] > b[0] and b[3] > b[1]
            and 0 <= b[0] < width and 0 <= b[1] < height
            and 0 < b[2] <= width and 0 < b[3] <= height)
    return res
