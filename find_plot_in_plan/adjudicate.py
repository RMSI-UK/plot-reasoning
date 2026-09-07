#!/usr/bin/env python3
"""When the locators disagree, ask a model to choose between them instead of searching again.

    python adjudicate.py --out <dir> [--model openai/gpt-5.6-terra] [--workers 12]

Measured on the blind 100 (78 pages with a hand-drawn boundary):

    the vote picks                        57/78 = 73%
    at least one candidate was right      69/78 = 88%
    unanimous pages   27, picked right 26 = 96%   -- nothing to fix here
    split pages       50, picked right 31 = 62%,  ceiling 42 = 84%

So 11 pages -- 14 points of the total -- are sitting in candidates already paid for, and no
reordering recovers them: seven rules were tried (smallest box, highest confidence, prefer luna,
median box, and so on) and none beat the 73% the current rule gets. The information is missing, not
mis-sorted.

Choosing among two or three drawn boxes is a different and much easier task than finding a plot on
a whole sheet: the answer is usually present, the alternatives are explicit, and the council's
address can be matched against a house number written on the paper. This asks exactly that
question, and only on the pages where the locators actually disagreed.
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
from pydantic import BaseModel, Field                             # noqa: E402
from pydantic_ai import Agent, BinaryContent, NativeOutput        # noqa: E402
from pydantic_ai.usage import UsageLimits                         # noqa: E402
from geoplanagent.utils import resolve_model                      # noqa: E402
from metadata import records_for                                  # noqa: E402

GEO = Path("/data/braintree/scan-processed/wp7-panels-sam3-seg/geometry")
SRC = Path("/data/braintree/scan-processed/scan-processed-wp7-multi-plan-panels/plan")
RATES = {"openai/gpt-5.6-terra": (1.00, 6.00), "openai/gpt-5.6-luna-pro": (0.10, 0.60),
         "openai/gpt-5.6-luna": (0.10, 0.60), "google/gemini-3.7-flash": (0.375, 1.875)}
LETTERS = "ABCD"
# BGR, deliberately far apart so a bilevel scan cannot be confused with a box
COLOURS = [(0, 0, 235), (200, 120, 0), (0, 160, 0), (200, 0, 200)]

PROMPT = """Several automatic locators were asked which plot a planning application concerns, and
they disagreed. You are shown the drawing twice: once clean, and once with each locator's answer
drawn as a labelled coloured rectangle.

Choose the rectangle that best contains the plot this application is about, and say why.

The rectangles are proposals, not boundaries. A rectangle is right if the plot is inside it and it
is not mostly other people's land; do not reject one for being a few pixels loose.

What actually decides it:
  - the council's address below. A house number or property name from it is very often written on
    the drawing, next to or inside the right plot.
  - the proposal text. It names what is being built, which is usually the hatched, stippled or
    newly outlined part, and that part lies inside the right plot.
  - an arrow or leader line from a label. Follow it to where it ENDS. The words themselves sit
    wherever there was room to write them and are often over a different, larger plot entirely.
  - a heavier or darker outline drawn round one plot.

Two traps, both seen on these drawings:
  - a rectangle that covers a whole terrace, close or block when the application is one house in
    it. Prefer the one holding a single property.
  - a rectangle round a large plot that merely happens to contain the label text. Prefer the one
    the arrow points AT.

If two rectangles both contain the right plot, choose the tighter one. If none of them contains it,
answer "none" and say where the plot actually is."""


class Choice(BaseModel):
    choice: str = Field(description='The letter of the best rectangle, or "none".')
    why: str = Field(description="In your own words, what decided it. Quote the marks you used.")
    house_number_seen: str = Field(
        description="Any house number or property name you can read on the drawing inside or "
                    "beside your chosen rectangle. Empty if none is legible.")
    confidence: float = Field(description="0.0 to 1.0")


def render(gray, boxes):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    th = max(2, int(round(max(h, w) / 480)))
    for i, b in enumerate(boxes):
        col = COLOURS[i % len(COLOURS)]
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)
        lab = LETTERS[i]
        fs = max(0.7, max(h, w) / 900)
        (tw, tht), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x, y = int(b[0]), max(tht + 4, int(b[1]) - 4)
        cv2.rectangle(vis, (x, y - tht - 4), (x + tw + 8, y + 4), col, -1)
        cv2.putText(vis, lab, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    return vis


def one(stem, cands, agent, model, model_name, case):
    gray = cv2.imread(str(next(iter(SRC.glob(f"{stem}.*")))), cv2.IMREAD_GRAYSCALE)
    boxes = [c["box"] for c in cands]
    bufs = []
    for img in (gray, render(gray, boxes)):
        b = io.BytesIO()
        Image.fromarray(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(b, "PNG")
        bufs.append(b.getvalue())
    listing = "\n".join(
        f"  {LETTERS[i]}  x0={c['box'][0]} y0={c['box'][1]} x1={c['box'][2]} y1={c['box'][3]}"
        f"   (that locator said: {(c.get('said') or '')[:150]})"
        for i, c in enumerate(cands))
    out = {"stem": stem, "model": model_name, "n_cands": len(cands),
           "letters": {LETTERS[i]: cands[i]["model"] for i in range(len(cands))}}
    t0 = time.time()
    try:
        run = agent.run_sync(
            [BinaryContent(data=bufs[0], media_type="image/png"),
             BinaryContent(data=bufs[1], media_type="image/png"),
             f"The drawing is {gray.shape[1]} x {gray.shape[0]} pixels.\n\nThe rectangles:\n"
             + listing + (case.as_prompt_block() if case else "")],
            model=model, usage_limits=UsageLimits(request_limit=3))
    except Exception as exc:                                       # noqa: BLE001
        out["error"] = f"{exc!s:.200}"
        return out
    o, u = run.output, run.usage()
    cin, cout = RATES.get(model_name, (1.0, 6.0))
    out.update({"seconds": round(time.time()-t0, 1), "choice": (o.choice or "").strip().upper()[:4],
                "why": o.why, "house_number_seen": o.house_number_seen,
                "confidence": o.confidence,
                "usd": round((u.input_tokens or 0)*cin/1e6 + (u.output_tokens or 0)*cout/1e6, 6)})
    idx = LETTERS.find(out["choice"][:1]) if out["choice"][:1] in LETTERS else -1
    if 0 <= idx < len(cands):
        out["picked_model"] = cands[idx]["model"]
        out["picked_box"] = cands[idx]["box"]
        out["picked_point"] = cands[idx]["point"]
    return out


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix*iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/u if u > 0 else 0.0


def is_split(cands):
    """True when the candidate boxes do not all agree -- the only pages worth adjudicating."""
    return any(box_iou(a["box"], b["box"]) < 0.5
               for a, b in ((cands[i], cands[j])
                            for i in range(len(cands)) for j in range(i+1, len(cands))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="openai/gpt-5.6-terra")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--runs", nargs="*",
                    default=["snapv2_blind100", "snapv2", "snapv2_noboundary"])
    args = ap.parse_args()

    jobs = {}
    for run in args.runs:
        p = GEO/run/"boundaries.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cands = [c for c in (r.get("locate") or []) if c.get("found") and c.get("box")]
            if len(cands) >= 2 and is_split(cands):
                jobs[r["stem"]] = cands

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out/f"adjudicate_{args.model.split('/')[-1]}.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["stem"])
    todo = [s for s in jobs if s not in done]
    print(f"{len(jobs)} split pages, {len(todo)} to run, model {args.model}", flush=True)
    if not todo:
        return 0
    cases = records_for(todo)
    agent = Agent("test", output_type=NativeOutput(Choice), retries=1, output_retries=0,
                  model_settings={"max_tokens": 4000, "timeout": 180}, instructions=PROMPT)
    model = resolve_model(args.model)
    lock = threading.Lock()
    t0 = time.time()
    with jsonl.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, s, jobs[s], agent, model, args.model, cases.get(s)): s
                for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False)+"\n")
                fh.flush()
            print(f"[{i}/{len(todo)}] {time.time()-t0:5.0f}s {r['stem'][:36]:36s} "
                  f"-> {r.get('choice','ERR')}  {r.get('house_number_seen','')[:18]}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
