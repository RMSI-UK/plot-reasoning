"""Draw what the model returned, and nothing else.

The point and box are drawn where the model put them. Nothing is repaired, snapped or traced --
if a coordinate is wrong, the overlay shows it wrong, which is the point of looking.

Colours are declared in BGR beside their meaning. cv2 takes BGR and an RGB-looking tuple
renders as the wrong colour; that mistake was made twice while this was being built, and both
times the legend said one thing while the pixels said another.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

POINT = (200, 30, 190)   # BGR -> magenta
BOX = (20, 150, 235)     # BGR -> amber
LONG_SIDE = 900          # output size; the originals are ~1 MP


def render_one(image_path: Path, row: dict, out_path: Path) -> bool:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return False
    h, w = gray.shape
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lw = max(2, min(h, w) // 300)
    fs = max(0.5, min(h, w) / 1500)

    def label(x: int, y: int, col, text: str) -> None:
        for colour, thick in ((255, 255, 255), lw + 2), (col, max(1, lw - 1)):
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        colour, thick, cv2.LINE_AA)

    box = row.get("box")
    if box and len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
        x0, y0 = max(0, box[0]), max(0, box[1])
        x1, y1 = min(w - 1, box[2]), min(h - 1, box[3])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), BOX, lw)
        label(x0, max(14, y0 - 6), BOX, f"box {box[0]},{box[1]} {box[2]},{box[3]}")

    point = row.get("point")
    if point:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            r = max(10, lw * 5)
            cv2.line(canvas, (x - r, y), (x + r, y), POINT, lw + 1, cv2.LINE_AA)
            cv2.line(canvas, (x, y - r), (x, y + r), POINT, lw + 1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), r, POINT, lw, cv2.LINE_AA)
            label(x + r + 5, max(14, y - 8), POINT, f"point {x},{y}")
        else:
            # out of range: say so on the image rather than dropping the marker silently
            label(10, 24, POINT, f"point {x},{y} is OUTSIDE this {w}x{h} image")

    scale = LONG_SIDE / max(h, w)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path),
                cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 89])
    return True


def render_all(src: Path, jsonl: Path, out_dir: Path) -> int:
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    made = 0
    for row in rows:
        if row.get("error"):
            continue
        for ext in (".jpg", ".png"):
            candidate = src / f"{row['stem']}{ext}"
            if candidate.exists():
                if render_one(candidate, row, out_dir / f"{row['stem']}.jpg"):
                    made += 1
                break
    return made


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render overlays from a results.jsonl")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(f"rendered {render_all(a.src, a.jsonl, a.out)} overlays to {a.out}")
