#!/usr/bin/env python3
"""Render a self-contained visual before/after report for snap_boundary v1.2."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np


FOCUS = {
    "89-00624-P_001-001_0001_p2",
    "93-01583-P_001-001_0001_p1",
    "93-00570-LB-93-767-P_001-002_0001_p2",
    "92-00392_001-001_0001_rec1",
    "91-01330-P-FHS_001-001_0002_p2",
}


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def read_rows(path: Path) -> dict[str, dict]:
    return {
        row["stem"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def status(row: dict) -> tuple[str, str]:
    if not row.get("vertices_page"):
        return "declined", "未输出"
    if row.get("auto_usable") is True:
        return "auto", "自动可用"
    if row.get("auto_usable") is False or row.get("flags"):
        return "review", "人工复核"
    return "auto", "检查通过"


def draw_box(image: np.ndarray, box: list | None, colour: tuple[int, int, int], label: str) -> None:
    if not box or len(box) != 4:
        return
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, 3, cv2.LINE_AA)
    cv2.putText(image, label, (x0 + 5, max(20, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, colour, 2, cv2.LINE_AA)


def overlay(gray: np.ndarray, row: dict) -> np.ndarray:
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    window = row.get("window")
    if window and len(window) == 4:
        x, y, w, h = map(int, window)
        draw_box(image, [x, y, x + w, y + h], (35, 155, 80), "TRACE WINDOW")
    draw_box(image, row.get("box_used"), (0, 160, 245), "LOCATE BOX")
    ring = row.get("vertices_page") or []
    if len(ring) >= 3:
        points = np.asarray(ring, np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], True, (25, 25, 225), 5, cv2.LINE_AA)
        for x, y in ring:
            cv2.circle(image, (int(x), int(y)), 4, (0, 220, 255), -1, cv2.LINE_AA)
    point = row.get("point")
    if point and point != [0, 0]:
        cv2.circle(image, tuple(map(int, point)), 10, (230, 150, 30), 4, cv2.LINE_AA)
    tip = row.get("tip")
    if tip and tip != [0, 0] and row.get("marking_spatial", True):
        cv2.drawMarker(image, tuple(map(int, tip)), (210, 25, 210), cv2.MARKER_CROSS,
                       28, 4, cv2.LINE_AA)
    return image


def bounds(row: dict, width: int, height: int) -> tuple[int, int, int, int]:
    points: list[tuple[int, int]] = []
    for x, y in row.get("vertices_page") or []:
        points.append((int(x), int(y)))
    box = row.get("box_used")
    if box and len(box) == 4:
        points.extend([(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))])
    for key in ("point", "tip"):
        point = row.get(key)
        if point and point != [0, 0]:
            points.append(tuple(map(int, point)))
    if not points:
        return 0, 0, width, height
    xs, ys = zip(*points)
    pad = max(35, int(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.18))
    return (max(0, min(xs) - pad), max(0, min(ys) - pad),
            min(width, max(xs) + pad), min(height, max(ys) + pad))


def metric(name: str, old: object, new: object) -> str:
    return (f"<tr><th>{esc(name)}</th><td>{esc(old)}</td><td>{esc(new)}</td></tr>")


def row_notes(row: dict) -> str:
    notes = list(row.get("flags") or [])
    if row.get("skipped"):
        notes.insert(0, row["skipped"])
    if not notes:
        notes = ["自动检查未发现阻断问题；仍不等于人工 ground truth。"]
    return "".join(f"<li>{esc(note)}</li>" for note in notes)


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    args = parser.parse_args()

    old_rows = read_rows(args.old / "boundaries.jsonl")
    new_rows = read_rows(args.new / "boundaries.jsonl")
    old_summary = json.loads((args.old / "summary.json").read_text())
    new_summary = json.loads((args.new / "summary.json").read_text())
    shots = args.new / "validation_shots"
    shots.mkdir(exist_ok=True)

    for stem, new_row in new_rows.items():
        gray = cv2.imread(str(args.src / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        for version, row in (("v11", old_rows.get(stem, {})), ("v12", new_row)):
            image = overlay(gray, row)
            x0, y0, x1, y1 = bounds(row, gray.shape[1], gray.shape[0])
            crop = image[y0:y1, x0:x1]
            cv2.imwrite(str(shots / f"{stem}__{version}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

    ordered = sorted(new_rows, key=lambda stem: (stem not in FOCUS, stem))
    cards: list[str] = []
    for stem in ordered:
        old = old_rows.get(stem, {})
        new = new_rows[stem]
        old_cls, old_label = status(old)
        new_cls, new_label = status(new)
        focus = "重点反馈" if stem in FOCUS else "随机样本"
        cards.append(f"""
<article class="case {new_cls}" data-status="{new_cls}">
  <header><div><span class="eyebrow">{focus}</span><h2>{esc(stem)}</h2></div><span class="pill">{new_label}</span></header>
  <div class="pair">
    <figure><figcaption>v1.1 · {old_label}</figcaption><img loading="lazy" src="validation_shots/{esc(stem)}__v11.jpg"></figure>
    <figure><figcaption>v1.2 · {new_label}</figcaption><img loading="lazy" src="validation_shots/{esc(stem)}__v12.jpg"></figure>
  </div>
  <div class="facts">
    <span>身份点 <b>{esc({True: '在环内', False: '不在环内', None: '不可检查'}.get(new.get('point_inside_ring'), '不可检查'))}</b></span>
    <span>空间提示 <b>{esc({True: '在环内', False: '不在环内', None: '不可检查'}.get(new.get('tip_inside_ring'), '不可检查'))}</b></span>
    <span>p90 <b>{fmt(new.get('ring_to_ink_p90_px'))} px</b></span>
    <span>空白边 <b>{fmt(100 * new.get('blank_share', 0)) if new.get('blank_share') is not None else '—'}%</b></span>
    <span>定位调用 <b>{fmt(new.get('locate_calls'))}</b></span>
  </div>
  <ul>{row_notes(new)}</ul>
</article>""")

    old_tip = f"{old_summary.get('tip_inside_ring', 0)}/{old_summary.get('tip_checkable', 0)}"
    new_tip = f"{new_summary.get('tip_inside_ring', 0)}/{new_summary.get('tip_checkable', 0)}"
    table = "".join([
        metric("产出多边形", f"{old_summary['traced']}/{old_summary['pages']}", f"{new_summary['traced']}/{new_summary['pages']}"),
        metric("自动检查通过", old_summary.get("pages_clean"), new_summary.get("pages_auto_usable")),
        metric("身份点在环内", "v1.1 未记录", f"{new_summary.get('point_inside_ring')}/{new_summary.get('point_checkable')}"),
        metric("空间提示点在环内", old_tip, new_tip),
        metric("ring→ink p90 中位数", f"{old_summary['ring_to_ink_p90_px']:.3f}px", f"{new_summary['ring_to_ink_p90_px']:.3f}px"),
        metric("3px 内占比中位数", f"{100*old_summary['frac_within_3px']:.1f}%", f"{100*new_summary['frac_within_3px']:.1f}%"),
        metric("需人工/拒绝", old_summary.get("pages_flagged"), new_summary.get("pages_needs_human")),
        metric("成本估算", f"£{old_summary['gbp_per_1000']:.2f}/千页", f"£{new_summary['gbp_per_1000']:.2f}/千页"),
    ])
    counts = {key: sum(status(row)[0] == key for row in new_rows.values())
              for key in ("auto", "review", "declined")}

    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>snap_boundary v1.2 · 同批 15 cases 前后验证</title><style>
:root{{--paper:#f1eee8;--card:#fffdfa;--ink:#20231f;--muted:#6c716b;--line:#d7d2c9;--green:#267248;--amber:#a66b00;--red:#ae352e;--blue:#285d88}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,system-ui,sans-serif}}main{{max-width:1420px;margin:auto;padding:42px 24px 72px}}
.kicker,.eyebrow{{font:700 11px/1.3 ui-monospace,monospace;letter-spacing:.11em;text-transform:uppercase;color:var(--red)}}h1{{font-size:clamp(30px,5vw,56px);line-height:1.02;letter-spacing:-.045em;margin:10px 0 18px}}.lead{{max-width:930px;color:var(--muted);font-size:17px}}
.summary{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:28px 0}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-weight:500}}td:last-child{{font-weight:700;color:var(--blue)}}
.method{{color:var(--muted)}}.filters{{display:flex;gap:8px;flex-wrap:wrap;margin:26px 0 18px}}button{{border:1px solid var(--line);border-radius:99px;background:var(--card);padding:8px 13px;cursor:pointer}}button.active{{background:var(--ink);color:white}}
.cases{{display:grid;gap:22px}}.case{{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--green);border-radius:12px;padding:18px}}.case.review{{border-left-color:var(--amber)}}.case.declined{{border-left-color:var(--red)}}header{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}h2{{font:700 16px/1.35 ui-monospace,monospace;margin:5px 0 0;overflow-wrap:anywhere}}.pill{{white-space:nowrap;border:1px solid currentColor;border-radius:99px;padding:4px 9px;color:var(--green)}}.review .pill{{color:var(--amber)}}.declined .pill{{color:var(--red)}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}}figure{{margin:0}}figcaption{{color:var(--muted);font-size:12px;margin-bottom:6px}}img{{display:block;width:100%;height:560px;object-fit:contain;background:#e8e4dc;border:1px solid var(--line);border-radius:7px}}
.facts{{display:flex;flex-wrap:wrap;gap:8px;margin-top:13px}}.facts span{{background:#f1eee8;border-radius:6px;padding:6px 9px;color:var(--muted)}}.facts b{{color:var(--ink)}}ul{{margin:12px 0 0;padding-left:20px;color:var(--muted)}}.foot{{margin-top:28px;color:var(--muted);font-size:12px}}
@media(max-width:800px){{main{{padding:24px 12px 50px}}.summary,.pair{{grid-template-columns:1fr}}img{{height:auto;max-height:680px}}}}
</style></head><body><main><div class="kicker">Same 15 cases · deterministic comparison</div><h1>v1.2 修复后验证</h1>
<p class="lead">同一批 15 张 plan、同一定位/描边模型，比较 v1.1 与 v1.2。v1.2 把定位点和标注提示直接传给 tracer，要求两次定位形成共识，并将无法证明为 parcel 的建筑/位置指示降级为人工处理。</p>
<section class="summary"><div class="panel"><table><thead><tr><th>指标</th><th>v1.1</th><th>v1.2</th></tr></thead><tbody>{table}</tbody></table></div><div class="panel method"><b>如何读这个报告</b><p>红线是输出边界，蓝圈是定位身份点，紫色叉是空间标注端点。重点先看最前面的 5 个用户反馈 case。线条贴合指标只能说明“沿着墨线”，不能单独证明“地块正确”。</p><p>v1.2：自动可用 {counts['auto']}，人工复核 {counts['review']}，拒绝输出 {counts['declined']}。没有人工 ground truth，因此“自动可用”仍是高置信候选，不是法律意义上的确认。</p></div></section>
<div class="filters"><button class="active" data-filter="all">全部 15</button><button data-filter="auto">自动可用 {counts['auto']}</button><button data-filter="review">人工复核 {counts['review']}</button><button data-filter="declined">拒绝 {counts['declined']}</button></div>
<section class="cases">{''.join(cards)}</section><p class="foot">完整输出：{esc(args.new)} · 运行耗时 {new_summary['wall_seconds']:.1f}s · 报告生成不调用模型。</p>
</main><script>document.querySelectorAll('button[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');const f=b.dataset.filter;document.querySelectorAll('.case').forEach(c=>c.hidden=f!=='all'&&c.dataset.status!==f)}})</script></body></html>"""
    output = args.new / "validation.html"
    output.write_text(document, encoding="utf-8")
    print(f"wrote {output} with {len(new_rows)} comparisons")


if __name__ == "__main__":
    main()
