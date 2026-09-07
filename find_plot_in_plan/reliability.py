#!/usr/bin/env python3
"""A per-page confidence for a traced boundary, computed from the page alone.

    python reliability.py --fit <run-dir>... --validate <run-dir>...

In production there is no hand-drawn ring to compare against: there is one scan, and the question
is how much to trust the polygon that came back. Everything used here is available at that moment.

What each signal is worth, measured against 132 hand-drawn boundaries (AUC, 0.5 = tells you
nothing about whether the plot is right):

    number of review flags        0.861      the strongest thing in the project, and it was
                                             written as a triage heuristic, not as a metric
    blank share                   0.785
    vertex count                  0.687
    snap IoU                      0.678
    tracer's own confidence       0.674      not saturated: median 0.78, monotone against truth
    arrow-tip containment         0.587
    ring-to-ink p90               0.545      noise
    LOCATOR confidence            0.568      noise, and badly calibrated: pages it marks 0.99
                                             are right 67% of the time. gemini answers 0.98 on
                                             almost everything and is right 52% of the time, so
                                             the three models' confidences do not share a scale
    frac within 3px               0.467      worse than a coin flip

The model self-reports are therefore the weakest inputs and the derived geometry is the strongest,
which is the opposite of what one would assume. The fit is a plain logistic regression -- with ~60
fitting rows anything larger would memorise -- and the coefficients are printed so the score can be
read rather than trusted.

Fitted on one page set and validated on another, always. Two figures in this project were produced
by a threshold chosen after seeing the answer and both had to be withdrawn.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
GEO = Path("/data/braintree/scan-processed/wp7-panels-sam3-seg/geometry")
GOOD_IOU = 0.70

FEATURES = [
    "n_flags", "unanimous", "cluster_frac", "n_locators",
    "blank_share", "ring_p90", "frac_within_3px", "n_vertices", "area_pct",
    "point_in_ring", "fallback", "scale", "cut_sides",
    "locate_conf", "trace_conf",
]


def truth_map():
    out = {}
    for f in ("hand_labels.json", "hand_labels_blind.json"):
        p = HERE / "labels" / f
        if not p.exists():
            continue
        for x in json.loads(p.read_text())["labels"]:
            if x.get("status") == "drawn" and x.get("vertices_page") \
                    and "v2" not in str(x.get("ring_source", "")):
                out[x["stem"]] = ("drawn", x["vertices_page"])
            elif x.get("status") == "no_boundary":
                out[x["stem"]] = ("none", None)
    return out


def fill(shape, ring):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(ring, np.int32).reshape(-1, 1, 2)], 1)
    return m


def featurise(r: dict) -> dict:
    """Everything here is knowable from the scan and the pipeline's own output."""
    e = r.get("edges") or {}
    return {
        "n_flags": len(r.get("flags") or []),
        "unanimous": 1.0 if r.get("unanimous") else 0.0,
        "cluster_frac": (r.get("vote_cluster") or 0) / max(1, r.get("vote_total") or 1),
        "n_locators": float(r.get("vote_total") or 0),
        "blank_share": float(r.get("blank_share") or 0),
        "ring_p90": float(r.get("ring_to_ink_p90_px") or 0),
        "frac_within_3px": float(r.get("frac_within_3px") or 0),
        "n_vertices": float(r.get("n_vertices") or len(r.get("vertices_page") or [])),
        "area_pct": float(r.get("area_pct") or 0),
        "point_in_ring": 1.0 if r.get("point_inside_ring") else 0.0,
        "fallback": 1.0 if r.get("source") == "building fallback" else 0.0,
        "scale": float(r.get("scale") or 1.0),
        "cut_sides": float(len(e.get("cut_sides") or [])),
        "locate_conf": float(r.get("locate_confidence") or 0.9),
        # absent in runs made before it was recorded; the median stands in so those rows still fit
        "trace_conf": float(r.get("trace_confidence") if r.get("trace_confidence") is not None
                            else 0.78),
    }


def rows_from(dirs, truth):
    out = []
    for d in dirs:
        p = GEO / d / "boundaries.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            t = truth.get(r["stem"])
            if t is None:
                continue
            kind, ring = t
            traced = bool(r.get("vertices_page"))
            if kind == "none":
                good = not traced          # the right answer on a blank page is to decline
            elif not traced:
                good = False
            else:
                sh = (r["page_h"], r["page_w"])
                a, b = fill(sh, ring), fill(sh, r["vertices_page"])
                u = np.count_nonzero(a | b)
                good = (np.count_nonzero(a & b) / u if u else 0) >= GOOD_IOU
            out.append({"stem": r["stem"], "good": bool(good), "traced": traced,
                        **featurise(r)})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k/n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-m)/d*100, (c+m)/d*100)


def auc(score, label):
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    if not pos or not neg:
        return float("nan")
    return sum((1 if a > b else .5 if a == b else 0) for a in pos for b in neg)/(len(pos)*len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--validate", nargs="+", required=True)
    ap.add_argument("--save", type=Path, default=HERE/"reliability_model.json")
    args = ap.parse_args()

    truth = truth_map()
    F = rows_from(args.fit, truth)
    V = rows_from(args.validate, truth)
    seen = {r["stem"] for r in F}
    V = [r for r in V if r["stem"] not in seen]      # no page may appear in both
    print(f"fit on {len(F)} pages ({sum(r['good'] for r in F)} good), "
          f"validate on {len(V)} disjoint pages ({sum(r['good'] for r in V)} good)\n")

    Xf = np.array([[r[k] for k in FEATURES] for r in F], float)
    yf = np.array([r["good"] for r in F], int)
    sc = StandardScaler().fit(Xf)
    clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(Xf), yf)

    print("what the fit leans on (standardised coefficients, + means 'more likely correct')")
    for k, c in sorted(zip(FEATURES, clf.coef_[0]), key=lambda t: -abs(t[1])):
        bar = "#" * int(round(abs(c)*14))
        print(f"  {k:18s} {c:+6.3f}  {bar}")

    Xv = np.array([[r[k] for k in FEATURES] for r in V], float)
    yv = np.array([r["good"] for r in V], int)
    pv = clf.predict_proba(sc.transform(Xv))[:, 1]
    pf = clf.predict_proba(sc.transform(Xf))[:, 1]
    print(f"\n  AUC on the fitting set   {auc(pf, yf):.3f}")
    print(f"  AUC on the HELD-OUT set  {auc(pv, yv):.3f}")
    base = np.array([r["n_flags"] == 0 for r in V], int)
    print(f"  AUC of 'zero flags' alone on the same held-out set  "
          f"{auc(base.astype(float), yv):.3f}")

    print(f"\n  calibration on the held-out set — what the score promises vs what it delivers")
    print(f"  {'score':>12}{'pages':>8}{'actually correct':>19}")
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        g = [(p, y) for p, y in zip(pv, yv) if lo <= p < hi]
        if g:
            k = sum(1 for _, y in g if y)
            l_, h_ = wilson(k, len(g))
            print(f"  {lo:.1f} - {hi:.1f} {len(g):8d}{k/len(g)*100:15.0f}%   [{l_:.0f}..{h_:.0f}]")

    print(f"\n  operating points on the held-out set")
    print(f"  {'threshold':>11}{'pages out':>11}{'coverage':>10}{'precision':>11}{'95% CI':>14}")
    for thr in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        g = [(p, y) for p, y in zip(pv, yv) if p >= thr]
        if not g:
            continue
        k = sum(1 for _, y in g if y)
        l_, h_ = wilson(k, len(g))
        print(f"  {thr:11.2f}{len(g):11d}{len(g)/len(V)*100:9.0f}%{k/len(g)*100:10.0f}%"
              f"   [{l_:3.0f}..{h_:3.0f}]")

    args.save.write_text(json.dumps({
        "features": FEATURES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        "good_iou": GOOD_IOU, "fit_on": args.fit, "validated_on": args.validate,
        "auc_heldout": round(auc(pv, yv), 4), "n_fit": len(F), "n_validate": len(V),
    }, indent=1))
    print(f"\n  saved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
