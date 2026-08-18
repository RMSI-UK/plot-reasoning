from __future__ import annotations

import numpy as np

import boundary as B
import snap_boundary_v1 as S


def _loc(box, point, confidence=0.8, found=True):
    return {"found": found, "box": box if found else None,
            "point": point if found else [0, 0], "tip": point if found else [0, 0],
            "confidence": confidence, "marking_kind": "hatching",
            "marking_spatial": found, "target_geometry": "curtilage",
            "said": "a hatched target", "error": None}


def test_locate_consensus_requires_two_answers():
    chosen, box, info = S.locate_consensus([
        _loc([10, 10, 40, 40], [20, 20]),
        _loc(None, None, found=False),
    ])
    assert chosen is None
    assert box is None
    assert info["members"] == []


def test_third_locate_can_rescue_one_failed_answer():
    first = _loc([10, 10, 40, 40], [20, 20], confidence=0.7)
    third = _loc([12, 11, 42, 41], [22, 21], confidence=0.9)
    chosen, box, info = S.locate_consensus([
        first, _loc(None, None, found=False), third,
    ])
    assert chosen is not None
    assert chosen["point"] == [22, 21]
    assert box == [10, 10, 42, 41]
    assert info["members"] == [0, 2]


def test_guidance_is_present_in_text_and_annotated_crop():
    win = B.Window(0, 0, 100, 100, 1.0)
    guidance = {"point": [50, 55], "tip": [48, 50], "box": [20, 20, 80, 80],
                "marking_kind": "leader", "marking_spatial": True,
                "target_geometry": "curtilage", "locate_said": "THE SITE arrow"}
    message = B.guidance_message(win, guidance)
    assert "MUST contain" in message
    assert "[50, 55]" in message
    assert "THE SITE arrow" in message
    marked = B.annotated_crop(np.full((100, 100), 255, np.uint8), win, guidance)
    assert marked is not None
    assert marked.shape == (100, 100, 3)
    assert not np.array_equal(marked[55, 50], [255, 255, 255])


def test_candidate_selection_prefers_correct_identity_over_old_area():
    old = {"source": "first crop", "identity_ok": False,
           "edges": {"cut_sides": ["B"]}, "fallback": False,
           "blank_share": 0.127, "frac_within_3px": 0.569,
           "ring_to_ink_p90_px": 11.91, "area_pct": 2.273}
    widened = {"source": "widened crop", "identity_ok": True,
               "edges": {"cut_sides": []}, "fallback": False,
               "blank_share": 0.0, "frac_within_3px": 1.0,
               "ring_to_ink_p90_px": 0.95, "area_pct": 0.645}
    selected, verdict = B.choose_candidate(old, widened)
    assert selected is widened
    assert "widened crop" in verdict


def test_non_spatial_text_does_not_require_tip_containment():
    ring = [[10, 10], [90, 10], [90, 90], [10, 90]]
    guidance = {"point": [50, 50], "tip": [150, 150], "marking_spatial": False}
    point_ok = B.point_inside_ring(guidance["point"], ring)
    tip_ok = (B.point_inside_ring(guidance["tip"], ring)
              if guidance["marking_spatial"] else None)
    identity_ok = point_ok is True and (
        tip_ok is True if guidance["marking_spatial"] else True)
    assert point_ok is True
    assert tip_ok is None
    assert identity_ok is True
