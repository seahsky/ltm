"""
Sanity tests for the CLIP semantic-frontier separation GATE
(``diagnose_clip_frontier_separation``).

The gate decides — for $0 of matrix compute — whether
``cos(CLIP_image(rgb), CLIP_text("a photo of a {goal}"))`` actually discriminates
goal-facing views from non-goal views on HM3D sim renders. A flat result is the
*expected* outcome (the project measured this twice) and is itself decision-relevant
(it protects the existing +0.2505 headline). These tests exercise the pure
separation/verdict/report functions with no torch / CLIP / habitat — the render path
is RACE-verified separately.

Invoke with::

    python embodied_memory/scripts/test_diagnose_clip_frontier_separation.py
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnose_clip_frontier_separation as dc  # noqa: E402


# ----------------------------------------------------------------------
# clip_value_separation — summary stats
# ----------------------------------------------------------------------


def case_separation_positive_when_goal_above_away():
    out = dc.clip_value_separation([0.30, 0.32, 0.28], [0.22, 0.23, 0.21])
    assert abs(out["mean_goal"] - 0.30) < 1e-6, out
    assert abs(out["mean_away"] - 0.22) < 1e-6, out
    assert abs(out["separation"] - 0.08) < 1e-6, out
    assert out["n_goal"] == 3 and out["n_away"] == 3, out
    print("  case separation_positive_when_goal_above_away: OK")


def case_separation_near_zero_when_goal_equals_away():
    out = dc.clip_value_separation([0.25, 0.25], [0.25, 0.25])
    assert abs(out["separation"] - 0.0) < 1e-9, out
    print("  case separation_near_zero_when_goal_equals_away: OK")


def case_separation_drops_nan_samples():
    # NaNs are dropped from the means; n_* count only finite samples.
    out = dc.clip_value_separation([0.30, float("nan"), 0.30], [0.20, float("inf")])
    assert out["n_goal"] == 2 and out["n_away"] == 1, out
    assert abs(out["mean_goal"] - 0.30) < 1e-6, out
    assert abs(out["mean_away"] - 0.20) < 1e-6, out
    assert abs(out["separation"] - 0.10) < 1e-6, out
    print("  case separation_drops_nan_samples: OK")


def case_separation_empty_inputs_nan_no_crash():
    out = dc.clip_value_separation([], [])
    assert math.isnan(out["separation"]), out
    assert math.isnan(out["mean_goal"]) and math.isnan(out["mean_away"]), out
    assert out["n_goal"] == 0 and out["n_away"] == 0, out
    print("  case separation_empty_inputs_nan_no_crash: OK")


def case_separation_single_element_lists_ok():
    # singleton lists must not crash and yield a finite separation.
    out = dc.clip_value_separation([0.27], [0.24])
    assert out["n_goal"] == 1 and out["n_away"] == 1, out
    assert abs(out["separation"] - 0.03) < 1e-6, out
    print("  case separation_single_element_lists_ok: OK")


def case_separation_one_side_empty_is_nan():
    # away side empty -> separation undefined (NaN), but mean_goal still finite.
    out = dc.clip_value_separation([0.30, 0.31], [])
    assert math.isfinite(out["mean_goal"]), out
    assert math.isnan(out["mean_away"]), out
    assert math.isnan(out["separation"]), out
    print("  case separation_one_side_empty_is_nan: OK")


# ----------------------------------------------------------------------
# _separation_verdict — GO / HOLD / INSUFFICIENT decision rule
# ----------------------------------------------------------------------


def case_verdict_go_when_sep_at_or_above_margin():
    assert dc._separation_verdict(0.08, margin=0.05) == "GO"
    # boundary: sep == margin is GO (>=).
    assert dc._separation_verdict(0.05, margin=0.05) == "GO"
    print("  case verdict_go_when_sep_at_or_above_margin: OK")


def case_verdict_hold_when_sep_below_margin():
    # the project's twice-measured flat case (~0.25 vs ~0.228 -> ~0.022 sep).
    assert dc._separation_verdict(0.022, margin=0.05) == "HOLD"
    assert dc._separation_verdict(0.0, margin=0.05) == "HOLD"
    assert dc._separation_verdict(-0.01, margin=0.05) == "HOLD"
    print("  case verdict_hold_when_sep_below_margin: OK")


def case_verdict_insufficient_on_nan():
    assert dc._separation_verdict(float("nan"), margin=0.05) == "INSUFFICIENT"
    assert dc._separation_verdict(float("inf"), margin=0.05) == "INSUFFICIENT"
    print("  case verdict_insufficient_on_nan: OK")


# ----------------------------------------------------------------------
# yaw_rotated_quat — the away-pose geometry (pure)
# ----------------------------------------------------------------------


def case_yaw_180_flips_identity_to_y_axis():
    # rotating identity [0,0,0,1] by pi about +Y -> [0, +-1, 0, 0] (a 180deg turn).
    q = dc.yaw_rotated_quat([0.0, 0.0, 0.0, 1.0], math.pi)
    assert abs(q[0]) < 1e-6 and abs(q[2]) < 1e-6, q
    assert abs(abs(q[1]) - 1.0) < 1e-6, q  # |y| == 1
    assert abs(q[3]) < 1e-6, q             # w == 0
    # output is unit-norm
    assert abs(math.sqrt(sum(c * c for c in q)) - 1.0) < 1e-6, q
    print("  case yaw_180_flips_identity_to_y_axis: OK")


def case_yaw_zero_is_identity():
    q = dc.yaw_rotated_quat([0.1, 0.2, 0.3, 0.9], 0.0)
    orig = [0.1, 0.2, 0.3, 0.9]
    norm = math.sqrt(sum(c * c for c in orig))
    expected = [c / norm for c in orig]
    for a, b in zip(q, expected):
        assert abs(a - b) < 1e-6, (q, expected)
    print("  case yaw_zero_is_identity: OK")


# ----------------------------------------------------------------------
# separation_report — prints the machine marker GATE_RESULT=...
# ----------------------------------------------------------------------


def case_report_prints_go_marker():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = dc.separation_report([0.30, 0.31], [0.20, 0.21], margin=0.05)
    out = buf.getvalue()
    assert "GATE_RESULT=GO" in out, out
    assert res["result"] == "GO", res
    print("  case report_prints_go_marker: OK")


def case_report_prints_hold_marker():
    # flat case: sep ~0.022 < margin 0.05 -> HOLD (protects the headline).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = dc.separation_report([0.250, 0.252], [0.228, 0.230], margin=0.05)
    out = buf.getvalue()
    assert "GATE_RESULT=HOLD" in out, out
    assert res["result"] == "HOLD", res
    assert "headline" in res["verdict"].lower(), res["verdict"]
    print("  case report_prints_hold_marker: OK")


def case_report_prints_insufficient_marker_on_empty():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = dc.separation_report([], [], margin=0.05)
    out = buf.getvalue()
    assert "GATE_RESULT=INSUFFICIENT" in out, out
    assert res["result"] == "INSUFFICIENT", res
    print("  case report_prints_insufficient_marker_on_empty: OK")


def main() -> int:
    print("CLIP semantic-frontier separation GATE sanity tests")
    case_separation_positive_when_goal_above_away()
    case_separation_near_zero_when_goal_equals_away()
    case_separation_drops_nan_samples()
    case_separation_empty_inputs_nan_no_crash()
    case_separation_single_element_lists_ok()
    case_separation_one_side_empty_is_nan()
    case_verdict_go_when_sep_at_or_above_margin()
    case_verdict_hold_when_sep_below_margin()
    case_verdict_insufficient_on_nan()
    case_yaw_180_flips_identity_to_y_axis()
    case_yaw_zero_is_identity()
    case_report_prints_go_marker()
    case_report_prints_hold_marker()
    case_report_prints_insufficient_marker_on_empty()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
