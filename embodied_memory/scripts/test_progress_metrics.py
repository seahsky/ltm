"""
TDD for the in-progress metric readout (progress_metrics.aggregate).

The numbers this prints are read mid-run to decide whether to keep burning GPU
or kill the job, so the failure that matters is a metric that LOOKS measured but
is not. Three rules are pinned here:

  * an absent metric is None, never 0.0 (a 0.0 Find-SR on an objectnav run would
    read as "the agent never succeeds" rather than "not measured here");
  * `success_1m` never contributes to SR — it is a STOP-independent reach
    diagnostic (CONTEXT.md);
  * Anomaly-response SR requires investigated AND resumed, and counts an aborted
    detour as a failure in the denominator, not as a missing measurement.

Run: PYTHONPATH=. python embodied_memory/scripts/test_progress_metrics.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile

# Load the module BY FILE PATH, not as embodied_memory.scripts.progress_metrics:
# the package __init__ imports numpy/habitat, and the whole point of this module
# is that it runs on a bare system python with no conda env active.
_SPEC = importlib.util.spec_from_file_location(
    "progress_metrics",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress_metrics.py"))
progress_metrics = importlib.util.module_from_spec(_SPEC)
sys.modules["progress_metrics"] = progress_metrics
_SPEC.loader.exec_module(progress_metrics)

aggregate = progress_metrics.aggregate
format_metrics = progress_metrics.format_metrics
load_episodes = progress_metrics.load_episodes


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return a is not None and abs(a - b) <= tol


def _objectnav_ep(idx: int, *, spl: float, success: bool, soft: float,
                  steps: int, reach: bool = False, finished_at: float = 0.0) -> dict:
    return {
        "episode_idx": idx, "scene_id": "TEEsavR23oF", "target_category": "chair",
        "spl": spl, "success": success, "soft_spl": soft, "n_steps": steps,
        "success_1m": reach, "min_distance_to_goal": 1.5, "finished_at": finished_at,
    }


def _anomaly_ep(idx: int, *, investigated: bool, resumed: bool, aborted: bool = False,
                primary_1m: bool = False, benign: int = 0) -> dict:
    ep = _objectnav_ep(idx, spl=0.0, success=False, soft=0.2, steps=400)
    ep["anomaly_report"] = {
        "primary_completed": False, "primary_completed_1m": primary_1m,
        "investigated": investigated, "resumed": resumed,
        "investigate_aborted": aborted, "n_benign_ignored": benign,
    }
    return ep


def case_empty_run_reports_zero_not_crash():
    m = aggregate([])
    assert m.n_episodes == 0 and m.benchmark_spl is None and m.soft_spl is None, m
    assert "no completed episodes yet" in format_metrics(m, "runs/x")
    print("  case_empty_run_reports_zero_not_crash: OK")


def case_objectnav_computes_the_three_navigation_metrics():
    eps = [_objectnav_ep(0, spl=0.0, success=False, soft=0.10, steps=500),
           _objectnav_ep(1, spl=0.6, success=True, soft=0.50, steps=100)]
    m = aggregate(eps)
    assert _close(m.benchmark_spl, 0.3), m.benchmark_spl
    assert _close(m.sr_01m, 0.5), m.sr_01m
    assert _close(m.soft_spl, 0.30), m.soft_spl
    assert _close(m.mean_steps, 300.0) and _close(m.median_steps, 300.0), m
    print("  case_objectnav_computes_the_three_navigation_metrics: OK")


def case_absent_controller_metrics_are_none_not_zero():
    """The load-bearing honesty rule: an objectnav run has no Find-SR at all."""
    m = aggregate([_objectnav_ep(0, spl=0.1, success=False, soft=0.2, steps=300)])
    assert m.n_controller_episodes == 0, m.n_controller_episodes
    assert m.find_sr_1m is None and m.anomaly_response_sr is None, m
    lines = {ln.split()[0]: ln for ln in format_metrics(m, "runs/r1v1-s1").splitlines()
             if ln.startswith("  ")}
    assert "n/a" in lines["FIND-SR"] and "0.0000" not in lines["FIND-SR"], lines["FIND-SR"]
    assert "n/a" in lines["ANOMALY-RESP"] and "0.0000" not in lines["ANOMALY-RESP"], lines
    print("  case_absent_controller_metrics_are_none_not_zero: OK")


def case_reach_1m_never_becomes_sr():
    """Every episode reaches within 1 m but none STOPs there: SR stays 0."""
    eps = [_objectnav_ep(i, spl=0.0, success=False, soft=0.3, steps=500, reach=True)
           for i in range(4)]
    m = aggregate(eps)
    assert _close(m.reach_1m, 1.0) and _close(m.sr_01m, 0.0), m
    assert "NOT a success rate" in format_metrics(m, "runs/x")
    print("  case_reach_1m_never_becomes_sr: OK")


def case_anomaly_response_sr_needs_investigated_and_resumed():
    eps = [_anomaly_ep(0, investigated=True, resumed=True),     # full loop
           _anomaly_ep(1, investigated=True, resumed=False),    # never resumed
           _anomaly_ep(2, investigated=False, resumed=False)]   # never reached source
    m = aggregate(eps)
    assert _close(m.anomaly_response_sr, 1.0 / 3.0), m.anomaly_response_sr
    assert _close(m.investigated_rate, 2.0 / 3.0), m.investigated_rate
    assert _close(m.resumed_rate, 1.0 / 3.0), m.resumed_rate
    print("  case_anomaly_response_sr_needs_investigated_and_resumed: OK")


def case_aborted_detour_counts_as_failure_not_missing():
    eps = [_anomaly_ep(0, investigated=True, resumed=True),
           _anomaly_ep(1, investigated=False, resumed=False, aborted=True)]
    m = aggregate(eps)
    assert m.n_controller_episodes == 2, m.n_controller_episodes
    assert _close(m.anomaly_response_sr, 0.5), m.anomaly_response_sr
    assert _close(m.aborted_rate, 0.5), m.aborted_rate
    print("  case_aborted_detour_counts_as_failure_not_missing: OK")


def case_find_sr_reads_the_1m_ring_from_the_report():
    eps = [_anomaly_ep(0, investigated=True, resumed=True, primary_1m=True),
           _anomaly_ep(1, investigated=True, resumed=True, primary_1m=False)]
    m = aggregate(eps)
    assert _close(m.find_sr_1m, 0.5), m.find_sr_1m
    assert _close(m.find_sr_01m, 0.0), m.find_sr_01m   # strict ring, never fires here
    print("  case_find_sr_reads_the_1m_ring_from_the_report: OK")


def case_benign_ignored_totals_across_episodes():
    m = aggregate([_anomaly_ep(0, investigated=True, resumed=True, benign=2),
                   _anomaly_ep(1, investigated=True, resumed=True, benign=3)])
    assert m.n_benign_ignored == 5, m.n_benign_ignored
    print("  case_benign_ignored_totals_across_episodes: OK")


def case_wallclock_comes_from_finished_at_spacing():
    eps = [_objectnav_ep(i, spl=0.0, success=False, soft=0.1, steps=10,
                         finished_at=1000.0 + i * 300.0) for i in range(5)]
    m = aggregate(eps)
    assert _close(m.mean_wallclock_min, 5.0), m.mean_wallclock_min      # 300 s spacing
    assert _close(m.total_wallclock_h, 1200.0 / 3600.0), m.total_wallclock_h
    print("  case_wallclock_comes_from_finished_at_spacing: OK")


def case_single_episode_has_no_rate_but_still_reports_metrics():
    m = aggregate([_objectnav_ep(0, spl=0.4, success=True, soft=0.5, steps=42,
                                 finished_at=1000.0)])
    assert m.mean_wallclock_min is None and m.total_wallclock_h is None, m
    assert _close(m.benchmark_spl, 0.4) and _close(m.mean_steps, 42.0), m
    print("  case_single_episode_has_no_rate_but_still_reports_metrics: OK")


def case_scene_coverage_is_counted():
    a = _objectnav_ep(0, spl=0.0, success=False, soft=0.1, steps=10)
    b = _objectnav_ep(1, spl=0.0, success=False, soft=0.1, steps=10)
    b["scene_id"] = "wcojb4TFT35"
    m = aggregate([a, b])
    assert m.n_scenes == 2 and m.last_scene == "wcojb4TFT35", m
    print("  case_scene_coverage_is_counted: OK")


def case_load_skips_error_files_and_unparseable_json():
    tmp = tempfile.mkdtemp(prefix="progmetrics")
    try:
        with open(os.path.join(tmp, "episode_000.json"), "w") as fh:
            json.dump(_objectnav_ep(0, spl=0.5, success=True, soft=0.5, steps=10), fh)
        with open(os.path.join(tmp, "episode_001_error.json"), "w") as fh:
            json.dump({"episode_idx": 1, "error": "boom"}, fh)
        with open(os.path.join(tmp, "episode_002.json"), "w") as fh:
            fh.write('{"episode_idx": 2, "spl":')          # caught mid-write
        eps = load_episodes(tmp)
        assert len(eps) == 1 and eps[0]["episode_idx"] == 0, eps
        assert _close(aggregate(eps).benchmark_spl, 0.5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  case_load_skips_error_files_and_unparseable_json: OK")


def main() -> int:
    print("running progress_metrics tests…")
    case_empty_run_reports_zero_not_crash()
    case_objectnav_computes_the_three_navigation_metrics()
    case_absent_controller_metrics_are_none_not_zero()
    case_reach_1m_never_becomes_sr()
    case_anomaly_response_sr_needs_investigated_and_resumed()
    case_aborted_detour_counts_as_failure_not_missing()
    case_find_sr_reads_the_1m_ring_from_the_report()
    case_benign_ignored_totals_across_episodes()
    case_wallclock_comes_from_finished_at_spacing()
    case_single_episode_has_no_rate_but_still_reports_metrics()
    case_scene_coverage_is_counted()
    case_load_skips_error_files_and_unparseable_json()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
