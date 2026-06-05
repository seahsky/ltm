"""
Sanity tests for the benchmark-success recompute in diagnose_pipeline.py.

Context (Run 15): neither previously-reported number is the HM3D benchmark
metric (STOP within 1.0 m of the goal). "8%" used the 0.1 m radius (10x too
strict); "67%" (success_1m) is STOP-independent reach. The true number is
recoverable from the episode_*.json logs: STOP terminates the episode, so the
final-step ``distance_to_goal`` IS distance-at-STOP, and
``benchmark_success = stopped AND distance_to_goal < 1.0``.

Pure-function tests only — no Habitat, no models. The report test writes
synthetic episode dicts into a tmp run dir (pattern of test_analyze_revisit).

Invoke with::

    python embodied_memory/scripts/test_diagnose_benchmark.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

_EMB_DIR = Path(__file__).resolve().parent.parent  # …/embodied_memory


def _load():
    path = _EMB_DIR / "scripts" / "diagnose_pipeline.py"
    spec = importlib.util.spec_from_file_location("diagnose_pipeline", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


dp = _load()


def _ep(scene="S", cat="chair", idx=0, action_stop=None, d2g=None,
        min_d2g=None, n_steps=100, steps=None, success_1m=None, success=False):
    ep = {
        "scene_id": scene, "target_category": cat, "episode_idx": idx,
        "n_steps": n_steps, "success": success,
    }
    if action_stop is not None:
        ep["action_stop"] = action_stop
    if d2g is not None:
        ep["distance_to_goal"] = d2g
    if min_d2g is not None:
        ep["min_distance_to_goal"] = min_d2g
    if steps is not None:
        ep["steps"] = steps
    if success_1m is not None:
        ep["success_1m"] = success_1m
    return ep


# ---- benchmark_success: stopped AND final d2g < radius ----

def case_stopped_within_radius():
    ep = _ep(action_stop=1, d2g=0.7)
    assert dp.benchmark_success(ep, radius=1.0) is True
    print("  case_stopped_within_radius: OK")


def case_stopped_outside_radius():
    ep = _ep(action_stop=1, d2g=1.3)
    assert dp.benchmark_success(ep, radius=1.0) is False
    print("  case_stopped_outside_radius: OK")


def case_reach_without_stop_is_not_success():
    # The crux: agent passed within 0.2m (so success_1m/reach is True) but
    # never issued STOP -> benchmark success MUST be False. This is exactly
    # what separates benchmark-SR from the reach@1m diagnostic.
    ep = _ep(action_stop=0, d2g=3.0, min_d2g=0.2, success_1m=True)
    assert dp.benchmark_success(ep, radius=1.0) is False
    print("  case_reach_without_stop_is_not_success: OK")


# ---- _episode_stopped precedence chain ----

def case_stopped_fallback_last_step_action():
    # action_stop missing (old log) -> last serialized step action==0 (STOP).
    ep = _ep(d2g=0.5, steps=[{"action": 1}, {"action": 0}])
    assert dp._episode_stopped(ep) is True
    assert dp.benchmark_success(ep, radius=1.0) is True
    ep2 = _ep(d2g=0.5, steps=[{"action": 1}, {"action": 2}])
    assert dp._episode_stopped(ep2) is False
    print("  case_stopped_fallback_last_step_action: OK")


def case_stopped_fallback_n_steps():
    # Both action_stop and steps missing -> early termination (n_steps below
    # the step budget) means the episode ended via STOP; exhausting the
    # budget means it timed out.
    ep = _ep(d2g=0.5, n_steps=120)
    assert dp._episode_stopped(ep, max_steps=250) is True
    ep_timeout = _ep(d2g=0.5, n_steps=250)
    assert dp._episode_stopped(ep_timeout, max_steps=250) is False
    print("  case_stopped_fallback_n_steps: OK")


def case_action_stop_takes_precedence():
    # action_stop present and 0 -> NOT stopped, even if n_steps is small.
    ep = _ep(action_stop=0, d2g=0.5, n_steps=10)
    assert dp._episode_stopped(ep) is False
    print("  case_action_stop_takes_precedence: OK")


# ---- radius sweep ----

def case_radius_sweep():
    ep = _ep(action_stop=2, d2g=0.8)
    assert dp.benchmark_success(ep, radius=1.0) is True
    assert dp.benchmark_success(ep, radius=0.5) is False
    print("  case_radius_sweep: OK")


# ---- report: aggregate + cold/warm split over a tmp run dir ----

def case_report_aggregate_cold_warm():
    eps = [
        # chair cold: stopped at 0.7m -> benchmark success
        _ep(idx=0, action_stop=1, d2g=0.7, min_d2g=0.5, n_steps=50,
            success_1m=True),
        # chair warm: stopped at 1.3m -> no
        _ep(idx=1, action_stop=1, d2g=1.3, min_d2g=1.1, n_steps=60),
        # chair warm: never stopped, reached 0.2m -> reach yes, benchmark no
        _ep(idx=2, action_stop=0, d2g=3.0, min_d2g=0.2, n_steps=250,
            success_1m=True),
        # bed cold: stopped at 0.4m -> success at both radii
        _ep(cat="bed", idx=3, action_stop=1, d2g=0.4, min_d2g=0.3, n_steps=40,
            success_1m=True),
    ]
    with tempfile.TemporaryDirectory() as d:
        for i, ep in enumerate(eps):
            with open(os.path.join(d, f"episode_{i:03d}.json"), "w") as f:
                json.dump(ep, f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dp.report_benchmark_success(d, radii=(1.0, 0.5))
        out = buf.getvalue()
    # headline metric named and aggregated: cold 2/2, warm 1/2... let's check
    # the overall: 2 of 4 episodes are benchmark successes at 1.0m.
    assert "benchmark_success@1.0m" in out, out
    assert "benchmark_success@0.5m" in out, out
    assert "stop_rate" in out, out
    # cold split: idx0 chair + idx3 bed are cold, both stopped <1.0 -> 1.000
    assert "cold" in out and "warm" in out, out
    # overall @1.0m = 2/4 = 0.500; warm = 0/2 = 0.000; cold = 2/2 = 1.000
    assert "0.500" in out, out
    assert "1.000" in out, out
    assert "0.000" in out, out
    # the contrast columns (reach diagnostics) must be present
    assert "succ@1m" in out, out
    print("  case_report_aggregate_cold_warm: OK")


def main() -> int:
    print("diagnose_pipeline benchmark-success sanity tests")
    case_stopped_within_radius()
    case_stopped_outside_radius()
    case_reach_without_stop_is_not_success()
    case_stopped_fallback_last_step_action()
    case_stopped_fallback_n_steps()
    case_action_stop_takes_precedence()
    case_radius_sweep()
    case_report_aggregate_cold_warm()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
