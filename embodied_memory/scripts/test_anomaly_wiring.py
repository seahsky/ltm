"""
TDD for the E5/E6/E7 anomaly-response WIRING seams that are locally verifiable
(no sim / GPU / Habitat). The end-to-end controller loop is RACE/sim-bound;
these pin the pure pieces the wiring depends on:

  * memory_bridge FrontierPhysicsScorer: source=="audio_investigate" -> 1.0 (E5-S6).
  * episode_runner._investigate_candidate shape (E5-S4).
  * _consume_memory_applies is False for anomaly_response (the injected divert
    candidate must survive the consume/anti-thrash filters).
  * audio_task.process_audio_step populates step.info["is_anomaly"] (the
    controller reads it); None on the default/gate-off path (byte-identical).
  * anomaly_controller.build_report shape (E5-S7 report hook is additive).

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_anomaly_wiring.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

from embodied_memory.memory_bridge import FrontierPhysicsScorer
from embodied_memory.frontier_planner import FrontierCandidate
from embodied_memory.episode_runner import _investigate_candidate, _consume_memory_applies
from embodied_memory import audio_task as at
from embodied_memory import anomaly_controller as ac


def _cand(source, raw_score=0.5, distance_m=3.0, bearing_rad=2.0):
    return FrontierCandidate(
        candidate_id=1,
        world_xy=np.array([1.0, 2.0], dtype=np.float32),
        grid_rc=(0, 0),
        distance_m=distance_m,
        bearing_rad=bearing_rad,
        cluster_size=1,
        raw_score=raw_score,
        source=source,
        metadata={},
    )


# ----------------------------------------------------------------------
# E5-S6 — rerank branch
# ----------------------------------------------------------------------
def case_rerank_branch_audio_investigate_returns_1():
    scorer = FrontierPhysicsScorer()
    cand = _cand("audio_investigate", raw_score=0.5, distance_m=3.0, bearing_rad=2.0)
    s = scorer.score("", None, {"frontier_candidate": cand})
    assert s == 1.0, f"investigate divert must win the rerank deterministically, got {s}"


def case_rerank_memory_low_cos_below_1():
    # proves the new branch is REACHED (not the memory branch): a memory cand at
    # the NULL cosine scores well below 1.0.
    scorer = FrontierPhysicsScorer()
    memc = _cand("memory", raw_score=0.30, distance_m=3.0)
    s = scorer.score("", None, {"frontier_candidate": memc})
    assert s < 1.0, f"low-cos memory cand should not saturate, got {s}"


def case_rerank_frontier_unchanged():
    # default frontier path is untouched by the new branch.
    scorer = FrontierPhysicsScorer()
    fc = _cand("frontier", raw_score=0.5, distance_m=2.0, bearing_rad=0.0)
    s = scorer.score("", None, {"frontier_candidate": fc})
    assert 0.0 <= s <= 1.0 and s != 1.0, f"frontier score should be its blend, got {s}"


# ----------------------------------------------------------------------
# E5-S4 — investigate candidate shape
# ----------------------------------------------------------------------
def case_investigate_candidate_shape():
    c = _investigate_candidate((1.0, 0.5, 2.0), (0.0, 0.5, 0.0))
    assert c.source == "audio_investigate", c.source
    assert c.metadata == {"investigate": True}, c.metadata
    assert float(c.world_xy[0]) == 1.0 and float(c.world_xy[1]) == 2.0, c.world_xy
    assert c.raw_score == 1.0
    assert c.cluster_size == 1
    assert abs(float(c.distance_m) - math.hypot(1.0, 2.0)) < 1e-5, c.distance_m


# ----------------------------------------------------------------------
# consume gate — injected divert must survive
# ----------------------------------------------------------------------
def case_consume_gate_noop_for_anomaly_response():
    assert _consume_memory_applies(False, "anomaly_response", False) is False
    assert _consume_memory_applies(False, "anomaly_response", True) is False


# ----------------------------------------------------------------------
# audio_task is_anomaly key
# ----------------------------------------------------------------------
def case_process_audio_step_is_anomaly_none_when_silent():
    cfg = at.AudioTaskConfig(enabled=True, onset_rms=0.05)
    st = at.AudioEpisodeState()
    diag = at.process_audio_step(None, 30, 48000, cfg, st, clap_encoder=None)
    assert diag["is_anomaly"] is None, diag


def case_process_audio_step_is_anomaly_true_under_gate():
    cfg = at.AudioTaskConfig(enabled=True, onset_rms=0.05, anomaly_gate=True)
    st = at.AudioEpisodeState()
    loud = np.full((2, 200), 0.1, dtype=np.float32)  # rms 0.1 >= onset
    _orig = at.is_anomaly
    try:
        at.is_anomaly = lambda *a, **k: (True, "alarm", {"margin": 0.1})
        diag = at.process_audio_step(loud, 30, 48000, cfg, st, clap_encoder=object())
    finally:
        at.is_anomaly = _orig
    assert diag["onset_fired"] is True, diag
    assert diag["is_anomaly"] is True, diag


def case_process_audio_step_is_anomaly_none_when_gate_off():
    # default audiogoal path (gate OFF) never sets is_anomaly -> stays None ->
    # never read (controller only runs for anomaly_response). Byte-identical.
    cfg = at.AudioTaskConfig(enabled=True, onset_rms=0.05, anomaly_gate=False)
    st = at.AudioEpisodeState()
    loud = np.full((2, 200), 0.1, dtype=np.float32)
    _orig_c = at.classify_anomaly
    try:
        at.classify_anomaly = lambda *a, **k: ("alarm", {})
        diag = at.process_audio_step(loud, 30, 48000, cfg, st, clap_encoder=object())
    finally:
        at.classify_anomaly = _orig_c
    assert diag["onset_fired"] is True
    assert diag["is_anomaly"] is None, diag


# ----------------------------------------------------------------------
# E5-S7 — report hook shape
# ----------------------------------------------------------------------
def case_build_report_shape():
    cfg = ac.AnomalyControllerConfig(enabled=True)
    st = ac.ControllerState()
    st.reset("chair")
    # SEARCH -> INVESTIGATE
    ac.step_controller(st, cfg, onset_fired=True, is_anomaly=True,
                       source_xyz=(1.0, 0.0, 2.0), arrived_at_source=False,
                       primary_goal_reached=False, anomaly_object="tv")
    # arrive -> CHECK
    ac.step_controller(st, cfg, onset_fired=False, is_anomaly=None,
                       source_xyz=(1.0, 0.0, 2.0), arrived_at_source=True,
                       primary_goal_reached=False, anomaly_class="alarm",
                       keyframe_caption="a television")
    rep = ac.build_report(st, primary_completed=True)
    for k in ("primary_completed", "investigated", "resumed", "anomaly_class",
              "source_xyz", "n_benign_ignored"):
        assert k in rep, f"report missing {k}"
    assert rep["investigated"] is True
    assert st.mode == ac.NavMode.REPORTED


def main() -> int:
    cases = [
        case_rerank_branch_audio_investigate_returns_1,
        case_rerank_memory_low_cos_below_1,
        case_rerank_frontier_unchanged,
        case_investigate_candidate_shape,
        case_consume_gate_noop_for_anomaly_response,
        case_process_audio_step_is_anomaly_none_when_silent,
        case_process_audio_step_is_anomaly_true_under_gate,
        case_process_audio_step_is_anomaly_none_when_gate_off,
        case_build_report_shape,
    ]
    print(f"running {len(cases)} anomaly_wiring cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
