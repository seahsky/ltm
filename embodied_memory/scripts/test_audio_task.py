"""
TDD for embodied_memory/audio_task.py — the pure per-step audio DECISION brain
of the AudioGoal task (M1). It owns onset-detection, once-per-episode CLAP
classification → object override, the lateral-sign DOA cue, and the audio-energy
STOP gate, so habitat_env only RENDERS (writes Step.audio) and episode_runner
only ORCHESTRATES. **audio_task NEVER imports habitat_sim** (two-env boundary).

All cases run in the ltm-embodied env (numpy+scipy, a FakeCLAP encoder, a
synthetic RIRGrid) — no sim, no CUDA, no Habitat. The package imports cleanly
locally (faiss present), so we import audio_task via the package directly.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_audio_task.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from embodied_memory import audio
from embodied_memory import audio_task as at
from embodied_memory.episode_source import AgentState, Step

_EMB_DIR = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


class FakeCLAP:
    """Deterministic stand-in for perception.CLAPAudioEncoder with call counters."""

    def __init__(self, audio_class: str):
        self._c = audio_class
        self.n_audio = 0
        self.n_text = 0
        self._basis = {c: i for i, c in enumerate(audio.ANOMALY_CLASSES)}

    def _vec(self, i: int) -> np.ndarray:
        v = np.zeros(512, dtype=np.float32)
        v[i] = 1.0
        return v

    def encode_audio(self, wav, sr):
        self.n_audio += 1
        return self._vec(self._basis[self._c])

    def encode_text(self, text):
        self.n_text += 1
        for c, i in self._basis.items():
            if audio.CLASS_TO_CLAP_PROMPT[c] == text:
                return self._vec(i)
        return np.zeros(512, dtype=np.float32)


def _binaural(overall_rms: float, n: int = 4000, lateral: int = 0) -> np.ndarray:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, n)).astype(np.float32)
    if lateral > 0:
        x[1] *= 1.6          # right louder → source right
    elif lateral < 0:
        x[0] *= 1.6
    cur = float(np.sqrt(np.mean(x ** 2)))
    return (x * (overall_rms / (cur + 1e-9))).astype(np.float32)


def _grid():
    cells = np.array([[0.0, 1.5, 0.0], [2.0, 1.5, 0.0], [4.0, 1.5, 0.0]], np.float32)
    irs = np.zeros((3, 2, 8), np.float32)
    irs[:, 0, 0] = 1.0
    irs[:, 1, 0] = 1.0
    return audio.RIRGrid(cells, np.array([0.0, 1.5, 0.0], np.float32), irs, 16000, "syn")


# ----------------------------------------------------------------------
# normalize_clip / render_step_audio
# ----------------------------------------------------------------------


def case_normalize_clip_rms_target():
    sr = 16000
    sine = (3.0 * np.sin(np.linspace(0, 200, sr))).astype(np.float32)
    out = at.normalize_clip(sine, target_db=-20.0)
    assert abs(audio.rms(out) - 10.0 ** (-20.0 / 20.0)) < 1e-3, audio.rms(out)
    silent = at.normalize_clip(np.zeros(1000, np.float32))
    assert np.allclose(silent, 0.0), "silence must stay silence (no div-by-zero)"
    print("  case normalize_clip_rms_target: OK")


def case_render_silence_before_t_anom():
    cfg = at.AudioTaskConfig(enabled=True, t_anom=30)
    clip = at.normalize_clip(np.ones(100, np.float32))
    out = at.render_step_audio(_grid(), np.zeros(3, np.float32), clip, step_idx=29, cfg=cfg)
    assert out is None, "before t_anom must be silence (None)"
    print("  case render_silence_before_t_anom: OK")


def case_render_after_t_anom():
    cfg = at.AudioTaskConfig(enabled=True, t_anom=30)
    g = _grid()
    clip = at.normalize_clip(np.array([1.0, 2.0, 3.0], np.float32))
    pos = np.array([2.0, 1.5, 0.0], np.float32)
    out = at.render_step_audio(g, pos, clip, step_idx=30, cfg=cfg)
    assert out is not None and out.shape[0] == 2, out
    assert np.allclose(out, audio.render_at_pose(g, pos, clip)), "must equal render_at_pose"
    print("  case render_after_t_anom: OK")


def case_render_disabled_returns_none():
    cfg = at.AudioTaskConfig(enabled=True, t_anom=0)
    clip = at.normalize_clip(np.ones(8, np.float32))
    assert at.render_step_audio(None, np.zeros(3, np.float32), clip, 5, cfg) is None
    assert at.render_step_audio(_grid(), np.zeros(3, np.float32), None, 5, cfg) is None
    print("  case render_disabled_returns_none: OK")


# ----------------------------------------------------------------------
# process_audio_step
# ----------------------------------------------------------------------


def case_process_none_audio_noop():
    st = at.AudioEpisodeState()
    diag = at.process_audio_step(None, 5, 16000, at.AudioTaskConfig(enabled=True),
                                 st, FakeCLAP("baby_cry"))
    assert diag["onset_fired"] is False and st.detected is False
    print("  case process_none_audio_noop: OK")


def case_process_below_onset_no_detect():
    st = at.AudioEpisodeState()
    cfg = at.AudioTaskConfig(enabled=True, onset_rms=0.05)
    quiet = _binaural(0.01)
    diag = at.process_audio_step(quiet, 5, 16000, cfg, st, FakeCLAP("baby_cry"))
    assert st.detected is False and diag["audio_target_override"] is None
    print("  case process_below_onset_no_detect: OK")


def case_process_onset_fires_once():
    st = at.AudioEpisodeState()
    cfg = at.AudioTaskConfig(enabled=True, onset_rms=0.05)
    enc = FakeCLAP("baby_cry")
    loud = _binaural(0.3)
    d1 = at.process_audio_step(loud, 10, 16000, cfg, st, enc)
    assert d1["onset_fired"] is True and st.detected is True and st.onset_step == 10
    assert enc.n_audio == 1, f"classify must fire once, got {enc.n_audio}"
    d2 = at.process_audio_step(loud, 11, 16000, cfg, st, enc)
    assert d2["onset_fired"] is False, "onset must not re-fire"
    assert enc.n_audio == 1, f"must NOT re-classify, got {enc.n_audio}"
    print("  case process_onset_fires_once: OK")


def case_process_sets_target_override():
    st = at.AudioEpisodeState()
    enc = FakeCLAP("baby_cry")
    diag = at.process_audio_step(_binaural(0.3), 10, 16000,
                                 at.AudioTaskConfig(enabled=True, onset_rms=0.05), st, enc)
    assert st.anomaly_class == "baby_cry", st.anomaly_class
    assert st.target_override == audio.CLASS_TO_OBJECT["baby_cry"] == "crib"
    assert diag["audio_class"] == "baby_cry" and diag["audio_target_override"] == "crib"
    print("  case process_sets_target_override: OK")


def case_process_clap_none_records_onset_no_class():
    st = at.AudioEpisodeState()
    diag = at.process_audio_step(_binaural(0.3), 10, 16000,
                                 at.AudioTaskConfig(enabled=True, onset_rms=0.05), st, None)
    assert st.detected is True and diag["onset_fired"] is True
    assert st.anomaly_class is None and st.target_override is None, "no CLAP → no class"
    print("  case process_clap_none_records_onset_no_class: OK")


def case_process_lateral_and_energy_in_diag():
    st = at.AudioEpisodeState()
    b = _binaural(0.3, lateral=+1)
    diag = at.process_audio_step(b, 10, 16000, at.AudioTaskConfig(enabled=True), st,
                                 FakeCLAP("baby_cry"))
    assert diag["audio_lateral_sign"] == audio.lateral_sign(b) == 1
    assert abs(diag["audio_energy"] - audio.rms(b)) < 1e-6
    print("  case process_lateral_and_energy_in_diag: OK")


# ----------------------------------------------------------------------
# retrieval target + STOP gate
# ----------------------------------------------------------------------


def case_target_fallback_when_undetected():
    st = at.AudioEpisodeState()
    assert at.audio_target_for_retrieval(st, "chair") == "chair", "undetected → fallback verbatim"
    print("  case target_fallback_when_undetected: OK")


def case_target_override_when_detected():
    st = at.AudioEpisodeState(detected=True, target_override="crib")
    assert at.audio_target_for_retrieval(st, "chair") == "crib"
    print("  case target_override_when_detected: OK")


def case_target_prefers_anomaly_object_override():
    # M2: a per-episode anomaly_object (the actual captioned object the source
    # sits near, e.g. 'bed') wins over the static CLASS_TO_OBJECT mapping.
    st = at.AudioEpisodeState(detected=True, target_override="crib",
                              anomaly_object_override="bed")
    assert at.audio_target_for_retrieval(st, "chair") == "bed"
    print("  case target_prefers_anomaly_object_override: OK")


def case_target_falls_back_to_class_to_object():
    st = at.AudioEpisodeState(detected=True, target_override="crib",
                              anomaly_object_override=None)
    assert at.audio_target_for_retrieval(st, "chair") == "crib"
    print("  case target_falls_back_to_class_to_object: OK")


def case_target_objectnav_byte_identical_with_override_field():
    # undetected + overrides present must STILL return the fallback verbatim
    st = at.AudioEpisodeState(detected=False, anomaly_object_override="bed",
                              target_override="crib")
    assert at.audio_target_for_retrieval(st, "chair") == "chair"
    print("  case target_objectnav_byte_identical_with_override_field: OK")


def case_state_reset_clears_anomaly_object_override():
    st = at.AudioEpisodeState(detected=True, anomaly_object_override="bed",
                              target_override="crib")
    st.reset()
    assert st.anomaly_object_override is None and st.target_override is None
    print("  case state_reset_clears_anomaly_object_override: OK")


def case_resolve_t_anom():
    # per-episode t_anom from episode.info overrides the run-level default;
    # absent/None → default (objectnav/revisit byte-identical).
    assert at.resolve_t_anom({"t_anom": 30}, 999) == 30
    assert at.resolve_t_anom({"t_anom": 10000}, 30) == 10000
    assert at.resolve_t_anom({}, 30) == 30
    assert at.resolve_t_anom(None, 42) == 42
    assert at.resolve_t_anom({"t_anom": None}, 7) == 7
    print("  case resolve_t_anom: OK")


def case_should_stop_true():
    cfg = at.AudioTaskConfig(enabled=True, energy_stop_db_rms=0.2, stop_distance_m=1.5)
    st = at.AudioEpisodeState(detected=True)
    assert at.should_audio_stop(st, energy=0.4, distance_to_goal=1.0, cfg=cfg) is True
    print("  case should_stop_true: OK")


def case_should_stop_false_far():
    cfg = at.AudioTaskConfig(enabled=True, energy_stop_db_rms=0.2, stop_distance_m=1.5)
    st = at.AudioEpisodeState(detected=True)
    assert at.should_audio_stop(st, energy=0.4, distance_to_goal=5.0, cfg=cfg) is False
    print("  case should_stop_false_far: OK")


def case_should_stop_false_quiet_and_undetected():
    cfg = at.AudioTaskConfig(enabled=True, energy_stop_db_rms=0.2, stop_distance_m=1.5)
    st = at.AudioEpisodeState(detected=True)
    assert at.should_audio_stop(st, energy=0.05, distance_to_goal=1.0, cfg=cfg) is False
    st2 = at.AudioEpisodeState(detected=False)
    assert at.should_audio_stop(st2, energy=0.9, distance_to_goal=0.1, cfg=cfg) is False
    print("  case should_stop_false_quiet_and_undetected: OK")


def case_state_reset_clears():
    st = at.AudioEpisodeState(detected=True, anomaly_class="baby_cry",
                              target_override="crib", onset_step=7,
                              last_energy=0.5, last_lateral=1)
    st.reset()
    assert (st.detected is False and st.anomaly_class is None and st.target_override is None
            and st.onset_step is None and st.last_energy == 0.0 and st.last_lateral == 0)
    print("  case state_reset_clears: OK")


# ----------------------------------------------------------------------
# Step.audio field + two-env boundary
# ----------------------------------------------------------------------


def _mk_step(**kw):
    base = dict(step_idx=0, rgb=np.zeros((2, 2, 3), np.uint8), depth=np.zeros((2, 2), np.float32),
                semantic=None, agent_state=AgentState(np.zeros(3, np.float32), 0.0),
                action=None, reward=0.0, done=False)
    base.update(kw)
    return Step(**base)


def case_step_audio_default_none():
    s = _mk_step()
    assert s.audio is None, "Step.audio must default to None (objectnav backward compat)"
    print("  case step_audio_default_none: OK")


def case_step_audio_field_roundtrip():
    a = np.zeros((2, 1000), np.float32)
    s = _mk_step(audio=a)
    assert s.audio.shape == (2, 1000) and s.audio.dtype == np.float32
    print("  case step_audio_field_roundtrip: OK")


def case_step_existing_positional_build():
    # full positional build identical to the current code path still works
    s = Step(0, np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.float32), None,
             AgentState(np.zeros(3, np.float32), 0.0), None, 0.0, False, {})
    assert s.audio is None
    print("  case step_existing_positional_build: OK")


def case_no_habitat_sim_import():
    # Two-env boundary: the runner-side audio modules must never IMPORT the audio
    # sim (mentions in comments are fine). Check for actual import statements.
    import ast
    for mod in ("audio.py", "audio_task.py"):
        tree = ast.parse((_EMB_DIR / mod).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not n.name.startswith("habitat_sim"), f"{mod} imports {n.name}"
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("habitat_sim"), \
                    f"{mod} imports from {node.module}"
    print("  case no_habitat_sim_import: OK")


# ----------------------------------------------------------------------
# habitat_env render seam (mock env — habitat imports are lazy)
# ----------------------------------------------------------------------


def _mock_source(**kw):
    from embodied_memory.habitat_env import HabitatObjectNavSource
    src = HabitatObjectNavSource(scene_id="syn", n_episodes=1, **kw)
    src._env = object()
    src._read_agent_state = lambda env: AgentState(np.array([2.0, 1.5, 0.0], np.float32), 0.0)
    return src


def _obs():
    return {"rgb": np.zeros((4, 4, 3), np.uint8),
            "depth": np.zeros((4, 4), np.float32), "semantic": None}


def case_habitat_default_objectnav():
    src = _mock_source()
    assert src.task == "objectnav" and src._rir_grid is None
    print("  case habitat_default_objectnav: OK")


def case_habitat_make_step_audio_gated():
    src = _mock_source(task="audiogoal", t_anom=5)
    src._rir_grid = _grid()
    src._anomaly_clip_norm = at.normalize_clip(np.ones(64, np.float32))
    src._audio_render_cfg = at.AudioTaskConfig(enabled=True, t_anom=5, sample_rate=16000)
    src._step_count = 3                      # before t_anom → silence
    assert src._make_step(_obs(), 1, 0.0, False, {}).audio is None
    src._step_count = 10                     # after t_anom → rendered
    s = src._make_step(_obs(), 1, 0.0, False, {})
    assert s.audio is not None and s.audio.shape[0] == 2, s.audio
    print("  case habitat_make_step_audio_gated: OK")


def case_habitat_objectnav_make_step_no_audio():
    src = _mock_source()                     # objectnav (audio gated off)
    src._step_count = 100
    assert src._make_step(_obs(), None, 0.0, False, {}).audio is None
    print("  case habitat_objectnav_make_step_no_audio: OK")


# ----------------------------------------------------------------------
# S1 onset-gate retrieval (LTM_AUDIO_DOA): make audio causally necessary
# ----------------------------------------------------------------------


def case_build_anomaly_clip_deterministic_and_normed():
    # the synthetic burst (no clip path) must be deterministic (seed 0) and
    # RMS-normalized to -20 dBFS (~0.1 linear) — the shared clip both habitat_env
    # and the onset-calibration diagnostic rely on for a matching energy scale.
    a = at.build_anomaly_clip(None, 48000)
    b = at.build_anomaly_clip(None, 48000)
    assert a.shape == b.shape and np.allclose(a, b), "synthetic burst must be deterministic"
    assert a.shape[0] == int(48000 * 0.5), a.shape
    assert abs(at.rms(a) - 0.1) < 0.02, at.rms(a)
    print("  case build_anomaly_clip_deterministic_and_normed: OK")


def case_resolve_anomaly_clip():
    import os
    import tempfile
    # explicit --anomaly-clip wins even over a (nonexistent) clip dir
    assert at.resolve_anomaly_clip("baby_cry", explicit_path="x.wav", clip_dir="/nope") == "x.wav"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "alarm.wav")
        open(p, "wb").close()
        assert at.resolve_anomaly_clip("alarm", clip_dir=d) == p          # staged class clip
        assert at.resolve_anomaly_clip("glass_break", clip_dir=d) is None  # not staged -> burst
        assert at.resolve_anomaly_clip(None, clip_dir=d) is None
    print("  case resolve_anomaly_clip: OK")


def case_onset_gate_suppresses_pre_onset():
    # flag on + not detected -> None (zero memory injection until heard)
    assert at.gate_retrieval_target("bed", onset_gate=True, detected=False) is None
    print("  case onset_gate_suppresses_pre_onset: OK")


def case_onset_gate_passes_after_onset():
    # flag on + detected -> resolved target flows through
    assert at.gate_retrieval_target("bed", onset_gate=True, detected=True) == "bed"
    print("  case onset_gate_passes_after_onset: OK")


def case_onset_gate_off_byte_identical():
    # flag OFF -> verbatim regardless of detection (default path unchanged)
    assert at.gate_retrieval_target("bed", onset_gate=False, detected=False) == "bed"
    assert at.gate_retrieval_target("bed", onset_gate=False, detected=True) == "bed"
    assert at.gate_retrieval_target(None, onset_gate=False, detected=False) is None
    print("  case onset_gate_off_byte_identical: OK")


def main() -> int:
    cases = [
        case_normalize_clip_rms_target,
        case_render_silence_before_t_anom,
        case_render_after_t_anom,
        case_render_disabled_returns_none,
        case_process_none_audio_noop,
        case_process_below_onset_no_detect,
        case_process_onset_fires_once,
        case_process_sets_target_override,
        case_process_clap_none_records_onset_no_class,
        case_process_lateral_and_energy_in_diag,
        case_target_fallback_when_undetected,
        case_target_override_when_detected,
        case_target_prefers_anomaly_object_override,
        case_target_falls_back_to_class_to_object,
        case_target_objectnav_byte_identical_with_override_field,
        case_state_reset_clears_anomaly_object_override,
        case_resolve_t_anom,
        case_should_stop_true,
        case_should_stop_false_far,
        case_should_stop_false_quiet_and_undetected,
        case_state_reset_clears,
        case_step_audio_default_none,
        case_step_audio_field_roundtrip,
        case_step_existing_positional_build,
        case_no_habitat_sim_import,
        case_habitat_default_objectnav,
        case_habitat_make_step_audio_gated,
        case_habitat_objectnav_make_step_no_audio,
        case_build_anomaly_clip_deterministic_and_normed,
        case_resolve_anomaly_clip,
        case_onset_gate_suppresses_pre_onset,
        case_onset_gate_passes_after_onset,
        case_onset_gate_off_byte_identical,
    ]
    print(f"running {len(cases)} audio_task cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
