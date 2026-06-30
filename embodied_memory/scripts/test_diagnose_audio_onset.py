"""
TDD for diagnose_audio_onset.py — the $0 onset-blocker diagnostic for a
--task anomaly_response (or audiogoal) run. Given a run dir it decides WHY
n_audio_onset_fired==0: ENERGY_TOO_LOW (rendered audio_energy never clears
onset_rms) vs GATE_SUPPRESSING (energy clears the bar but the forced CLAP
anomaly-gate / is_anomaly rejects the onset). Pure logic + JSON/log parsing,
no sim/GPU.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_diagnose_audio_onset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_audio_onset as d  # noqa: E402


# ----------------------------------------------------------------------
# classify_onset_blocker — the verdict
# ----------------------------------------------------------------------
def case_classify_fires_when_onset_fired():
    v, _ = d.classify_onset_blocker(n_onset_fired=3, max_energy=0.2, onset_rms=0.05)
    assert v == "ONSET_FIRES", v


def case_classify_gate_when_energy_clears_but_no_onset():
    # energy >= onset_rms but onset never fired => the gate (is_anomaly) rejected it
    v, rec = d.classify_onset_blocker(n_onset_fired=0, max_energy=0.12, onset_rms=0.05)
    assert v == "GATE_SUPPRESSING", v
    assert "gate" in rec.lower()


def case_classify_energy_when_below_threshold():
    v, rec = d.classify_onset_blocker(n_onset_fired=0, max_energy=0.01, onset_rms=0.05)
    assert v == "ENERGY_TOO_LOW", v
    assert "onset" in rec.lower()


def case_classify_gate_rejected_counter_is_authoritative():
    # the per-tick n_gate_rejected counter PROVES the gate suppressed onset, even
    # if the keyframe-sparse max_energy looks 0 (the loud ticks weren't keyframes).
    v, rec = d.classify_onset_blocker(n_onset_fired=0, max_energy=0.0,
                                      onset_rms=0.05, n_gate_rejected=7)
    assert v == "GATE_SUPPRESSING", v
    assert "gate" in rec.lower()


def case_classify_unknown_threshold_when_onset_rms_none():
    v, _ = d.classify_onset_blocker(n_onset_fired=0, max_energy=0.01, onset_rms=None)
    assert v == "UNKNOWN_THRESHOLD", v


def case_classify_boundary_equal_is_gate():
    # energy exactly at the threshold counts as "clears the bar" (onset uses >=)
    v, _ = d.classify_onset_blocker(n_onset_fired=0, max_energy=0.05, onset_rms=0.05)
    assert v == "GATE_SUPPRESSING", v


# ----------------------------------------------------------------------
# episode_energy_stats — per-episode audio_energy from a loaded ep_log
# ----------------------------------------------------------------------
def case_energy_stats_basic():
    ep = {"steps": [
        {"audio_energy": 0.0}, {"audio_energy": 0.03}, {"audio_energy": 0.11},
        {"audio_energy": None}, {"caption": "no audio key"},
    ]}
    s = d.episode_energy_stats(ep)
    assert s["n_steps"] == 5
    assert s["n_audible"] == 2          # 0.03 and 0.11 are > 0
    assert abs(s["max_energy"] - 0.11) < 1e-9
    assert s["max_energy"] >= s["mean_energy"] > 0.0


def case_energy_stats_no_audio():
    ep = {"steps": [{"caption": "x"}, {"audio_energy": None}]}
    s = d.episode_energy_stats(ep)
    assert s["n_audible"] == 0
    assert s["max_energy"] == 0.0


# ----------------------------------------------------------------------
# find_onset_rms — parse the calibrated/pinned value from the run log text
# ----------------------------------------------------------------------
def case_find_onset_rms_recommend_line():
    txt = "blah\nRECOMMEND_ONSET_RMS=0.065432\nmore"
    assert abs(d.find_onset_rms(txt) - 0.065432) < 1e-9


def case_find_onset_rms_driver_line():
    txt = "  onset_rms (calibrated for 4.0 m audible radius) = 0.05\n"
    assert abs(d.find_onset_rms(txt) - 0.05) < 1e-9


def case_find_onset_rms_pinned_line():
    txt = "  onset_rms pinned (override) = 0.008\n"
    assert abs(d.find_onset_rms(txt) - 0.008) < 1e-9


def case_find_onset_rms_prefers_last():
    # the driver prints RECOMMEND_ONSET_RMS then the final "onset_rms ... = X";
    # both should agree, but the LAST value is the one actually used.
    txt = "RECOMMEND_ONSET_RMS=0.07\n  onset_rms (calibrated for 4.0 m audible radius) = 0.07\n"
    assert abs(d.find_onset_rms(txt) - 0.07) < 1e-9


def case_find_onset_rms_none_when_absent():
    assert d.find_onset_rms("no threshold here") is None


def main() -> int:
    cases = [
        case_classify_fires_when_onset_fired,
        case_classify_gate_when_energy_clears_but_no_onset,
        case_classify_energy_when_below_threshold,
        case_classify_gate_rejected_counter_is_authoritative,
        case_classify_unknown_threshold_when_onset_rms_none,
        case_classify_boundary_equal_is_gate,
        case_energy_stats_basic,
        case_energy_stats_no_audio,
        case_find_onset_rms_recommend_line,
        case_find_onset_rms_driver_line,
        case_find_onset_rms_pinned_line,
        case_find_onset_rms_prefers_last,
        case_find_onset_rms_none_when_absent,
    ]
    print(f"running {len(cases)} diagnose_audio_onset cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
