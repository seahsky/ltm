"""
TDD for ONSET PROVENANCE — the check that separates a working interrupt from a
vacuum cleaner.

On 2026-07-16 a run exited 0 and printed CONTROLLER_VERDICT=CONTROLLER_RAN,
FEASIBILITY_RESULT=GO and warm S3-S1 +0.4454 (p=0.000). Every one of those was
measuring something else: all eight onsets fired at step 0-10 against t_anom=30, so
the alarm never triggered a single interrupt — the background bed did. Every counter
was green. The evidence lived in one `[audio] onset @step` LOG line that nothing
read, and would have been lost had the log not been pasted by hand.

`n_audio_onset_fired` counts onsets, not CAUSES. `n_audio_gate_rejected == 0` does
NOT mean the gate had nothing to reject — onset is one-shot, so 0 means the gate
ACCEPTED the first over-threshold tick. Neither can say WHAT fired.

The only signal that can is `onset_step` vs `t_anom`: an onset BEFORE t_anom cannot
be the anomaly, because the anomaly is not playing yet. So the runner must surface
both into `summary.episodes` (NOT the keyframe-sparse ep_log["steps"], whose
sparsity already defeated diagnose_audio_onset once), and the census must refuse to
certify a run whose interrupts predate the sound.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        /opt/anaconda3/envs/ltm-embodied/bin/python \
        embodied_memory/scripts/test_onset_provenance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_anomaly_controller as dac  # noqa: E402
from embodied_memory.episode_runner import RunSummary  # noqa: E402


def _row(**kw):
    """A census row in episode_row()'s shape, with the provenance fields merged in."""
    base = {
        "name": "r", "episode_id": "0", "n_steps": 100, "success_1m": True,
        "has_report": True, "investigated": True, "resumed": True,
        "investigate_aborted": False, "primary_completed": False,
        "primary_completed_1m": True, "n_benign_ignored": 0,
        "anomaly_class": "alarm", "expected_interrupt": None,
        "n_audio_onset_fired": 1, "audio_onset_step": 40, "audio_t_anom": 30,
    }
    base.update(kw)
    return base


# ----------------------------------------------------------------------
# the runner must SURFACE the provenance (summary.episodes, not the sparse trace)
# ----------------------------------------------------------------------


def case_summary_carries_onset_step_and_t_anom():
    d = RunSummary().to_dict()
    for k in ("audio_onset_step", "audio_t_anom"):
        assert k in d, f"{k} must be in summary.json; got {sorted(d.keys())}"
    print("  case summary_carries_onset_step_and_t_anom: OK")


def case_summary_onset_step_defaults_to_none_not_zero():
    # 0 is a REAL onset step (the bed fired at step 0). A default of 0 would be
    # indistinguishable from the worst false-fire there is.
    s = RunSummary()
    assert s.audio_onset_step is None, s.audio_onset_step
    assert s.to_dict()["audio_onset_step"] is None
    s.audio_onset_step = 0
    assert s.to_dict()["audio_onset_step"] == 0, "step 0 must survive as 0, not None"
    print("  case summary_onset_step_defaults_to_none_not_zero: OK")


# ----------------------------------------------------------------------
# onset_provenance — the pure verdict
# ----------------------------------------------------------------------


def case_onset_at_or_after_t_anom_is_the_anomaly():
    assert dac.onset_provenance(_row(audio_onset_step=40, audio_t_anom=30)) == "ANOMALY"
    assert dac.onset_provenance(_row(audio_onset_step=30, audio_t_anom=30)) == "ANOMALY"
    print("  case onset_at_or_after_t_anom_is_the_anomaly: OK")


def case_onset_before_t_anom_is_a_false_fire():
    # The real runs/anomresp-bed-s3 numbers: onsets at 8/0/0/7 against t_anom=30.
    for step in (0, 7, 8, 29):
        assert dac.onset_provenance(_row(audio_onset_step=step, audio_t_anom=30)) \
            == "FALSE_FIRE", step
    print("  case onset_before_t_anom_is_a_false_fire: OK")


def case_silent_cold_pass_onset_is_a_false_fire():
    # The cold seed is silent (t_anom=10000), so ANY onset there is the bed. This
    # is the episode that first gave the game away.
    assert dac.onset_provenance(_row(audio_onset_step=8, audio_t_anom=10000)) \
        == "FALSE_FIRE"
    print("  case silent_cold_pass_onset_is_a_false_fire: OK")


def case_no_onset_is_not_a_false_fire():
    # A silent episode is not a false fire — it is simply no interrupt.
    r = _row(n_audio_onset_fired=0, audio_onset_step=None)
    assert dac.onset_provenance(r) == "NO_ONSET", dac.onset_provenance(r)
    print("  case no_onset_is_not_a_false_fire: OK")


def case_missing_fields_are_unknown_not_clean():
    # Archived runs (pre-provenance summaries) must read UNKNOWN, never ANOMALY —
    # the whole point is refusing to certify what we cannot see.
    assert dac.onset_provenance({"n_audio_onset_fired": 1}) == "UNKNOWN"
    assert dac.onset_provenance(_row(audio_t_anom=None)) == "UNKNOWN"
    assert dac.onset_provenance(_row(audio_onset_step=None)) == "UNKNOWN"
    print("  case missing_fields_are_unknown_not_clean: OK")


# ----------------------------------------------------------------------
# aggregate + verdict must REFUSE to certify a bed-triggered run
# ----------------------------------------------------------------------


def case_aggregate_counts_provenance():
    rows = [_row(audio_onset_step=40), _row(audio_onset_step=0),
            _row(n_audio_onset_fired=0, audio_onset_step=None)]
    agg = dac.aggregate(rows)
    assert agg["n_onset_anomaly"] == 1, agg
    assert agg["n_onset_false_fire"] == 1, agg
    print("  case aggregate_counts_provenance: OK")


def case_verdict_refuses_when_every_interrupt_predates_the_sound():
    # Reproduces runs/anomresp-bed-s3 exactly: investigated AND resumed (so the old
    # rule certifies CONTROLLER_RAN) but every onset fired before t_anom.
    rows = [
        _row(episode_id="0", audio_onset_step=8, investigated=False, resumed=False),
        _row(episode_id="1", audio_onset_step=0, investigated=True, resumed=True),
        _row(episode_id="2", audio_onset_step=0, investigated=False, resumed=True),
        _row(episode_id="3", audio_onset_step=7, investigated=False, resumed=True),
    ]
    v, why = dac.verdict(dac.aggregate(rows))
    assert v == "FALSE_FIRE", (
        f"a run whose interrupts all predate the anomaly must NOT certify; got {v}")
    assert "t_anom" in why or "predate" in why.lower(), why
    print("  case verdict_refuses_when_every_interrupt_predates_the_sound: OK")


def case_verdict_certifies_a_genuine_loop():
    rows = [
        _row(episode_id="0", n_audio_onset_fired=0, audio_onset_step=None,
             investigated=False, resumed=False),
        _row(episode_id="1", audio_onset_step=34, investigated=True, resumed=True),
        _row(episode_id="2", audio_onset_step=31, investigated=False, resumed=True),
    ]
    assert dac.verdict(dac.aggregate(rows))[0] == "CONTROLLER_RAN"
    print("  case verdict_certifies_a_genuine_loop: OK")


def case_one_genuine_loop_survives_a_stray_false_fire():
    # A single bed-triggered episode must not condemn a run that also has a real
    # anomaly-triggered loop — refuse only when NO interrupt was genuine.
    rows = [
        _row(episode_id="0", audio_onset_step=2, investigated=True, resumed=True),
        _row(episode_id="1", audio_onset_step=35, investigated=True, resumed=True),
    ]
    assert dac.verdict(dac.aggregate(rows))[0] == "CONTROLLER_RAN"
    print("  case one_genuine_loop_survives_a_stray_false_fire: OK")


def case_archived_runs_without_provenance_verdict_as_before():
    # Back-compat: the census runs against archived summaries. No provenance fields
    # => UNKNOWN => the pre-existing behaviour, never a spurious FALSE_FIRE.
    rows = [{"name": "r", "episode_id": "0", "n_steps": 10, "has_report": True,
             "investigated": True, "resumed": True}]
    assert dac.verdict(dac.aggregate(rows))[0] == "CONTROLLER_RAN"
    print("  case archived_runs_without_provenance_verdict_as_before: OK")


def main() -> int:
    cases = [
        case_summary_carries_onset_step_and_t_anom,
        case_summary_onset_step_defaults_to_none_not_zero,
        case_onset_at_or_after_t_anom_is_the_anomaly,
        case_onset_before_t_anom_is_a_false_fire,
        case_silent_cold_pass_onset_is_a_false_fire,
        case_no_onset_is_not_a_false_fire,
        case_missing_fields_are_unknown_not_clean,
        case_aggregate_counts_provenance,
        case_verdict_refuses_when_every_interrupt_predates_the_sound,
        case_verdict_certifies_a_genuine_loop,
        case_one_genuine_loop_survives_a_stray_false_fire,
        case_archived_runs_without_provenance_verdict_as_before,
    ]
    print(f"running {len(cases)} onset_provenance cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
