"""The one-shot onset threshold, and §3.1's provenance invariants — which raise.

The detector itself is small: the agent's own measured RMS crosses ``onset_rms`` once,
and the step it happened is the onset step. Everything the old tree had around it —
the temporal step detector ADR-0004 called the technically correct fix, ``bg_gain``,
the ``AUDIBLE``/``OUT_OF_COVERAGE`` cell screen — is gone, because an unrendered bed
made the pre-onset signal flat at every pose (§2.4, ADR-0009).

**The provenance check changed job, and that is the substance of this module.**
Under ADR-0009 ``onset_step < t_anom`` is structurally impossible: the bed is below
threshold and position-invariant, the anomaly contributes exactly zero before
``t_anom``, and there is no second source. So the check can no longer be a diagnostic
read afterwards — it is an **asserted invariant that raises**, because the only ways it
can now fail are the bed level drifting or the source starting early, and both are
silent-fabrication bugs of exactly the kind this map keeps finding. This is ticket 12's
discipline applied to the signal instead of the mesh.

That is the `anommxv` break closed by construction rather than by a heuristic: the
interrupt firing on the wrong thing invalidated an n=64 headline, and it was visible
only in hindsight because the check was a log line.

Two things this module deliberately does not do:

- It does not decide whether the sound is *anomalous*. That is ``clap.is_anomaly``
  composed with ``normality``, and ticket 25 wires them; onset is energy alone.
- It holds no state of its own. ``OnsetState`` is frozen and every transition returns a
  new one, so the runner owns the episode's mutable state in one place and a leaked
  detector cannot carry an onset across episodes — which is the bug ``AudioEpisodeState.
  reset()`` existed to paper over.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

__all__ = ["ProvenanceError", "OnsetState", "observe_step", "assert_provenance"]


class ProvenanceError(RuntimeError):
    """A §3.1 invariant failed: the measured signal is not what the task built.

    Fatal on purpose. Every episode after this one would be measuring a mixture nobody
    specified, and the funnel counts would still look reasonable.
    """


@dataclass(frozen=True)
class OnsetState:
    """The episode's onset, and the evidence that it is honest.

    ``onset_step`` is ``None`` until the threshold is crossed and never changes again —
    "one-shot" is a property of this type rather than of a flag the caller has to
    respect. ``pre_onset_rms`` is the last reading taken before ``t_anom``, kept because
    §5.2 requires the *measured* pre-onset bed RMS on the audit record: the assertion
    passing is not the same artefact as the number it passed on.
    """

    onset_step: Optional[int] = None
    onset_rms_measured: Optional[float] = None
    pre_onset_rms: Optional[float] = None
    n_pre_onset_readings: int = 0

    @property
    def fired(self) -> bool:
        return self.onset_step is not None


def observe_step(
    state: OnsetState,
    *,
    step: int,
    measured_rms: float,
    t_anom: int,
    onset_rms: float,
    bed_rms: float,
    tolerance: float,
) -> OnsetState:
    """Fold one step's measured RMS into the onset state. Raises on a §3.1 violation.

    ``measured_rms`` is ``clips.rms`` of ``bed.heard_signal`` — the agent's own signal,
    the same number the per-step record carries and CLAP is handed. Passing anything
    else (the IR's energy, a windowed slice) would silently move the domain the
    threshold was calibrated in.

    ``tolerance`` is **relative** to ``bed_rms``. Absolute would need re-picking every
    time the bed level moved, and the bed level is a free constant (§2.3).

    This is §3.1's **first** invariant: before ``t_anom``, the measured RMS is the bed
    level. The anomaly is not mixed in at all before then, so any deviation means the
    bed drifted or something else is sounding. Checked on every pre-``t_anom`` step
    rather than once, because "the source started early" and "the bed decayed at step
    40" are different bugs and only a per-step check separates them.

    The second — ``onset_step >= t_anom`` — is ``assert_provenance``, on the recorded
    state, because it cannot fail here: this function's own control flow makes an early
    onset unrepresentable, and a branch that cannot be reached is not a check.

    A zero bed level makes the relative tolerance vacuous, so it is handled as the
    special case it is: with no bed there is nothing to drift, and the honest assertion
    is that the pre-onset signal is silent.
    """
    if step < int(t_anom):
        expected = float(bed_rms)
        allowed = abs(expected) * float(tolerance)
        if expected == 0.0:
            allowed = float(tolerance)
        if abs(float(measured_rms) - expected) > allowed:
            raise ProvenanceError(
                "step {}: pre-onset RMS {:.6g} is not the bed level {:.6g} "
                "(tolerance {:.3%}, t_anom {}). The anomaly is not mixed in before "
                "t_anom, so either the bed drifted or something started sounding "
                "early. §3.1 raises on this rather than logging it: the `anommxv` "
                "matrix ran to completion with the interrupt firing on the wrong "
                "sound.".format(step, float(measured_rms), expected, float(tolerance), int(t_anom))
            )
        return replace(
            state,
            pre_onset_rms=float(measured_rms),
            n_pre_onset_readings=state.n_pre_onset_readings + 1,
        )

    if state.fired or float(measured_rms) < float(onset_rms):
        return state
    return replace(
        state, onset_step=int(step), onset_rms_measured=float(measured_rms)
    )


def assert_provenance(
    state: OnsetState, *, t_anom: int, bed_rms: float, tolerance: float
) -> None:
    """§3.1's second invariant, plus the one ticket 16 taught: did it prove anything?

    Called on the recorded state — at the end of an episode, and before the audit record
    is written — rather than inside the per-step fold. Its subject is the *artefact*: an
    ``OnsetState`` that reached here with an impossible onset step, or with a pre-onset
    reading that no longer matches the bed, is what an analyst would otherwise quote.

    Three checks:

    1. ``onset_step >= t_anom``. Unrepresentable through ``observe_step`` and asserted
       anyway, because a state can be built by any caller and this is the invariant the
       spec states.
    2. The recorded ``pre_onset_rms`` still equals the bed level. Cheap, and it catches
       a state assembled from the wrong episode's readings.
    3. **The invariant was exercised.** With ``t_anom > 0`` and zero pre-onset readings,
       check 1 of ``observe_step`` never ran: invariant 1 is *unverified*, not
       satisfied. This is ticket 16's log canary in a different costume — a guard whose
       evidence never arrived reports that, rather than green.
    """
    if state.fired and int(state.onset_step) < int(t_anom):
        raise ProvenanceError(
            "onset recorded at step {} but t_anom is {} — structurally impossible under "
            "ADR-0009 (one source, unrendered bed, zero anomaly contribution before "
            "t_anom), so this state did not come from the task that was built".format(
                int(state.onset_step), int(t_anom)
            )
        )
    if state.pre_onset_rms is not None:
        expected = float(bed_rms)
        allowed = abs(expected) * float(tolerance) if expected != 0.0 else float(tolerance)
        if abs(float(state.pre_onset_rms) - expected) > allowed:
            raise ProvenanceError(
                "recorded pre-onset RMS {:.6g} is not the bed level {:.6g} within "
                "{:.3%}".format(float(state.pre_onset_rms), expected, float(tolerance))
            )
    if int(t_anom) > 0 and state.n_pre_onset_readings == 0:
        raise ProvenanceError(
            "t_anom is {} but no pre-onset reading was taken, so §3.1's first invariant "
            "never ran. It is unverified, not satisfied — the episode either never "
            "stepped before t_anom or the runner did not fold its readings in".format(
                int(t_anom)
            )
        )
