"""``AudioConfig`` — the audio layer's frozen sub-config (ADR-0013).

Composed into ``RunConfig`` by ticket 25 and built from ``argparse``; there is no
environment flag anywhere in this layer (ADR-0008, and ``test_no_env_flags.py`` holds
it).

**``onset_rms`` is deliberately absent.** Task spec §2.3 derives it at run start from
the calibration sweep, so putting it here would let an operator set by hand the one
number the map insists is measured — the "hand-nudged threshold" the spec names as the
wrong correction. What lives here is the *input* to that derivation: the chosen bed
level and the band to sweep. The output is a ``CalibrationResult``
(``audio/calibration.py``), carried on the audit record with its separation margin.

Three of these are task spec §9's "left to the builder" numbers, set against nothing
yet. They carry a **provenance tag** in the comment above each, per ADR-0014: `box` and
`source` are measured, `fake` and `runtime` are not, and an unmeasured constant is set
generously rather than tightly. The first box calibration is what turns ``bed_rms`` and
``audible_band_m`` from choices into measurements — or fails the gate and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

__all__ = ["AudioConfig", "WindowPolicy"]


# **Why the policy enum lives in a config module and not beside ``window.plan_window``.**
# ADR-0013 gives `earshot/config.py` exactly ("audio.config", "agent.config", "types")
# as its import edge (`tests/mac/_tree.py:133`). `RunConfig` carries the policy as a
# TOP-LEVEL field -- it has to, because `dataclasses.asdict` leaves an Enum on a
# sub-config as an Enum object and `RunConfig.as_dict` would stop being JSON -- so
# `earshot/config.py` must be able to name this type, and `audio.window` is not on that
# edge. `window.py` imports it from here and re-exports it, so callers still have one
# import site.
class WindowPolicy(Enum):
    """How long the source sounds before the offset step (ADR-0017).

    ``CONTINUOUS`` is the source that never stops: ``playing = step >= t_anom`` and
    nothing closes it, which is the pre-ADR-0017 behaviour. It is kept as the **named
    control arm** rather than as a fallback. Every funnel delta a windowed run reports
    crosses two changes at once -- the offset step and the accumulating renderer -- and
    this repo's own rule is that a claim that X broke because of a change needs the arm
    where the change is absent. The hermeticity gate already called a pre-existing
    failure a leak for want of exactly that control run.

    ``FIXED_STEPS`` is one duration on every episode, read off ``RunConfig.
    sounding_steps``. The only policy whose value can be read off a single run without
    first estimating a distribution.

    ``BUDGET_FRACTION`` is ``floor(fraction * max_steps)`` -- a run-level constant,
    while ``t_anom`` is per episode, so the window's *length* stops varying with the
    episode's geometry even though its opening does not.

    ``DRAWN`` is uniform on ``[lo, hi]`` steps, drawn per episode as a pure function of
    ``(seed, episode_index)``. What SAVi and SAVN-CE do (SAVN-CE: onset uniform on
    [0, 5] s, duration Gaussian, mean 15 s).

    **Which one is right is an open question this build does not answer.** See
    ``audio/window.py``'s module docstring: the mechanism keeps all four reachable and
    the default is provisional.
    """

    CONTINUOUS = "continuous"
    FIXED_STEPS = "fixed_steps"
    BUDGET_FRACTION = "budget_fraction"
    DRAWN = "drawn"


@dataclass(frozen=True)
class AudioConfig:
    """Every number the audio layer needs that is not derived at run start."""

    # provenance: fake — a CHOICE, not a calibration (task spec §2.3: "the bed level is
    # a chosen constant … it is our signal, unrendered and position-invariant, so there
    # is nothing to calibrate it against"). Its validity is decided by the calibration
    # gate, which either separates it from the anomaly distribution or fails and names
    # `global_volume` as the correction. Low, because the bed's whole job is to be a
    # flat floor the anomaly rises out of: it is what makes §3.1's "pre-onset RMS equals
    # the bed level" a check with content rather than an assertion about silence.
    bed_rms: float = 1e-3

    # provenance: fake — the geodesic band the §2.3 sweep sends the anomaly across, as
    # (near, far) metres. Not the audible band the *task* has: §2.5 refuses to screen
    # audibility at build time, so this is only where the calibration LOOKS. Far end
    # generous: ticket 06 walked LOS and non-LOS out to several metres and still scored
    # a climbable gradient, and a too-narrow sweep would set the threshold off a
    # distribution the episode never visits.
    audible_band_m: Tuple[float, float] = (1.0, 8.0)

    # provenance: fake — how many poses the sweep renders. Each is one live render
    # (~27 ms at ticket 06's preset), so this is cheap; the cost that matters is that
    # a percentile over too few samples is not a distribution.
    sweep_poses: int = 16

    # provenance: fake — §3.1's tolerance, RELATIVE to the bed level. The bed is
    # generated once per EPISODE at exactly the render's length and RMS-normalised
    # (`runner.py:481`; this comment said "per step" and `bed.py` said "per run", and a
    # future reader would have sized a buffer against one of them), so a healthy
    # pre-onset reading is the bed level to float precision and this could be far
    # tighter. Kept generous on ADR-0014's rule: it has never been measured against a
    # live render, and the assertion RAISES, so a tight bound would stop a run for a
    # reason that is not the fabrication it exists to catch.
    pre_onset_rms_tol: float = 0.05

    # provenance: box — `AudioSensorSpec().acousticsConfig.sampleRate` reads 44100.0 on
    # our branch (ticket 04). Held at the branch default on purpose: ESC-50 is 44.1 kHz,
    # so the clip needs no resample and `clips.load_anomaly_clip`'s resample path stays
    # unexercised on the standard route.
    sample_rate: int = 44100

    # provenance: fake — HOW MUCH AUDIO TIME ONE SIMULATOR STEP IS, and the tree had no
    # such number before ADR-0017. `AgentSpec` is 0.25 m and 30 deg per step
    # (`sim/world.py:141-142`); nothing anywhere mapped a step to a span of seconds, and
    # the accumulation buffer cannot exist without one — each step's convolution has to
    # be written at an offset, and that offset IS this number.
    #
    # 1.0 s, chosen and not derived. Deriving it (0.25 m at a walking pace) would imply a
    # measurement of a speed nobody took and would read as `source`. Round because it is
    # also the cross-quote: 500 steps is 500 s, so SAVN-CE's 15 s mean duration reads as
    # 15 steps without arithmetic. The other reading — 0.25 m at ~1 m/s, so 0.25 s — is
    # rejected here because it makes the ramp below four times as long, and a ramp that
    # wide eats any candidate window.
    #
    # What it costs, measured on this Mac (numpy, a 5 s white-noise clip at 44100 and a
    # synthetic 72300-sample IR, hop 44100): the received level RAMPS over
    # ceil(N/hop) = 5 steps to within a few percent of its settled value and is fully
    # settled after ceil((N + L - 1)/hop) = 7, then DECAYS to exactly the bed over the
    # same 7 steps after the offset step. Halving it doubles both. The ramp is why it is
    # not smaller.
    step_seconds: float = 1.0

    # provenance: box — measured 1.0 on our branch (ticket 04), NOT the 0.25 ticket 01
    # quoted from `main`. This is the ONLY correction the calibration gate may apply
    # when the bed and anomaly distributions overlap (§2.3); moving `onset_rms` by hand
    # is named in the spec as the wrong fix.
    global_volume: float = 1.0

    # `None` keeps `spec.ACOUSTICS_PRESET`'s 500. This overrides that one key and nothing
    # else, because it is the only knob in the preset that trades ACCURACY for speed
    # rather than speed alone.
    #
    # **Why it is now a run-time knob.** `yield-1` and `detour-1` ran the same scene under
    # the same configuration and 4 of 20 episodes came out with different outcomes. The
    # onset step was identical in all twenty, so the trigger is deterministic; the render
    # is not — the calibration threshold moved up to 13%, separation 2.5 dB, and the live
    # RMS at the trigger pose 24%. The navmesh `seed` is fixed and does not touch this:
    # it seeds pose DRAWS, and the same poses were rendering differently.
    #
    # `indirectRayCount` is the reason. Ticket 06 cut it 5000 -> 500 for a 63x speedup,
    # which is a Monte Carlo estimate with a tenth of the samples, and the variance is the
    # price that was paid without being measured. It is the dominant cost lever and
    # roughly linear, so this is also the one field where a number can be traded back:
    # criterion 7's ceiling is 0.5 s/step and `detour-1` measured mean 0.059, max 0.099.
    #
    # Recorded on the audit through `run_config`, so a run's variance can always be read
    # against the ray count that produced it.
    indirect_ray_count: Optional[int] = None

    # provenance: source — carried from the old tree's `--anomaly-clip` default. The
    # ESC-50 staging directory `clips.fetch_esc50_clips` writes and
    # `clips.resolve_anomaly_clip` reads.
    clip_dir: str = "data/anomaly_audio"

    # provenance: source — carried verbatim (`audio_task.AudioTaskConfig`). The clip is
    # RMS-normalised once per run so the anomaly's level is a property of the run rather
    # than of which ESC-50 recording was staged.
    target_norm_rms_db: float = -20.0
