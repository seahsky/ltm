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
from typing import Tuple

__all__ = ["AudioConfig"]


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
    # generated per step at exactly the render's length and RMS-normalised, so a healthy
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

    # provenance: box — measured 1.0 on our branch (ticket 04), NOT the 0.25 ticket 01
    # quoted from `main`. This is the ONLY correction the calibration gate may apply
    # when the bed and anomaly distributions overlap (§2.3); moving `onset_rms` by hand
    # is named in the spec as the wrong fix.
    global_volume: float = 1.0

    # provenance: source — carried from the old tree's `--anomaly-clip` default. The
    # ESC-50 staging directory `clips.fetch_esc50_clips` writes and
    # `clips.resolve_anomaly_clip` reads.
    clip_dir: str = "data/anomaly_audio"

    # provenance: source — carried verbatim (`audio_task.AudioTaskConfig`). The clip is
    # RMS-normalised once per run so the anomaly's level is a property of the run rather
    # than of which ESC-50 recording was staged.
    target_norm_rms_db: float = -20.0
