"""The background bed, and what the agent hears once it is mixed in (ADR-0009).

The bed is a **fixed-level diotic signal, generated directly and mixed after
rendering**. It never touches the RIR, so it is position-invariant *by construction*
rather than by a calibration that has to hold — which is what `CONTEXT.md` has always
defined it as and what the grid-era implementation was not.

Three consequences the rest of the layer depends on, none of which is a coincidence:

- **§2.4's absolute threshold is sound.** ADR-0004 ruled an absolute RMS threshold
  impossible because a 1.4x temporal step sat inside an 8x spatial swing. That argument
  needed the bed to be *rendered*. Unrendered, the pre-onset signal is flat at the bed
  level at every pose in every scene, and the whole spatial swing lives in the
  post-onset term.
- **§3.1's provenance check has content.** "Pre-onset measured RMS equals the bed level
  within tolerance" is only an assertion worth raising on because the expected value is
  a constant we chose, not a function of where the agent happens to stand.
- **`diotic_collapse` does not carry.** The bed is generated diotic at source; there is
  nothing to collapse from a render. Both ears get the same samples, so the bed
  contributes exactly zero to `lateral.lateral_sign` — it cannot bias the cue the
  controller turns on.

``heard_signal`` is the composition point: §7 says ``process_audio_step``'s per-step
orchestration is "rewritten rather than ported", and this is what it reduced to once
the grid lookup was gone. Pure, so ticket 25's runner wires it rather than reimplements
it, and so the onset detector and CLAP are guaranteed to be looking at the same signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from earshot.audio.clips import render_through_ir, rms

__all__ = ["bed_signal", "mix_bed", "heard_signal"]

# The bed is noise rather than a tone, and the seed is fixed. A tone would put all its
# energy in one band, where a room mode or an ESC-50 clip that happens to share it makes
# the pre-onset level pose-dependent after all. Broadband noise at a normalised RMS has
# the flat statistics §2.4 needs. Fixed seed because a red run that cannot be reproduced
# is not evidence (ticket 21's navmesh seed, same reason).
BED_SEED = 20260804


def bed_signal(n_samples: int, level_rms: float, seed: int = BED_SEED) -> np.ndarray:
    """``(2, n_samples)`` of diotic noise at exactly ``level_rms``.

    Diotic: the same samples in both ears, so the bed carries no interaural level
    difference and cannot move the lateral cue.

    Normalised to the requested RMS *after* generation, so the returned buffer's RMS is
    the level to float precision rather than in expectation. That is what lets
    ``AudioConfig.pre_onset_rms_tol`` be a tolerance on drift rather than a slack budget
    for sampling noise.

    Generated once per run at the clip's length, not once per step: it is time-invariant
    as well as position-invariant, and regenerating it per step would make the pre-onset
    reading a fresh random draw — the very variance the assertion is trying to see.
    """
    n = int(n_samples)
    if n <= 0:
        raise ValueError("bed length must be positive, got {}".format(n_samples))
    level = float(level_rms)
    if level < 0.0:
        raise ValueError("bed level must not be negative, got {}".format(level_rms))
    mono = np.random.default_rng(int(seed)).standard_normal(n).astype(np.float32)
    current = rms(mono)
    if current > 1e-12 and level > 0.0:
        mono = (mono * (level / current)).astype(np.float32)
    elif level == 0.0:
        mono = np.zeros(n, dtype=np.float32)
    return np.stack([mono, mono]).astype(np.float32)


def mix_bed(rendered: Any, bed: Any) -> np.ndarray:
    """Add the bed to a rendered binaural signal. Lengths must already agree.

    Deliberately not tolerant of a mismatch. The bed is generated at the clip's length
    and ``clips.render_through_ir`` trims to the same, so a mismatch means one of those
    two changed — and the tempting fixes (tile the bed, crop the render) would each
    silently change the RMS the onset threshold was calibrated against.
    """
    signal = np.asarray(rendered, dtype=np.float32)
    floor = np.asarray(bed, dtype=np.float32)
    if signal.shape != floor.shape:
        raise ValueError(
            "bed is {} but the rendered signal is {} — both are built at the clip's "
            "length, so a mismatch means the clip changed under one of them".format(
                floor.shape, signal.shape
            )
        )
    return (signal + floor).astype(np.float32)


def heard_signal(ir: Any, clip: Any, bed: Any, *, playing: bool) -> np.ndarray:
    """``(2, N)`` — everything the agent hears this step.

    ``playing`` is whether the anomaly source is sounding, which is ``step >= t_anom``
    and nothing else. Before that the anomaly contributes **exactly zero** (§3.1) — not
    a scaled-down render, not a fade: the clip is simply not mixed, which is what makes
    ``onset_step < t_anom`` structurally impossible rather than merely unlikely.

    **There is no pose parameter, and that is the whole difference from the grid.**
    ``render_step_audio`` took ``agent_pos`` because it had to choose a cell, which is
    where ``nearest`` could fabricate audio for a pose it had no data for (ADR-0003).
    Here the geometry is already inside the IR the simulator just rendered, and the bed
    is position-invariant, so there is nowhere for a coordinate to enter.
    """
    if not playing:
        return np.asarray(bed, dtype=np.float32)
    return mix_bed(render_through_ir(ir, clip), bed)
