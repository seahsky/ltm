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

    Generated once per EPISODE, not once per step: it is time-invariant as well as
    position-invariant, and regenerating it per step would make the pre-onset reading a
    fresh random draw — the very variance the assertion is trying to see. This line used
    to say "once per run" while ``audio/config.py`` said "per step"; both were wrong and
    a future reader would have sized a buffer against one of them.

    **Since ADR-0019's split readout the runner builds TWO beds from this function**, one
    at ``hop`` for ``tail.heard_step`` and one at ``len(clip)`` for
    ``tail.heard_clip_window``, each normalised at its own length. A ``hop``-length SLICE
    of the clip-length bed was rejected, and the arithmetic is why: a slice of ``n``
    Gaussian samples carries a relative RMS error of about ``1/sqrt(2n)``, and ``n`` is
    ``hop``, which is a free parameter (``step_seconds`` × ``sample_rate``). Measured
    against the fixed ``BED_SEED`` — at the shipped hop of 44100 the last-hop slice
    deviates 0.0023% and the worst disjoint slice 0.3107%, harmless; at the runner
    fixture's hop of 441 the last-hop slice deviates 3.8960% and the worst disjoint slice
    6.7906%; at the tail fixture's hop of 100 the worst disjoint slice is 17.7320%.
    ``AudioConfig.pre_onset_rms_tol`` is 0.05, so a slice would raise ``ProvenanceError``
    outright at two configurations this tree ships tests at and would spend 78% of the
    tolerance at a third — and the cost scales the wrong way, since the smaller the step
    the worse it gets. The paragraph above promises that normalising after generation is
    what makes that tolerance a bound on DRIFT rather than a slack budget for sampling
    noise; slicing spends exactly what it promised not to. Two beds are exact to within
    a measured 3.7e-08 relative at every length.

    **The two beds are NOT sample-aligned**, so nothing may compare their samples: they
    share ``BED_SEED`` and therefore their leading samples, but each is scaled by its own
    RMS. Nothing does compare them — the cue bed feeds the onset and the clip bed feeds
    CLAP, and neither is diffed against the other.
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

    Deliberately not tolerant of a mismatch, and since ADR-0019 the refusal is
    load-bearing rather than defensive. There are now exactly TWO lengths a caller can
    legitimately be at — ``hop``, the cue readout's width, and ``len(clip)``, the clip
    readout's — so a mismatch means the bed was built for the OTHER readout. That is what
    catches ``tail.heard_step(bed_cue=bed_clip)``, which would otherwise compose a signal
    of the wrong length and, worse, of the wrong domain.

    The tempting fixes (tile the bed, crop the render) would each silently change the RMS
    the onset threshold was calibrated against, and cropping in particular is the exact
    move ``bed_signal``'s docstring measures at 6.79% error against a 5% tolerance.
    """
    signal = np.asarray(rendered, dtype=np.float32)
    floor = np.asarray(bed, dtype=np.float32)
    if signal.shape != floor.shape:
        raise ValueError(
            "bed is {} but the signal is {} — a bed is built either at hop (the cue "
            "readout's width) or at len(clip) (the clip readout's), so a mismatch means "
            "the bed was built for the other readout. Pass bed_cue to heard_step and "
            "bed_clip to heard_clip_window; never slice one from the other.".format(
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

    **Since ADR-0017 the runner composes through ``audio.tail.heard_step`` instead.** A
    bounded sounding window cut to the bed with no tail is unphysical: this function's
    not-playing branch returns the bed itself, so the silence arrives as a hard step. It
    is retained, unchanged and un-called by the runner, as the **continuous-source
    composition and the named control** the tail's Mac tests measure their decay
    against — the way ``test_rising_window.py`` keeps ``OLD_EPS``. A control that is
    deleted is a comparison that cannot be made twice.
    """
    if not playing:
        return np.asarray(bed, dtype=np.float32)
    return mix_bed(render_through_ir(ir, clip), bed)
