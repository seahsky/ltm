"""§2.3's calibration sweep: where ``onset_rms`` comes from, and the gate on it.

The threshold is **derived from measurement at run start**, which is why it is not on
``AudioConfig``. The procedure, verbatim from the task spec:

1. Render the anomaly at a spread of poses across the intended audible band.
2. Take the bed level ``B`` and the distribution of anomaly RMS over those poses.
3. Set ``onset_rms`` strictly between ``B`` and the low percentile of the anomaly
   distribution.
4. Report the separation between the two distributions as the gate number, in the
   pattern of ticket 13's EER and the CapRL separation gate.

**If the distributions overlap the gate fails, and the correction is ``globalVolume``,
never a hand-nudged threshold.** That sentence is the whole reason this module raises
instead of returning a best-effort number: a threshold nudged until the smoke passes is
a threshold that means nothing, and the map's record has a matrix that ran to
completion on exactly that kind of number.

The old ``onset_rms`` of 0.065 does not carry. It was calibrated against a grid render
with a *rendered* bed, and neither of those exists here.

Everything is injected — the sweep takes a ``render_at`` callable and a list of poses —
so the arithmetic is Mac-testable and only the poses come from the box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from earshot.audio.clips import render_through_ir, rms

__all__ = [
    "CalibrationError",
    "CalibrationResult",
    "ANOMALY_LOW_PERCENTILE",
    "MIN_SEPARATION_DB",
    "SCATTER_REPEATS",
    "sweep_anomaly_rms",
    "sweep_render_scatter",
    "render_scatter_of",
    "calibrate_onset",
    "band_poses",
]


class CalibrationError(RuntimeError):
    """The gate failed: the bed and the anomaly are not separable at these levels."""


# provenance: fake — which end of the anomaly distribution the threshold must clear. The
# 10th percentile rather than the minimum: one pose behind a closed door is the tail
# §2.5 deliberately refuses to screen out at build time, and letting it set the
# threshold would drag it down towards the bed for every other pose in the episode. That
# attrition belongs in the funnel's stage 3, not in the threshold.
ANOMALY_LOW_PERCENTILE = 10.0

# provenance: fake — how much daylight the gate demands, in dB, between the bed and that
# percentile. 6 dB is a factor of two in amplitude; below it the "strictly between"
# placement has no room and a small drift in either distribution re-crosses it. Set here
# rather than at zero so the gate fails while the fix is still `globalVolume` rather
# than after a run has been quoted.
MIN_SEPARATION_DB = 6.0


@dataclass(frozen=True)
class CalibrationResult:
    """The derived threshold and the evidence for it. Lands on the audit record (§5.2).

    ``separation_db`` is the gate number — the thing to paste back, in the pattern of
    ticket 13's EER 0.00 and the CapRL gate's separation. ``passed`` is always ``True``
    on a returned result, because ``calibrate_onset`` raises otherwise; it exists so the
    serialised record says so explicitly rather than by absence.
    """

    onset_rms: float
    bed_rms: float
    anomaly_low: float
    anomaly_median: float
    anomaly_min: float
    anomaly_max: float
    separation_db: float
    n_poses: int
    global_volume: float
    passed: bool = True
    render_scatter: Optional[float] = None
    scatter_repeats: int = 0
    # (distance-to-source, received RMS) for every pose the sweep visited — the FIELD
    # PROFILE. The sweep already renders this curve to place the threshold and used to
    # keep only four percentiles of it, which is the distance axis thrown away: the
    # percentiles say how loud the anomaly gets, and the pairs say whether it gets louder
    # as you approach, which is the entire premise of an energy-gradient climb.
    profile: Tuple[Tuple[float, float], ...] = ()

    def as_dict(self) -> dict:
        return {
            "profile": [[float(d), float(r)] for d, r in self.profile],
            "onset_rms": self.onset_rms,
            "bed_rms": self.bed_rms,
            "anomaly_low": self.anomaly_low,
            "anomaly_median": self.anomaly_median,
            "anomaly_min": self.anomaly_min,
            "anomaly_max": self.anomaly_max,
            "separation_db": self.separation_db,
            "n_poses": self.n_poses,
            "global_volume": self.global_volume,
            "passed": self.passed,
            "render_scatter": (
                None if self.render_scatter is None else float(self.render_scatter)
            ),
            "scatter_repeats": int(self.scatter_repeats),
        }


def sweep_anomaly_rms(
    poses: Sequence[Any], render_at: Callable[[Any], Any], clip: Any
) -> List[float]:
    """The anomaly's RMS at each pose. One live render per pose, nothing else.

    ``render_at(pose)`` returns that pose's raw binaural IR — in the runner, seat the
    agent and call ``guarded_observe``. The bed is **not** mixed in: this measures the
    anomaly's own distribution, and the bed is the other side of the comparison.

    Measured through ``clips.render_through_ir``, not on the IR's own energy, because
    the threshold is applied to what the agent hears. An IR and a received signal differ
    by the clip's own level, so calibrating on one and thresholding on the other is a
    silent unit error — the kind that shows up as a threshold that never fires.
    """
    return [rms(render_through_ir(render_at(pose), clip)) for pose in poses]


# provenance: fake — how many times one pose is re-rendered to size the renderer's own
# non-determinism.
#
# **Three was too few, and the consumer is why.** A sample SD carries a relative error of
# about 1/sqrt(2(n-1)) and a low bias of (1 - c4): at n=3 that is ~50% and ~11%, so the
# estimate swings roughly threefold between episodes on nothing but its own noise. This
# number is not reported — it is the `eps` in the climb's rising test, so that swing is a
# CONTROL PARAMETER moving several-fold across episodes of one run, and two episodes in
# the same room can be running materially different controllers. At twelve the relative
# error is ~21% and the bias ~2%.
#
# The cost stays negligible because it is renders, not steps: eleven extra per episode is
# ~6.6 s against `detour-2`'s 8m26s over 20 episodes, ~1.3%, and criterion 7's per-step
# audio ceiling is untouched — the sweep runs once, before the episode.
SCATTER_REPEATS = 12


def sweep_render_scatter(
    pose: Any, render_at: Callable[[Any], Any], clip: Any, repeats: int = SCATTER_REPEATS
) -> List[float]:
    """The received RMS at **one** pose, rendered ``repeats`` times.

    This is the measurement `detour-1` and `detour-2` both wanted and neither had. The
    sweep above samples 16 poses at *different* distances, so its spread is the distance
    gradient — the signal. Re-rendering a *fixed* pose holds distance, geometry and clip
    constant, so everything left is the ray-traced renderer disagreeing with itself, which
    is the quantity a "did it get louder?" test has to clear.

    Same path as `sweep_anomaly_rms` deliberately: measured through `render_through_ir`,
    because a threshold derived on the IR's own energy and applied to a received signal is
    the silent unit error that file already warns about.
    """
    n = int(repeats)
    if n < 2:
        raise CalibrationError(
            "render scatter needs at least 2 repeats to have a spread at all, got "
            "{}".format(n)
        )
    return [rms(render_through_ir(render_at(pose), clip)) for _ in range(n)]


def render_scatter_of(samples: Sequence[float]) -> float:
    """Sample standard deviation of repeats at one pose. The noise floor of a comparison.

    Sample (``n - 1``) rather than population, because these are a handful of draws from
    the renderer's distribution and not the whole of it. With `SCATTER_REPEATS = 3` the
    difference is a factor of 1.22 — not decorative at this n.

    Zero is a legitimate return and is **not** special-cased: a renderer that agreed with
    itself exactly across repeats would be a finding, and burying it under a floor would
    hide it. The caller decides what a zero threshold means.
    """
    values = [float(v) for v in samples]
    if len(values) < 2:
        raise CalibrationError(
            "render scatter needs at least 2 samples, got {}".format(len(values))
        )
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear-interpolated percentile. numpy would do, but this stays arithmetic.

    ``np.percentile`` is one import away and identical; keeping it explicit means the
    gate number can be re-derived by hand from the printed distribution, which is what
    made ticket 16's box measurements re-usable by three later tickets.
    """
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise CalibrationError("no anomaly samples — the sweep rendered nothing")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def calibrate_onset(
    bed_rms: float,
    anomaly_rms: Sequence[float],
    *,
    global_volume: float = 1.0,
    low_percentile: float = ANOMALY_LOW_PERCENTILE,
    min_separation_db: float = MIN_SEPARATION_DB,
    scatter_samples: Sequence[float] = (),
    profile: Sequence[Tuple[float, float]] = (),
) -> CalibrationResult:
    """Place ``onset_rms`` strictly between the bed and the anomaly, or fail the gate.

    The placement is the **geometric** mean of the two, not the arithmetic one: these
    are signal levels, the comparison that matters is a ratio, and the arithmetic
    midpoint of 0.001 and 0.1 sits 34 dB above the bed and 6 dB below the anomaly rather
    than half way between them. Geometrically it is equidistant in dB from both, so the
    same margin protects against the bed drifting up and the anomaly arriving quieter.

    Raises ``CalibrationError`` when the distributions do not separate, and the message
    names ``globalVolume`` because that is the correction §2.3 allows. Deliberately
    *not* a returned ``passed=False`` result: a caller who can carry on past a failed
    gate is a caller who will.

    ``scatter_samples`` are repeats at ONE pose (`sweep_render_scatter`), and they do not
    touch the threshold — they ride along because the episode's own record is where a
    per-episode noise estimate belongs, and because the climb downstream needs it. Empty
    is allowed and leaves ``render_scatter`` None, which every consumer must read as "not
    measured" rather than "zero".

    ``profile`` is the same for the distance axis: ``(distance, rms)`` for each swept
    pose, carried through untouched so the episode's record holds the curve rather than
    four percentiles of it. It does not enter the threshold either. Empty is allowed, and
    means the caller did not measure the distances — a record written before this existed
    reads as absent rather than as a flat field.
    """
    bed = float(bed_rms)
    samples = [float(v) for v in anomaly_rms]
    if not samples:
        raise CalibrationError(
            "the calibration sweep produced no samples, so there is no distribution to "
            "place a threshold in"
        )
    low = _percentile(samples, low_percentile)
    if bed <= 0.0:
        raise CalibrationError(
            "bed level is {:.6g}. With no bed the pre-onset signal is silence, §3.1's "
            "first invariant has nothing to assert, and there is no lower distribution "
            "to separate from — set AudioConfig.bed_rms above zero".format(bed)
        )
    if low <= bed:
        raise CalibrationError(
            "the distributions OVERLAP: the anomaly's {:.0f}th percentile is {:.6g}, at "
            "or below the bed level {:.6g} over {} poses (anomaly min {:.6g}, max "
            "{:.6g}). §2.3's correction is globalVolume — currently {:.3g}, measured 1.0 "
            "on our branch — never a hand-nudged threshold.".format(
                float(low_percentile), low, bed, len(samples),
                min(samples), max(samples), float(global_volume),
            )
        )
    separation_db = 20.0 * math.log10(low / bed)
    if separation_db < float(min_separation_db):
        raise CalibrationError(
            "separation is {:.2f} dB, below the {:.2f} dB the gate requires (bed "
            "{:.6g}, anomaly p{:.0f} {:.6g}, {} poses). There is room for a threshold "
            "but not for a margin, so a small drift in either distribution re-crosses "
            "it. Raise globalVolume (currently {:.3g}).".format(
                separation_db, float(min_separation_db), bed,
                float(low_percentile), low, len(samples), float(global_volume),
            )
        )
    scatter = [float(v) for v in scatter_samples]
    ordered = sorted(samples)
    return CalibrationResult(
        render_scatter=render_scatter_of(scatter) if len(scatter) >= 2 else None,
        scatter_repeats=len(scatter),
        profile=tuple((float(d), float(r)) for d, r in profile),
        onset_rms=math.sqrt(bed * low),
        bed_rms=bed,
        anomaly_low=low,
        anomaly_median=_percentile(ordered, 50.0),
        anomaly_min=ordered[0],
        anomaly_max=ordered[-1],
        separation_db=separation_db,
        n_poses=len(samples),
        global_volume=float(global_volume),
    )


def band_poses(
    near_far: Tuple[float, float], n_poses: int
) -> List[float]:
    """The geodesic distances the sweep should sample, log-spaced across the band.

    Log rather than linear because level falls roughly with the log of distance, so
    linear spacing crowds the samples into the quiet far end and under-samples exactly
    the near band where the threshold has to sit.

    Returns distances, not poses: turning a distance into a navigable pose needs the
    navmesh, which lives behind ``sim/`` and reaches ticket 25's runner, not this layer
    (ADR-0013). This is the half of the sweep plan that can be decided without a
    simulator, and it is here so that it is decided once.
    """
    near, far = (float(near_far[0]), float(near_far[1]))
    count = int(n_poses)
    if not (0.0 < near < far):
        raise ValueError(
            "audible band must be (near, far) with 0 < near < far, got {}".format(near_far)
        )
    if count < 2:
        raise ValueError("a distribution needs at least 2 poses, got {}".format(n_poses))
    step = (math.log(far) - math.log(near)) / (count - 1)
    return [math.exp(math.log(near) + step * i) for i in range(count)]
