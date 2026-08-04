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
from typing import Any, Callable, List, Sequence, Tuple

from earshot.audio.clips import render_through_ir, rms

__all__ = [
    "CalibrationError",
    "CalibrationResult",
    "ANOMALY_LOW_PERCENTILE",
    "MIN_SEPARATION_DB",
    "sweep_anomaly_rms",
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

    def as_dict(self) -> dict:
        return {
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
    ordered = sorted(samples)
    return CalibrationResult(
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
