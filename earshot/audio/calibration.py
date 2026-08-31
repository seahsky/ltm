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

**Which DOMAIN the sweep measures in is the recurring hazard here, and ADR-0019 moved it
for the second time.** A threshold derived on one signal and applied to another is a
silent unit error: it shows up as a threshold that never fires, or as one that fires on
the bed. Since the split readout the agent reads ``hop`` samples per step, so
``sweep_cue_rms`` is what places ``onset_rms`` and ``sweep_anomaly_rms`` is kept only as
the pre-ADR-0017 control. The two functions that used to take an ``Optional[hop]`` and
switch domain on it were SPLIT into four named functions rather than extended to a third
domain — an Optional that switches measurement domain is exactly the error this module
exists to refuse, and a control arm is only a control while its body is unchanged.

Everything is injected — the sweep takes a ``render_at`` callable and a list of poses —
so the arithmetic is Mac-testable and only the poses come from the box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from earshot.audio.clips import as_binaural, render_through_ir, rms
from earshot.audio.tail import (
    advance_tail,
    clip_readout,
    cue_crest,
    cue_level,
    cue_min_ratio,
    cue_readout,
    open_tail,
    steady_state_cue_rms,
)

__all__ = [
    "CalibrationError",
    "CalibrationResult",
    "CueSweepSample",
    "LoopScatter",
    "ANOMALY_LOW_PERCENTILE",
    "MIN_SEPARATION_DB",
    "SCATTER_REPEATS",
    "CUE_PHASE_AGGREGATION",
    "sweep_anomaly_rms",
    "sweep_cue_rms",
    "sweep_render_scatter",
    "sweep_loop_scatter",
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

# How ``sweep_cue_rms`` turns one pose's ``phase_folds`` cue readings into the single
# level the threshold is placed against. Written onto the record so a reader knows which
# aggregation produced ``onset_rms`` rather than inferring it.
#
# A string constant rather than an Enum arm: ADR-0008 types behaviour on frozen configs
# and makes arms Enums, but there is exactly one value here, and an unexercised second
# arm would be dead code that reads as a choice an operator can make.
CUE_PHASE_AGGREGATION = "quadratic_mean_over_loop_phases"


@dataclass(frozen=True)
class CueSweepSample:
    """One pose's cue measurement: the level the threshold uses, and the loop behind it.

    A frozen pair rather than a positional tuple, and the reason is the whole risk of the
    split readout: ``level`` and ``phases`` are two float sequences of the same shape
    family, and a caller that swaps a level for a phase (or a cue level for a clip one)
    gets a plausible number and a silently moved threshold. ``level`` is computed once,
    inside the sweep, by ``tail.cue_level`` -- so no call site can aggregate differently.
    """

    level: float
    phases: Tuple[float, ...]


@dataclass(frozen=True)
class LoopScatter:
    """The two readouts' held-pose samples, off the SAME folds. Same reason as above.

    ``cue`` is what ``climb_eps`` reads because the cue is what ``is_rising`` compares.
    ``clip`` is the ADR-0017 arm and costs nothing -- same folds, second readout -- which
    is what makes it affordable to keep as a control.
    """

    cue: Tuple[float, ...]
    clip: Tuple[float, ...]


@dataclass(frozen=True)
class CalibrationResult:
    """The derived threshold and the evidence for it. Lands on the audit record (§5.2).

    ``separation_db`` is the gate number — the thing to paste back, in the pattern of
    ticket 13's EER 0.00 and the CapRL gate's separation. ``passed`` is always ``True``
    on a returned result, because ``calibrate_onset`` raises otherwise; it exists so the
    serialised record says so explicitly rather than by absence.

    **THREE named scatter arms, and ``climb_eps`` reads the cue one.** They are three
    estimates of one sentence -- "the spread of the reading the climb compares" -- and
    that sentence stayed true across ADR-0019 while the reading itself changed length,
    which is exactly what would have let the domain move under a stable name. So
    ``render_scatter`` is RENAMED to ``cue_render_scatter`` rather than redefined in
    place: every number on disk under the old key is the clip-loop estimate, and a reader
    differencing across the change would be subtracting two domains under one label.

    - ``cue_render_scatter`` -- successive CUE readouts of the accumulator, each folding
      a fresh render. What ``is_rising`` compares, therefore what ``climb_eps`` reads.
    - ``clip_render_scatter`` -- the ADR-0017 arm, off the SAME folds, so it is free.
    - ``single_render_scatter`` -- the pre-ADR-0017 arm, independent whole-clip renders,
      the only one that costs extra renders. Every historic ``eps`` on disk is this
      number, so without it a reader cannot price the change against a past run.

    **THE PRE-REGISTERED PREDICTION, and a Mac measurement that already doubts it.** The
    model behind ``climb_eps`` says the cue readout averages ``cue_tail_steps`` renders
    where the clip readout averages ``clip_tail_steps``, so the three should order
    ``single_render > cue_render > clip_render``. The box is what settles it. On this Mac,
    against three synthetic renderer-noise models at the box's numbers (5 s white-noise
    clip, ``L = 72300`` at RT60 0.8 s, 200 repeats), the ordering came out
    ``cue >= single > clip`` under two of three: ``single/cue`` 0.929, 1.128 and 0.630,
    against ``single/clip`` 2.055, 2.132 and 2.109. The mechanism is measurable -- 100.0%
    of that IR's energy sits inside the first hop, so the cue window holds essentially
    ONE render and its spread is one render's spread, not an average of three. **If the
    box reproduces that, the averaging model behind ``climb_eps`` is wrong for a second
    time and the epsilon is again the wrong size.**

    **AND THE CUE ARM CARRIES THE LOOP PHASE, WHICH THE CLIP ARM DOES NOT.** Measured at
    the box's numbers with a perfectly DETERMINISTIC renderer, so the only moving part is
    the clip's own envelope: ``cue`` SD/level is 2.93e-03 for 5 s of white noise and
    **2.33** for a 0.6 s transient, against a ``clip`` SD/level of 1e-16 in both. For a
    bursty clip -- which most of ESC-50 is -- ``cue_render_scatter`` is not renderer
    non-determinism at all, it is ``cue_crest`` in another costume, and ``climb_eps``
    would hand ``is_rising`` a floor of about twice the level itself. ``cue_phase_crest``
    and ``cue_phase_min_ratio`` are on this record so that case is identifiable rather
    than merely suffered; nothing is gated on them here, deliberately.
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
    cue_render_scatter: Optional[float] = None
    cue_scatter_repeats: int = 0
    # The ADR-0017 arm, read off the same folds as the cue arm and therefore free. `None`
    # means the arm was not run, never that the two agreed.
    clip_render_scatter: Optional[float] = None
    clip_scatter_repeats: int = 0
    # The pre-ADR-0017 arm, and the only thing that makes the change priceable against
    # every `eps` already on disk: the same pose, measured the old way.
    single_render_scatter: Optional[float] = None
    single_render_repeats: int = 0
    # The loop, summarised. `cue_phase_folds` is how many distinct readings one held pose
    # cycles through; crest and min_ratio are the MEDIAN over the swept poses. The crest
    # is a property of the clip's envelope against the hop and is very nearly
    # pose-independent (measured identical to four decimal places across IRs at one
    # configuration), so the median is a robust summary -- and a large spread across
    # poses would itself be a finding worth a later look.
    cue_phase_folds: int = 0
    cue_phase_crest: Optional[float] = None
    cue_phase_min_ratio: Optional[float] = None
    cue_phase_aggregation: Optional[str] = None
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
            "cue_render_scatter": (
                None
                if self.cue_render_scatter is None
                else float(self.cue_render_scatter)
            ),
            "cue_scatter_repeats": int(self.cue_scatter_repeats),
            "clip_render_scatter": (
                None
                if self.clip_render_scatter is None
                else float(self.clip_render_scatter)
            ),
            "clip_scatter_repeats": int(self.clip_scatter_repeats),
            "single_render_scatter": (
                None
                if self.single_render_scatter is None
                else float(self.single_render_scatter)
            ),
            "single_render_repeats": int(self.single_render_repeats),
            "cue_phase_folds": int(self.cue_phase_folds),
            "cue_phase_crest": (
                None if self.cue_phase_crest is None else float(self.cue_phase_crest)
            ),
            "cue_phase_min_ratio": (
                None
                if self.cue_phase_min_ratio is None
                else float(self.cue_phase_min_ratio)
            ),
            "cue_phase_aggregation": self.cue_phase_aggregation,
        }


def sweep_anomaly_rms(
    poses: Sequence[Any],
    render_at: Callable[[Any], Any],
    clip: Any,
) -> List[float]:
    """The anomaly's whole-clip RMS at each pose. One live render per pose, nothing else.

    ``render_at(pose)`` returns that pose's raw binaural IR — in the runner, seat the
    agent and call ``guarded_observe``. The bed is **not** mixed in: this measures the
    anomaly's own distribution, and the bed is the other side of the comparison.

    Measured through ``clips.render_through_ir``, not on the IR's own energy, because
    the threshold is applied to what the agent hears. An IR and a received signal differ
    by the clip's own level, so calibrating on one and thresholding on the other is a
    silent unit error — the kind that shows up as a threshold that never fires.

    **This is now purely the pre-ADR-0017 CONTROL ARM and has no production caller.**
    ``sweep_cue_rms`` is what places ``onset_rms``. The ``Optional[hop]`` this function
    used to carry was SPLIT out rather than extended to a third domain, because an
    Optional that switches measurement domain is precisely the silent unit error the
    paragraph above warns about, and splitting is what keeps this arm byte-identical to
    what every historic number was measured with.
    """
    return [rms(render_through_ir(render_at(pose), clip)) for pose in poses]


def sweep_cue_rms(
    poses: Sequence[Any],
    render_at: Callable[[Any], Any],
    clip: Any,
    *,
    hop: int,
) -> List[CueSweepSample]:
    """The CUE level at each pose, and the loop phases behind it. THIS places ``onset_rms``.

    One live render per pose -- ``render_at`` is called exactly ``len(poses)`` times, the
    same bill ``sweep_anomaly_rms`` pays -- and the IR is then reused by
    ``tail.steady_state_cue_rms``, which folds it ``clip_tail_steps + 1 + phase_folds``
    times (13 at the box's numbers against ``steady_state_render``'s 8). Only numpy time
    moves; the sweep is still dominated by its 16 habitat renders.

    **The level is the quadratic mean over the loop's phases, and ``onset_rms`` does not
    move because of it.** The ``phase_folds`` cue windows are disjoint, consecutive and
    tile the settled period an integer number of times, so their quadratic mean EQUALS
    the clip readout's RMS -- measured at ratio 1.000000000000 in all four configurations
    this tree ships. The sweep changed domain and the threshold's LEVEL stayed put, which
    is what makes ADR-0019 reviewable as one number that must not move beside several
    that must.

    The phases ride along untouched, the way ``profile`` does: they do not enter the
    threshold, the gate or any refusal. They are what ``cue_phase_crest`` and
    ``cue_phase_min_ratio`` are computed from, and for a bursty clip they are the
    difference between a source that is audible on one fold in five and a source that is
    audible.
    """
    stride = int(hop)
    samples: List[CueSweepSample] = []
    for pose in poses:
        phases = steady_state_cue_rms(render_at(pose), clip, hop=stride)
        samples.append(CueSweepSample(level=cue_level(phases), phases=tuple(phases)))
    return samples


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
    pose: Any,
    render_at: Callable[[Any], Any],
    clip: Any,
    repeats: int = SCATTER_REPEATS,
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

    **This is the pre-ADR-0017 estimator, byte-for-byte, and it has no production
    consumer of its own.** ``sweep_loop_scatter`` is what the runner reads. The
    ``Optional[hop]`` that used to switch this function between two measurement domains
    was SPLIT out at ADR-0019 rather than extended to a third, for the reason the module
    docstring gives: an Optional that switches domain is a silent unit error waiting to
    happen, and a control arm is only a control while it is unchanged. Every ``eps`` on
    disk before ADR-0017 is this number.

    **One residual, named rather than fixed.** This sweep measures the SOURCE alone, as it
    always has, while the runner's reading is `mix_bed(readout, bed)`. Before ADR-0017 the
    render had a fixed alignment, so the cross-term with the (fixed) bed was a constant and
    dropped out; the accumulator's readout is a different rotation of the period every
    step, so that cross-term now jitters. That term scales as ``1/sqrt(n)`` in the window
    length, so reading it off the CUE window instead of the clip window makes it
    ``sqrt(N/hop)`` = ``sqrt(5)`` larger at the shipped defaults. Still small while the
    source is well above the bed; it is the near-threshold regime, exactly where
    ``onset_rms`` sits, where it matters most, and it is unmeasured after the split. It is
    not folded in here because `sweep_anomaly_rms` does not mix the bed either and
    `onset_rms` is placed against both distributions; moving one and not the other is the
    unit error this file exists to warn about.
    """
    n = int(repeats)
    if n < 2:
        raise CalibrationError(
            "render scatter needs at least 2 repeats to have a spread at all, got "
            "{}".format(n)
        )
    return [rms(render_through_ir(render_at(pose), clip)) for _ in range(n)]


def sweep_loop_scatter(
    pose: Any,
    render_at: Callable[[Any], Any],
    clip: Any,
    repeats: int = SCATTER_REPEATS,
    *,
    hop: int,
) -> LoopScatter:
    """Both readouts' spread at **one** pose, off the SAME folds. ``climb_eps`` reads ``.cue``.

    The loop simulated at a held pose: one accumulator, a FRESH ``render_at(pose)`` folded
    on every step, settled ``clip_tail_steps`` folds -- the LONGER of the two settles, so
    both readouts are settled -- and then ``repeats`` consecutive folds with both readouts
    taken off each one.

    **The cost is exactly what the previous loop arm cost.** ``render_at`` is called
    ``clip_tail_steps + repeats`` times, 7 + 12 = 19 at the box's numbers, and the clip
    arm is free: same folds, second readout. That is the whole reason it is affordable to
    keep the ADR-0017 estimator as a control beside the cue one.

    **`tail.steady_state_render` was the wrong instrument here and the error was 2.4x**,
    which is the history that put a loop arm in this file at all. That function folds ONE
    render ``clip_tail_steps + 1`` times, so its spread across repeats is one render's
    spread, while the runner's loop folds independent renders into every reading. The
    clip arm below is that measurement, kept.

    **What the CUE arm actually contains, measured, because it is not only the
    renderer.** Two terms, and at the box's numbers the second can be the larger:

    - the renderer's own disagreement, which the clip arm averages down and the cue arm
      does not. Measured on this Mac at the box's numbers over 200 repeats with three
      synthetic noise models, ``cue/clip`` SD ratios 2.213, 1.890 and 3.350 -- but
      ``single/cue`` 0.929, 1.128 and 0.630, i.e. the cue arm is NOT below the
      pre-ADR-0017 arm, because 100.0% of a 72300-sample RT60-0.8 s IR's energy sits
      inside one hop and the cue window therefore holds essentially one render;
    - **the LOOP PHASE**, which the clip arm cannot see at all. With a perfectly
      deterministic renderer at the box's numbers the clip arm's SD/level is 1e-16 and
      the cue arm's is 2.93e-03 for 5 s of white noise and **2.33** for a 0.6 s transient.
      For a bursty clip this term swamps the renderer entirely.

    Both are honest answers to "what is the spread of the reading the climb compares",
    because ``is_rising`` really does compare readings that cycle with the loop. Neither
    is what the name ``render`` suggests. ``CalibrationResult`` carries ``cue_phase_crest``
    and ``cue_phase_min_ratio`` beside the three scatters so the two terms are separable
    after the fact; nothing is corrected here, deliberately, because a correction would be
    a policy change riding in an audio commit.

    The samples are consecutive and therefore autocorrelated by construction -- the clip
    arm's share ``(N - hop)/N`` of their samples exactly (measured lag-1 autocorrelation
    +0.806 against the cue arm's -0.068 on one noise model). That is a property of the
    quantity, not a defect of the estimator.
    """
    n = int(repeats)
    if n < 2:
        raise CalibrationError(
            "loop scatter needs at least 2 repeats to have a spread at all, got "
            "{}".format(n)
        )
    stride = int(hop)
    window = int(len(clip))
    first = as_binaural(render_at(pose))
    state = open_tail(
        window=window, hop=stride, headroom=max(0, int(first.shape[1]) - 1)
    )
    state = advance_tail(state, ir=first, clip=clip, sounding=True)
    # `TailState.clip_tail_steps` is readable now that a fold has happened, and reading it
    # rather than re-deriving the expression is what keeps a wider IR on a more
    # reverberant scene lengthening the settle instead of silently under-settling it. It
    # is the CLIP tail on purpose: it is the longer of the two, so settling on it leaves
    # both readouts settled, where settling on `cue_tail_steps` would leave the clip
    # readout still ramping and its arm measuring the ramp.
    for _ in range(max(0, state.clip_tail_steps - 1)):
        state = advance_tail(state, ir=render_at(pose), clip=clip, sounding=True)
    cue_samples: List[float] = []
    clip_samples: List[float] = []
    for _ in range(n):
        state = advance_tail(state, ir=render_at(pose), clip=clip, sounding=True)
        cue_samples.append(rms(cue_readout(state)))
        clip_samples.append(rms(clip_readout(state)))
    return LoopScatter(cue=tuple(cue_samples), clip=tuple(clip_samples))


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
    cue_scatter_samples: Sequence[float] = (),
    clip_scatter_samples: Sequence[float] = (),
    single_render_samples: Sequence[float] = (),
    profile: Sequence[Tuple[float, float]] = (),
    cue_phases: Sequence[Sequence[float]] = (),
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

    ``anomaly_rms`` takes the per-pose LEVELS, which since ADR-0019 are
    ``sweep_cue_rms``'s ``CueSweepSample.level``. The threshold arithmetic above is
    UNCHANGED by that split, and the identity in ``tail.cue_level`` is why: the quadratic
    mean of a pose's loop phases equals its clip-readout RMS exactly, so ``onset_rms``
    lands where it landed before.

    ``cue_scatter_samples`` / ``clip_scatter_samples`` are repeats at ONE pose
    (`sweep_loop_scatter`), and ``single_render_samples`` are the same pose measured the
    pre-ADR-0017 way (`sweep_render_scatter`). None of the three touches the threshold —
    they ride along because the episode's own record is where a per-episode noise estimate
    belongs, and because the climb downstream reads the cue one. Empty leaves the
    matching field ``None``, which every consumer must read as "not measured" rather than
    "zero"; the arm was not run, which is a different fact from the arms agreeing.

    Three arms rather than one because ``climb_eps``' input changed domain twice and the
    field name would otherwise have stayed still under it. Every number on disk under the
    legacy ``render_scatter`` key is a clip-loop estimate if the record also carries
    ``single_render_scatter``, and a whole-clip estimate if it does not.

    ``cue_phases`` is one tuple per pose, and it rides along exactly the way ``profile``
    does: it does not touch the threshold, the gate or any refusal. It fills
    ``cue_phase_folds`` from the first pose and ``cue_phase_crest`` /
    ``cue_phase_min_ratio`` from the MEDIAN over poses, and it stamps
    ``cue_phase_aggregation`` so the record says which aggregation produced the threshold
    rather than leaving it to be inferred. Empty leaves those fields None/0, which reads
    as not measured.

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
    cue_scatter = [float(v) for v in cue_scatter_samples]
    clip_scatter = [float(v) for v in clip_scatter_samples]
    single = [float(v) for v in single_render_samples]
    per_pose = [tuple(float(v) for v in phases) for phases in cue_phases]
    per_pose = [phases for phases in per_pose if phases]
    ordered = sorted(samples)
    return CalibrationResult(
        cue_render_scatter=(
            render_scatter_of(cue_scatter) if len(cue_scatter) >= 2 else None
        ),
        cue_scatter_repeats=len(cue_scatter),
        clip_render_scatter=(
            render_scatter_of(clip_scatter) if len(clip_scatter) >= 2 else None
        ),
        clip_scatter_repeats=len(clip_scatter),
        single_render_scatter=(
            render_scatter_of(single) if len(single) >= 2 else None
        ),
        single_render_repeats=len(single),
        cue_phase_folds=len(per_pose[0]) if per_pose else 0,
        cue_phase_crest=(
            _percentile([cue_crest(phases) for phases in per_pose], 50.0)
            if per_pose
            else None
        ),
        cue_phase_min_ratio=(
            _percentile([cue_min_ratio(phases) for phases in per_pose], 50.0)
            if per_pose
            else None
        ),
        cue_phase_aggregation=CUE_PHASE_AGGREGATION if per_pose else None,
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
