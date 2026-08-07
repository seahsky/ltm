"""Why the detour ended — read off the per-step audit records.

    python -m earshot.tools.detour_report runs/yield-1/ziup5kvtCCR

**The question this exists to answer is one the yield-1 sweep raised and could not
settle.** Twelve of that run's twenty episodes in ``ziup5kvtCCR`` spent *exactly*
``investigate_max_steps`` and abandoned the investigation, every one of them resuming at
``onset_step + 120``. Two diagnoses fit that identically and imply opposite fixes:

- **short of steps** — the climb was walking at the source and the budget cut it off.
  Then the budget is the bug, and 120 (provenance ``fake``, argued from one synthetic
  5.4 m source) is a number chosen against the wrong scale.
- **the climb wandered** — the agent moved, and did not get closer. Then a bigger budget
  buys a longer wander at the cost of the primary find, and the climb is the bug.

Nothing separated them, because the record held ``measured_rms`` (an energy proxy for
where the agent was) and ``displacement_m`` (that it moved at all) but never *where*.
``StepRecord.position`` closed that, and this reads the pairing back out.

**The abandoned and the reached episodes are the two arms**, and they are reported side
by side deliberately: this map's rule is that a claim about why something failed needs
the arm where it did not. "The abandoned detours walked 18 m to close 3 m" means nothing
until the reached ones are shown walking 6 m to close 5 m.

``aggregate()`` is pure, so the arithmetic is Mac-testable against injected records while
the runs that feed it need a GPU — the same split ``yield_report`` is built on.

**No thresholds, and no verdict.** Every number here is a measurement; there is no
invented constant deciding "converging" from "wandering", because the two are separated
by an order of magnitude in ``walked_per_metre_closed`` and a classifier would only hide
which. Box results on this map have been decidable exactly when they printed what they
measured (ADR-0014).

The plateau half — why the terminal approach stalls
---------------------------------------------------

``detour-1`` left one question open and named it the next lever: seven of twelve
abandoned detours settle in a tight 2.06–2.76 m band, and nothing said why. Reading the
carried rule (``agent/controller.realizable_investigate_step``) narrows it to one line::

    rising = current > previous + eps
    if visual_confirm and not rising:  STOP
    if rising:                         FORWARD
    else:                              TURN

**No branch advances a plateaued agent.** Not rising and not confirmed is a turn, and
``visual_confirm`` is the ``OracleDetector`` at ``oracle_radius_m`` — 1.0 m, *geodesic*,
carried over from Find-SR's primary ring. ``detour-1``'s empty gap, 0.78 m to 2.06 m,
straddles exactly that radius.

So the stall has two candidate mechanisms and they imply opposite fixes:

- **the plateau is real** — the binaural gradient genuinely flattens out around 2 m, no
  forward would have raised it, and the lever is the arrival criterion.
- **the plateau is spurious** — the gradient is still climbing, but ``rising`` is a
  *single-step* comparison and ``detour-1`` measured the live render moving 24% between
  identical runs. One unlucky reading on a rising gradient sends the agent into a turn,
  and the lever is the estimator, not the acoustics.

``measured_rms`` is the same value the controller fed into ``energy_history``, so
``rising`` is **exactly reconstructible** from the record and the plateau windows below
are the controller's own predicate rather than a band someone chose. Within each window
this reports the slope of ``measured_rms`` against distance-to-source, its residual
scatter, and how far the agent actually travelled while plateaued.

**A window the agent barely moved through cannot answer the question**, and that is
reported rather than regressed: with the agent turning in place the distance series is
near-constant, the regression is ill-conditioned, and a slope computed from it would be
noise wearing a number's clothes. Those windows are counted as ``static`` — which is
itself a third finding, because an agent that never translated never tested the field.

Still no verdict here. This prints the slope distribution and the reader decides.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import pathlib
import statistics
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from earshot.agent.controller import (
    ACT_FORWARD,
    ACT_STOP,
    ACT_TURN_LEFT,
    ACT_TURN_RIGHT,
    realizable_investigate_step,
)
from earshot.report.artifacts import ENV_REPORT_NAME, episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit, FunnelStage, StepRecord

__all__ = [
    "ABANDONED",
    "REACHED",
    "NO_DETOUR",
    "RISING_EPS",
    "DEFAULT_MIN_SPAN_M",
    "rising_flags",
    "rule_action",
    "plateau_windows",
    "fit_slope",
    "trace_one",
    "aggregate",
    "format_report",
    "load_traces",
    "main",
]

ABANDONED = "abandoned"  # entered INVESTIGATE, spent the budget, never reached the source
REACHED = "reached"      # got there; the control arm for the above
NO_DETOUR = "no_detour"  # onset never fired, or the episode ended before it could

# READ from the carried rule's own signature, never re-spelled here. Two copies of one
# constant is the drift trap this tree has already paid for once: a replay that
# reconstructs `rising` with a different epsilon than the agent used is not a replay, it
# is a second controller that happens to resemble the first. `test_detour_report` holds
# this against the signature so a change there fails loudly rather than silently
# re-defining every plateau ever measured.
RISING_EPS = float(
    inspect.signature(realizable_investigate_step).parameters["eps"].default
)

# provenance: fake — the distance a plateau window must span before its slope is worth
# fitting, in metres. NOT a verdict threshold: it decides whether a regression is
# *defined*, not whether a plateau is real. A turning agent holds distance almost
# constant, so the denominator of the slope goes to zero and the fit explodes; below this
# span the window is reported as `static` and its slope withheld. Both counts are printed,
# so nothing is dropped silently, and `--min-span` moves it.
DEFAULT_MIN_SPAN_M = 0.25

# Where the histogram's buckets fall, in steps. The 1 bucket is its own because a
# one-step window is the single-step test flickering rather than a stall, and 20+ is its
# own because that is long enough for the agent to have translated somewhere had it
# wanted to — `RISING_WINDOW` is 5, so a window four times that is not the estimator.
LENGTH_BUCKETS = (1, 2, 5, 10, 20)

# At or above this, a window is "long": counted separately so the share of plateaued
# steps living in genuine stalls is readable next to the share living in dropouts.
LONG_WINDOW_STEPS = 10


def _length_histogram(windows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, int]]:
    """Window lengths bucketed, as ``(label, count)`` in ascending order.

    Buckets rather than raw lengths because the question is which of two mechanisms owns
    the mass, not what the exact distribution is. A list of pairs rather than a dict so
    the order survives JSON without the reader having to re-sort it.
    """
    edges = list(LENGTH_BUCKETS)
    counts = [0] * len(edges)
    for window in windows:
        steps = int(window.get("n_steps") or 0)
        for index in reversed(range(len(edges))):
            if steps >= edges[index]:
                counts[index] += 1
                break
    labels: List[str] = []
    for index, edge in enumerate(edges):
        upper = edges[index + 1] if index + 1 < len(edges) else None
        if upper is None:
            labels.append("{}+".format(edge))
        elif upper == edge + 1:
            labels.append(str(edge))
        else:
            labels.append("{}-{}".format(edge, upper - 1))
    return list(zip(labels, counts))


def _median(values: Sequence[float]) -> Optional[float]:
    """Median, or None for an empty sample. Median rather than mean because n is ~10 and
    one episode that walked into a corner should not move the number the fix is chosen
    against."""
    clean = [float(v) for v in values if v is not None]
    return statistics.median(clean) if clean else None


def rising_flags(rms: Sequence[float], *, eps: float = RISING_EPS) -> List[bool]:
    """``realizable_investigate_step``'s own ``rising``, recomputed per step. Pure.

    Carried verbatim from the rule::

        rising = previous is None or current > previous + eps

    **Computed over the WHOLE episode, then sliced to the detour** — never over the
    window alone. The controller's ``energy_history`` has been accumulating since step 0,
    so at the first detour step ``previous`` is the reading from the step *before* the
    onset, and a window-local recomputation would call that step rising by default and
    invent a forward the agent never took. ``ENERGY_HISTORY`` is 8 and the rule reads two
    entries, so the runner's trimming cannot affect this.
    """
    flags: List[bool] = []
    previous: Optional[float] = None
    for value in rms:
        current = float(value)
        flags.append(previous is None or current > previous + float(eps))
        previous = current
    return flags


def rule_action(rising: bool, lateral_sign: Optional[int]) -> str:
    """What the carried rule answers, given ``rising`` and the lateral sign. Pure.

    The rule's third input, ``visual_confirm``, is **not recorded per step** — and for
    the arm this tool exists to explain it does not need to be. ``visual_confirm`` can
    only change the answer where ``rising`` is false, and there it produces a STOP; an
    abandoned episode by definition never STOPped, so the confirm was false at every one
    of its steps and the answer is determined by these two alone.

    That is why the check this feeds is exact on the abandoned arm and only a lower bound
    on the reached one, whose final step may legitimately differ.
    """
    if rising:
        return ACT_FORWARD
    if lateral_sign is not None and int(lateral_sign) > 0:
        return ACT_TURN_RIGHT
    return ACT_TURN_LEFT  # negative, zero, and absent all scan left (the rule's default)


def plateau_windows(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """Maximal runs of consecutive not-rising steps, as ``[start, end)`` index pairs. Pure.

    This is the plateau *as the controller sees it*: the stretch over which the rule had
    stopped answering FORWARD. A window of one step is still a window — the agent turned
    once and recovered — and it is reported rather than filtered, because the difference
    between one long stall and forty brief ones is the difference between the two
    diagnoses this tool exists to separate.
    """
    windows: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, rising in enumerate(flags):
        if not rising and start is None:
            start = index
        elif rising and start is not None:
            windows.append((start, index))
            start = None
    if start is not None:
        windows.append((start, len(flags)))
    return windows


def fit_slope(
    distances: Sequence[float], rms: Sequence[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Least-squares slope of ``rms`` on ``distance``, and the residual SD. Pure.

    Returns ``(slope, residual_sd)`` in RMS-units per metre. **A negative slope is a live
    gradient** — louder as the distance falls — and that sign convention is the whole
    reading: flat means the cue is exhausted, clearly negative means the cue was still
    there and the single-step test missed it.

    ``(None, None)`` when the fit is not defined: fewer than two points, or a distance
    series with no spread at all. The residual SD is ``None`` at exactly two points,
    where a line through both leaves nothing to scatter — reported as absent rather than
    as a scatter of zero, which would read as a noiseless measurement.

    Written out rather than taken from ``statistics``: ``linear_regression`` and
    ``covariance`` are 3.10, and this suite runs on 3.9 (ADR-0014).
    """
    xs = [float(x) for x in distances]
    ys = [float(y) for y in rms]
    n = len(xs)
    if n < 2 or n != len(ys):
        return None, None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None, None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    if n == 2:
        return slope, None
    intercept = mean_y - slope * mean_x
    residuals = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return slope, math.sqrt(residuals / (n - 2))


def _plateau_rows(
    steps: Sequence[StepRecord],
    distances: Sequence[Optional[float]],
    flags: Sequence[bool],
    *,
    min_span_m: float,
) -> List[Dict[str, Any]]:
    """One row per plateau window inside the detour. Pure."""
    rows: List[Dict[str, Any]] = []
    for start, end in plateau_windows(flags):
        window = list(range(start, end))
        known = [i for i in window if distances[i] is not None]
        span = None
        if known:
            values = [float(distances[i]) for i in known]
            span = max(values) - min(values)
        rms = [float(steps[i].measured_rms) for i in window]
        row: Dict[str, Any] = {
            "start_step": int(steps[start].step),
            "n_steps": len(window),
            "d_start_m": distances[start],
            "d_span_m": span,
            "rms_span": (max(rms) - min(rms)) if rms else None,
            "n_forward": sum(1 for i in window if steps[i].action == ACT_FORWARD),
            # A window with no measurable travel is `static`: the agent turned in place,
            # never translated, and so never put the field to the test. Its slope is
            # withheld rather than fitted — see DEFAULT_MIN_SPAN_M.
            "static": span is None or span < float(min_span_m),
        }
        if row["static"] or len(known) < 2:
            row["slope_per_m"] = None
            row["residual_sd"] = None
        else:
            row["slope_per_m"], row["residual_sd"] = fit_slope(
                [float(distances[i]) for i in known],
                [float(steps[i].measured_rms) for i in known],
            )
        # **The rays-1 gate, in one ratio.** How much RMS the gradient delivers across
        # this window (|slope| x the distance actually travelled) against how much the
        # render scatters around the line. Well above 1 means a windowed estimator could
        # have recovered the cue from these very traces and the fix is the estimator;
        # well below 1 means the cue is buried at 500 rays and ray count becomes a lever.
        # A ratio, so it needs no constant to read.
        #
        # `None` where the scatter is absent OR exactly zero. Zero is the two-point fit
        # and the synthetic trace, never a rendered one — a ray-traced window with no
        # residual at all would itself be the finding, and reporting an unbounded ratio
        # there (or an `Infinity` that is not valid JSON) would bury it. The window's
        # slope and residual are both printed, so the reader sees the case directly.
        slope, scatter = row["slope_per_m"], row["residual_sd"]
        row["signal_to_scatter"] = (
            (abs(float(slope)) * float(span) / float(scatter))
            if slope is not None and scatter and span else None
        )
        rows.append(row)
    return rows


def trace_one(
    audit: EpisodeAudit,
    *,
    budget: Optional[int] = None,
    min_span_m: float = DEFAULT_MIN_SPAN_M,
) -> Dict[str, Any]:
    """One episode's detour, as measurements. Pure.

    The detour **window** is ``[onset_step, onset_step + budget]``, clipped to the
    episode. It is inferred rather than read, because no record marks where INVESTIGATE
    ended — and for the episodes this tool exists for, inference is exact: an abandoned
    detour ends at the budget by definition. For a reached episode the window overshoots
    into the resumed primary search, which is stated rather than corrected; the numbers
    that matter there (``d_min``, ``gap_closed_m``) are minima and are reached before the
    overshoot, and ``walked_m`` is flagged as an upper bound.
    """
    onset = audit.onset
    # Attribute access, not getattr-with-a-default: a default would turn a renamed field
    # into "no episode ever diverted", which is a diagnosis rather than an error.
    onset_step = None if onset is None else onset.onset_step
    stage = audit.funnel_stage
    if onset_step is None or stage < FunnelStage.INVESTIGATE_ENTERED:
        outcome = NO_DETOUR
    elif stage >= FunnelStage.SOURCE_REACHED:
        outcome = REACHED
    else:
        outcome = ABANDONED

    row: Dict[str, Any] = {
        "episode": int(audit.episode_index),
        "outcome": outcome,
        "onset_step": onset_step,
        "n_steps": len(audit.steps),
        "walked_is_upper_bound": outcome == REACHED,
    }
    if outcome == NO_DETOUR:
        return row

    steps = [row_ for row_ in audit.steps if row_.step >= int(onset_step)]
    if budget is not None:
        steps = [row_ for row_ in steps if row_.step <= int(onset_step) + int(budget)]
    row["detour_steps"] = len(steps)

    distances = audit.distance_to_source_history
    by_step = {r.step: d for r, d in zip(audit.steps, distances)}
    window = [by_step.get(r.step) for r in steps]
    known = [d for d in window if d is not None]
    if known:
        row["d_onset_m"] = window[0]
        row["d_min_m"] = min(known)
        row["d_end_m"] = known[-1]
        # None, not 0.0, when the start distance is unknown: a gap that could not be
        # measured and a gap of zero are different claims.
        row["gap_closed_m"] = (
            None if window[0] is None else float(window[0]) - float(min(known)))
    else:
        # Every record written before `StepRecord.position` landed. Reported as absent so
        # a pre-yield-1 run cannot read as a detour that closed nothing.
        for key in ("d_onset_m", "d_min_m", "d_end_m", "gap_closed_m"):
            row[key] = None

    walked = sum(float(r.displacement_m or 0.0) for r in steps)
    row["walked_m"] = walked
    closed = row.get("gap_closed_m")
    # The whole diagnosis in one number: metres walked per metre of gap closed. Near 1 is
    # a straight line at the source; large is a wander. A ratio, so it needs no threshold
    # to read and no constant to justify.
    row["walked_per_metre_closed"] = (
        (walked / closed) if isinstance(closed, float) and closed > 0 else None)

    moves = [r for r in steps if r.displacement_m is not None]
    row["n_moves"] = len(moves)
    row["n_collided"] = sum(1 for r in moves if r.collided)
    rms = [float(r.measured_rms) for r in steps]
    row["rms_onset"] = rms[0] if rms else None
    row["rms_max"] = max(rms) if rms else None

    # --- the plateau half ------------------------------------------------
    # `rising` over the WHOLE episode, then keyed back to the detour by step number.
    # See `rising_flags`: a window-local recomputation invents a forward at the onset.
    all_flags = rising_flags([r.measured_rms for r in audit.steps])
    flag_by_step = {r.step: f for r, f in zip(audit.steps, all_flags)}
    flags = [flag_by_step[r.step] for r in steps]

    plateaus = _plateau_rows(steps, window, flags, min_span_m=min_span_m)
    row["plateaus"] = plateaus
    row["n_plateaus"] = len(plateaus)
    row["plateau_steps"] = sum(int(p["n_steps"]) for p in plateaus)
    row["longest_plateau_steps"] = max(
        (int(p["n_steps"]) for p in plateaus), default=0)
    row["n_static_plateaus"] = sum(1 for p in plateaus if p["static"])
    # The consistency signal that survives without `realizable_action`: the carried rule
    # answers FORWARD only while rising, so the follower's forwards should thin out
    # inside the windows. A ratio rather than a test — it corroborates the reconstruction
    # without being able to prove it.
    in_plateau = sum(int(p["n_forward"]) for p in plateaus)
    row["forward_in_plateau"] = in_plateau
    row["forward_total"] = sum(1 for r in steps if r.action == ACT_FORWARD)

    # The exact check, armed only on records that carry what the CUE said. Absent on
    # every run before `StepRecord.realizable_action` landed, and absent is reported as
    # unvalidated rather than as agreement — an unchecked reconstruction that reads as a
    # checked one is the failure mode this field exists to close.
    stops = sum(1 for r in steps if r.realizable_action == ACT_STOP)
    checkable = [
        (r, f) for r, f in zip(steps, flags)
        if r.realizable_action is not None and r.realizable_action != ACT_STOP
    ]
    row["n_rule_checked"] = len(checkable)
    row["n_rule_agree"] = sum(
        1 for r, f in checkable if rule_action(f, r.lateral_sign) == r.realizable_action)
    # Never re-derivable from the two above: the rule's STOP needs `visual_confirm`,
    # which no record carries, so these steps are excluded from the check by name.
    row["n_rule_stop"] = stops
    return row


def aggregate(traces: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pool per-episode traces into the two arms and their medians. Pure."""
    arms: Dict[str, List[Mapping[str, Any]]] = {ABANDONED: [], REACHED: [], NO_DETOUR: []}
    for trace in traces:
        arms.setdefault(str(trace.get("outcome")), []).append(trace)

    def _arm(name: str) -> Dict[str, Any]:
        rows = arms.get(name) or []
        return {
            "n": len(rows),
            "detour_steps": _median([r.get("detour_steps") for r in rows]),
            "d_onset_m": _median([r.get("d_onset_m") for r in rows]),
            "d_min_m": _median([r.get("d_min_m") for r in rows]),
            "gap_closed_m": _median([r.get("gap_closed_m") for r in rows]),
            "walked_m": _median([r.get("walked_m") for r in rows]),
            "walked_per_metre_closed": _median(
                [r.get("walked_per_metre_closed") for r in rows]),
            "collision_rate": _median([
                (r["n_collided"] / r["n_moves"]) if r.get("n_moves") else None
                for r in rows]),
        }

    def _plateau(name: str) -> Dict[str, Any]:
        """The plateau windows of one arm, pooled over its episodes."""
        rows = arms.get(name) or []
        windows: List[Mapping[str, Any]] = []
        for trace in rows:
            windows.extend(trace.get("plateaus") or [])
        fitted = [w for w in windows if w.get("slope_per_m") is not None]
        forward_in = sum(int(t.get("forward_in_plateau") or 0) for t in rows)
        forward_all = sum(int(t.get("forward_total") or 0) for t in rows)
        plateau_steps = sum(int(t.get("plateau_steps") or 0) for t in rows)
        detour_steps = sum(int(t.get("detour_steps") or 0) for t in rows)
        return {
            "n_windows": len(windows),
            "n_static": sum(1 for w in windows if w.get("static")),
            "n_fitted": len(fitted),
            "window_steps": _median([w.get("n_steps") for w in windows]),
            "longest_steps": _median([t.get("longest_plateau_steps") for t in rows]),
            # **The median is 1 and hides everything.** `detour-2` reported a median
            # window of one step with zero travel while 60% of detour steps sat inside a
            # window — arithmetic that only works if a long tail exists that the median
            # cannot show. A tail of long, genuinely-stalled windows and a hail of
            # one-step dropouts are different mechanisms with different fixes, so the
            # distribution is reported rather than a middle value standing in for it.
            "length_histogram": _length_histogram(windows),
            "steps_in_long_windows": sum(
                int(w.get("n_steps") or 0) for w in windows
                if int(w.get("n_steps") or 0) >= LONG_WINDOW_STEPS),
            "d_start_m": _median([w.get("d_start_m") for w in windows]),
            "d_span_m": _median([w.get("d_span_m") for w in windows]),
            "slope_per_m": _median([w.get("slope_per_m") for w in fitted]),
            "residual_sd": _median([w.get("residual_sd") for w in fitted]),
            "signal_to_scatter": _median([w.get("signal_to_scatter") for w in fitted]),
            # Fraction of the detour spent with the rule no longer answering FORWARD.
            "plateau_step_share": (
                (plateau_steps / detour_steps) if detour_steps else None),
            # The corroborating signal: forwards inside the windows against forwards
            # overall. The rule cannot answer FORWARD while plateaued, so a
            # reconstruction that is right predicts this sits well below the step share.
            "forward_share_in_plateau": (
                (forward_in / forward_all) if forward_all else None),
        }

    positioned = sum(1 for t in traces if t.get("d_onset_m") is not None)
    checked = sum(int(t.get("n_rule_checked") or 0) for t in traces)
    agreed = sum(int(t.get("n_rule_agree") or 0) for t in traces)
    return {
        "n_episodes": len(traces),
        "n_with_position": positioned,
        "arms": {name: _arm(name) for name in (ABANDONED, REACHED, NO_DETOUR)},
        "plateaus": {name: _plateau(name) for name in (ABANDONED, REACHED)},
        # The abort condition, pooled. `n_rule_checked` is zero on every run written
        # before `StepRecord.realizable_action`, and zero here means UNVALIDATED — the
        # reconstruction was never put to a test — which format_report says in words
        # rather than leaving a 0/0 to be read as agreement.
        "rule_check": {
            "n_checked": checked,
            "n_agree": agreed,
            "n_stop_excluded": sum(int(t.get("n_rule_stop") or 0) for t in traces),
            "agreement": (agreed / checked) if checked else None,
        },
        "per_episode": sorted(traces, key=lambda r: int(r.get("episode") or 0)),
    }


def _fmt(value: Any, spec: str = "{:.2f}") -> str:
    return "n/a" if value is None else spec.format(value)


def _wrap(text: str) -> List[str]:
    """A prose paragraph at the table's width, indented to sit under it.

    The verdict sentences are the part of this report a reader acts on, and an unwrapped
    300-character line in a terminal is one they skim past.
    """
    return textwrap.wrap(text, width=78, initial_indent="  ", subsequent_indent="  ")


def format_report(agg: Mapping[str, Any]) -> str:
    lines = ["ep  outcome     detour  d_onset  d_min  closed   walked  walked/closed  coll",
             "-" * 78]
    for row in agg["per_episode"]:
        moves = row.get("n_moves") or 0
        lines.append("{:<3} {:<11} {:>5}  {:>7}  {:>5}  {:>6}  {:>7}  {:>13}  {:>4}".format(
            row.get("episode", "?"),
            str(row.get("outcome", "?")),
            row.get("detour_steps", "-"),
            _fmt(row.get("d_onset_m")),
            _fmt(row.get("d_min_m")),
            _fmt(row.get("gap_closed_m")),
            _fmt(row.get("walked_m")),
            _fmt(row.get("walked_per_metre_closed"), "{:.1f}"),
            "{}/{}".format(row.get("n_collided", 0), moves)))
    lines.append("-" * 78)

    lines.append("")
    lines.append("medians by arm (the abandoned detours, against the ones that got there):")
    lines.append("  {:<12} {:>3}  {:>7}  {:>8}  {:>7}  {:>8}  {:>14}  {:>5}".format(
        "arm", "n", "steps", "d_onset", "closed", "walked", "walked/closed", "coll"))
    for name in (ABANDONED, REACHED):
        arm = agg["arms"][name]
        lines.append("  {:<12} {:>3}  {:>7}  {:>8}  {:>7}  {:>8}  {:>14}  {:>5}".format(
            name, arm["n"], _fmt(arm["detour_steps"], "{:.0f}"), _fmt(arm["d_onset_m"]),
            _fmt(arm["gap_closed_m"]), _fmt(arm["walked_m"]),
            _fmt(arm["walked_per_metre_closed"], "{:.1f}"),
            _fmt(arm["collision_rate"], "{:.0%}")))
    no_detour = agg["arms"][NO_DETOUR]["n"]
    if no_detour:
        lines.append("  ({} episode(s) never entered INVESTIGATE and are excluded)".format(
            no_detour))

    if agg["n_with_position"] < agg["n_episodes"]:
        lines.append("")
        lines.append(
            "  {} of {} episode(s) carry no per-step position, so their distances read "
            "n/a rather than zero — records written before StepRecord.position landed. "
            "Re-run to measure them.".format(
                agg["n_episodes"] - agg["n_with_position"], agg["n_episodes"]))

    lines.extend(_plateau_lines(agg))
    return "\n".join(lines)


def _histogram_lines(plateaus: Mapping[str, Any]) -> List[str]:
    """Window length against count, per arm. What the median could not show."""
    arms = [(name, plateaus.get(name) or {}) for name in (ABANDONED, REACHED)]
    histograms = [(name, arm.get("length_histogram") or []) for name, arm in arms]
    if not any(rows for _name, rows in histograms):
        return []
    labels = [label for label, _count in histograms[0][1]]
    lines = ["", "  window length in steps, by count — a 1 is the test flickering, a long",
             "  one is an agent that stayed put with room to move:"]
    lines.append("  {:<12} {}".format(
        "arm", " ".join("{:>6}".format(label) for label in labels)))
    for name, rows in histograms:
        lines.append("  {:<12} {}".format(
            name, " ".join("{:>6}".format(count) for _label, count in rows)))
    for name, arm in arms:
        long_steps = arm.get("steps_in_long_windows")
        if long_steps is None:
            continue
        lines.append(
            "  {:<12} {} plateaued step(s) sit in windows of {}+ steps".format(
                name, long_steps, LONG_WINDOW_STEPS))
    return lines


def _plateau_lines(agg: Mapping[str, Any]) -> List[str]:
    """The plateau half of the report: is the cue exhausted, or is the test missing it?"""
    plateaus = agg.get("plateaus") or {}
    if not plateaus:
        return []
    lines = ["", "plateau windows (maximal runs where the carried rule stopped answering",
             "FORWARD — its own predicate, recomputed from measured_rms):"]
    lines.append("  {:<12} {:>4}  {:>6}  {:>6}  {:>7}  {:>7}  {:>9}  {:>8}  {:>7}".format(
        "arm", "wins", "static", "steps", "d_start", "d_span", "slope/m", "resid", "sig/sc"))
    for name in (ABANDONED, REACHED):
        arm = plateaus.get(name) or {}
        lines.append(
            "  {:<12} {:>4}  {:>6}  {:>6}  {:>7}  {:>7}  {:>9}  {:>8}  {:>7}".format(
                name,
                arm.get("n_windows", 0),
                "{}/{}".format(arm.get("n_static", 0), arm.get("n_windows", 0)),
                _fmt(arm.get("window_steps"), "{:.0f}"),
                _fmt(arm.get("d_start_m")),
                _fmt(arm.get("d_span_m")),
                _fmt(arm.get("slope_per_m"), "{:+.2e}"),
                _fmt(arm.get("residual_sd"), "{:.1e}"),
                _fmt(arm.get("signal_to_scatter"), "{:.2f}")))
    lines.append("")
    lines.append("  slope is measured_rms against distance-to-source: NEGATIVE is a live")
    lines.append("  gradient (louder as the gap closes), flat is a cue that has run out.")
    lines.append("  sig/sc is |slope| x d_span over the residual SD — above 1 the cue was")
    lines.append("  recoverable from these traces and the single-step test is what missed")
    lines.append("  it; below 1 the render buries it at this ray count.")
    lines.extend(_histogram_lines(plateaus))

    for name in (ABANDONED, REACHED):
        arm = plateaus.get(name) or {}
        share = arm.get("plateau_step_share")
        forward = arm.get("forward_share_in_plateau")
        if share is None and forward is None:
            continue
        lines.append(
            "  {:<12} {} of detour steps plateaued; {} of its forwards fell inside "
            "a window".format(name, _fmt(share, "{:.0%}"), _fmt(forward, "{:.0%}")))

    check = agg.get("rule_check") or {}
    lines.append("")
    if not check.get("n_checked"):
        # NOT a silent pass. Nothing here was validated, and a reader who cannot tell
        # that from a validated run will trust a reconstruction no one checked.
        lines.extend(_wrap(
            "RECONSTRUCTION UNVALIDATED — no record carries "
            "StepRecord.realizable_action, so what the cue SAID was never compared "
            "against what was recomputed. Every plateau above rests on an unchecked "
            "model of the controller. Re-run to arm the check."))
    else:
        lines.extend(_wrap(
            "reconstruction checked on {} step(s): {} agree ({}). {} STOP step(s) "
            "excluded — the rule's STOP needs visual_confirm and no record carries "
            "it.".format(
                check["n_checked"], check["n_agree"],
                _fmt(check.get("agreement"), "{:.1%}"),
                check.get("n_stop_excluded", 0))))
        if (check.get("agreement") or 0.0) < 1.0:
            lines.extend(_wrap(
                "DISAGREEMENT IS THE FINDING, not a rounding error: the model of the "
                "controller above is wrong wherever these differ, and nothing derived "
                "from it should be read until that is explained."))
    return lines


def load_traces(
    run_dir: str, *, min_span_m: float = DEFAULT_MIN_SPAN_M
) -> List[Dict[str, Any]]:
    """Every episode's detour trace, with the budget read from the run's own env_report."""
    from earshot.task.smoke import episode_indices

    root, _ = run_paths(run_dir)
    budget = None
    env_path = root / ENV_REPORT_NAME
    if env_path.exists():
        payload = json.loads(env_path.read_text(encoding="utf-8"))
        controller = (payload.get("run_config") or {}).get("controller") or {}
        raw = controller.get("investigate_max_steps")
        budget = int(raw) if isinstance(raw, int) else None
    traces = []
    for index in episode_indices(str(root)):
        _, audit_path = episode_paths(root, index)
        traces.append(
            trace_one(read_audit(audit_path), budget=budget, min_span_m=min_span_m))
    return traces


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a directory `python -m earshot` wrote")
    parser.add_argument("--json", action="store_true", help="emit the aggregate as JSON")
    parser.add_argument(
        "--min-span", type=float, default=DEFAULT_MIN_SPAN_M,
        help="metres a plateau window must span before its slope is fitted rather than "
             "reported static (default {:.2f})".format(DEFAULT_MIN_SPAN_M))
    args = parser.parse_args(argv)

    if not pathlib.Path(args.run_dir).is_dir():
        print("no such run directory: {}".format(args.run_dir))
        return 2
    traces = load_traces(args.run_dir, min_span_m=args.min_span)
    if not traces:
        print("no episode records under {} — nothing to trace".format(args.run_dir))
        return 2
    agg = aggregate(traces)
    print(json.dumps(agg, indent=2) if args.json else format_report(agg))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
