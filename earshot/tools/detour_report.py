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
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence

from earshot.report.artifacts import ENV_REPORT_NAME, episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit, FunnelStage

__all__ = [
    "ABANDONED",
    "REACHED",
    "NO_DETOUR",
    "trace_one",
    "aggregate",
    "format_report",
    "load_traces",
    "main",
]

ABANDONED = "abandoned"  # entered INVESTIGATE, spent the budget, never reached the source
REACHED = "reached"      # got there; the control arm for the above
NO_DETOUR = "no_detour"  # onset never fired, or the episode ended before it could


def _median(values: Sequence[float]) -> Optional[float]:
    """Median, or None for an empty sample. Median rather than mean because n is ~10 and
    one episode that walked into a corner should not move the number the fix is chosen
    against."""
    clean = [float(v) for v in values if v is not None]
    return statistics.median(clean) if clean else None


def trace_one(audit: EpisodeAudit, *, budget: Optional[int] = None) -> Dict[str, Any]:
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

    positioned = sum(1 for t in traces if t.get("d_onset_m") is not None)
    return {
        "n_episodes": len(traces),
        "n_with_position": positioned,
        "arms": {name: _arm(name) for name in (ABANDONED, REACHED, NO_DETOUR)},
        "per_episode": sorted(traces, key=lambda r: int(r.get("episode") or 0)),
    }


def _fmt(value: Any, spec: str = "{:.2f}") -> str:
    return "n/a" if value is None else spec.format(value)


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
    return "\n".join(lines)


def load_traces(run_dir: str) -> List[Dict[str, Any]]:
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
        traces.append(trace_one(read_audit(audit_path), budget=budget))
    return traces


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="a directory `python -m earshot` wrote")
    parser.add_argument("--json", action="store_true", help="emit the aggregate as JSON")
    args = parser.parse_args(argv)

    if not pathlib.Path(args.run_dir).is_dir():
        print("no such run directory: {}".format(args.run_dir))
        return 2
    traces = load_traces(args.run_dir)
    if not traces:
        print("no episode records under {} — nothing to trace".format(args.run_dir))
        return 2
    agg = aggregate(traces)
    print(json.dumps(agg, indent=2) if args.json else format_report(agg))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
