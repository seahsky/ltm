"""How much of the outcome is the agent and how much is the renderer.

    python -m earshot.tools.flip_report runs/rays-1-r500-* runs/rays-1-r2500-*

**The finding this exists to size.** ``yield-1`` and ``detour-1`` ran the same scene under
the same configuration, and 4 of 20 episodes came out with different outcomes. Both
funnels said 8/20 source-reached; they were not the same eight. The onset step was
identical in all twenty, so the trigger is deterministic — the render is not. Calibration
thresholds moved up to 13% between the runs, separation 2.5 dB, and the live RMS at the
trigger pose 24%.

**And it is a knob, not a fact.** ``spec.ACOUSTICS_PRESET`` sets ``indirectRayCount`` to
500, cut from habitat's 5000 by ticket 06 for a 63x speedup. That is a Monte Carlo
estimate with a tenth of the samples; the variance is the price, and it was paid without
being measured. ``AudioConfig.indirect_ray_count`` makes it settable, and this reads back
what buying some of it returns.

**Two numbers, and the second is the one that matters.**

- The *aggregate* rate (how many episodes reached the source) can be stable while the
  membership churns underneath it — which is exactly what the two runs above did. An
  aggregate compared across arms is the shape every ablation on this map uses, so its
  stability is not optional.
- The *flip rate* — episodes whose outcome is not unanimous across repeats — is what
  decides how many repeats a matrix cell needs before a paired delta means anything.
  ``yield-1`` against ``detour-1`` puts it near 20% at 500 rays.

``compare()`` is pure, so the arithmetic is Mac-testable against injected records while
the runs that feed it need a GPU — the same split ``yield_report`` and ``detour_report``
are built on. No verdict and no threshold: the ray counts are the arms, and which one to
run is a decision with a wall-clock price that belongs to a person.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

from earshot.report.artifacts import ENV_REPORT_NAME, episode_paths, read_audit, run_paths
from earshot.report.audit import FunnelStage

__all__ = ["outcomes_of", "compare", "format_report", "main"]

# The stage an episode has to reach for the anomaly-response loop to have run (§6, and
# CONTEXT.md's Anomaly-response SR). Below it the episode is an abandoned investigation.
LOOP_STAGE = FunnelStage.PRIMARY_RESUMED


def outcomes_of(run_dir: str) -> Dict[str, Any]:
    """One run's per-episode loop outcomes, plus the ray count that produced them."""
    from earshot.task.smoke import episode_indices

    root, _ = run_paths(run_dir)
    rays: Optional[int] = None
    env_path = root / ENV_REPORT_NAME
    if env_path.exists():
        payload = json.loads(env_path.read_text(encoding="utf-8"))
        audio = (payload.get("run_config") or {}).get("audio") or {}
        raw = audio.get("indirect_ray_count")
        # `None` in the record means the run took the preset's default rather than
        # setting one, which is a different fact from "500 was chosen" only in intent —
        # and intent is what a comparison across arms is reading.
        rays = int(raw) if isinstance(raw, int) else None
    reached = {}
    for index in episode_indices(str(root)):
        _, audit_path = episode_paths(root, index)
        reached[int(index)] = read_audit(audit_path).funnel_stage >= LOOP_STAGE
    return {"run": pathlib.Path(run_dir).name, "rays": rays, "reached": reached}


def compare(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Group runs by ray count and measure both stabilities. Pure.

    An episode counts as *flipped* when its outcome is not unanimous across the repeats of
    its arm. Unanimity rather than a pairwise count, because with three or more repeats
    "2 of 3 agree" is still an episode whose outcome depends on the roll.

    Episodes missing from some repeat of an arm are reported rather than dropped: a repeat
    that built a different episode set is not a repeat, and silently intersecting would
    turn that into a smaller and quieter flip rate.
    """
    arms: Dict[Any, List[Mapping[str, Any]]] = {}
    for run in runs:
        arms.setdefault(run.get("rays"), []).append(run)

    rows = []
    for rays in sorted(arms, key=lambda r: (r is None, r)):
        members = arms[rays]
        per_run_reached = [sum(1 for v in m["reached"].values() if v) for m in members]
        per_run_n = [len(m["reached"]) for m in members]
        every_episode = sorted({e for m in members for e in m["reached"]})
        complete = [e for e in every_episode
                    if all(e in m["reached"] for m in members)]
        ragged = [e for e in every_episode if e not in complete]
        flipped = [e for e in complete
                   if len({m["reached"][e] for m in members}) > 1]
        rows.append({
            "rays": rays,
            "n_runs": len(members),
            "runs": [m["run"] for m in members],
            "n_episodes": per_run_n,
            "reached": per_run_reached,
            # None, not 0.0, with a single repeat: one run cannot disagree with itself,
            # and a flip rate of zero would read as stability rather than as no evidence.
            "flip_rate": (len(flipped) / len(complete))
            if len(members) > 1 and complete else None,
            "n_flipped": len(flipped) if len(members) > 1 else None,
            "n_compared": len(complete),
            "flipped": flipped,
            "ragged": ragged,
        })
    return {"arms": rows}


def format_report(comparison: Mapping[str, Any]) -> str:
    lines = ["rays    runs  episodes  reached per run      flipped  flip rate",
             "-" * 68]
    for row in comparison["arms"]:
        rate = row["flip_rate"]
        lines.append("{:<6}  {:>4}  {:>8}  {:<19}  {:>7}  {:>9}".format(
            "preset" if row["rays"] is None else row["rays"],
            row["n_runs"],
            ",".join(str(n) for n in sorted(set(row["n_episodes"]))),
            " ".join(str(n) for n in row["reached"]),
            "n/a" if row["n_flipped"] is None else "{}/{}".format(
                row["n_flipped"], row["n_compared"]),
            "n/a" if rate is None else "{:.0%}".format(rate)))
    lines.append("-" * 68)
    for row in comparison["arms"]:
        if row["flipped"]:
            lines.append("  {}: episodes not unanimous: {}".format(
                "preset" if row["rays"] is None else row["rays"],
                ", ".join(str(e) for e in row["flipped"])))
        if row["ragged"]:
            lines.append("  {}: episode(s) MISSING from some repeat, excluded from the "
                         "rate: {}".format(
                             "preset" if row["rays"] is None else row["rays"],
                             ", ".join(str(e) for e in row["ragged"])))
        if row["n_runs"] < 2:
            lines.append("  {}: one run only — a flip rate needs a repeat, so this arm "
                         "reports n/a rather than 0%".format(
                             "preset" if row["rays"] is None else row["rays"]))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", help="run directories to compare")
    parser.add_argument("--json", action="store_true", help="emit the comparison as JSON")
    args = parser.parse_args(argv)

    runs = []
    for run_dir in args.run_dirs:
        if not pathlib.Path(run_dir).is_dir():
            print("no such run directory: {}".format(run_dir))
            return 2
        runs.append(outcomes_of(run_dir))
    if not any(run["reached"] for run in runs):
        print("no episode records under any of those directories — nothing to compare")
        return 2
    comparison = compare(runs)
    print(json.dumps(comparison, indent=2, default=sorted) if args.json
          else format_report(comparison))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
