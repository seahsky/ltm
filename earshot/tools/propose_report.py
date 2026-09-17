"""ADR-0026's counters, read off a finished sweep. Read-only, no GPU, seconds.

WHY THIS IS A MODULE AND NOT FORTY LINES INSIDE THE DRIVER. `propose_sweep.sh` shipped
its mechanism check as an inline Python heredoc, and the heredoc globbed
`<arm>/<scene>/episodes/<N>/audit.json`. `report/artifacts.episode_paths` writes
`<scene>/episodes/ep0000.audit.json`, a FLAT file with no directory per episode. The glob
matched nothing, every counter printed 0, and `propose-1`'s readout reported "the
mechanism never ran" over 1128 episodes whose audits were on disk the whole time.

That is `dream-1`'s failure repeated: a central quantity written to disk that no reader
could reach. `pilot-1` is the other precedent, and the rule the tree already had is the
one that was broken here -- a reader is a tested module in `tools/`, never a string inside
a bash script, because nothing in the suite can see the string.

WHAT IT DECIDES. ADR-0026's fourth branch: if eq. 26 ranked the memory's proposal first on
under 5% of the steps where it was eligible, the rail or the store is suppressing the
mechanism and the run is NOT a result about memory, whatever the contrast says. The
contrast is not interpretable before this section is read.

  python -m earshot.tools.propose_report runs/<tag>
  python -m earshot.tools.propose_report runs/<tag> --arms "propose-a propose-b"

Exits 2 if no arm recorded the counters anywhere: unreadable is not "the mechanism was
inert", which is the distinction this file exists because of.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

from earshot.report.artifacts import episode_paths, read_audit, run_paths

__all__ = ["ARM_PREFIX", "COUNTERS", "arm_totals", "format_report", "main"]

# Only these arms can carry the counters: `runner.py` writes them when
# `MemoryContext.proposes` is on, which `propose_sweep.sh` sets from a `propose-` prefix.
ARM_PREFIX = "propose"

# The six the runner writes with explicit zeros, plus the rail, which is a setting rather
# than a count and is reported separately for that reason.
COUNTERS: Tuple[str, ...] = (
    "memory_propose_eligible",
    "memory_propose_ranked_first",
    "memory_propose_emitted",
    "memory_propose_railed",
    "memory_propose_unrouted",
    "memory_propose_no_acoustic",
)
RAIL_KEY = "memory_propose_rail_m"

# ADR-0026's fourth branch.
SUPPRESSED_BELOW = 0.05


def _scene_dirs(arm_dir: pathlib.Path) -> Tuple[pathlib.Path, ...]:
    if not arm_dir.is_dir():
        return ()
    return tuple(
        sorted(p for p in arm_dir.iterdir() if p.is_dir() and run_paths(p)[1].is_dir())
    )


def arm_totals(arm_dir: str) -> Dict[str, float]:
    """Summed counters over every episode under one arm directory. Touches the disk.

    ``episodes`` is the number of audits read and ``scenes`` the number of scene
    directories, both reported so a zero counter can be told apart from a zero read.
    """
    root = pathlib.Path(arm_dir)
    totals: Dict[str, float] = {key: 0.0 for key in COUNTERS}
    totals["episodes"] = 0.0
    totals["episodes_with_counters"] = 0.0
    rails = set()
    scenes = _scene_dirs(root)
    for scene in scenes:
        _, episodes = run_paths(scene)
        for path in sorted(episodes.glob("ep*.audit.json")):
            index = int(path.name[2:6])
            audit = read_audit(episode_paths(scene, index)[1])
            metrics: Mapping[str, float] = audit.metrics or {}
            totals["episodes"] += 1.0
            if any(key in metrics for key in COUNTERS):
                totals["episodes_with_counters"] += 1.0
            for key in COUNTERS:
                totals[key] += float(metrics.get(key, 0.0))
            if RAIL_KEY in metrics:
                rails.add(float(metrics[RAIL_KEY]))
    totals["scenes"] = float(len(scenes))
    if len(rails) == 1:
        totals[RAIL_KEY] = rails.pop()
    elif len(rails) > 1:
        # Two rails inside one arm is two experiments in one directory, and it must not be
        # averaged into a single number that describes neither.
        totals["rail_disagreement"] = float(len(rails))
    return totals


def format_report(by_arm: Mapping[str, Mapping[str, float]]) -> str:
    """The printed report. Pure, so the branch arithmetic is assertable without a run."""
    lines = ["ADR-0026 — what the memory's proposal did", "-" * 72]
    if not by_arm:
        lines.append("  NO PROPOSING ARM ON DISK. This sweep has no `{}*` directory, so "
                     "there is".format(ARM_PREFIX))
        lines.append("  nothing here to read. That is not a null.")
        return "\n".join(lines)
    for arm in sorted(by_arm):
        t = by_arm[arm]
        eligible = t.get("memory_propose_eligible", 0.0)
        first = t.get("memory_propose_ranked_first", 0.0)
        lines.append("")
        lines.append("  {}  ({:.0f} episode(s) over {:.0f} scene(s))".format(
            arm, t.get("episodes", 0.0), t.get("scenes", 0.0)))
        if t.get("episodes", 0.0) == 0.0:
            lines.append("    NO EPISODE ON DISK under this arm. Check "
                         "<arm>/<scene>/episodes/.")
            continue
        if t.get("episodes_with_counters", 0.0) == 0.0:
            # The distinction this module exists for.
            lines.append("    COUNTERS NOT RECORDED on any of its {:.0f} episode(s). The "
                         "arm ran".format(t["episodes"]))
            lines.append("    WITHOUT --memory-proposes, or predates PR #133. Unreadable "
                         "is not inert.")
            continue
        lines.append("    steps the proposal was IN THE POOL:      {:>8.0f}".format(
            eligible))
        lines.append("    steps eq. 26 RANKED IT FIRST:            {:>8.0f}{}".format(
            first,
            "" if eligible <= 0 else "   ({:.2%} of eligible)".format(first / eligible)))
        lines.append("    emitted:   {:>8.0f}".format(
            t.get("memory_propose_emitted", 0.0)))
        lines.append("    SUPPRESSED by the rail:  {:>8.0f}".format(
            t.get("memory_propose_railed", 0.0)))
        lines.append("    suppressed, unrouted:    {:>8.0f}".format(
            t.get("memory_propose_unrouted", 0.0)))
        lines.append("    suppressed, no acoustic estimate to compare against: {:>4.0f}"
                     .format(t.get("memory_propose_no_acoustic", 0.0)))
        if RAIL_KEY in t:
            lines.append("    rail: {}".format(
                "REMOVED" if t[RAIL_KEY] < 0 else "{:.2f} m".format(t[RAIL_KEY])))
        if "rail_disagreement" in t:
            lines.append("    RAIL DISAGREEMENT: {:.0f} different rails inside one arm, "
                         "so this".format(t["rail_disagreement"]))
            lines.append("    directory holds more than one experiment.")
        lines.append("")
        if t.get("memory_propose_emitted", 0.0) == 0.0:
            lines.append("    THE PROPOSAL NEVER ENTERED A POOL. Every consultation was "
                         "suppressed,")
            lines.append("    so this arm ran the acoustic estimate alone and the "
                         "contrast is a")
            lines.append("    contrast against NO MEMORY, not against a ranked one. Read "
                         "the")
            lines.append("    suppression reasons above: they say which gate did it.")
        elif eligible <= 0:
            lines.append("    EMITTED BUT NEVER RANKED. The proposal reached "
                         "`reachable_pool` and")
            lines.append("    was filtered out by the navmesh every time, so eq. 26 "
                         "never saw it.")
        elif first / eligible < SUPPRESSED_BELOW:
            lines.append("    UNDER {:.0%} — ADR-0026's FOURTH BRANCH. The rail or the "
                         "store is".format(SUPPRESSED_BELOW))
            lines.append("    suppressing the mechanism, and this run is NOT a result "
                         "about memory")
            lines.append("    until that is fixed. Do not read the contrast yet.")
        else:
            lines.append("    THE MECHANISM IS LIVE. eq. 26 preferred the memory's "
                         "waypoint on")
            lines.append("    {:.2%} of the steps it was offered on, so a difference "
                         "against the".format(first / eligible))
            lines.append("    replace arm is attributable to the ranking.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ADR-0026's proposal counters, off a finished sweep. Read-only.")
    parser.add_argument("tag_dir", help="runs/<tag>")
    parser.add_argument(
        "--arms", default=None,
        help="space-separated arm directory names (default: every `{}*` directory "
             "present)".format(ARM_PREFIX))
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = pathlib.Path(args.tag_dir)
    if not root.is_dir():
        print("{} is not a directory".format(args.tag_dir))
        return 2
    if args.arms:
        names = tuple(args.arms.split())
    else:
        names = tuple(sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name.startswith(ARM_PREFIX)
        ))
    by_arm = {name: arm_totals(str(root / name)) for name in names}
    print(format_report(by_arm))
    if not by_arm or all(
        t.get("episodes_with_counters", 0.0) == 0.0 for t in by_arm.values()
    ):
        # Unreadable is not inert, and it must not exit 0 and look like a finding.
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
