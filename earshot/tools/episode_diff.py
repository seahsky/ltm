"""Pair the EPISODES, not the scenes: a McNemar test over two sweeps.

`funnel_diff` compares two runs one scene at a time, because `summary.json` records how
many episodes reached the source and never which. That throws away almost everything: a
twenty-scene sweep of 365 episodes yields at most twenty comparisons, of which the ones
that moved are fewer still, and a real effect of a dozen episodes cannot clear a sign test
over sixteen. `cast-1` against `arrive-2` is exactly that case — +13 episodes, p = 0.45.

The information is on disk. Every episode writes an `audit.json` carrying its own
`funnel_stage`, and the builder is deterministic, so episode 7 of one scene is the SAME
TASK in both arms. Pairing them turns twenty comparisons into 365.

**The pairing is verified, never assumed.** Two runs of a changed builder would put a
different task at the same index and the comparison would be silently meaningless, so a
pair is formed only when both audits agree on the scene and on `source_xyz`; anything else
is counted, named and dropped. That is the same discipline `funnel_diff` applies when it
refuses to subtract sweeps that built different scene sets.

**McNemar needs no noise model, which is the other reason to prefer it.** `funnel_diff`
has to compare a net against `sqrt(FLIP_RATE * built)` using a flip rate measured once, on
one scene, in `detour-1`. Here the flips ARE the discordant pairs: under a null of no
effect the two arms have the same per-episode outcome distribution, so a disagreement is
equally likely to fall either way and the discordant pairs are a fair coin by
construction. Nothing about the renderer has to be estimated.

**What it does assume is that episodes are independent**, which they are not — episodes
inside one scene share a room, a source and a renderer. If the disagreements cluster into
a few scenes, the p is anti-conservative. So the per-scene split of the discordant pairs
is printed beside the total, and a reader who sees the whole imbalance coming from two
rooms should believe the scene-level sign test instead.
"""

from __future__ import annotations

import argparse
import pathlib
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from earshot.report.artifacts import episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.tools.funnel_diff import HEADLINE_STAGE, two_sided_exact_binomial

__all__ = [
    "SOURCE_TOLERANCE_M",
    "load_outcomes",
    "pair_episodes",
    "mcnemar",
    "format_report",
    "main",
]

# Two audits describe the same task when their source positions agree to here. They are
# written from the same float and read back through the same JSON, so an exact compare
# would work today; the tolerance is for the day something rounds on the way out, and it
# is far tighter than any two distinct objects in a scene could be.
SOURCE_TOLERANCE_M = 1e-6


def _source_key(audit: EpisodeAudit) -> Optional[Tuple[float, ...]]:
    source = audit.source_xyz
    return None if source is None else tuple(float(v) for v in source.as_tuple())


def _same_source(
    left: Optional[Sequence[float]], right: Optional[Sequence[float]]
) -> bool:
    """A missing source on either side is NOT a match. An episode that never recorded
    where the sound was cannot be verified against one that did, and pairing them anyway
    is the exact failure this function exists to prevent."""
    if left is None or right is None:
        return False
    return all(abs(a - b) <= SOURCE_TOLERANCE_M for a, b in zip(left, right))


def load_outcomes(
    tag_dir: str, *, stage: FunnelStage = HEADLINE_STAGE
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """`{scene: {episode_index: {"reached", "source", "stage"}}}` for one sweep's tag
    directory. The only function here that touches the disk.

    A scene directory that built nothing is present with an empty mapping rather than
    absent, so a sweep's shape survives into the diff: `mL8ThkuaVTM` yields zero episodes
    in every run this repo has, and a reader must be able to tell that from a scene that
    was never attempted.
    """
    from earshot.task.smoke import episode_indices

    root, _ = run_paths(tag_dir)
    if not root.is_dir():
        raise ValueError("{} is not a directory".format(tag_dir))
    if run_paths(root)[1].is_dir():
        raise ValueError(
            "{} looks like a SCENE directory, not a sweep — pass the tag directory "
            "that holds one subdirectory per scene".format(tag_dir)
        )
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        episodes: Dict[int, Dict[str, Any]] = {}
        for index in episode_indices(str(scene_dir)):
            _, audit_path = episode_paths(scene_dir, index)
            audit = read_audit(audit_path)
            episodes[int(index)] = {
                "reached": audit.funnel_stage >= stage,
                "source": _source_key(audit),
                "stage": audit.funnel_stage.name,
            }
        out[scene_dir.name] = episodes
    return out


def pair_episodes(
    before: Mapping[str, Mapping[int, Mapping[str, Any]]],
    after: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Match episodes across two sweeps by `(scene, index)` and verify the source. Pure.

    Everything that could not be paired comes back named. A comparison that quietly
    dropped half its episodes and reported a clean p over the rest is the failure mode
    worth more than the convenience of a shorter return type.
    """
    scenes_both = sorted(set(before) & set(after))
    pairs: List[Dict[str, Any]] = []
    unmatched_index: List[str] = []
    mismatched_source: List[str] = []
    for scene in scenes_both:
        b_eps, a_eps = before[scene], after[scene]
        for index in sorted(set(b_eps) | set(a_eps)):
            if index not in b_eps or index not in a_eps:
                unmatched_index.append("{}#{}".format(scene, index))
                continue
            b, a = b_eps[index], a_eps[index]
            if not _same_source(b.get("source"), a.get("source")):
                mismatched_source.append("{}#{}".format(scene, index))
                continue
            pairs.append({
                "scene": scene,
                "episode": index,
                "before": bool(b["reached"]),
                "after": bool(a["reached"]),
            })
    return {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "scenes_paired": scenes_both,
        "scenes_before_only": sorted(set(before) - set(after)),
        "scenes_after_only": sorted(set(after) - set(before)),
        "unmatched_index": unmatched_index,
        "mismatched_source": mismatched_source,
        "n_dropped": len(unmatched_index) + len(mismatched_source),
    }


def mcnemar(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The exact McNemar test over paired episode outcomes. Pure.

    `lost` is reached-before-only, `gained` is reached-after-only. The concordant pairs —
    both reached, neither reached — carry no information about a difference and are
    excluded from the test, which is the whole point: they are where the scene-level
    reading spends its power.
    """
    gained = [p for p in pairs if p["after"] and not p["before"]]
    lost = [p for p in pairs if p["before"] and not p["after"]]
    both = sum(1 for p in pairs if p["before"] and p["after"])
    neither = sum(1 for p in pairs if not p["before"] and not p["after"])
    discordant = len(gained) + len(lost)
    per_scene: Dict[str, Dict[str, int]] = {}
    for p in gained:
        per_scene.setdefault(p["scene"], {"gained": 0, "lost": 0})["gained"] += 1
    for p in lost:
        per_scene.setdefault(p["scene"], {"gained": 0, "lost": 0})["lost"] += 1
    return {
        "n_gained": len(gained),
        "n_lost": len(lost),
        "n_both": both,
        "n_neither": neither,
        "n_discordant": discordant,
        "net": len(gained) - len(lost),
        "p_value": two_sided_exact_binomial(max(len(gained), len(lost)), discordant),
        "per_scene": per_scene,
        "scenes_with_discordance": len(per_scene),
    }


def _wrap(text: str) -> List[str]:
    return textwrap.wrap(" ".join(text.split()), width=72,
                         initial_indent="  ", subsequent_indent="  ")


def format_report(
    pairing: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    labels: Tuple[str, str] = ("before", "after"),
    stage: FunnelStage = HEADLINE_STAGE,
) -> str:
    """The printed report. Pure, so the arithmetic is assertable without a run on disk."""
    before_label, after_label = labels
    lines = [
        "{} — {} against {}, PAIRED BY EPISODE".format(
            stage.name, before_label, after_label),
        "-" * 72,
    ]
    n = int(result["n_discordant"])
    lines.append("  paired episodes      {}".format(pairing["n_pairs"]))
    lines.append("  reached in both      {}".format(result["n_both"]))
    lines.append("  reached in neither   {}".format(result["n_neither"]))
    lines.append("  {:<20} {}".format(after_label + " only", result["n_gained"]))
    lines.append("  {:<20} {}".format(before_label + " only", result["n_lost"]))
    lines.append("-" * 72)
    lines.append("  net {:+d} over {} discordant pair(s)".format(int(result["net"]), n))
    if result["p_value"] is None:
        lines.append("  NO DISCORDANT PAIRS — the two arms agreed on every episode.")
    else:
        lines.append("  exact McNemar p = {:.4f}".format(result["p_value"]))
    lines.append("")
    lines.extend(_wrap(
        "The concordant pairs are excluded by construction: an episode both arms "
        "reached, or neither reached, says nothing about a difference between them. "
        "Under a null the two arms have the same per-episode outcome distribution, so a "
        "disagreement is equally likely to fall either way and no flip rate has to be "
        "estimated for this test."))
    lines.append("")
    lines.extend(_wrap(
        "INDEPENDENCE IS THE ASSUMPTION. Episodes inside one scene share a room, a "
        "source and a renderer, so if the disagreements cluster the p above is "
        "anti-conservative. They are spread over {} of the {} paired scene(s) "
        "below; read the scene-level sign test instead if that looks like one "
        "room.".format(result["scenes_with_discordance"], len(pairing["scenes_paired"]))))
    per_scene = result["per_scene"]
    if per_scene:
        lines.append("")
        lines.extend(_wrap(
            "gained = reached in {} only; lost = reached in {} only.".format(
                after_label, before_label)))
        lines.append("  scene                {:>8} {:>8} {:>8}".format(
            "gained", "lost", "net"))
        for scene in sorted(per_scene):
            row = per_scene[scene]
            lines.append("  {:<20} {:>8} {:>8} {:>+8d}".format(
                scene[:20], row["gained"], row["lost"], row["gained"] - row["lost"]))
    lines.extend(_dropped_lines(pairing))
    return "\n".join(lines)


def _dropped_lines(pairing: Mapping[str, Any]) -> List[str]:
    """What could not be paired, always printed. A silent drop turns a partial comparison
    into a confident one, which is the way this kind of report lies."""
    lines = ["", "  " + "-" * 70]
    dropped = int(pairing["n_dropped"])
    if not dropped and not pairing["scenes_before_only"] and not pairing["scenes_after_only"]:
        lines.extend(_wrap(
            "EVERY EPISODE PAIRED. Both sweeps built the same scenes, the same episode "
            "indices, and every pair agreed on where the sound was."))
        return lines
    for key, label in (("scenes_before_only", "scene(s) only the first sweep built"),
                       ("scenes_after_only", "scene(s) only the second sweep built")):
        if pairing[key]:
            lines.extend(_wrap("{}: {}".format(label, ", ".join(pairing[key]))))
    if pairing["unmatched_index"]:
        lines.extend(_wrap(
            "{} episode(s) present in one sweep and not the other, DROPPED: {}".format(
                len(pairing["unmatched_index"]), ", ".join(pairing["unmatched_index"]))))
    if pairing["mismatched_source"]:
        lines.extend(_wrap(
            "{} episode(s) share an index but NOT a source position, DROPPED: {}. The "
            "builder put a different task at that index in the two sweeps, so they are "
            "not the same episode and subtracting them would compare two "
            "problems.".format(
                len(pairing["mismatched_source"]), ", ".join(pairing["mismatched_source"]))))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-episode McNemar test between two sweeps' tag directories.")
    parser.add_argument("before", help="the tag directory WITHOUT the change")
    parser.add_argument("after", help="the tag directory WITH it")
    parser.add_argument(
        "--stage", default=HEADLINE_STAGE.name,
        help="funnel stage counted as a success (default {})".format(HEADLINE_STAGE.name))
    args = parser.parse_args(argv)

    try:
        stage = FunnelStage[args.stage]
    except KeyError:
        print("unknown stage {!r}; expected one of {}".format(
            args.stage, ", ".join(s.name for s in FunnelStage)))
        return 2
    try:
        before = load_outcomes(args.before, stage=stage)
        after = load_outcomes(args.after, stage=stage)
    except ValueError as exc:
        print(str(exc))
        return 2

    pairing = pair_episodes(before, after)
    if not pairing["n_pairs"]:
        print("no episode paired between {} and {} — nothing to test".format(
            args.before, args.after))
        return 2
    result = mcnemar(pairing["pairs"])
    labels = (pathlib.Path(args.before).name, pathlib.Path(args.after).name)
    print(format_report(pairing, result, labels=labels, stage=stage))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
