"""How much of HM3D can express a decoupled anomaly response — read off the run records.

    python -m earshot.tools.yield_report runs/yield-<tag>

**The number this exists to produce is a denominator.** §2.1's builder refuses an episode
whose scene cannot place an anomaly source far enough from every primary goal, on the same
floor as both the anchor and the agent's start, at a real view point. Every refusal is
correct. But the refusal rate is the fraction of HM3D the task can be posed in at all, and
it therefore bounds every ``n`` an experiment matrix can quote — a matrix planned against
20 scenes and delivered against 9 is a different paper.

Nobody has measured it. The smoke ran one scene and skipped 1 of 2, which is a sample of
one and consistent with anything from 10% to 90%.

``aggregate()`` is pure, so the arithmetic is Mac-testable against injected records while
the runs that feed it need a GPU. It reports **per rule**, not just per scene, because
"the yield is 45%" and "the yield is 45% and two thirds of the loss is the floor rule" are
different findings: the first prices the matrix, the second says which rule to revisit.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["RULE_PATTERNS", "aggregate", "format_report", "main"]

# The builder writes its per-rule counts into the skip reason as prose — "(rejected: 11
# too near, 4 on another floor, 0 with no view point)" — so this parses them back out.
# Deliberately tolerant: an unparsed reason is counted as `unattributed` rather than
# dropped, because a rule that stops matching must show up as a gap in the total and not
# as a silently smaller loss. The alternative, having the builder emit structured counts,
# is the better fix and is noted on the ticket rather than done here.
RULE_PATTERNS = (
    ("too_near", re.compile(r"(\d+)\s+too near")),
    # Distinct from `too_near`: that one is "on top of the GOAL", this one is "on top of
    # the AGENT". Pooling them would name the wrong rule to revisit, which is the whole
    # reason this report is per-rule and not just per-scene.
    ("at_the_start", re.compile(r"(\d+)\s+at the start")),
    ("on_another_floor", re.compile(r"(\d+)\s+on another floor")),
    ("no_view_point", re.compile(r"(\d+)\s+with no view point")),
)


def _rule_counts(reason: str) -> Dict[str, int]:
    return {name: int(m.group(1)) for name, pat in RULE_PATTERNS
            for m in [pat.search(reason or "")] if m}


def aggregate(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pool `summary.json` records into a yield, per scene and per rejection rule."""
    per_scene: List[Dict[str, Any]] = []
    built = skipped = 0
    rules: Dict[str, int] = {}
    unattributed = 0

    for s in summaries:
        n_built = int(s.get("n_episodes") or 0)
        skips = list(s.get("skipped") or [])
        built += n_built
        skipped += len(skips)
        scene_rules: Dict[str, int] = {}
        for entry in skips:
            counts = _rule_counts(str(entry.get("reason") or ""))
            if not counts:
                unattributed += 1
            for name, n in counts.items():
                scene_rules[name] = scene_rules.get(name, 0) + n
                rules[name] = rules.get(name, 0) + n
        offered = n_built + len(skips)
        per_scene.append({
            "scene": s.get("scene") or "?",
            "built": n_built,
            "skipped": len(skips),
            "offered": offered,
            "yield": (n_built / offered) if offered else None,
            "rules": scene_rules,
        })

    offered = built + skipped
    return {
        "n_scenes": len(summaries),
        "built": built,
        "skipped": skipped,
        "offered": offered,
        # None, not 0.0, when nothing was offered: a yield of zero and no data are
        # different claims, and only one of them is a measurement.
        "yield": (built / offered) if offered else None,
        "rules": rules,
        "unattributed_skips": unattributed,
        "per_scene": sorted(per_scene, key=lambda r: r["scene"]),
    }


def format_report(agg: Mapping[str, Any]) -> str:
    lines = ["scene                     built  skipped   yield  rules",
             "-" * 72]
    for row in agg["per_scene"]:
        y = row["yield"]
        rules = ", ".join("{} {}".format(v, k) for k, v in sorted(row["rules"].items()))
        lines.append("{:<24} {:>5}  {:>7}  {:>6}  {}".format(
            row["scene"][:24], row["built"], row["skipped"],
            "n/a" if y is None else "{:.0%}".format(y), rules or "-"))
    lines.append("-" * 72)
    y = agg["yield"]
    lines.append("{:<24} {:>5}  {:>7}  {:>6}".format(
        "TOTAL ({} scenes)".format(agg["n_scenes"]), agg["built"], agg["skipped"],
        "n/a" if y is None else "{:.0%}".format(y)))
    if agg["rules"]:
        lines.append("")
        lines.append("rejections by rule (objects rejected, summed over episodes):")
        for name, n in sorted(agg["rules"].items(), key=lambda kv: -kv[1]):
            lines.append("  {:<20} {}".format(name, n))
    if agg["unattributed_skips"]:
        lines.append("")
        lines.append("  {} skip(s) whose reason did not parse — the per-rule totals "
                     "below-count by that much".format(agg["unattributed_skips"]))
    return "\n".join(lines)


def load_summaries(root: str) -> List[Dict[str, Any]]:
    """Every `summary.json` under `root`, one level down or at the root itself."""
    base = pathlib.Path(root)
    found = sorted(base.glob("*/summary.json")) + sorted(base.glob("summary.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in found]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="a sweep directory holding <scene>/summary.json")
    parser.add_argument("--json", action="store_true", help="emit the aggregate as JSON")
    args = parser.parse_args(argv)

    summaries = load_summaries(args.root)
    if not summaries:
        print("no summary.json under {} — did the runs write one? (a run from before "
              "summary.json landed will not have)".format(args.root))
        return 2
    agg = aggregate(summaries)
    print(json.dumps(agg, indent=2) if args.json else format_report(agg))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
