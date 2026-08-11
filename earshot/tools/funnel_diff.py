"""Two sweeps, scene by scene, funnel stage by funnel stage — the subtraction that scores a change.

    python -m earshot.tools.funnel_diff runs/yield-2 runs/eps-1

**The measurement a run report cannot make.** A sweep prints its own funnel and nothing
else, so `eps-1` came back GREEN at 33% source-reached with no arm beside it and no way to
say whether the estimator fix moved anything. CLAUDE.md's rule is that a claim about a
change needs the arm where the change is absent; this is the arithmetic for reading the
two arms together, and it costs nothing because both sweeps already wrote `summary.json`.

**It refuses to subtract things that are not comparable.** Two sweeps are only an arm-pair
if they built the same episodes, and `summary.json` carries exactly enough to check that:
a scene present in one and missing from the other, or built to a different episode count,
is reported as UNPAIRED and left out of the total. A per-scene delta over a different
episode set is not a delta, and pooling one into a headline is how a builder change gets
read as a controller result.

**Pairing here is at the scene, not the episode.** `summary.json` holds counts, not
outcomes, so this cannot tell which episodes flipped — and `detour-1` measured ~20% of
them flipping between identical runs on render non-determinism alone. That bound is
printed with the total rather than left for a reader to remember. A genuinely paired test
needs the per-episode funnel stage out of each run's `audit.json`, which is a different
tool and a bigger read.

``diff()`` is pure, so the arithmetic is Mac-testable against injected records while the
runs that feed it need a GPU — the same split `yield_report` and `detour_report` are built
on. No verdict: it prints the deltas and the reader decides.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from earshot.report.audit import FunnelStage
from earshot.tools.yield_report import load_summaries

__all__ = ["HEADLINE_STAGE", "diff", "format_report", "main", "two_sided_exact_binomial"]

# The stage the headline delta is taken on. Stage 5 is arrival at the anomaly source —
# what a controller change is trying to move, and the numerator of Anomaly-response SR.
# Every stage is reported; this one gets the per-scene column.
HEADLINE_STAGE = FunnelStage.SOURCE_REACHED

# `detour-1` ran the same scene twice under the same configuration and 4 of 20 episodes
# changed arm — the renderer is non-deterministic and there is no seed for it.
#
# **This is the flip RATE, and the net delta is not compared against it.** The first
# version of this file printed "a delta near this many episodes is not a result", which is
# far too conservative and would have argued away a real one: flips go BOTH directions, so
# under a null of no effect each episode contributes -1, 0 or +1 with variance `FLIP_RATE`,
# and the net has mean zero and SD `sqrt(FLIP_RATE * built)`. At 365 episodes that is 8.5,
# not 73. The count is still printed, as the scale of the churn underneath a net.
FLIP_RATE = 0.20


def _by_scene(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Keyed by scene label. A sweep writes one directory per scene, so a repeat means
    two runs were pooled under one tag, which the yield report has been fooled by before —
    it is raised rather than silently overwritten."""
    out: Dict[str, Dict[str, Any]] = {}
    for summary in summaries:
        scene = str(summary.get("scene") or "")
        if scene in out:
            raise ValueError(
                "scene {!r} appears twice in one sweep — two runs pooled under one tag, "
                "so neither one's funnel is that scene's funnel".format(scene)
            )
        out[scene] = dict(summary)
    return out


def two_sided_exact_binomial(extreme: int, n: int) -> Optional[float]:
    """The exact two-sided p for `extreme` of `n` independent outcomes falling one way
    under a fair coin. ``None`` when nothing moved, because no p exists for zero trials.

    Shared with `episode_diff`'s McNemar rather than reimplemented there: the sign test
    over scenes and the McNemar over episodes are the SAME arithmetic on different units,
    and two copies of an exact-tail sum is a place for them to drift apart. It is exact
    rather than normal-approximated because these `n` are small — 16 scenes moved in the
    largest comparison this repo has run, where a normal approximation is not honest.
    """
    if n <= 0:
        return None
    extreme = max(int(extreme), n - int(extreme))
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _sign_test(deltas: Sequence[int]) -> Dict[str, Any]:
    """Scenes up, down, flat, and the exact two-sided sign-test p. Pure.

    **The statistic this design can actually support.** Episodes inside one scene share a
    room, a source and a renderer, so they are not independent draws and an aggregate rate
    over them carries a noise model nobody has fitted. Which DIRECTION each scene moved is
    a much weaker thing to assume about, and 15 of 16 scenes falling one way is decisive
    where the same total concentrated in three scenes would not be.

    Ties are dropped rather than split, which is the conservative convention: a scene that
    did not move is no evidence either way, and counting it as half a success would
    manufacture confidence out of scenes where nothing happened.
    """
    up = sum(1 for d in deltas if d > 0)
    down = sum(1 for d in deltas if d < 0)
    flat = sum(1 for d in deltas if d == 0)
    n = up + down
    return {
        "up": up,
        "down": down,
        "flat": flat,
        "n": n,
        "p_value": two_sided_exact_binomial(max(up, down), n),
    }


def _stage_counts(summary: Mapping[str, Any]) -> Dict[str, int]:
    funnel = summary.get("funnel") or {}
    return {stage.name: int(funnel.get(stage.name, 0)) for stage in FunnelStage}


def diff(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    stage: FunnelStage = HEADLINE_STAGE,
) -> Dict[str, Any]:
    """Pool two sweeps' `summary.json` records into a per-scene and per-stage delta. Pure.

    ``before`` is the arm without the change. The sign convention follows from that:
    positive is the change helping.
    """
    left, right = _by_scene(before), _by_scene(after)
    paired: List[Dict[str, Any]] = []
    unpaired: List[Dict[str, Any]] = []

    for scene in sorted(set(left) | set(right)):
        a, b = left.get(scene), right.get(scene)
        if a is None or b is None:
            unpaired.append({
                "scene": scene,
                "reason": "only in {}".format("after" if a is None else "before"),
                "before_built": None if a is None else int(a.get("n_episodes") or 0),
                "after_built": None if b is None else int(b.get("n_episodes") or 0),
            })
            continue
        a_built, b_built = int(a.get("n_episodes") or 0), int(b.get("n_episodes") or 0)
        if a_built != b_built:
            # The builder moved, so the two are different episode sets wearing one scene
            # label. ADR-0015's rule, applied: a pre/post claim needs both arms built the
            # same way, and `at_the_start` has already invalidated one such comparison.
            unpaired.append({
                "scene": scene,
                "reason": "built {} against {} — different episode sets".format(
                    a_built, b_built),
                "before_built": a_built,
                "after_built": b_built,
            })
            continue
        a_stages, b_stages = _stage_counts(a), _stage_counts(b)
        paired.append({
            "scene": scene,
            "built": a_built,
            "before": a_stages[stage.name],
            "after": b_stages[stage.name],
            "delta": b_stages[stage.name] - a_stages[stage.name],
            "stages": {
                name: (a_stages[name], b_stages[name]) for name in a_stages
            },
        })

    built = sum(int(row["built"]) for row in paired)
    before_total = sum(int(row["before"]) for row in paired)
    after_total = sum(int(row["after"]) for row in paired)
    stages: List[Dict[str, Any]] = []
    for name in (s.name for s in FunnelStage):
        a_sum = sum(int(row["stages"][name][0]) for row in paired)
        b_sum = sum(int(row["stages"][name][1]) for row in paired)
        stages.append({"stage": name, "before": a_sum, "after": b_sum,
                       "delta": b_sum - a_sum})

    return {
        "stage": stage.name,
        "per_scene": paired,
        "unpaired": unpaired,
        "n_paired_scenes": len(paired),
        "built": built,
        "before": before_total,
        "after": after_total,
        "delta": after_total - before_total,
        # Rates, and `None` rather than 0.0 on an empty pairing: a delta of zero and a
        # delta that could not be computed are different answers to the question.
        "before_rate": (before_total / built) if built else None,
        "after_rate": (after_total / built) if built else None,
        "delta_rate": ((after_total - before_total) / built) if built else None,
        # What running the same code twice would produce. The COUNT is the churn; the SD
        # is what a net has to clear, and they differ by a square root — see FLIP_RATE.
        "flip_noise_episodes": FLIP_RATE * built,
        "delta_sd": math.sqrt(FLIP_RATE * built) if built else None,
        "delta_z": (
            (after_total - before_total) / math.sqrt(FLIP_RATE * built) if built else None
        ),
        # The test that needs no noise model at all: how the per-scene deltas fall. Scene
        # outcomes are independent draws in a way episodes within a scene are not, and a
        # consistent SIGN across scenes is evidence an aggregate cannot give — 15 of 16
        # scenes moving one way is decisive where the same total spread over 3 scenes
        # would not be.
        "sign_test": _sign_test([int(row["delta"]) for row in paired]),
        "stages": stages,
    }


def _fmt_rate(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:.1%}".format(value)


def _wrap(text: str) -> List[str]:
    """A prose paragraph at the table's width, indented to sit under it. The caveats are
    the part of this report that stops a number being quoted wrongly, and an unwrapped
    300-character line in a terminal is one a reader skims past."""
    return textwrap.wrap(text, width=72, initial_indent="  ", subsequent_indent="  ")


def format_report(agg: Mapping[str, Any], *, labels: Tuple[str, str] = ("before", "after")) -> str:
    before_label, after_label = labels
    stage = agg.get("stage")
    lines = [
        "{} — {} against {}".format(stage, before_label, after_label),
        "-" * 72,
        "{:<24} {:>6}  {:>9}  {:>9}  {:>7}".format(
            "scene", "built", before_label[:9], after_label[:9], "delta"),
    ]
    for row in agg["per_scene"]:
        lines.append("{:<24} {:>6}  {:>9}  {:>9}  {:>+7}".format(
            str(row["scene"])[:24], row["built"], row["before"], row["after"],
            row["delta"]))
    lines.append("-" * 72)
    lines.append("{:<24} {:>6}  {:>9}  {:>9}  {:>+7}".format(
        "TOTAL ({} scenes)".format(agg["n_paired_scenes"]), agg["built"],
        agg["before"], agg["after"], agg["delta"]))
    lines.append("{:<24} {:>6}  {:>9}  {:>9}  {:>7}".format(
        "", "", _fmt_rate(agg["before_rate"]), _fmt_rate(agg["after_rate"]),
        _fmt_rate(agg["delta_rate"])))

    lines.append("")
    lines.append("the whole ladder, over the paired scenes:")
    lines.append("  {:<22} {:>9}  {:>9}  {:>7}".format(
        "stage", before_label[:9], after_label[:9], "delta"))
    for row in agg["stages"]:
        lines.append("  {:<22} {:>9}  {:>9}  {:>+7}".format(
            row["stage"], row["before"], row["after"], row["delta"]))

    if agg["unpaired"]:
        lines.append("")
        lines.append("NOT SUBTRACTED — these scenes are not an arm-pair and are excluded")
        lines.append("from every total above:")
        for row in agg["unpaired"]:
            lines.append("  {:<24} {}".format(str(row["scene"])[:24], row["reason"]))

    sign = agg.get("sign_test") or {}
    if sign.get("n"):
        lines.append("")
        lines.append("  scenes: {} down, {} up, {} unchanged — sign test p = {:.4f}".format(
            sign["down"], sign["up"], sign["flat"], sign["p_value"]))
        lines.extend(_wrap(
            "The direction each scene moved, which assumes nothing about the renderer. "
            "Episodes inside one scene share a room and a source and are not independent; "
            "scenes are closer to it."))

    if agg["built"]:
        lines.append("")
        lines.extend(_wrap(
            "~{:.0f} episode(s) of this pairing would change arm between two runs of the "
            "SAME code — detour-1 measured a {:.0%} per-episode flip rate against a "
            "renderer that has no seed. Flips go BOTH ways, so what a net delta has to "
            "clear is their SD of {:.1f} episodes, not their count: this delta is "
            "{:+.1f} of those.".format(
                agg["flip_noise_episodes"], FLIP_RATE, agg["delta_sd"], agg["delta_z"])))
        lines.extend(_wrap(
            "These are scene-level counts, not paired episodes: summary.json records how "
            "many reached, never which. A per-episode test needs each run's audit.json."))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("before", help="the sweep WITHOUT the change: runs/<tag>")
    parser.add_argument("after", help="the sweep with it: runs/<tag>")
    parser.add_argument(
        "--stage", default=HEADLINE_STAGE.name,
        choices=[s.name for s in FunnelStage],
        help="which funnel stage the per-scene column reports (default {})".format(
            HEADLINE_STAGE.name))
    parser.add_argument("--json", action="store_true", help="emit the diff as JSON")
    args = parser.parse_args(argv)

    loaded = []
    for root in (args.before, args.after):
        summaries = load_summaries(root)
        if not summaries:
            print("no summary.json under {} — did the runs write one? (a run from before "
                  "summary.json landed will not have)".format(root))
            return 2
        loaded.append(summaries)

    agg = diff(loaded[0], loaded[1], stage=FunnelStage[args.stage])
    if args.json:
        print(json.dumps(agg, indent=2))
    else:
        labels = (pathlib.Path(args.before).name, pathlib.Path(args.after).name)
        print(format_report(agg, labels=labels))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
