"""The acceptance gate for a SAVN-CE reproduced-reference run (ADR-0015).

`earshot.task.smoke` judges an earshot run against the nine acceptance criteria. This is
the same shape for a run of somebody else's code: seven criteria, tallied over one run
directory, printed with their measurements, nonzero exit if any is not green.

Three of this repo's rules shape it, and each one is a scar:

  * **A criterion that could not be evaluated is never green.** A missing probe key
    reads `NOT_RUN`, and `NOT_RUN` is red. A probe that skipped once reported success.
  * **A capability is exercised, never proxied.** Criterion 3 does not check that a
    checkpoint file exists; it checks that the loaded state dict actually covers the
    goal-descriptor and memory-transformer submodules. `EVAL.USE_CKPT_CONFIG` is `True`
    by default and their released-checkpoint invocation turns three `pretrained` flags
    off, so "the eval ran" and "the eval ran the trained model" are genuinely different
    events, and only the second one produces a number worth reporting.
  * **Criterion 5 exists because of `anommxv`.** A zero-geometry audio context returns
    plausible-looking audio, and that fabricated audio invalidated a headline. Silence
    is therefore a hard failure, not a warning.

The judging is a pure function over two dicts so the Mac can exercise **both arms** of
every criterion (ADR-0014) without a simulator, a GPU, or MP3D. The box arm runs the
same function over a real run.

Aggregation mirrors `savnce_baselines/magnet/ppo/ppo_trainer.py:1001-1013` exactly,
including its four skipped keys: we recompute their table rather than scraping their
log lines, because they already dump per-episode stats to
`<model-dir>/tb/<split>_stats_<seed>.json` and a per-episode file is evidence a mean is
not.
"""

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

GREEN = "GREEN"
RED = "RED"
NOT_RUN = "NOT_RUN"

# `ppo_trainer.py:1003` skips these four when it aggregates. Mirrored rather than
# reinvented: a table that aggregates a different key set is not their table.
EXCLUDED_STAT_KEYS: Tuple[str, ...] = (
    "audio_duration",
    "gt_na",
    "descriptor_pred_gt",
    "view_point_goals",
)

# The submodules whose weights the released checkpoint must actually carry. Their
# README's released-checkpoint command sets `RL.PPO.GOAL_DESCRIPTOR.use_pretrained
# False` and `RL.PPO.SCENE_MEMORY_TRANSFORMER.use_pretrained False`, which stops the
# separately-pretrained weights loading — so if the main checkpoint does not carry them
# either, the eval scores a partly random model and says nothing about it.
#
# The two names are read off their policy, not guessed: `savnce_baselines/magnet/ppo/
# policy.py` builds `self.goal_descriptor_encoder` and `self.smt_state_encoder`, and
# `ppo_trainer.py:157` saves `self.agent.state_dict()` under the key `state_dict`.
REQUIRED_CKPT_SUBMODULES: Tuple[str, ...] = ("goal_descriptor_encoder", "smt_state_encoder")

# arXiv 2603.19660, Table 1, clean environments, MAGNet. The paper reports percentages;
# the per-episode stats are 0-1 fractions, so the comparison scales by this.
PUBLISHED_PERCENT: Dict[str, float] = {"success": 37.7, "spl": 32.9}
PERCENT = 100.0

# Pre-registered before the first run (ADR-0015). Reproduced means inside this band on
# the arm the paper reports: `test`, 1000 episodes.
ACCEPTANCE_BAND_SR_POINTS = 2.0
ACCEPTANCE_SPLIT = "test"
ACCEPTANCE_EPISODES = 1000


class Criterion(NamedTuple):
    """One judged criterion. `measurement` is printed whether green or red."""

    number: int
    name: str
    verdict: str
    measurement: str

    @property
    def is_green(self) -> bool:
        return self.verdict == GREEN


def _finite(value: object) -> Optional[float]:
    """A float, or None if the value is absent, non-numeric, NaN or infinite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flag(probe: Dict[str, object], key: str) -> Optional[bool]:
    value = probe.get(key)
    return value if isinstance(value, bool) else None


def _string_list(probe: Dict[str, object], key: str) -> Optional[List[str]]:
    value = probe.get(key)
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def aggregate(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean of every numeric per-episode stat, their key set and their denominator.

    Pure. `stats` is their `<split>_stats_<seed>.json`: one entry per episode, keyed
    `"<scene>,<episode_id>"`.
    """
    if not stats:
        return {}
    episodes = list(stats.values())
    keys = [
        key
        for key in episodes[0].keys()
        if key not in EXCLUDED_STAT_KEYS and _finite(episodes[0][key]) is not None
    ]
    totals: Dict[str, float] = {}
    for key in keys:
        values = [_finite(episode.get(key)) for episode in episodes]
        if any(value is None for value in values):
            continue
        totals[key] = sum(value for value in values if value is not None) / len(episodes)
    return totals


def judge(
    probe: Dict[str, object],
    stats: Optional[Dict[str, Dict[str, float]]],
    requested_episodes: int,
) -> List[Criterion]:
    """The seven criteria. Pure: no filesystem, no imports, no simulator.

    `stats` is None when the eval produced no per-episode file at all, which is the
    honest state after a crash and must not be confused with an empty one.
    """
    criteria: List[Criterion] = []

    # 1 - the environment is the one the run needs, patch included.
    audio_capable = _flag(probe, "habitat_sim_audio_capable")
    patched = _flag(probe, "multi_audio_sensor_patch")
    if audio_capable is None or patched is None:
        criteria.append(Criterion(1, "env", NOT_RUN, "env probe absent"))
    else:
        verdict = GREEN if (audio_capable and patched) else RED
        criteria.append(
            Criterion(
                1,
                "env",
                verdict,
                "audio_capable={} multi_sensor_patch={}".format(audio_capable, patched),
            )
        )

    # 2 - every path the config names resolves, and there is at least one episode.
    missing = _string_list(probe, "missing_paths")
    available = _finite(probe.get("episodes_available"))
    if missing is None or available is None:
        criteria.append(Criterion(2, "data", NOT_RUN, "data probe absent"))
    else:
        verdict = GREEN if (not missing and available >= 1) else RED
        detail = "missing_paths={} episodes_available={:.0f}".format(len(missing), available)
        if missing:
            detail += " first_missing={}".format(missing[0])
        criteria.append(Criterion(2, "data", verdict, detail))

    # 3 - the checkpoint loaded AND covers the submodules, not merely exists.
    loaded = _flag(probe, "ckpt_loaded")
    absent_submodules = _string_list(probe, "ckpt_missing_submodules")
    if loaded is None or absent_submodules is None:
        criteria.append(Criterion(3, "ckpt", NOT_RUN, "checkpoint probe absent"))
    else:
        verdict = GREEN if (loaded and not absent_submodules) else RED
        detail = "loaded={} missing_submodules={}".format(
            loaded, ",".join(absent_submodules) if absent_submodules else "none"
        )
        criteria.append(Criterion(3, "ckpt", verdict, detail))

    # 4 - a short run is red. Not a warning, not a footnote.
    if stats is None:
        criteria.append(Criterion(4, "episodes", NOT_RUN, "no per-episode stats file"))
    else:
        completed = len(stats)
        verdict = GREEN if completed == requested_episodes else RED
        criteria.append(
            Criterion(
                4,
                "episodes",
                verdict,
                "completed={} requested={}".format(completed, requested_episodes),
            )
        )

    # 5 - the audio channel carried something. See the module docstring.
    loudest = _finite(probe.get("max_abs_audio"))
    if loudest is None:
        criteria.append(Criterion(5, "audio", NOT_RUN, "no audio observation recorded"))
    else:
        verdict = GREEN if loudest > 0.0 else RED
        criteria.append(Criterion(5, "audio", verdict, "max_abs_audio={:.6g}".format(loudest)))

    # 6 - the headline numbers exist and are numbers.
    table = aggregate(stats) if stats else {}
    success = _finite(table.get("success"))
    spl = _finite(table.get("spl"))
    if not stats:
        criteria.append(Criterion(6, "metrics", NOT_RUN, "no episodes to aggregate"))
    elif success is None or spl is None:
        criteria.append(
            Criterion(
                6,
                "metrics",
                RED,
                "success={} spl={}".format(table.get("success"), table.get("spl")),
            )
        )
    else:
        criteria.append(
            Criterion(
                6,
                "metrics",
                GREEN,
                "SR={:.1f} SPL={:.1f} (percent)".format(success * PERCENT, spl * PERCENT),
            )
        )

    # 7 - the number that decides whether the full 1000-episode run is affordable.
    #
    # Episodes per hour, not fps, is the headline: it is always computable from the
    # wall clock and the completed count, and it answers the actual question ("can we
    # afford the full arm?") in one step. Simulator fps rides along when their `na`
    # (number of actions) metric is present, because it is the number comparable to
    # SAVN-CE's own reported ~200 fps on a 128-thread box.
    per_hour = _finite(probe.get("episodes_per_hour"))
    fps = _finite(probe.get("fps"))
    if per_hour is None:
        criteria.append(Criterion(7, "throughput", NOT_RUN, "wall clock not recorded"))
    else:
        verdict = GREEN if per_hour > 0.0 else RED
        detail = "episodes_per_hour={:.1f}".format(per_hour)
        detail += " fps={:.2f}".format(fps) if fps is not None else " fps=unavailable(no 'na' metric)"
        if per_hour > 0.0:
            detail += " -> {:.1f} h for {} episodes".format(
                ACCEPTANCE_EPISODES / per_hour, ACCEPTANCE_EPISODES
            )
        criteria.append(Criterion(7, "throughput", verdict, detail))

    return criteria


def acceptance(
    stats: Optional[Dict[str, Dict[str, float]]], split: str, requested_episodes: int
) -> Tuple[bool, str]:
    """The pre-registered band from ADR-0015. Applicable only on the reported arm.

    Returns (applicable, sentence). A smoke is never judged against it, because a
    2-scene number is indefensible for the same reason `val_mini` never entered Table 1.
    """
    if split != ACCEPTANCE_SPLIT or requested_episodes != ACCEPTANCE_EPISODES:
        return (
            False,
            "acceptance not applicable: pre-registered on split={} at {} episodes, this run is {} at {}".format(
                ACCEPTANCE_SPLIT, ACCEPTANCE_EPISODES, split, requested_episodes
            ),
        )
    table = aggregate(stats) if stats else {}
    success = _finite(table.get("success"))
    if success is None:
        return False, "acceptance not evaluable: no success metric in the per-episode stats"
    measured = success * PERCENT
    published = PUBLISHED_PERCENT["success"]
    delta = measured - published
    inside = abs(delta) <= ACCEPTANCE_BAND_SR_POINTS
    return (
        inside,
        "SR {:.1f} vs published {:.1f}, delta {:+.1f} points, band +/-{:.1f} -> {}".format(
            measured, published, delta, ACCEPTANCE_BAND_SR_POINTS, "REPRODUCED" if inside else "MISS"
        ),
    )


def render(criteria: Sequence[Criterion], acceptance_line: str) -> str:
    """The printed tally. Box tests print their measurements."""
    width = max(len(criterion.name) for criterion in criteria)
    lines = ["", "SAVN-CE reproduced reference - acceptance gate", ""]
    for criterion in criteria:
        lines.append(
            "  {:>1}  {:<{width}}  {:<7}  {}".format(
                criterion.number, criterion.name, criterion.verdict, criterion.measurement, width=width
            )
        )
    green = sum(1 for criterion in criteria if criterion.is_green)
    lines.append("")
    lines.append("  {}/{} green   (NOT_RUN counts as red)".format(green, len(criteria)))
    lines.append("  {}".format(acceptance_line))
    lines.append("")
    return "\n".join(lines)


def _load_json(path: pathlib.Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="runs/savnce-<tag>")
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    probe = _load_json(run_dir / "probe.json")
    if probe is None:
        sys.stderr.write(
            "FATAL: {}/probe.json is missing or unreadable. The eval driver writes it "
            "even when the eval itself dies, so its absence means the driver never "
            "started.\n".format(run_dir)
        )
        return 2

    split = str(probe.get("split", "unknown"))
    requested = int(_finite(probe.get("requested_episodes")) or 0)
    stats_path = probe.get("stats_file")
    stats: Optional[Dict[str, Dict[str, float]]] = None
    if isinstance(stats_path, str):
        raw = _load_json(run_dir / stats_path) if not pathlib.Path(stats_path).is_absolute() else _load_json(pathlib.Path(stats_path))
        if raw is not None:
            stats = {key: value for key, value in raw.items() if isinstance(value, dict)}

    criteria = judge(probe, stats, requested)
    _, acceptance_line = acceptance(stats, split, requested)
    sys.stdout.write(render(criteria, acceptance_line))

    (run_dir / "gate.json").write_text(
        json.dumps(
            {
                "criteria": [criterion._asdict() for criterion in criteria],
                "aggregate": aggregate(stats) if stats else {},
                "acceptance": acceptance_line,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0 if all(criterion.is_green for criterion in criteria) else 1


if __name__ == "__main__":
    raise SystemExit(main())
