"""Writes the `probe.json` that `savnce_gate.py` judges (ADR-0015).

Two subcommands, because the facts arrive at two different times:

  `pre`   before the eval — the env, the data paths, the checkpoint's contents, and one
          real audio render. Written FIRST and unconditionally, so that a run which dies
          mid-eval still leaves a probe the gate can judge. A crashed run must produce a
          red gate, not a missing file that reads as "not run yet".
  `post`  after the eval — the wall clock, and the throughput derived from it.

Nothing here parses a log line. Their trainer already dumps per-episode stats to
`<model-dir>/tb/<split>_stats_<seed>.json` (`ppo_trainer.py:995-999`), so the gate
recomputes their table from per-episode evidence instead of scraping a mean.
"""

import argparse
import gzip
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence

from earshot.tools.savnce_gate import REQUIRED_CKPT_SUBMODULES
from earshot.tools.savnce_verify import find_scene, render_audio

SECONDS_PER_HOUR = 3600.0


def count_episodes(split_root: pathlib.Path) -> int:
    """Episodes in a habitat-style split: the top-level file plus every content shard."""
    total = 0
    candidates: List[pathlib.Path] = []
    top = split_root / "{}.json.gz".format(split_root.name)
    if top.is_file():
        candidates.append(top)
    content = split_root / "content"
    if content.is_dir():
        candidates.extend(sorted(content.glob("*.json.gz")))
    for path in candidates:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        episodes = payload.get("episodes") if isinstance(payload, dict) else None
        if isinstance(episodes, list):
            total += len(episodes)
    return total


def inspect_checkpoint(path: pathlib.Path) -> Dict[str, object]:
    """Did it load, and does it carry the submodules the eval will otherwise randomise?"""
    if not path.is_file():
        return {"ckpt_loaded": False, "ckpt_missing_submodules": list(REQUIRED_CKPT_SUBMODULES)}
    try:
        import torch

        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — the message is the deliverable
        sys.stderr.write("checkpoint load failed: {}: {}\n".format(type(exc).__name__, exc))
        return {"ckpt_loaded": False, "ckpt_missing_submodules": list(REQUIRED_CKPT_SUBMODULES)}
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        return {"ckpt_loaded": False, "ckpt_missing_submodules": list(REQUIRED_CKPT_SUBMODULES)}
    keys = "\n".join(state.keys())
    missing = [name for name in REQUIRED_CKPT_SUBMODULES if name not in keys]
    return {"ckpt_loaded": True, "ckpt_missing_submodules": missing}


def probe_pre(args: argparse.Namespace) -> int:
    run_dir = pathlib.Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path(args.data_root)

    probe: Dict[str, object] = {
        "split": args.split,
        "requested_episodes": args.episodes,
        "stats_file": args.stats_file,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    missing: List[str] = []
    scenes_dir = data_root / "scene_datasets" / "mp3d"
    split_root = data_root / "datasets" / "savnce_dataset" / "mp3d" / "v1" / args.split
    checkpoint = pathlib.Path(args.ckpt)
    for label, path in (("scenes_dir", scenes_dir), ("split_root", split_root), ("checkpoint", checkpoint)):
        if not path.exists():
            missing.append("{}={}".format(label, path))
    probe["missing_paths"] = missing
    probe["episodes_available"] = count_episodes(split_root) if split_root.is_dir() else 0

    probe.update(inspect_checkpoint(checkpoint))

    failures: List[str] = []
    try:
        import numpy
        import quaternion  # noqa: F401  must precede habitat_sim (issue #1813)

        import habitat_sim
        import habitat_sim.sensor

        layout = getattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType", None)
        probe["habitat_sim_audio_capable"] = bool(layout is not None and hasattr(layout, "Binaural"))
        scene = find_scene(data_root)
        if scene is None:
            failures.append("no .glb under {}".format(data_root))
        else:
            loudest = render_audio(habitat_sim, numpy, scene, failures)
            if loudest is not None:
                probe["max_abs_audio"] = loudest
            # The render used a non-default sensor uuid, so reaching an observation at
            # all IS the patch working. A KeyError leaves max_abs_audio absent, and the
            # gate reads that as NOT_RUN rather than as a pass.
            probe["multi_audio_sensor_patch"] = loudest is not None
    except Exception as exc:  # noqa: BLE001
        failures.append("{}: {}".format(type(exc).__name__, exc))
        probe.setdefault("habitat_sim_audio_capable", False)

    probe["pre_failures"] = failures
    (run_dir / "probe.json").write_text(json.dumps(probe, indent=2, sort_keys=True), encoding="utf-8")
    for problem in failures:
        sys.stderr.write("  probe: {}\n".format(problem))
    return 0


def probe_post(args: argparse.Namespace) -> int:
    run_dir = pathlib.Path(args.run_dir)
    probe_path = run_dir / "probe.json"
    if not probe_path.is_file():
        sys.stderr.write("FATAL: {} absent — `probe pre` never ran\n".format(probe_path))
        return 2
    probe = json.loads(probe_path.read_text(encoding="utf-8"))

    wall = float(args.wall_clock_s)
    probe["wall_clock_s"] = wall

    stats: Dict[str, Dict[str, float]] = {}
    stats_path = run_dir / str(probe.get("stats_file", ""))
    if stats_path.is_file():
        try:
            raw = json.loads(stats_path.read_text(encoding="utf-8"))
            stats = {key: value for key, value in raw.items() if isinstance(value, dict)}
        except ValueError:
            stats = {}

    completed = len(stats)
    if wall > 0.0:
        probe["episodes_per_hour"] = completed / (wall / SECONDS_PER_HOUR)
        # `na` is their per-episode action count. Present, this gives simulator fps,
        # the number comparable to SAVN-CE's own reported ~200 fps. Absent, no guess.
        actions = [value.get("na") for value in stats.values()]
        if actions and all(isinstance(count, (int, float)) for count in actions):
            probe["fps"] = float(sum(actions)) / wall
    probe["completed_episodes"] = completed

    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    pre = sub.add_parser("pre", help="write probe.json before the eval")
    pre.add_argument("--run-dir", required=True)
    pre.add_argument("--data-root", required=True)
    pre.add_argument("--ckpt", required=True)
    pre.add_argument("--split", required=True)
    pre.add_argument("--episodes", type=int, required=True)
    pre.add_argument("--stats-file", required=True, help="path to their stats json, relative to the run dir")
    pre.set_defaults(func=probe_pre)

    post = sub.add_parser("post", help="merge the wall clock and throughput in")
    post.add_argument("--run-dir", required=True)
    post.add_argument("--wall-clock-s", required=True)
    post.set_defaults(func=probe_post)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
