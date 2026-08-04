"""``python -m earshot`` — argparse to ``RunConfig``, then the run.

ADR-0013 gives this file three jobs and no others: parse, compose, dispatch. Everything
it can decide is a flag with a default from the module configs, so the CLI surface is a
projection of ``RunConfig`` rather than a second place numbers live.

**There is no environment flag and no config file.** ADR-0008 removed the flag surface
the old tree had (``LTM_REALIZABLE_LOCALIZATION`` was read at the runner), and a
YAML/JSON layer was considered and rejected in ADR-0013 for roughly eight numbers and two
enums. What was kept instead is the record: ``task/runner.run`` writes the resolved
configuration into ``env_report.json`` beside the environment probes, so "what was this
run" is answerable from the run directory rather than from a shell history.

``assert_env()`` is called inside ``run()`` rather than here, which is a small deviation
from the ADR's one-line sketch of this file and it is deliberate: the assertion's report
has to reach the artefact writer, and a second call site would run the probes twice —
including CLAP's 153.5 M-parameter construction.

Two flags are worth reading twice:

``--overwrite`` exists because ``report/artifacts.py`` refuses to overwrite by default,
and that refusal is an answer to a real incident (committed run directories holding a
different run's data). The flag is how someone says "yes, replace it" out loud.

``--clap`` costs a model load and changes what the report can say: without it the anomaly
verdict is ``None``, which ``step_controller`` reads as "nothing conditioned this, so any
onset interrupts", and ``anomaly_class`` stays null rather than being copied off the
dataset. The smoke runs without it (§4.3: one sound, the anomaly by construction).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from earshot.audio.clips import ANOMALY_CLASSES
from earshot.config import Detector, Localization, RunConfig

__all__ = ["build_parser", "config_from_args", "main"]


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Defaults come from ``RunConfig``, so there is one home for each number."""
    defaults = RunConfig(run_dir="")
    parser = argparse.ArgumentParser(
        prog="python -m earshot",
        description=(
            "Run the anomaly-response task on live in-sim SoundSpaces 2.0 audio. "
            "Linux + CUDA + the `ss2` env only."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="where the artefacts land: runs/<tag>/{env_report.json,episodes/}",
    )
    parser.add_argument("--split", default=defaults.split, help="ObjectNav split")
    parser.add_argument(
        "--data-root",
        default=defaults.data_root,
        help="root the dataset and mesh directories are searched under",
    )
    parser.add_argument(
        "--scene",
        default=defaults.scene,
        help="scene label; empty means the first in the split whose mesh is present",
    )
    parser.add_argument(
        "--category", default=None, help="restrict the PRIMARY goal to one category"
    )
    parser.add_argument("--n-episodes", type=int, default=defaults.n_episodes)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument(
        "--t-anom",
        type=int,
        default=defaults.t_anom,
        help="the step the anomaly source starts PLAYING (§2.5: not when it is heard)",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--localization",
        choices=[arm.value for arm in Localization],
        default=defaults.localization.value,
        help="realizable (the smoke, §8) or oracle (the bisection tool)",
    )
    parser.add_argument(
        "--detector",
        choices=[arm.value for arm in Detector],
        default=defaults.detector.value,
        help="oracle (the smoke) or caption (R2; needs earshot/vlm.py, not yet built)",
    )
    parser.add_argument(
        "--anomaly-class",
        choices=list(ANOMALY_CLASSES),
        default=defaults.anomaly_class,
    )
    parser.add_argument(
        "--anomaly-clip", default=None, help="explicit .wav path, overriding the class"
    )
    parser.add_argument(
        "--clap",
        action="store_true",
        help="load CLAP and classify the heard clip (0.7 GiB; the smoke does not need it)",
    )
    parser.add_argument("--min-source-sep-m", type=float, default=defaults.min_source_sep_m)
    parser.add_argument("--max-source-dy-m", type=float, default=defaults.max_source_dy_m)
    parser.add_argument(
        "--audio-step-ceiling-s",
        type=float,
        default=defaults.audio_step_ceiling_s,
        help="smoke criterion 7's per-step audio ceiling; recorded, asserted by the smoke",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace artefacts already in --run-dir (they are refused by default)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """``Namespace`` to ``RunConfig``. Pure, so the mapping is unit-testable.

    Separated from ``main`` for the reason ``env_check.judge`` is separated from its
    probes: the part that can be checked without a box is the part worth checking, and
    a CLI whose flags quietly stop reaching the config is a class of bug that only shows
    up as a run that ignored what it was told.
    """
    return RunConfig(
        run_dir=args.run_dir,
        split=args.split,
        data_root=args.data_root,
        scene=args.scene,
        category=args.category,
        n_episodes=int(args.n_episodes),
        max_steps=int(args.max_steps),
        t_anom=int(args.t_anom),
        seed=int(args.seed),
        localization=Localization(args.localization),
        detector=Detector(args.detector),
        anomaly_class=args.anomaly_class,
        anomaly_clip=args.anomaly_clip,
        clap=bool(args.clap),
        min_source_sep_m=float(args.min_source_sep_m),
        max_source_dy_m=float(args.max_source_dy_m),
        audio_step_ceiling_s=float(args.audio_step_ceiling_s),
        overwrite=bool(args.overwrite),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, build, run. Returns a process exit code.

    ``task/runner.py`` is imported inside this function, not at module scope: it reaches
    ``sim/world.py`` and therefore habitat-sim, and ``build_parser`` /
    ``config_from_args`` have to stay importable on a machine that cannot load the
    simulator so the Mac suite can check them.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = config_from_args(args)

    from earshot.task.runner import run

    def say(message: str) -> None:
        # In-thread, between steps, and flushed — which ADR-0013 established is safe
        # against the guard's fd capture. What is forbidden is a CONCURRENT writer, so
        # this deliberately starts no thread and no progress bar.
        print(message, flush=True)

    run(config, progress=say)
    return 0


if __name__ == "__main__":
    sys.exit(main())
