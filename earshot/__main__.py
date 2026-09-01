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
import dataclasses
import sys
from typing import Optional, Sequence

from earshot.audio.clips import ANOMALY_CLASSES
from earshot.audio.config import WindowPolicy
from earshot.config import (
    CastPolicy,
    ClimbRule,
    Detector,
    IrPolicy,
    LateralCue,
    Localization,
    RunConfig,
)

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
        help="pin the step the anomaly source starts PLAYING (§2.5: not when it is "
        "heard). Omit to derive one per episode from its own start-to-goal distance, "
        "which is what keeps the onset inside the find it interrupts",
    )
    parser.add_argument(
        "--sounding-policy",
        choices=[policy.value for policy in WindowPolicy],
        default=defaults.sounding_policy.value,
        help="how long the source sounds before the offset step (ADR-0017). "
        "`continuous` is the CONTROL ARM: the source never stops, which is the "
        "pre-ADR-0017 behaviour, so a windowed run's funnel delta has an arm to be "
        "measured against. `fixed_steps` / `budget_fraction` / `drawn` all close it",
    )
    parser.add_argument(
        "--sounding-steps",
        type=int,
        default=defaults.sounding_steps,
        help="the FIXED_STEPS duration. The default is PROVISIONAL and has no sweep "
        "behind it; it is set generously because a window that closes before the agent "
        "is in earshot is silent attrition rather than a harder task",
    )
    parser.add_argument(
        "--sounding-budget-fraction",
        type=float,
        default=defaults.sounding_budget_fraction,
        help="the BUDGET_FRACTION duration, as a fraction of --max-steps",
    )
    parser.add_argument(
        "--sounding-draw-steps",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(defaults.sounding_draw_steps),
        help="the DRAWN duration's inclusive range, drawn per episode as a pure "
        "function of (--seed, episode index)",
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
        "--climb-rule",
        choices=[arm.value for arm in ClimbRule],
        default=defaults.climb_rule.value,
        help="live (the energy climb steers INVESTIGATE) or off (the climb is never "
        "consulted, so the agent runs the scan/cast cycle alone) — ADR-0018's matrix",
    )
    parser.add_argument(
        "--lateral-cue",
        choices=[arm.value for arm in LateralCue],
        default=defaults.lateral_cue.value,
        help="live (the interaural sign steers turns) or off (the sign is treated as "
        "ambiguous, so the turn decision falls to its zero/absent default)",
    )
    parser.add_argument(
        "--cast-policy",
        choices=[arm.value for arm in CastPolicy],
        default=defaults.cast_policy.value,
        help="cast (a leg is walked once the climb goes dead) or scan_only (every dead "
        "step turns instead, the pre-`eps-1` control arm)",
    )
    parser.add_argument(
        "--ir-policy",
        choices=[arm.value for arm in IrPolicy],
        default=defaults.ir_policy.value,
        help="full (the room's real IR) or anechoic (every rendered IR is replaced by "
        "a flat, reverberation-free stand-in at all three render sites)",
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
    parser.add_argument("--min-source-start-sep-m", type=float,
                        default=defaults.min_source_start_sep_m)
    parser.add_argument(
        "--audio-step-ceiling-s",
        type=float,
        default=defaults.audio_step_ceiling_s,
        help="smoke criterion 7's per-step audio ceiling; recorded, asserted by the smoke",
    )
    parser.add_argument(
        "--indirect-ray-count",
        type=int,
        default=defaults.audio.indirect_ray_count,
        help="override the acoustics preset's indirectRayCount (default 500). The one "
             "knob that trades RENDER ACCURACY for speed: two runs of the same scene at "
             "500 disagreed on 4 of 20 episode outcomes. Roughly linear in cost",
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
        t_anom=None if args.t_anom is None else int(args.t_anom),
        sounding_policy=WindowPolicy(args.sounding_policy),
        sounding_steps=int(args.sounding_steps),
        sounding_budget_fraction=float(args.sounding_budget_fraction),
        # `nargs=2` yields a LIST, and `RunConfig` is compared by equality against its
        # own defaults in `tests/mac/test_config.py` — a list here fails that for a
        # reason that has nothing to do with the value.
        sounding_draw_steps=tuple(int(value) for value in args.sounding_draw_steps),
        seed=int(args.seed),
        localization=Localization(args.localization),
        detector=Detector(args.detector),
        climb_rule=ClimbRule(args.climb_rule),
        lateral_cue=LateralCue(args.lateral_cue),
        cast_policy=CastPolicy(args.cast_policy),
        ir_policy=IrPolicy(args.ir_policy),
        anomaly_class=args.anomaly_class,
        anomaly_clip=args.anomaly_clip,
        clap=bool(args.clap),
        min_source_sep_m=float(args.min_source_sep_m),
        max_source_dy_m=float(args.max_source_dy_m),
        min_source_start_sep_m=float(args.min_source_start_sep_m),
        audio_step_ceiling_s=float(args.audio_step_ceiling_s),
        audio=dataclasses.replace(
            RunConfig(run_dir="").audio,
            indirect_ray_count=(None if args.indirect_ray_count is None
                                else int(args.indirect_ray_count)),
        ),
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
