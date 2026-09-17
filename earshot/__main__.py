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
from typing import Dict, Optional, Sequence

from earshot.audio.clips import ANOMALY_CLASSES, SOUNDING_CLASSES
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
from earshot.memory.consolidate import ImportanceWeights
from earshot.memory.store import MemoryCondition
from earshot.task.dream import DreamKnobs
from earshot.task.plan import PlanWeights

__all__ = ["build_parser", "config_from_args", "memory_kwargs_from_args", "main"]


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
        # The three carried emergency names, plus ADR-0018's bank of record. The bank is
        # what ADR-0022's matrix draws from: it spans FOUR anchor categories against the
        # emergency bank's one, which is what makes a recalled category a prediction
        # rather than a constant. Sorted so `--help` reads as a list rather than as two.
        choices=sorted(set(ANOMALY_CLASSES) | set(SOUNDING_CLASSES)),
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
    parser.add_argument(
        "--memory-condition",
        choices=[condition.value for condition in MemoryCondition],
        default=MemoryCondition.NONE.value,
        help="ADR-0018's matrix cell. `none` (default) runs with no memory arm at all, "
             "byte-identical to a build with no --memory-* flags. Any other value needs "
             "--memory-store and --clap: the store is queried with a CLAP embedding of "
             "the heard clip, and the recalled category is resolved through the scene "
             "under test's own prior-tour points where the prior pass reached them, "
             "falling back to its ObjectNav ground truth where it did not",
    )
    parser.add_argument(
        "--memory-store",
        default=None,
        help="the JSON file a prior pass wrote (`memory_build.dump_stores`); required "
             "when --memory-condition is not `none`",
    )
    parser.add_argument(
        "--memory-k",
        type=int,
        default=5,
        help="neighbours the semantic store votes over per recall (default 5); recorded "
             "on the audit, never silently defaulted at the vote itself",
    )
    parser.add_argument(
        "--memory-abstain-below",
        type=float,
        default=None,
        help="decline a recall whose vote scores below this mean cosine, recording "
             "`low_confidence` instead of a place. Default None = the vote always "
             "answers, which is every result measured before this flag existed. THE "
             "VALUE IS PER STORE: matrix-1's own distributions separated fully at "
             "0.7842, but that is a fact about that run's confidences and has to be "
             "re-measured (`matrix_audit` section E) rather than carried over",
    )
    parser.add_argument(
        "--memory-proposes",
        action="store_true",
        help="ADR-0026: the recalled place is emitted as a SECOND investigate candidate "
             "and eq. 26 chooses, instead of REPLACING the acoustic estimate outright "
             "(the default, and what every result before this flag was measured under). "
             "**NEEDS --dream TO DO ANYTHING**: both candidates are diverts, so the "
             "structural override cannot separate them and without a memory term there "
             "is nothing to choose with. Without --dream the arm correctly reduces to "
             "the acoustic behaviour rather than quietly becoming replacement",
    )
    parser.add_argument(
        "--memory-propose-max-offset",
        type=float,
        default=6.0,
        help="THE SAFETY RAIL (ADR-0026). A proposal further than this, by NAVMESH route, "
             "from the acoustic estimate is not emitted at all. matrix-2 measured a "
             "wrong prior with nothing above it costing 10.3 points; ranking fixes the "
             "ordering and not the magnitude, and an unbounded proposal is that failure "
             "one rank lower. Negative removes the rail, which is an arm to measure "
             "against and not a default. Default 6.0 m is a FIRST CHOICE the run prices",
    )
    parser.add_argument(
        "--dream",
        action="store_true",
        help="run DREAM (ICRA2027_Memory eq. 4-27): short-term memory, end-of-episode "
             "consolidation, a three-level long-term memory that GROWS across the "
             "episodes of this run, and memory-weighted plan scoring. Needs --clap, and "
             "EVERY --dream-* knob below: the paper values none of them, so a default "
             "here would be a number nobody chose reaching an artefact nobody reads",
    )
    for flag, kind, equation in DREAM_KNOB_FLAGS:
        parser.add_argument(
            "--dream-{}".format(flag.replace("_", "-")),
            type=kind,
            default=None,
            help="{} (eq. {}). Required with --dream.".format(flag, equation),
        )
    # NOT in DREAM_KNOB_FLAGS, deliberately. Every entry in that table is a knob the
    # paper leaves unvalued and `DreamKnobs.as_metrics` writes onto every episode; these
    # two are paths to a file on a particular box and belong on no audit. Keeping them
    # out is also what keeps "--dream needs every knob" true: a chain is optional.
    parser.add_argument(
        "--dream-memory-in",
        default=None,
        help="restore M^E from this file before the first episode (written by "
             "--dream-memory-out). Requires --dream. A MISSING file is an error and "
             "never a silent empty memory: the two are indistinguishable afterwards.",
    )
    parser.add_argument(
        "--dream-memory-out",
        default=None,
        help="write M^E here after the last episode, overwriting. Requires --dream. "
             "Point both flags at one path to chain scenes into a lifelong memory.",
    )
    return parser


# The fifteen numbers `ICRA2027_Memory` leaves open, as CLI flags (fourteen it names and
# does not value, plus `max_retained`, which caps eq. 13 -- see ADR-0024). The
# table drives both the parser and `dream_kwargs_from_args`, so a knob cannot reach one
# and not the other -- which is the failure `config_from_args`'s own docstring describes
# ("a CLI whose flags quietly stop reaching the config").
DREAM_KNOB_FLAGS = (
    ("stm_horizon", int, "4"),
    ("stm_decay", float, "19"),
    ("present_weight", float, "19"),
    ("coherence", float, "9"),
    ("min_segment", int, "9"),
    ("max_segment", int, "9"),
    ("alpha", float, "10"),
    ("beta", float, "10"),
    ("gamma", float, "10"),
    ("eta", float, "13"),
    ("max_retained", int, "13"),
    ("min_support", int, "16"),
    ("k_experience", int, "20"),
    ("k_pattern", int, "21"),
    ("k_knowledge", int, "22"),
    ("temperature", float, "23"),
    ("lambda_plan", float, "26"),
    ("lambda_memory", float, "26"),
    ("lambda_feasibility", float, "26"),
)


def dream_kwargs_from_args(args: argparse.Namespace) -> Dict[str, object]:
    """The one ``run()`` DREAM keyword, from ``--dream`` plus its knobs.

    ``{}`` when ``--dream`` is absent, for the reason ``memory_kwargs_from_args`` returns
    ``{}`` on a bare invocation: a run with no DREAM flags must reach ``run()`` with no
    DREAM keyword at all, so the audit records the absence rather than a `None` a reader
    would have to interpret.

    **EVERY KNOB IS REQUIRED AND THE ERROR NAMES ALL THE MISSING ONES AT ONCE.** Fourteen
    flags is a lot to get right one `argparse` error at a time, and a sweep driver that
    discovered them one failed launch after another would burn a box slot per knob.
    """
    if not getattr(args, "dream", False):
        # The chain flags are meaningless without the mechanism they carry, and a run
        # that accepted them silently would write nothing and report success -- which is
        # the shape of every incident in this repo's convention list.
        stray = [
            name for name in ("dream_memory_in", "dream_memory_out")
            if getattr(args, name, None) is not None
        ]
        if stray:
            raise SystemExit(
                "{} need --dream; without it there is no M^L to carry and the run would "
                "quietly ignore them".format(
                    " ".join("--{}".format(name.replace("_", "-")) for name in stray)
                )
            )
        return {}
    missing = [
        flag for flag, _kind, _eq in DREAM_KNOB_FLAGS
        if getattr(args, "dream_{}".format(flag), None) is None
    ]
    if missing:
        raise SystemExit(
            "--dream needs every knob; missing: {}. The paper values none of them, so "
            "there is nothing to default to.".format(
                " ".join("--dream-{}".format(name.replace("_", "-")) for name in missing)
            )
        )
    value = {flag: getattr(args, "dream_{}".format(flag)) for flag, _k, _e in DREAM_KNOB_FLAGS}
    return {
        "dream_knobs": DreamKnobs(
            stm_horizon=int(value["stm_horizon"]),
            stm_decay=float(value["stm_decay"]),
            present_weight=float(value["present_weight"]),
            coherence=float(value["coherence"]),
            min_segment=int(value["min_segment"]),
            max_segment=int(value["max_segment"]),
            importance=ImportanceWeights(
                alpha=float(value["alpha"]),
                beta=float(value["beta"]),
                gamma=float(value["gamma"]),
            ),
            eta=float(value["eta"]),
            max_retained=int(value["max_retained"]),
            min_support=int(value["min_support"]),
            k_experience=int(value["k_experience"]),
            k_pattern=int(value["k_pattern"]),
            k_knowledge=int(value["k_knowledge"]),
            temperature=float(value["temperature"]),
            plan_weights=PlanWeights(
                plan=float(value["lambda_plan"]),
                memory=float(value["lambda_memory"]),
                feasibility=float(value["lambda_feasibility"]),
            ),
        ),
        # Passed through as given, including `None`. `run()` starts from an empty `M^E`
        # on `None` in, and writes nothing on `None` out, so a DREAM run with neither
        # flag is byte-identical to one from before they existed.
        "dream_memory_in": args.dream_memory_in,
        "dream_memory_out": args.dream_memory_out,
    }


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


def memory_kwargs_from_args(args: argparse.Namespace) -> Dict[str, object]:
    """The three ``run()`` memory keywords, from the CLI's three ``--memory-*`` flags.

    Separate from ``config_from_args`` on purpose: ``memory.store.MemoryCondition``'s own
    docstring is why the cell is not a ``RunConfig`` field (ADR-0013's layer graph gives
    ``config`` no edge to ``memory/``), so this returns a plain keyword dict for ``run()``
    rather than growing the config to carry a value it cannot act on.

    ``{}`` when ``--memory-condition`` is left at its default. A bare invocation must
    reach ``run()`` with no memory keywords at all -- not with ``memory_condition=None``
    typed out -- so that path and every pre-matrix caller stay indistinguishable at the
    call site, which is what ``run()``'s own byte-identical guarantee depends on.

    Raises a usage error (``SystemExit``, via ``argparse``'s own convention) rather than
    ``run()``'s ``ValueError`` for the missing-store case: this is a CLI mistake, caught
    before the environment probe or the scene load are paid for, not a caller composing
    ``run()`` wrong in code.
    """
    condition = MemoryCondition(args.memory_condition)
    if condition is MemoryCondition.NONE:
        return {}
    if not args.memory_store:
        raise SystemExit(
            "--memory-condition {} needs --memory-store <path to a store "
            "memory_build.dump_stores wrote>".format(condition.value)
        )
    from earshot.task.memory_build import load_stores

    semantic, episodic, _provenance = load_stores(args.memory_store)
    return {
        "memory_condition": condition,
        "memory_prior_stores": (semantic, episodic),
        "memory_k": int(args.memory_k),
        "memory_min_confidence": (
            None if args.memory_abstain_below is None
            else float(args.memory_abstain_below)
        ),
        "memory_proposes": bool(args.memory_proposes),
        # Negative is the operator's way to say "no rail", because argparse cannot take
        # `None` on a float flag without a sentinel and a string sentinel would have to be
        # parsed twice. `MemoryContext.__post_init__` refuses zero and negatives, so the
        # translation happens exactly here and the dataclass keeps one meaning per value.
        "memory_propose_max_offset_m": (
            None if float(args.memory_propose_max_offset) < 0.0
            else float(args.memory_propose_max_offset)
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, build, run. Returns a process exit code.

    ``task/runner.py`` is imported inside this function, not at module scope: it reaches
    ``sim/world.py`` and therefore habitat-sim, and ``build_parser`` /
    ``config_from_args`` have to stay importable on a machine that cannot load the
    simulator so the Mac suite can check them.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = config_from_args(args)
    memory_kwargs = memory_kwargs_from_args(args)
    dream_kwargs = dream_kwargs_from_args(args)

    from earshot.task.runner import run

    def say(message: str) -> None:
        # In-thread, between steps, and flushed — which ADR-0013 established is safe
        # against the guard's fd capture. What is forbidden is a CONCURRENT writer, so
        # this deliberately starts no thread and no progress bar.
        print(message, flush=True)

    run(config, progress=say, **memory_kwargs, **dream_kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
