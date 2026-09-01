"""``RunConfig`` — the whole run in one frozen value, composed from the module configs.

ADR-0013: each layer defines its own frozen config beside itself (``AudioConfig``,
``PlannerConfig``, ``ControllerConfig``, ``DetectorConfig``) and this composes them.
``__main__.py`` builds one from ``argparse``; nothing anywhere reads the environment
(ADR-0008, held by ``tests/mac/test_no_env_flags.py``).

**The two surviving experimental arms are enums, not booleans.** ``Localization`` and
``Detector`` are ADR-0008's two, and a third option is addable to either without a flag
explosion — which is the shape the old tree's ``LTM_REALIZABLE_LOCALIZATION`` did not
have.

**ADR-0017's ``sounding_policy`` is the third arm, and its DEFAULT IS PROVISIONAL.** How
long the source sounds before the offset step is an open question the mechanism does not
answer: a fixed step count, a fraction of the step budget, and a per-episode draw are all
reachable, and ``CONTINUOUS`` is the fourth value and the pre-ADR-0017 control arm. The
default is ``FIXED_STEPS`` at 60 steps, tagged ``provenance: fake``, with no sweep behind
it. ``audio/window.py``'s module docstring names the evidence the first sweep at that
default produces to settle it.

**``onset_rms`` is deliberately absent**, as it is from ``AudioConfig``: task spec §2.3
derives it at run start from the calibration sweep, and a config field would let an
operator hand-set the one number the spec insists is measured. What lives here is the
*input* to the derivation — the bed level and the band — inside ``AudioConfig``.

**This value is what the run record carries.** ``task/runner.py`` writes
``as_dict()`` into ``env_report.json`` beside ``assert_env()``'s probes, because
``agent/config.py``'s argument for ``DetectorConfig`` existing at all is that a number
which gates a STOP and appears in no artefact is the class of thing this map keeps
finding after the fact. That is one artefact holding two kinds of fact — the resolved
environment and the resolved configuration — and both answer the same question about a
run directory a year from now.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from earshot.agent.config import ControllerConfig, DetectorConfig, PlannerConfig
from earshot.audio.config import AudioConfig, WindowPolicy

__all__ = [
    "Localization",
    "Detector",
    "ClimbRule",
    "LateralCue",
    "CastPolicy",
    "IrPolicy",
    "RunConfig",
]


class Localization(Enum):
    """How the agent reaches the anomaly source (ADR-0011).

    ``REALIZABLE`` climbs live binaural RMS with the lateral cue and a visual confirm,
    and is what the smoke runs (§8): an oracle-localization smoke would leave the whole
    live-audio path unexercised in the one episode that exists to prove it. ``ORACLE``
    point-goals to the true source position and is retained as a bisection tool — if the
    smoke fails, running it separates audio from controller in one step.
    """

    REALIZABLE = "realizable"
    ORACLE = "oracle"


class Detector(Enum):
    """How "is object X here" is answered (ADR-0008's seam, ``agent/detector.py``).

    ``ORACLE`` reads the simulator's geometry and is what the smoke runs, with the
    disclosure that rides with it: an oracle STOP deletes the ``stop_miss`` half of the
    0.031 benchmark-SPL decomposition, so smoke find numbers are not capability numbers.
    ``CAPTION`` is the R2 arm and ships live but unexercised until R2.
    """

    ORACLE = "oracle"
    CAPTION = "caption"


class ClimbRule(Enum):
    """Whether the energy climb steers INVESTIGATE (ADR-0018's memory matrix).

    ``OFF`` prices how much of localization the gradient is worth: ``is_rising``'s
    verdict is not consulted, so the cue counts as dead at every step and the agent runs
    the scan/cast cycle alone. ``LIVE`` is what ships today.
    """

    LIVE = "live"
    OFF = "off"


class LateralCue(Enum):
    """Whether the interaural sign steers INVESTIGATE's turns.

    ``OFF`` treats the sign as ``0`` (ambiguous) inside the controller, so the turn
    decision always takes its documented zero/absent default rather than reading the
    binaural cue. ``LIVE`` is what ships today.
    """

    LIVE = "live"
    OFF = "off"


class CastPolicy(Enum):
    """How INVESTIGATE moves once the climb goes dead.

    ``SCAN_ONLY`` collapses the cast leg to zero steps, which is the pre-``eps-1``
    behaviour and a control arm: every dead step turns instead of walking a leg.
    ``CAST`` is what ships today.
    """

    CAST = "cast"
    SCAN_ONLY = "scan_only"


class IrPolicy(Enum):
    """Which impulse response the sensor's audio is rendered through.

    ``ANECHOIC`` replaces every rendered IR with ``audio.ir.anechoic_like``'s flat,
    reverberation-free stand-in before it is convolved, at all three call sites that
    render one, so the calibration sweep and the step loop stay on the same path
    (ADR-0017). ``FULL`` is what ships today: the room's real, reverberant IR.
    """

    FULL = "full"
    ANECHOIC = "anechoic"


@dataclass(frozen=True)
class RunConfig:
    """One run: where it writes, what it loads, which arms, and the module configs."""

    # -- where the artefacts land ---------------------------------------
    # No default: a run directory is a decision, and the failure mode of a default is a
    # second run quietly landing on top of the first. `report/artifacts.py` refuses to
    # overwrite for the same reason.
    run_dir: str

    # -- what to load ---------------------------------------------------
    # provenance: source — the ObjectNav split and the search root `task/episodes.py`
    # resolves its candidates under. `scene` empty means "the first scene in the split
    # whose mesh is on this machine", which is what a box with a partial download needs.
    split: str = "val"
    data_root: str = "."
    scene: str = ""
    category: Optional[str] = None

    # -- the episode loop -----------------------------------------------
    n_episodes: int = 1

    # provenance: source — the old tree's ObjectNav budget. The step budget is the
    # primary task's; the detour has its own sub-budget in `ControllerConfig`.
    max_steps: int = 500

    # provenance: runtime — the step the anomaly source starts playing. Task spec §2.5:
    # this is when the source starts SOUNDING, not when the agent hears it, so an agent
    # that never gets close enough produces an episode with no onset and that attrition
    # is a funnel stage rather than a screened-out episode.
    #
    # `None` derives it per episode from that episode's own start-to-goal distance
    # (`task.dataset.derive_t_anom`), which is why the tag is no longer `fake`. It was a
    # constant 30, justified against the 500-step budget — and under an oracle STOP the
    # budget is not what ends an episode. The smoke's second box episode found its bed at
    # step 30, the same step the source started sounding, and the loop under test never
    # ran. An integer here pins the old behaviour on every episode.
    t_anom: Optional[int] = None

    # -- the sounding window (ADR-0017) ---------------------------------
    # All four are TOP-LEVEL rather than on a sub-config, and the reason is threefold.
    # (1) It is an experimental ARM, and ADR-0008 says arms are enums on the config:
    # `Localization` and `Detector`, this tree's only two, are both top level. (2) A
    # sub-config CANNOT hold it — `as_dict` passes sub-configs through
    # `dataclasses.asdict`, which leaves an Enum member as an Enum object and makes
    # `json.dumps` raise; the top-level enums below are hand-mapped to `.value` instead.
    # (3) A fifth sub-config would widen a set `tests/mac/test_config.py` pins to exactly
    # the four module configs, and that is a diff nobody should make to dodge writing
    # four flags.
    #
    # provenance: fake — HOW THE SOUNDING WINDOW'S DURATION IS CHOSEN, and this default
    # is PROVISIONAL. ADR-0017 leaves the choice open between a fixed step count, a
    # fraction of the step budget, and a per-episode draw (which is what SAVi and SAVN-CE
    # do). All three are reachable here; `CONTINUOUS` is the fourth and is the
    # pre-ADR-0017 control arm.
    #
    # FIXED_STEPS by default because it is the only policy whose value can be read off a
    # single run without a distribution, and the first sweep's job is to answer the
    # question rather than to average over it. What answers it: `onset_delay_steps` and
    # `heard_within_window` on the episode's metrics bag.
    sounding_policy: WindowPolicy = WindowPolicy.FIXED_STEPS

    # provenance: fake — no sweep behind this number. 60 steps is 60 s at
    # `AudioConfig.step_seconds`, four times SAVN-CE's 15 s mean, and it is GENEROUS on
    # purpose (ADR-0014: an unmeasured constant is set generously). The failure mode of a
    # short window is not a harder task — it is that the agent never gets within earshot
    # before the source stops, `onset.fired` stays False forever (the level after the
    # tail is the bed, and `calibrate_onset` separates those by >= 6 dB by construction),
    # the funnel caps at T_ANOM_REACHED, CLAP is never handed a clip, and NOTHING RAISES.
    # That reads as ordinary attrition. It also clears the accumulator's own 5-step ramp
    # with 55 steps to spare, and against `max_steps = 500` it leaves hundreds of steps
    # of silent phase, which is the phase the whole ADR is for.
    sounding_steps: int = 60

    # provenance: fake — 0.12 * 500 = 60, the same duration FIXED_STEPS gives, so
    # switching policy at the defaults changes the VARIANCE of the duration and not its
    # level.
    sounding_budget_fraction: float = 0.12

    # provenance: fake — (30, 90), mean 60, the same reason. Drawn per episode as a pure
    # function of (`seed`, episode index), never from a global RNG: `tools/episode_diff.py`
    # pairs the same episode index across two sweeps, so a duration that depended on
    # anything else would put a different task on each side of the pair.
    sounding_draw_steps: Tuple[int, int] = (30, 90)

    # provenance: fake — seeds the navmesh's random point draws, which are the
    # calibration sweep's poses. A red run that cannot be reproduced is not evidence.
    seed: int = 20260805

    # -- the arms -------------------------------------------------------
    localization: Localization = Localization.REALIZABLE
    detector: Detector = Detector.ORACLE

    # -- the ablation arms (the memory matrix's controls) ----------------
    # provenance: fake — no sweep behind any of the four defaults; each names the
    # behaviour that ships today, so the default configuration is byte-identical to the
    # pre-arm runner and a delta is attributable to the flag that was set.
    climb_rule: ClimbRule = ClimbRule.LIVE
    lateral_cue: LateralCue = LateralCue.LIVE
    cast_policy: CastPolicy = CastPolicy.CAST
    ir_policy: IrPolicy = IrPolicy.FULL

    # -- the anomaly ----------------------------------------------------
    # provenance: source — one of `audio.clips.ANOMALY_CLASSES`. It selects the ESC-50
    # recording staged at `<AudioConfig.clip_dir>/<class>.wav`; `anomaly_clip` overrides
    # the path outright. There is no synthetic fallback (ticket 22): a run whose staging
    # failed raises rather than classifying CLAP against a noise burst.
    anomaly_class: str = "alarm"
    anomaly_clip: Optional[str] = None

    # provenance: box — CLAP is 153.5 M params and 0.713 GiB of VRAM (ticket 15), and
    # the smoke does not exercise the gate at all (§4.3: one sound, the anomaly by
    # construction). Off by default, so the cost is paid only by runs that use it — and
    # `assert_env(clap=...)` is asked the same question at the same time.
    clap: bool = False

    # -- the dataset builder (ADR-0010) ---------------------------------
    # provenance: source — carried from `make_anomaly_response_smoke`. The xz separation
    # that decouples the source from the primary goal, and the same-floor band.
    min_source_sep_m: float = 3.0
    max_source_dy_m: float = 1.0
    # provenance: MEASURED — `detour-1`, the mirror of `min_source_sep_m`: that keeps the
    # source off the GOAL, this keeps it off the AGENT.
    #
    # Spelled again rather than imported: ADR-0013 puts `config` at
    # ("audio.config", "agent.config", "types"), so it cannot name `task.dataset`, where
    # the rule and the measurement that set it live. All three of these builder numbers
    # are duplicated that way and only this one is held — `tests/mac/test_config.py`
    # asserts every one against `task.dataset`, which is the same mechanism
    # `test_report_artifacts` uses for the notifier's copy of `summary.json`.
    min_source_start_sep_m: float = 2.0

    # -- smoke criterion 7 ----------------------------------------------
    # provenance: fake — the per-step wall-clock ceiling, and §9 leaves it to the
    # builder. Set generously and NOT at ticket 06's 27.2 ms, exactly as criterion 7
    # says: ticket 06 measured 2.3x pose variance against ticket 04 on the same scene,
    # and this budget also carries the convolution, the bed mix and the guard's two
    # tempfiles, none of which were in that 27.2 ms. A tight bound would fail for a
    # reason that is not a regression. The runner records; ticket 26's smoke asserts.
    audio_step_ceiling_s: float = 0.5

    # -- writing --------------------------------------------------------
    # `report/artifacts.py` refuses to overwrite an existing artefact, because re-using a
    # tag mixes two runs into one directory with nothing on disk saying so — which is how
    # a set of committed results came to be quoted against numbers from a different run.
    # This is how someone says "yes, replace it" out loud, and it lands in the run record
    # so the replacement is itself recorded.
    overwrite: bool = False

    # -- the module configs ---------------------------------------------
    audio: AudioConfig = field(default_factory=AudioConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    # `detector_config`, not `detector`: the arm above owns that name, and two fields one
    # letter apart on the same object is the near-miss this tree keeps paying for.
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)

    def as_dict(self) -> Dict[str, Any]:
        """The serialised form, JSON-ready. Lands inside ``env_report.json``.

        Enums become their ``value`` and the sub-configs go through
        ``dataclasses.asdict``, so adding a field to ``PlannerConfig`` shows up in the
        run record without a second edit here — which is the drift that would otherwise
        make this a partial copy of the configuration rather than the configuration.
        """
        return {
            "run_dir": self.run_dir,
            "split": self.split,
            "data_root": self.data_root,
            "scene": self.scene,
            "category": self.category,
            "n_episodes": int(self.n_episodes),
            "max_steps": int(self.max_steps),
            "t_anom": None if self.t_anom is None else int(self.t_anom),
            "sounding_policy": self.sounding_policy.value,
            "sounding_steps": int(self.sounding_steps),
            "sounding_budget_fraction": float(self.sounding_budget_fraction),
            "sounding_draw_steps": [
                int(self.sounding_draw_steps[0]),
                int(self.sounding_draw_steps[1]),
            ],
            "seed": int(self.seed),
            "localization": self.localization.value,
            "detector": self.detector.value,
            "climb_rule": self.climb_rule.value,
            "lateral_cue": self.lateral_cue.value,
            "cast_policy": self.cast_policy.value,
            "ir_policy": self.ir_policy.value,
            "anomaly_class": self.anomaly_class,
            "anomaly_clip": self.anomaly_clip,
            "clap": bool(self.clap),
            "min_source_sep_m": float(self.min_source_sep_m),
            "max_source_dy_m": float(self.max_source_dy_m),
            "min_source_start_sep_m": float(self.min_source_start_sep_m),
            "audio_step_ceiling_s": float(self.audio_step_ceiling_s),
            "overwrite": bool(self.overwrite),
            "audio": dataclasses.asdict(self.audio),
            "planner": dataclasses.asdict(self.planner),
            "controller": dataclasses.asdict(self.controller),
            "detector_config": dataclasses.asdict(self.detector_config),
        }
