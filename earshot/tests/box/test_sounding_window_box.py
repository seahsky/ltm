#!/usr/bin/env python3
"""ADR-0019's SPLIT READOUT against an IR the renderer actually produced. V100 + ``ss2``.

    conda activate ss2
    bash earshot/tools/box_gate.sh

**F29: the accumulation buffer shipped with no box arm at all.** ``audio/tail.py``'s
module docstring states a wall-clock bill and concludes that it *spends none of*
criterion 7's margin; ``task/runner.py`` repeats that sentence as the justification for
folding the accumulator INSIDE the ``audio_s`` bracket criterion 7 audits. Every number
in both sentences is a Mac number, taken on a synthetic IR, on a machine that has no
renderer to put beside them. Criterion 7's ceiling is 0.5 s and has already been breached
once at 0.5335 s.

**ADR-0019 then moved what the agent reads, and every number this file used to assert
became a number about the wrong quantity.** The buffer had ONE readout, the last
``N = len(clip)`` samples; ``N`` is 5 s and a step is 1 s, so the "instantaneous" RMS was
a five-second moving average and its post-offset decay was the analysis window emptying
rather than the room. The repair is a second readout off the same buffer:

    clip readout   ``buffer[:, :N]``            (2, N)     CLAP only, unchanged
    cue  readout   ``buffer[:, N - hop : N]``   (2, hop)   what arrived DURING this step

So this file now measures BOTH, at the real numbers -- hop 44100 against IRs this box
rendered, at poses it drew itself -- and PRINTS the per-step bill, both ramps, both
tails, the settled level, both post-offset decay curves, the loop's phase profile and the
three scatter arms (ADR-0014). The Mac tests run at ``window = 800, hop = 100`` on
synthetic IRs; nothing here is a re-run of them at a different scale, because none of
these numbers exist until a scene is loaded.

**THE CLAIM THIS FILE EXISTS TO SETTLE.** ``cue_tail_steps`` is asserted, on the box, to
be evidence that the geometric acoustics did work -- and the only way to make that a
measurement rather than a sentence is an anechoic control. ``TestTheCueTailIsReverberation``
folds the same real clip through a 1-sample IR (a room with literally no reverberation)
and shows the two halves of the defect in one test: the anechoic control **reproduces the
real room's CLIP decay curve to a point or two**, which is why that curve was never
reverb, and it **collapses the CUE decay to a single hard step** where the real room's
outlives it, which is why this one is. The Mac can state that arithmetic; only the box can
state it against a room.

**Both arms** (ADR-0014), and the forced failures are picked so they cannot come back
green by accident. This tree's scar is a CLAP arm that was handed the real checkpoint and
reported PASS, so each one below first asserts the precondition that makes it capable of
firing -- that the real IR is wider than one sample, that the last sounding fold emitted
a loud chunk, that there is real energy past the read window -- and then fires it:

- ``TestTheCueTailIsReverberation`` runs the anechoic control described above.
- ``TestTheForcedFailureArms.test_a_truncating_buffer...`` folds the same real IR through
  a buffer capped at the read window, which is the implementation ``open_tail`` refuses to
  be. Since the split its damage has a name: the truncated reverb is exactly the samples
  that would have slid into the CUE window on the next silent fold, so a truncating buffer
  makes the room read ANECHOIC -- ``cue`` zero one fold after the offset step, against the
  honest fold's ``cue_tail_steps``.
- ``...test_the_silent_phase_guard_refuses...`` takes the record this box's own fold
  produces, cripples one field, and shows ``silent_phase_tally`` raising rather than
  publishing an SWS -- with the healthy record passing through the same call.

**WHAT THE MAC CANNOT REACH, AND WHETHER THIS DOES.** F20: ``run() -> summary.json`` is
the only path an SWS reaches disk by, and a Mac cannot walk it. ``TestTheSwsReachesDisk``
walks it here -- one short real episode, into a temp run directory -- and reads back
``summary.json``, the episode's ``audit.json`` and ``env_report.json``. It is the only
test in this file that needs the ObjectNav dataset and the staged ESC-50 clip, and it
SKIPS (loudly, naming the command) when either is absent, or when the scene can place no
episode at all. A skip there is a NOT_RUN and NOT_RUN is red: read it as this box not
being provisioned, never as the path being green.

**Cost.** Two scene loads: one in ``setUpModule`` for the IRs, one inside ``run()``.
``setUpModule`` also renders a HELD pose repeatedly, because ``sweep_render_scatter`` and
``sweep_loop_scatter`` are about the renderer disagreeing with itself and that question
cannot be asked of a stored IR. Those renders are served back to the two sweeps through
their own ``render_at`` seam -- the same seam ``calibrate_episode`` passes a live closure
to -- so the sweeps under test are the production functions and the IRs are live.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import time
import unittest
from typing import Any, Callable, List, Optional, Sequence, Tuple

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which
# habitat-sim reads at import time.
import earshot  # noqa: F401
import numpy as np

from earshot.audio.bed import bed_signal, heard_signal, mix_bed
from earshot.audio.calibration import (
    SCATTER_REPEATS,
    render_scatter_of,
    sweep_loop_scatter,
    sweep_render_scatter,
)
from earshot.audio.clips import (
    load_anomaly_clip,
    render_through_ir,
    resolve_anomaly_clip,
    rms,
    synthetic_burst,
)
from earshot.audio.config import AudioConfig, WindowPolicy
from earshot.audio.sensor import AudioSensorHandle
from earshot.audio.spec import audio_sensor_spec
from earshot.audio.tail import (
    CUE_RAMP_STEPS,
    advance_tail,
    clip_readout,
    cue_crest,
    cue_level,
    cue_min_ratio,
    cue_readout,
    heard_clip_window,
    heard_step,
    hop_samples,
    open_tail,
    phase_folds,
    steady_state_cue_rms,
    steady_state_render,
)
from earshot.audio.window import plan_window
from earshot.config import RunConfig
from earshot.metrics import post_offset_audible_steps
from earshot.report.artifacts import (
    ENV_REPORT_NAME,
    RUN_SUMMARY_NAME,
    episode_paths,
    run_paths,
)
from earshot.report.audit import (
    EpisodeAudit,
    FunnelStage,
    SoundingWindowRecord,
    StepRecord,
)
from earshot.task.dataset import EmptyDatasetError
from earshot.task.episodes import (
    available_scenes,
    find_scenes_dir,
    find_split_dir,
    load_scene,
)
from earshot.task.runner import (
    RunSummary,
    TailNotActiveError,
    run,
    silent_phase_tally,
    tail_is_active,
)
from earshot.types import Xyz

SPLIT = os.environ.get("SS2_SPLIT", "val")
PLACEMENT_SEED = 20260830
# Near enough to stay in one room and to keep the IR strong. Nothing here is about the
# lateral cue, so the bearing does not matter -- what matters is that the source is close
# enough that a silent IR means a broken context rather than a long walk.
SOURCE_OFFSET_M = 2.0
# Eight poses, because the IR is trimmed to its own decay and its WIDTH is therefore a
# per-pose measurement (ticket 06, finding 4). One pose cannot show a width change.
N_POSES = 8
# `RunConfig.anomaly_class`'s default, so the clip is the one a real run plays.
ANOMALY_CLASS = "alarm"
# ESC-50 recordings are 5 s; the synthetic stand-in matches so the hop/window ratio is
# the shipped one either way.
FALLBACK_CLIP_SECONDS = 5.0
# Spare held-pose renders past what the two scatter sweeps consume. The pool is sized
# from this box's own measured IR width rather than from a constant, so a more
# reverberant scene lengthens `sweep_loop_scatter`'s settle and the pool follows it.
HELD_POSE_SPARE_RENDERS = 4

# `run()`'s arm, sized so the window CLOSES inside the episode: SWS's denominator counts
# episodes that ran past their own offset step, and an episode that ends first is not
# eligible and would leave `sws_status` at "not_run" for a reason that is not a defect.
RUN_T_ANOM = 2
RUN_SOUNDING_STEPS = 8
RUN_MAX_STEPS = 40

_SCENE = None
# The content-file stem, kept beside the dataset because they are not the same string:
# `load_scene` reads `content/<stem>.json.gz` and takes `scene_label` off the EPISODES
# inside it. `RunConfig.scene` is fed back into `load_scene`, so it must be the stem.
_SCENE_STEM = ""
_CLIP = None
_CLIP_LABEL = ""
_CLIP_PATH = None
_ARMING_IR_SHAPE = None
_IRS: List[np.ndarray] = []
_POSES: List[Xyz] = []
_RENDER_MS: List[float] = []
# One pose, rendered again and again. `sweep_render_scatter` and `sweep_loop_scatter`
# measure the renderer disagreeing with ITSELF, which a stored IR cannot express.
_HELD_POSE: Optional[Xyz] = None
_HELD_RENDERS: List[np.ndarray] = []


def _median(values: Sequence[float]) -> float:
    """The middle element, matching ``test_audio_box``'s convention exactly."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def setUpModule():
    """Render the IRs this file measures against, then CLOSE the simulator.

    The world is opened once, sampled at ``N_POSES`` navigable poses and once more at a
    HELD pose, then closed before the first test runs. Two reasons, and the second is the
    one that would bite. The accumulator is pure numpy, so nothing below needs a live
    simulator once real IRs are in hand; and ``TestTheSwsReachesDisk`` calls
    ``task.runner.run``, which builds a ``World`` of its own -- a module-scope world held
    open across the file would put two simulators and two audio contexts in one process
    for no gain.

    The render bill is therefore sampled here, where the world is, and printed by the
    test that judges it. Scene selection is ``test_audio_box``'s: the first ObjectNav
    scene whose mesh is on this box, or ``SS2_SCENE_LABEL``.

    **The held-pose pool is sized from this box's own first render**, never from ticket
    06's 72300: ``sweep_loop_scatter`` settles ``clip_tail_steps`` folds and that number
    is ``ceil((N + L - 1) / hop)`` for the L this scene produces.
    """
    global _SCENE, _SCENE_STEM, _CLIP, _CLIP_LABEL, _CLIP_PATH, _ARMING_IR_SHAPE
    global _HELD_POSE

    config = AudioConfig()
    split_dir = find_split_dir(SPLIT)
    scenes_dir = find_scenes_dir()
    override = os.environ.get("SS2_SCENE_LABEL")
    for label in [override] if override else list(available_scenes(split_dir)):
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            _SCENE, _SCENE_STEM = dataset, label
            break
    if _SCENE is None:
        raise unittest.SkipTest(
            "no ObjectNav {} scene has its mesh on this box (looked under {})".format(
                SPLIT, scenes_dir
            )
        )
    print("\n  scene: {}".format(_SCENE.scene_label), flush=True)
    print("  mesh:  {}".format(_SCENE.scene_path), flush=True)

    # The real recording if it is staged. The fallback is named rather than silent: the
    # decay curves below are a property of the CLIP as much as of the room (`audio/tail.py`
    # measures a transient cliffing where noise decays, and the CUE readout makes that
    # louder rather than quieter), so a reader has to know which one produced the numbers.
    # `load_anomaly_clip` itself has no fallback on purpose.
    _CLIP_PATH = resolve_anomaly_clip(ANOMALY_CLASS, None, config.clip_dir)
    if _CLIP_PATH is None:
        _CLIP = synthetic_burst(
            config.sample_rate, FALLBACK_CLIP_SECONDS, config.target_norm_rms_db
        )
        _CLIP_LABEL = "SYNTHETIC {:.1f} s burst -- no {}.wav under {}".format(
            FALLBACK_CLIP_SECONDS, ANOMALY_CLASS, config.clip_dir
        )
    else:
        _CLIP = load_anomaly_clip(
            _CLIP_PATH, config.sample_rate, config.target_norm_rms_db
        )
        _CLIP_LABEL = _CLIP_PATH
    print("  clip:  {}".format(_CLIP_LABEL), flush=True)
    print("  clip:  {} samples ({:.3f} s at {} Hz)  rms {:.6g}".format(
        len(_CLIP), len(_CLIP) / float(config.sample_rate), config.sample_rate,
        rms(_CLIP)), flush=True)

    hop = hop_samples(
        step_seconds=config.step_seconds, sample_rate=config.sample_rate
    )
    print("  hop:   {} samples ({:.3f} s)   phase_folds {}   clip_ramp_steps {}   "
          "CUE_RAMP_STEPS {}".format(
              hop, hop / float(config.sample_rate),
              phase_folds(window=len(_CLIP), hop=hop),
              int(math.ceil(len(_CLIP) / float(hop))), CUE_RAMP_STEPS), flush=True)

    from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs

    spec, binaural = audio_spec_parts()
    audio_sensor_spec(spec, config, binaural)
    world = World(
        _SCENE.scene_path,
        camera_sensor_specs(width=256, height=256) + [spec],
    )
    try:
        world.seed_navmesh(PLACEMENT_SEED)
        point = world.random_navigable_point()
        # Seat the agent BEFORE arming: `arm_audio_context` owns the first render and
        # rejects a silent IR, and habitat's default pose is wherever the scene puts it.
        world.set_pose(point)
        handle = AudioSensorHandle(
            world.sensor_handle(str(spec.uuid)),
            world.observe,
            Xyz(point.x + SOURCE_OFFSET_M, point.y + 0.5, point.z),
            uuid=str(spec.uuid),
        )
        _ARMING_IR_SHAPE = handle.report.ir_shape
        print("  arming render: IR {}  ({} vertices)".format(
            _ARMING_IR_SHAPE, handle.report.n_vertices), flush=True)

        tries = 0
        while len(_IRS) < N_POSES and tries < N_POSES * 4:
            tries += 1
            drawn = world.random_navigable_point()
            world.set_pose(drawn)
            # The source rides with the listener so every pose is a 2 m source in a
            # different room. A fixed source and a wandering listener would measure
            # occlusion, which is a different question and produces silent IRs.
            handle.set_source(
                Xyz(drawn.x + SOURCE_OFFSET_M, drawn.y + 0.5, drawn.z)
            )
            started = time.perf_counter()
            observation, _guard = handle.observe()
            impulse = np.asarray(handle.audio_of(observation), dtype=np.float32)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if float(np.max(np.abs(impulse))) <= 0.0:
                print("  pose {}: silent IR, redrawing".format(tries), flush=True)
                continue
            _RENDER_MS.append(elapsed_ms)
            _IRS.append(impulse)
            _POSES.append(drawn)

        if _IRS:
            held = _POSES[0]
            world.set_pose(held)
            handle.set_source(Xyz(held.x + SOURCE_OFFSET_M, held.y + 0.5, held.z))
            observation, _guard = handle.observe()
            first = np.asarray(handle.audio_of(observation), dtype=np.float32)
            _HELD_RENDERS.append(first)
            settle = int(math.ceil(
                (len(_CLIP) + int(first.shape[1]) - 1) / float(hop)
            ))
            # `sweep_render_scatter` takes `repeats`; `sweep_loop_scatter` takes
            # `clip_tail_steps + repeats`. Both are served from this one pool.
            needed = 2 * SCATTER_REPEATS + settle + HELD_POSE_SPARE_RENDERS
            while len(_HELD_RENDERS) < needed:
                observation, _guard = handle.observe()
                _HELD_RENDERS.append(
                    np.asarray(handle.audio_of(observation), dtype=np.float32)
                )
            _HELD_POSE = held
    finally:
        world.close()

    if len(_IRS) < 2:
        raise unittest.SkipTest(
            "only {} non-silent IR(s) in {} draws on {} -- the audio context is not "
            "producing renders to measure".format(len(_IRS), tries, _SCENE.scene_label)
        )
    widths = [int(ir.shape[1]) for ir in _IRS]
    print("  {} usable IR(s) in {} draws; widths {} to {} samples".format(
        len(_IRS), tries, min(widths), max(widths)), flush=True)
    held_widths = [int(ir.shape[1]) for ir in _HELD_RENDERS]
    identical = sum(
        1 for ir in _HELD_RENDERS[1:]
        if ir.shape == _HELD_RENDERS[0].shape
        and bool(np.array_equal(ir, _HELD_RENDERS[0]))
    )
    print("  held pose ({:.2f}, {:.2f}, {:.2f}): {} repeat renders; widths {} to {}; "
          "{} of {} byte-identical to the first".format(
              _POSES[0].x, _POSES[0].y, _POSES[0].z, len(_HELD_RENDERS),
              min(held_widths), max(held_widths), identical,
              len(_HELD_RENDERS) - 1), flush=True)


def _hop() -> int:
    config = AudioConfig()
    return hop_samples(
        step_seconds=config.step_seconds, sample_rate=config.sample_rate
    )


def _expected_clip_tail_steps(window: int, ir_samples: int, hop: int) -> int:
    """``ceil((N + L - 1) / hop)``, written out because ``TailState.clip_tail_steps``
    cannot answer it until a sounding fold has widened ``max_ir_samples``."""
    return int(math.ceil((int(window) + int(ir_samples) - 1) / float(hop)))


def _expected_cue_tail_steps(ir_samples: int, hop: int) -> int:
    """``ceil((hop + L - 1) / hop)``, for the same reason and at the other width.

    This is the number ADR-0019 made the fence post: it is ``1`` exactly when the IR fits
    inside one step, so ``> 1`` is the first thing on the record that is evidence the
    geometric acoustics did any work.
    """
    return int(math.ceil((int(hop) + int(ir_samples) - 1) / float(hop)))


def _anechoic_like(impulse: np.ndarray) -> np.ndarray:
    """A ``(2, 1)`` IR at the same peak as ``impulse`` -- a room with no reverberation.

    Scaled to the real IR's peak only so the printed absolute levels stay in the same
    range; every comparison against it is on a curve normalised by its own settled level,
    where the scale cancels exactly.
    """
    peak = float(np.max(np.abs(np.asarray(impulse, dtype=np.float32))))
    return np.full((2, 1), max(peak, 1e-12), dtype=np.float32)


def _phase_energies(clip: Any, hop: int) -> Tuple[float, ...]:
    """Energy of the ``hop``-sample chunk each loop phase emits, oldest phase first.

    Fold ``j`` emits phase ``j % phase_folds``, because two folds emit the same chunk
    exactly when their indices differ by a multiple of ``N // gcd(N, hop)``.
    """
    signal = np.asarray(clip, dtype=np.float32).reshape(-1)
    n = int(signal.size)
    stride = int(hop)
    energies = []
    for index in range(phase_folds(window=n, hop=stride)):
        start = (index * stride) % n
        chunk = signal[(start + np.arange(stride)) % n]
        energies.append(float(np.sum(np.square(chunk))))
    return tuple(energies)


def _folds_ending_on_the_loudest_phase(settle: int, energies: Sequence[float]) -> int:
    """How many sounding folds to run so the LAST one emits the clip's loudest chunk.

    ADR-0014's precondition rule applied to the loop. The cue readout is one step wide,
    so a bursty clip whose last ring landed four folds ago reads exactly zero at the
    offset step -- which is honest, and which would make every post-offset arm below
    vacuous by accident rather than by measurement. Aligning the offset step to the
    loudest phase is what lets those arms assert that the room, not the clip, is what
    the cue tail carries.
    """
    folds = len(energies)
    loudest = int(max(range(folds), key=lambda index: energies[index]))
    return int(settle) + ((loudest - (int(settle) - 1)) % folds)


def _sequential_render_at(
    renders: Sequence[np.ndarray],
) -> Tuple[Callable[[Any], np.ndarray], List[np.ndarray]]:
    """A ``render_at`` seam serving pre-rendered HELD-POSE IRs, in order, once each.

    Not a proxy for the renderer (ADR-0014 forbids that): every IR served is a live
    render of the same pose taken in ``setUpModule``, and ``render_at`` is the seam
    ``calibrate_episode`` itself passes a closure to. What is replaced is only *when* the
    render happens, because the world is closed by the time these tests run and holding
    it open across ``run()`` would put two audio contexts in one process.

    Exhaustion RAISES rather than cycling. A cycling pool would silently understate the
    scatter it is measuring -- it would feed the same render into two repeats -- which is
    exactly the shape of quiet-and-plausible failure this file exists to refuse.
    """
    remaining = list(renders)
    consumed: List[np.ndarray] = []

    def render_at(_pose: Any) -> np.ndarray:
        if not remaining:
            raise AssertionError(
                "the held-pose render pool ran out after {} calls. It is sized in "
                "setUpModule from this scene's own IR width; raise "
                "HELD_POSE_SPARE_RENDERS.".format(len(consumed))
            )
        impulse = remaining.pop(0)
        consumed.append(impulse)
        return impulse

    return render_at, consumed


class TestTheAccumulatorBillAgainstCriterion7(unittest.TestCase):
    """``audio/tail.py``'s wall-clock claim, remeasured where there is a renderer."""

    def test_the_per_step_bill_is_measured_against_the_ceiling_it_claims_to_clear(self):
        """The whole point of the file: the Mac numbers had no renderer beside them.

        Reported AND gated, unlike ``test_audio_box``'s render bill. That one is
        report-only because pose variance (2.3x, ticket 06) would fail a tight bound for
        a reason that is not a regression; this one is gated against criterion 7's own
        0.5 s ceiling, which is the number ``RunConfig.audio_step_ceiling_s`` holds and
        the runner enforces per step. A bound that generous cannot fail for pose variance
        -- it can only fail if the accumulator costs a different order of magnitude here
        than it does on a Mac, which is exactly the thing nobody has measured.

        **Since ADR-0019 there are two per-step arms and the bill should have FALLEN.**
        The shipped path composes through the CUE readout, ``hop`` samples wide; the
        ADR-0017 path composed through the CLIP readout, ``N`` samples wide, so one
        ``N``-length copy and one ``N``-length mix left the per-step path. The Mac's
        before/after is 5.54 ms -> 5.23 ms against 19.56 ms for the whole-clip render.
        Both arms are timed here and the delta is printed, because "it should have
        fallen" is a claim and this is where it stops being one.

        ``heard_clip_window`` is timed too: it is what CLAP is handed, ONCE per episode
        rather than once per step, so its cost is priced per episode and never enters the
        bracket.
        """
        config = AudioConfig()
        clip = _CLIP
        impulse = _IRS[0]
        n_ir = int(impulse.shape[1])
        window = len(clip)
        hop = _hop()
        bed_cue = bed_signal(hop, config.bed_rms)
        bed_clip = bed_signal(window, config.bed_rms)
        ceiling_ms = float(RunConfig(run_dir="unused").audio_step_ceiling_s) * 1000.0

        state = open_tail(window=window, hop=hop, headroom=max(0, n_ir - 1))
        cue_ms: List[float] = []
        silent_ms: List[float] = []
        clip_ms: List[float] = []
        whole_clip_ms: List[float] = []
        clap_ms: List[float] = []
        # The two per-step arms are INTERLEAVED, not run back to back. The delta between
        # them is a few percent of a few milliseconds, and two consecutive nine-sample
        # blocks on a shared box see different CPU and cache states -- which would show
        # up as a saving or a loss that is really the scheduler. Alternating them puts
        # both arms under the same conditions sample for sample.
        for _ in range(9):
            started = time.perf_counter()
            state, _cue = heard_step(
                state, ir=impulse, clip=clip, bed_cue=bed_cue, sounding=True
            )
            cue_ms.append((time.perf_counter() - started) * 1000.0)
            # The ADR-0017 arm: the same fold, composed through the clip readout. This is
            # what the per-step path cost before the split -- the "before" number.
            started = time.perf_counter()
            state = advance_tail(state, ir=impulse, clip=clip, sounding=True)
            _clip_signal = mix_bed(clip_readout(state), bed_clip)
            clip_ms.append((time.perf_counter() - started) * 1000.0)
        for _ in range(9):
            started = time.perf_counter()
            state, _cue = heard_step(
                state, ir=None, clip=clip, bed_cue=bed_cue, sounding=False
            )
            silent_ms.append((time.perf_counter() - started) * 1000.0)
        for _ in range(9):
            started = time.perf_counter()
            heard_signal(impulse, clip, bed_clip, playing=True)
            whole_clip_ms.append((time.perf_counter() - started) * 1000.0)
        for _ in range(9):
            started = time.perf_counter()
            heard_clip_window(state, bed_clip=bed_clip)
            clap_ms.append((time.perf_counter() - started) * 1000.0)

        # The first of each is dropped: it pays for the FFT plan and the first allocation
        # of a buffer this wide, and no step in a run after the first pays either.
        sounding = _median(cue_ms[1:])
        silent = _median(silent_ms[1:])
        clip_step = _median(clip_ms[1:])
        whole_clip = _median(whole_clip_ms[1:])
        clap_readout = _median(clap_ms[1:])
        render = _median(_RENDER_MS[1:]) if len(_RENDER_MS) > 1 else _RENDER_MS[0]
        bracket = render + sounding

        print("\n  --- F29/ADR-0019: the per-step bill on this box ---", flush=True)
        print("  IR {}  ({:.3f} s)   clip {} samples   hop {} samples".format(
            tuple(impulse.shape), n_ir / float(config.sample_rate), window, hop),
            flush=True)
        print("  accumulator FFT over hop + L = {} ; whole-clip FFT over N + L = "
              "{}".format(hop + n_ir, window + n_ir), flush=True)
        print("  guarded render      (median of {}): {:8.2f} ms".format(
            max(1, len(_RENDER_MS) - 1), render), flush=True)
        print("  step composed through the CUE  readout: {:8.2f} ms   "
              "(the shipped path; Mac claimed 5.23)".format(sounding), flush=True)
        print("  step composed through the CLIP readout: {:8.2f} ms   "
              "(the ADR-0017 path; Mac claimed 5.54)".format(clip_step), flush=True)
        print("  the split saved {:+.2f} ms per step ({:+.1f}%) -- one N-length copy "
              "and one N-length mix".format(
                  sounding - clip_step,
                  100.0 * (sounding - clip_step) / max(clip_step, 1e-9)), flush=True)
        print("  accumulator step, silent              : {:8.2f} ms   "
              "(no convolution on a silent step)".format(silent), flush=True)
        print("  whole-clip heard_signal               : {:8.2f} ms   "
              "(Mac claimed 19.56; the pre-ADR-0017 path)".format(whole_clip),
              flush=True)
        print("  heard_clip_window (CLAP, ONCE per episode): {:8.2f} ms   "
              "-- never in the bracket".format(clap_readout), flush=True)
        print("  runner audio_s bracket = render + cue step: {:8.2f} ms".format(
            bracket), flush=True)
        print("  criterion 7 ceiling {:8.2f} ms   bracket uses {:.1f}% of it   "
              "headroom {:.1f}x".format(
                  ceiling_ms, 100.0 * bracket / ceiling_ms,
                  ceiling_ms / max(bracket, 1e-9)), flush=True)
        print("  the ceiling has been breached once at 533.50 ms; the accumulator alone "
              "uses {:.1f}% of it".format(100.0 * sounding / ceiling_ms), flush=True)
        print("  {:.1f} s per 500-step episode for the bracket".format(
            bracket * 500.0 / 1000.0), flush=True)
        print("  buffer: {} x {} float32 = {:.2f} MB   cue bed {:.1f} kB   clip bed "
              "{:.2f} MB".format(
                  state.buffer.shape[0], state.buffer.shape[1],
                  state.buffer.nbytes / 1e6, bed_cue.nbytes / 1e3,
                  bed_clip.nbytes / 1e6), flush=True)

        self.assertLess(
            bracket,
            ceiling_ms,
            "the per-step audio bill is at or over criterion 7's ceiling BEFORE any "
            "controller work. The ceiling has already been breached once at 0.5335 s, "
            "and the accumulator now sits inside the same bracket.",
        )
        self.assertLess(
            sounding,
            0.1 * ceiling_ms,
            "audio/tail.py says the accumulator 'spends none of that margin' against "
            "criterion 7. Here it spends more than a tenth of the ceiling, so that "
            "sentence and runner.py's copy of it are false on this hardware.",
        )
        self.assertLess(
            sounding,
            whole_clip,
            "the accumulator step is DEARER than the whole-clip render it replaced, "
            "which inverts audio/tail.py's cost claim and removes the argument for "
            "folding it inside the audio_s bracket. Its FFT is over hop + L = {} against "
            "N + L = {}, so this is a measurement about this box rather than about the "
            "arithmetic.".format(hop + n_ir, window + n_ir),
        )
        self.assertLess(
            silent,
            sounding,
            "a silent step costs as much as a sounding one, so the convolution is not "
            "being skipped when the source is not emitting",
        )
        # Direction of travel, and the bound is deliberately far looser than the effect.
        # The Mac's saving is 0.31 ms of 5.54 -- about 6% -- and a median of eight
        # interleaved timings on a shared box cannot resolve 6%. Asserting at 1.1x would
        # buy a flaky box gate, and a flaky gate on this repo costs a box trip. What the
        # 1.5x bound refuses is the only failure worth a red here: a cue path that is
        # GROSSLY dearer, which is what it would look like if the N-length copy and the
        # N-length mix had not actually left the per-step path. The measured delta is
        # printed above for a reader to judge, whichever way it lands.
        self.assertLess(
            sounding,
            1.5 * clip_step,
            "the CUE-composed step costs half again what the CLIP-composed one it "
            "replaced did. ADR-0019's cost claim is that an N-length copy and an "
            "N-length mix left the per-step path; measured here they did not.",
        )


class TestTheBufferNeverTruncates(unittest.TestCase):
    """The IR width is a per-pose measurement, so the buffer has to grow rather than clip.

    ``spec.py:81-86`` sets no ``maxIRLength`` on purpose, so there is no width to size
    against; ticket 06's ``[2, 72300]`` is one scene at one pose. The width the buffer is
    preallocated from is read off the guard's arming render here exactly as
    ``run_episode`` reads it (``handle.report.ir_shape``), never assumed.
    """

    def test_a_single_fold_holds_the_whole_convolution_at_every_pose(self):
        """Sample-for-sample against an independent convolution, per pose.

        Total energy would be the weak check and it is the wrong one: for a fast-decaying
        IR the reverb past the read window can be a thousandth of the fold's energy, so a
        truncating buffer would pass an energy tolerance loose enough to survive float32
        FFT error. The comparison is therefore against scipy's ``fftconvolve`` sample by
        sample -- a different implementation, in float64 -- and the energy living beyond
        the read window is PRINTED as what truncation would silently cost.

        The comparison starts at ``window - hop``, which is where a sounding fold writes
        and is therefore also where the CUE readout begins: the first ``hop`` samples of
        the reference ARE the cue window after one fold, which the next test asserts.
        """
        from scipy.signal import fftconvolve

        clip = _CLIP
        window = len(clip)
        hop = _hop()
        headroom = (
            0 if _ARMING_IR_SHAPE is None else max(0, int(_ARMING_IR_SHAPE[1]) - 1)
        )
        chunk = np.asarray(clip, dtype=np.float64).reshape(-1)[:hop]

        print("\n  --- F29: the buffer against every rendered IR width ---", flush=True)
        print("  arming ir_shape {} -> preallocated headroom {} samples".format(
            _ARMING_IR_SHAPE, headroom), flush=True)
        print("  pose  IR samples   buffer   grew   cue_tail   clip_tail   "
              "max sample err / peak   energy past the read window", flush=True)
        for index, impulse in enumerate(_IRS):
            n_ir = int(impulse.shape[1])
            folded = advance_tail(
                open_tail(window=window, hop=hop, headroom=headroom),
                ir=impulse,
                clip=clip,
                sounding=True,
            )
            reference = np.stack([
                fftconvolve(chunk, np.asarray(impulse[channel], dtype=np.float64))
                for channel in (0, 1)
            ])
            start = window - hop
            held = np.asarray(
                folded.buffer[:, start : start + reference.shape[1]], dtype=np.float64
            )
            peak = float(np.max(np.abs(reference)))
            error = float(np.max(np.abs(held - reference))) / max(peak, 1e-12)
            total = float(np.sum(np.square(folded.buffer)))
            beyond = float(np.sum(np.square(folded.buffer[:, window:])))
            print("  {:>4}  {:>10}   {:>6}   {:>4}   {:>8}   {:>9}   {:>21.3e}   "
                  "{:.4f}% of the fold".format(
                      index, n_ir, int(folded.buffer.shape[1]),
                      "yes" if folded.n_grows else "no",
                      folded.cue_tail_steps, folded.clip_tail_steps, error,
                      100.0 * beyond / max(total, 1e-30)), flush=True)

            self.assertGreaterEqual(
                int(folded.buffer.shape[1]),
                window + n_ir - 1,
                "the buffer is narrower than window + L - 1 at pose {}, so the fold's "
                "last {} reverb samples had nowhere to go".format(index, n_ir - 1),
            )
            self.assertLess(
                error,
                1e-4,
                "the accumulated fold at pose {} does not match an independent "
                "convolution of the same hop through the same IR -- the buffer dropped "
                "or misplaced samples".format(index),
            )
            self.assertGreater(
                beyond,
                0.0,
                "no energy at all past the read window at pose {}, so a truncating "
                "buffer would be indistinguishable here and this check proves "
                "nothing".format(index),
            )
            # The two readouts come off ONE buffer and must never diverge. This is the
            # cheapest place to say so against a real IR: the cue window IS the clip
            # window's last `hop` samples, at every pose, byte for byte.
            np.testing.assert_array_equal(
                cue_readout(folded), clip_readout(folded)[:, -hop:]
            )
            self.assertEqual(
                folded.cue_tail_steps, _expected_cue_tail_steps(n_ir, hop)
            )
            self.assertEqual(
                folded.clip_tail_steps,
                _expected_clip_tail_steps(window, n_ir, hop),
            )

    def test_the_widest_ir_seen_is_what_the_buffer_is_sized_to(self):
        """One tail across every pose, which is the shape an episode actually walks."""
        clip = _CLIP
        window = len(clip)
        hop = _hop()
        widths = [int(ir.shape[1]) for ir in _IRS]
        headroom = (
            0 if _ARMING_IR_SHAPE is None else max(0, int(_ARMING_IR_SHAPE[1]) - 1)
        )

        state = open_tail(window=window, hop=hop, headroom=headroom)
        for impulse in _IRS:
            state = advance_tail(state, ir=impulse, clip=clip, sounding=True)
            self.assertGreaterEqual(
                int(state.buffer.shape[1]), window + int(state.max_ir_samples) - 1
            )

        print("\n  --- F29: the width across {} poses ---".format(len(_IRS)), flush=True)
        # The poses beside the widths, because the width IS the pose's measurement and a
        # reader who wants to reproduce a spread needs to know where it was taken.
        print("  poses: {}".format(", ".join(
            "({:.2f}, {:.2f}, {:.2f})".format(pose.x, pose.y, pose.z)
            for pose in _POSES)), flush=True)
        print("  IR widths: {}".format(widths), flush=True)
        print("  narrowest {}  widest {}  spread {} samples ({:.3f} s)".format(
            min(widths), max(widths), max(widths) - min(widths),
            (max(widths) - min(widths)) / float(AudioConfig().sample_rate)), flush=True)
        print("  max_ir_samples {}  n_grows {}  buffer {} samples".format(
            state.max_ir_samples, state.n_grows, int(state.buffer.shape[1])), flush=True)
        print("  clip_tail_steps {}  clip_ramp_steps {}  cue_tail_steps {}  "
              "CUE_RAMP_STEPS {}  phase_folds {}".format(
                  state.clip_tail_steps, state.clip_ramp_steps, state.cue_tail_steps,
                  CUE_RAMP_STEPS, state.phase_folds), flush=True)
        if max(widths) == min(widths):
            print("  NOTE: every pose rendered the same width on this scene, so the "
                  "growth path was not exercised by pose variation here -- the "
                  "headroom-0 arm below exercises it against the same real width.",
                  flush=True)
        self.assertEqual(int(state.max_ir_samples), max(widths))
        self.assertEqual(
            state.clip_tail_steps,
            _expected_clip_tail_steps(window, max(widths), hop),
        )
        self.assertEqual(
            state.cue_tail_steps, _expected_cue_tail_steps(max(widths), hop)
        )
        self.assertEqual(state.clip_ramp_steps, int(math.ceil(window / float(hop))))
        self.assertEqual(state.phase_folds, phase_folds(window=window, hop=hop))

    def test_an_unpreallocated_buffer_grows_to_the_real_width_rather_than_truncating(
        self,
    ):
        """``headroom=0`` is the path ``run_episode`` takes when the guard has no shape.

        ADR-0017's word is "preallocated"; ``open_tail`` is preallocated-with-a-grow, and
        the grow is the safety net for a wider pose. This is the net under load: the same
        real IR into a buffer that was sized for none of it.
        """
        clip = _CLIP
        window = len(clip)
        hop = _hop()
        impulse = _IRS[0]
        n_ir = int(impulse.shape[1])

        grown = advance_tail(
            open_tail(window=window, hop=hop, headroom=0),
            ir=impulse,
            clip=clip,
            sounding=True,
        )
        preallocated = advance_tail(
            open_tail(window=window, hop=hop, headroom=max(0, n_ir - 1)),
            ir=impulse,
            clip=clip,
            sounding=True,
        )
        print("\n  --- F29: the growth path against a real IR ---", flush=True)
        print("  headroom 0 -> buffer {} samples, n_grows {}".format(
            int(grown.buffer.shape[1]), grown.n_grows), flush=True)
        print("  headroom {} -> buffer {} samples, n_grows {}".format(
            n_ir - 1, int(preallocated.buffer.shape[1]), preallocated.n_grows),
            flush=True)
        self.assertEqual(grown.n_grows, 1)
        self.assertEqual(preallocated.n_grows, 0)
        self.assertEqual(int(grown.buffer.shape[1]), window + n_ir - 1)
        np.testing.assert_array_equal(grown.buffer, preallocated.buffer)
        # The readouts must be identical across the two allocation paths as well, and the
        # cue one especially: `clip_readout`'s aliasing scar was found on exactly the
        # headroom-0 state, whose buffer is `window` wide and C-contiguous.
        np.testing.assert_array_equal(cue_readout(grown), cue_readout(preallocated))
        self.assertFalse(np.shares_memory(cue_readout(grown), grown.buffer))
        self.assertFalse(np.shares_memory(clip_readout(grown), grown.buffer))


class TestTheRampTheTailAndTheSettledLevel(unittest.TestCase):
    """The shapes ``audio/tail.py``'s "measured shape" block states, on a real room.

    Both readouts, both ramps, both tails. Before ADR-0019 there was one of each and the
    one there was described the analysis window rather than the audio.
    """

    def test_the_two_ramps_are_measured_and_only_one_of_them_exists(self):
        """``CUE_RAMP_STEPS`` is 1 by construction; the clip window fills over 5 folds.

        The contrast is asserted in the form that is independent of the clip AND of the
        room: a sounding fold writes from ``window - hop`` onward, so after ``j + 1``
        folds every clip-readout sample before ``window - (j + 1) * hop`` is EXACTLY
        zero, while the cue readout lies entirely inside the region the last fold wrote.
        A level-based assertion would be neither -- a bursty recording ramps in level
        whatever the buffer does.

        FILL and LEVEL-SETTLE are two different numbers and this test is about FILL only.
        The cue's LEVEL still approaches steady state over ``cue_tail_steps`` folds,
        because the room's reverberation builds up over ``L`` samples; the next test
        measures that. They coincide for the clip readout only because ``N >> L``.
        """
        clip = _CLIP
        window = len(clip)
        hop = _hop()
        impulse = _IRS[0]
        n_ir = int(impulse.shape[1])
        ramp_steps = int(math.ceil(window / float(hop)))
        folds = phase_folds(window=window, hop=hop)

        state = open_tail(window=window, hop=hop, headroom=max(0, n_ir - 1))
        cue_levels: List[float] = []
        clip_levels: List[float] = []
        fills: List[float] = []
        leading_zeros: List[int] = []
        for index in range(max(ramp_steps, folds) + 1):
            state = advance_tail(state, ir=impulse, clip=clip, sounding=True)
            cue_levels.append(rms(cue_readout(state)))
            clip_levels.append(rms(clip_readout(state)))
            fills.append(float(state.clip_source_fill))
            unwritten = max(0, window - (index + 1) * hop)
            leading_zeros.append(unwritten)
            self.assertTrue(
                bool(np.all(clip_readout(state)[:, :unwritten] == 0.0)),
                "after {} sounding folds the clip readout has content before sample "
                "{}, which no fold has written into yet".format(index + 1, unwritten),
            )

        print("\n  --- ADR-0019: the two ramps ---", flush=True)
        print("  N {}  hop {}  L {}   clip_ramp_steps ceil(N/hop) = {}   "
              "CUE_RAMP_STEPS = {}".format(
                  window, hop, n_ir, ramp_steps, CUE_RAMP_STEPS), flush=True)
        print("  fold   clip readout unwritten prefix   clip_source_fill   "
              "clip rms        cue rms", flush=True)
        for index in range(len(cue_levels)):
            print("  {:>4}   {:>28}   {:>16.4f}   {:>10.6g}   {:>10.6g}".format(
                index, leading_zeros[index], fills[index], clip_levels[index],
                cue_levels[index]), flush=True)
        print("  the clip window is {:.0%} full after one fold and full after {}; the "
              "cue window is written whole by fold 0 and has no partial state to be "
              "in".format(fills[0], ramp_steps), flush=True)

        self.assertEqual(CUE_RAMP_STEPS, 1)
        self.assertEqual(state.clip_ramp_steps, ramp_steps)
        self.assertGreater(
            leading_zeros[0],
            0,
            "one fold already wrote the whole clip read window, so there is no fill "
            "ramp to contrast the cue against and this arm proves nothing. That means "
            "hop >= window, which open_tail refuses.",
        )
        self.assertEqual(leading_zeros[ramp_steps - 1], 0)
        for index in range(ramp_steps):
            self.assertAlmostEqual(
                fills[index],
                min(1.0, float((index + 1) * hop) / float(window)),
                places=9,
                msg="clip_source_fill is what the CLAP deferral reads and it did not "
                    "advance one hop per sounding fold at fold {}".format(index),
            )
        self.assertGreater(
            max(cue_levels[:folds]),
            0.0,
            "not one of the loop's {} phases put any energy in the cue window, so this "
            "clip is silent and every arm below is vacuous".format(folds),
        )

    def test_the_two_tails_the_settled_level_and_both_decay_curves_are_printed(self):
        """Both post-offset curves side by side. That pair IS ADR-0019's argument.

        The settled level over bare ``render_through_ir`` is the one with a consequence
        attached: ``onset_rms`` comes off the calibration sweep and the runner applies it
        to an accumulated readout, so if those two domains differ the threshold is
        applied in a domain it was not measured in. Since ADR-0019 the sweep measures the
        CUE level and ``steady_state_render`` is the CLIP control the identity is written
        against; both are checked here, on a real room.

        The number of sounding folds is aligned so the LAST one emits the clip's loudest
        phase (see ``_folds_ending_on_the_loudest_phase``). Without that, a bursty ESC-50
        recording can put its last ring four folds before the offset step, the cue reads
        exactly zero there, and every post-offset assertion below passes vacuously.
        """
        config = AudioConfig()
        clip = _CLIP
        impulse = _IRS[0]
        n_ir = int(impulse.shape[1])
        window = len(clip)
        hop = _hop()
        bed_cue = bed_signal(hop, config.bed_rms)
        clip_tail_steps = _expected_clip_tail_steps(window, n_ir, hop)
        cue_tail_steps = _expected_cue_tail_steps(n_ir, hop)
        energies = _phase_energies(clip, hop)
        sounding_folds = _folds_ending_on_the_loudest_phase(
            clip_tail_steps + 2, energies
        )

        state = open_tail(window=window, hop=hop, headroom=max(0, n_ir - 1))
        readings: List[Tuple[int, float]] = []
        step = 0
        for _ in range(sounding_folds):
            state, cue = heard_step(
                state, ir=impulse, clip=clip, bed_cue=bed_cue, sounding=True
            )
            readings.append((step, rms(cue)))
            step += 1
        offset_step = step
        settled_clip = rms(clip_readout(state))
        settled_cue = rms(cue_readout(state))
        bare = rms(render_through_ir(impulse, clip))
        steady = rms(steady_state_render(impulse, clip, hop=hop))
        level = cue_level(steady_state_cue_rms(impulse, clip, hop=hop))

        cue_decay: List[float] = []
        clip_decay: List[float] = []
        buffer_energy: List[float] = []
        for _ in range(clip_tail_steps + 1):
            state, cue = heard_step(
                state, ir=None, clip=clip, bed_cue=bed_cue, sounding=False
            )
            cue_decay.append(rms(cue_readout(state)))
            clip_decay.append(rms(clip_readout(state)))
            buffer_energy.append(float(np.sum(np.square(state.buffer))))
            readings.append((step, rms(cue)))
            step += 1
        audible = post_offset_audible_steps(
            readings=readings,
            offset_step=offset_step,
            bed_rms=config.bed_rms,
            tolerance=config.pre_onset_rms_tol,
        )

        print("\n  --- ADR-0019: the two tails and the settled level ---", flush=True)
        print("  clip {}".format(_CLIP_LABEL), flush=True)
        print("  N {}  L {}  hop {}".format(window, n_ir, hop), flush=True)
        print("  cue  tail ceil((hop + L - 1)/hop) = {}   <- the ROOM".format(
            cue_tail_steps), flush=True)
        print("  clip tail ceil((N   + L - 1)/hop) = {}   <- the analysis window "
              "emptying".format(clip_tail_steps), flush=True)
        print("  loop phase energies (fold {} emits phase {}, the loudest): {}".format(
            sounding_folds - 1, (sounding_folds - 1) % len(energies),
            " ".join("{:.4g}".format(value) for value in energies)), flush=True)
        print("  settled clip readout rms     {:.6g}".format(settled_clip), flush=True)
        print("  settled cue  readout rms     {:.6g}   (one phase of {})".format(
            settled_cue, len(energies)), flush=True)
        print("  bare render_through_ir rms   {:.6g}   settled_clip/bare {:.4f}x".format(
            bare, settled_clip / max(bare, 1e-30)), flush=True)
        print("  steady_state_render rms      {:.6g}   settled_clip/steady {:.6f}x   "
              "<- the CLIP control".format(
                  steady, settled_clip / max(steady, 1e-30)), flush=True)
        print("  cue_level(steady_state_cue_rms) {:.6g}   level/steady {:.9f}x   "
              "<- THE IDENTITY: onset_rms did not move".format(
                  level, level / max(steady, 1e-30)), flush=True)
        print("  post-offset CUE  readout over settled_cue : {}".format(
            " ".join("{:.4f}".format(value / max(settled_cue, 1e-30))
                     for value in cue_decay)), flush=True)
        print("  post-offset CLIP readout over settled_clip: {}".format(
            " ".join("{:.4f}".format(value / max(settled_clip, 1e-30))
                     for value in clip_decay)), flush=True)
        print("  post_offset_audible_steps (bed {:.3g}, tol {:.0%}): {} of {} silent "
              "steps{}".format(
                  config.bed_rms, config.pre_onset_rms_tol, audible, len(cue_decay),
                  "   <- the silent phase was a HARD CUT for this clip"
                  if audible == 0 else ""), flush=True)
        print("  (measured on the CUE trace since the split, so it counts steps at "
              "which the ROOM was still audible. Its values FALL against every "
              "pre-split run and that is the correction, not a regression. Printed and "
              "not asserted; SilentPhaseTally prints it beside every SWS.)", flush=True)

        self.assertEqual(state.clip_tail_steps, clip_tail_steps)
        self.assertEqual(state.cue_tail_steps, cue_tail_steps)
        self.assertLess(
            cue_tail_steps,
            clip_tail_steps,
            "the cue tail is not shorter than the clip tail on this room, so the split "
            "bought nothing here and every claim below about the analysis window is "
            "about a window that is not longer than a step",
        )
        self.assertAlmostEqual(
            settled_clip / max(steady, 1e-30),
            1.0,
            places=3,
            msg="the fold does not converge to steady_state_render, which is the CLIP "
                "control the ADR-0019 identity is written against.",
        )
        self.assertAlmostEqual(
            level / max(steady, 1e-30),
            1.0,
            places=6,
            msg="cue_level(steady_state_cue_rms(...)) != rms(steady_state_render(...)) "
                "on this room. That identity is the ONLY proof that onset_rms did not "
                "move when the calibration sweep changed domain: the phase_folds cue "
                "windows are disjoint, consecutive and tile the settled period, so "
                "their quadratic mean is the clip readout's RMS exactly. If it fails "
                "here, the threshold is being placed in one domain and applied in "
                "another -- onset.py:81-84's named failure.",
        )
        self.assertGreater(settled_clip, 0.95 * bare)
        self.assertLess(
            settled_clip,
            1.5 * bare,
            "the accumulated level is more than 1.5x bare render_through_ir on this "
            "room. audio/tail.py measured 1.0014x to 1.0146x on synthetic IRs; a real "
            "room this far out moves every threshold calibrated on the bare render.",
        )
        # The fence posts. Exactly zero is exactly right: the readout is a window over a
        # buffer that slides by `hop` a step, so both go to zero at a computable fold and
        # neither may outlive it.
        self.assertEqual(
            cue_decay[cue_tail_steps - 1],
            0.0,
            "the CUE readout still holds energy cue_tail_steps folds after the last "
            "sounding step. That is the fence post smoke criterion 4 measures the "
            "silent phase's level from since ADR-0019.",
        )
        self.assertEqual(
            clip_decay[clip_tail_steps - 1],
            0.0,
            "the CLIP readout still holds energy clip_tail_steps folds after the last "
            "sounding step, so the CLAP window outlives what the record says it can",
        )
        # Conditional, and the branch is printed rather than hidden. At
        # `cue_tail_steps == 1` the cue readout is zero one fold after the offset step BY
        # CONSTRUCTION -- the room fits inside a step -- so asserting otherwise would go
        # red for a scene fact. That scene fact gets its own finding in
        # TestTheCueTailIsReverberation; it is not restated as a failure here.
        if cue_tail_steps > 1:
            self.assertGreater(
                cue_decay[0],
                0.0,
                "the cue readout was ALREADY silent one fold after the offset step, on "
                "a run whose last sounding fold was aligned to the clip's loudest "
                "phase. Either this room's reverb is under one hop -- which a "
                "cue_tail_steps of {} denies -- or the alignment above stopped "
                "working.".format(cue_tail_steps),
            )
        else:
            print("  NOTE: cue_tail_steps is 1, so this room's IR fits inside one "
                  "simulator step and the silent phase is an honest hard cut. There is "
                  "no cue decay to assert; TestTheCueTailIsReverberation reports that "
                  "as its own finding.", flush=True)
        # NON-STRICT, and the loosening is a correction rather than a concession. A
        # silent fold slides the buffer left by `hop` and drops whatever was leftmost; if
        # the clip's loop put a silent hop there, the buffer loses exactly nothing and two
        # consecutive folds report bit-identical energy. Measured here on a 0.6 s
        # transient looped on a 5 s period: 11176216.0 twice in a row. The claim this
        # assertion is for is the one its message states -- energy never ENTERS on a
        # silent step -- and the strict drop is carried by the two fence posts below,
        # which force the buffer from nonzero to exactly zero inside one fold.
        for earlier, later in zip(buffer_energy, buffer_energy[1:]):
            if earlier == 0.0:
                self.assertEqual(later, 0.0)
            else:
                self.assertLessEqual(
                    later, earlier, "energy entered the buffer on a silent step"
                )
        self.assertGreater(
            buffer_energy[clip_tail_steps - 2],
            0.0,
            "the buffer emptied a step EARLY: at clip_tail_steps - 1 folds there is "
            "still source in it, and tail_is_active reads that fence post",
        )
        self.assertEqual(buffer_energy[clip_tail_steps - 1], 0.0)


class TestTheCueTailIsReverberation(unittest.TestCase):
    """The anechoic control, on a real room. This is the claim ADR-0019 rests on.

    ``cue_tail_steps`` is documented as *the first number on the record that is evidence
    the geometric acoustics did any work*, and ``clip_tail_steps`` is documented as
    evidence of nothing at all. Both halves are testable and the test is the same one: a
    1-sample IR, which is a room with literally no reverberation, folded through the same
    clip at the same hop.

    The Mac measures this on synthetic IRs. Only here is the room real, which is why the
    file exists (ADR-0014: a capability is exercised, never proxied).
    """

    def test_an_anechoic_ir_reproduces_the_clip_tail_and_collapses_the_cue_tail(self):
        """Both arms, one test, because the two curves are only meaningful together.

        HEALTHY: with the real IR, ``cue_tail_steps > 1`` and the cue readout is still
        loud one fold after the offset step and exactly zero at ``cue_tail_steps``.

        FORCED FAILURE: with a 1-sample IR, ``cue_tail_steps == 1`` and the cue readout
        is EXACTLY zero on the first silent fold -- an honest hard cut -- while its CLIP
        decay curve tracks the real room's to a point or two. That second half is the
        defect stated as a measurement: the curve the pre-ADR-0019 agent read as a reverb
        tail is reproduced by a room that has no reverb.
        """
        clip = _CLIP
        impulse = _IRS[0]
        n_ir = int(impulse.shape[1])
        window = len(clip)
        hop = _hop()
        anechoic = _anechoic_like(impulse)
        energies = _phase_energies(clip, hop)

        self.assertGreaterEqual(
            n_ir,
            2,
            "this scene rendered a 1-sample IR, so the real room IS the anechoic "
            "control and this test cannot tell them apart",
        )

        curves = {}
        for name, ir in (("room", impulse), ("anechoic", anechoic)):
            n_this = int(ir.shape[1])
            clip_tail = _expected_clip_tail_steps(window, n_this, hop)
            cue_tail = _expected_cue_tail_steps(n_this, hop)
            folds = _folds_ending_on_the_loudest_phase(clip_tail + 2, energies)
            state = open_tail(window=window, hop=hop, headroom=max(0, n_this - 1))
            for _ in range(folds):
                state = advance_tail(state, ir=ir, clip=clip, sounding=True)
            settled_cue = rms(cue_readout(state))
            settled_clip = rms(clip_readout(state))
            cue_curve: List[float] = []
            clip_curve: List[float] = []
            for _ in range(clip_tail + 1):
                state = advance_tail(state, ir=None, clip=clip, sounding=False)
                cue_curve.append(rms(cue_readout(state)) / max(settled_cue, 1e-30))
                clip_curve.append(rms(clip_readout(state)) / max(settled_clip, 1e-30))
            curves[name] = {
                "L": n_this,
                "cue_tail": cue_tail,
                "clip_tail": clip_tail,
                "cue": cue_curve,
                "clip": clip_curve,
                "settled_cue": settled_cue,
                "settled_clip": settled_clip,
            }

        room = curves["room"]
        dead = curves["anechoic"]
        # Only where BOTH clip curves are still above zero: past that point one of them
        # has hit its fence post and a difference there is the fence post, not the room.
        overlap = [
            index
            for index in range(min(len(room["clip"]), len(dead["clip"])))
            if room["clip"][index] > 0.0 and dead["clip"][index] > 0.0
        ]
        gaps = [abs(room["clip"][index] - dead["clip"][index]) for index in overlap]

        print("\n  --- ADR-0019 CONTROL: is the tail the room, or the read window? ---",
              flush=True)
        print("  room     IR L {:>7}   cue_tail {}   clip_tail {}".format(
            room["L"], room["cue_tail"], room["clip_tail"]), flush=True)
        print("  anechoic IR L {:>7}   cue_tail {}   clip_tail {}".format(
            dead["L"], dead["cue_tail"], dead["clip_tail"]), flush=True)
        print("  post-offset CLIP decay, room     : {}".format(
            " ".join("{:.4f}".format(value) for value in room["clip"])), flush=True)
        print("  post-offset CLIP decay, anechoic : {}".format(
            " ".join("{:.4f}".format(value) for value in dead["clip"])), flush=True)
        print("  worst CLIP gap over the {} steps both are alive: {:.4f} "
              "({:.2f} points)".format(
                  len(overlap), max(gaps) if gaps else 0.0,
                  100.0 * (max(gaps) if gaps else 0.0)), flush=True)
        print("  post-offset CUE  decay, room     : {}".format(
            " ".join("{:.4f}".format(value) for value in room["cue"])), flush=True)
        print("  post-offset CUE  decay, anechoic : {}".format(
            " ".join("{:.4f}".format(value) for value in dead["cue"])), flush=True)
        print("  the room's cue is {:.4f} of settled one fold after the offset step "
              "where the anechoic control is {:.4f}; CUE gap {:.4f} against CLIP gap "
              "{:.4f}".format(
                  room["cue"][0], dead["cue"][0], room["cue"][0] - dead["cue"][0],
                  max(gaps) if gaps else 0.0), flush=True)
        print("  READ IT THIS WAY: the CLIP curves agree, so that curve was never "
              "reverb -- it is the {} sample read window emptying {} samples at a time. "
              "The CUE curves do not agree, so that one is.".format(window, hop),
              flush=True)

        # HEALTHY ARM: the room outlives a step, and it is measurably still there.
        self.assertGreater(
            room["cue_tail"],
            1,
            "cue_tail_steps is 1 on this room: its IR fits inside one step, the silent "
            "phase is an honest hard cut, and this scene cannot demonstrate a reverb "
            "tail at all. That is a scene fact and it is worth knowing, but it makes "
            "the arm below vacuous.",
        )
        self.assertGreater(
            room["cue"][0],
            0.0,
            "the room's cue readout is already exactly zero one fold after the offset "
            "step, which contradicts a cue_tail_steps of {}".format(room["cue_tail"]),
        )
        self.assertEqual(
            room["cue"][room["cue_tail"] - 1],
            0.0,
            "the room's cue readout outlives cue_tail_steps folds",
        )

        # FORCED FAILURE ARM: the control, and it must fire on both halves.
        self.assertEqual(dead["cue_tail"], 1)
        self.assertEqual(
            dead["cue"][0],
            0.0,
            "a 1-sample IR left something in the cue window one fold after the offset "
            "step. Nothing can be there -- the fold wrote hop samples and the buffer "
            "slid hop -- so this control is not the control it claims to be.",
        )
        self.assertGreaterEqual(
            len(overlap),
            2,
            "fewer than two silent steps where both CLIP curves are alive, so the "
            "'anechoic reproduces the clip decay' half cannot be measured here",
        )
        self.assertLess(
            max(gaps),
            0.05,
            "the anechoic control's CLIP decay is more than 5 points away from the real "
            "room's. On this Mac the gap is 1.3 points and that closeness IS the defect "
            "ADR-0019 repairs -- the clip readout's decay is the analysis window "
            "emptying. A large gap here would mean this room's reverb really does "
            "dominate a 5 s window, which would make L >= N and contradict the guard.",
        )
        # THE CONTRAST, and it is scale-free on purpose. A fixed threshold on the cue
        # gap would be a guess about how reverberant an HM3D room is; what ADR-0019
        # actually claims is that the anechoic control is a WORSE approximation of the
        # room in the cue domain than in the clip domain, and that is a comparison
        # between two numbers this test already has.
        self.assertGreater(
            room["cue"][0] - dead["cue"][0],
            max(gaps),
            "the anechoic control tracks the room's CUE decay at least as closely as it "
            "tracks the CLIP decay ({:.4f} against {:.4f}), so the split bought no new "
            "evidence about the room on this scene and ADR-0019's central claim is "
            "unsupported here.".format(room["cue"][0] - dead["cue"][0], max(gaps)),
        )


class TestTheCuePhaseProfile(unittest.TestCase):
    """How intermittent the agent's per-step reading is, for the clip a run actually plays.

    Nothing else in the tree can produce this number: it needs a real IR and the staged
    ESC-50 recording together. ``cue_crest`` and ``cue_min_ratio`` ride on the calibration
    record for exactly this reason, and this is where their scale is established.
    """

    def test_the_loop_phases_the_crest_and_the_identity_are_printed(self):
        """The identity in a fourth configuration, and the crest that prices the risk.

        ``cue_level`` over the loop's phases equals ``rms(steady_state_render(...))``
        exactly, because the ``phase_folds`` cue windows are disjoint, consecutive and
        tile the settled period an integer number of times. The Mac measures ratio
        1.000000000000 at three synthetic configurations; this is the fourth, with a
        room and a recording.

        The crest is the honest cost of the split, recorded and deliberately NOT gated: a
        clip whose energy sits inside one hop is loud on one fold and near-silent on the
        others, so the gate can pass on the quadratic mean while four folds in five read
        near the bed. Refusing such a clip would be a second change and would make four
        of ESC-50's five classes unusable.
        """
        clip = _CLIP
        impulse = _IRS[0]
        window = len(clip)
        hop = _hop()
        folds = phase_folds(window=window, hop=hop)

        phases = steady_state_cue_rms(impulse, clip, hop=hop)
        level = cue_level(phases)
        crest = cue_crest(phases)
        min_ratio = cue_min_ratio(phases)
        steady = rms(steady_state_render(impulse, clip, hop=hop))
        energies = _phase_energies(clip, hop)

        print("\n  --- ADR-0019: the loop's phase profile for {} ---".format(
            _CLIP_LABEL), flush=True)
        print("  phase_folds = N // gcd(N, hop) = {} // gcd({}, {}) = {}".format(
            window, window, hop, folds), flush=True)
        print("  cue rms by phase, oldest first: {}".format(
            " ".join("{:.6g}".format(value) for value in phases)), flush=True)
        print("  same, over the level:           {}".format(
            " ".join("{:.4f}".format(value / max(level, 1e-30))
                     for value in phases)), flush=True)
        print("  emitted chunk energy by phase:  {}".format(
            " ".join("{:.4g}".format(value) for value in energies)), flush=True)
        print("  cue_level (quadratic mean)  {:.6g}".format(level), flush=True)
        print("  rms(steady_state_render)    {:.6g}   ratio {:.12f}   <- THE IDENTITY"
              .format(steady, level / max(steady, 1e-30)), flush=True)
        print("  cue_crest      {:.4f}   (1.0 = flat across the loop; 2.24 for a 0.6 s "
              "transient on a 5 s loop)".format(crest), flush=True)
        print("  cue_min_ratio  {:.4f}   (how nearly silent the quietest fold is)"
              .format(min_ratio), flush=True)
        print("  CONSEQUENCE: the onset is one-shot and monotone-latching, so this "
              "delays the first crossing by at most phase_folds - 1 = {} steps and can "
              "never prevent it. That bound reaches the metrics bag as "
              "sounding_phase_folds.".format(folds - 1), flush=True)
        print("  OPEN HAZARD (not fixed here): RISING_WINDOW is 5 and phase_folds is "
              "{}. They are equal at the shipped defaults, so each of is_rising's two "
              "windows spans exactly one loop period and the phase cancels in the mean "
              "BY ARITHMETIC COINCIDENCE. Change step_seconds or the clip length and "
              "the windows beat against the period.".format(folds), flush=True)

        self.assertEqual(len(phases), folds)
        self.assertAlmostEqual(
            level / max(steady, 1e-30),
            1.0,
            places=6,
            msg="the quadratic mean of the loop's cue RMSs is not the clip readout's "
                "RMS on this room and this recording. That identity is what makes "
                "onset_rms invariant across ADR-0019; without it the threshold moved "
                "and every historic gate number is unpriceable.",
        )
        self.assertGreaterEqual(crest, 1.0)
        self.assertLessEqual(min_ratio, 1.0)
        self.assertLessEqual(
            crest,
            math.sqrt(folds) + 1e-6,
            "the crest cannot exceed sqrt(phase_folds) -- that is the value when one "
            "fold carries all the energy -- so a larger one means cue_level is not the "
            "quadratic mean it is documented to be",
        )


class TestTheThreeScatterArms(unittest.TestCase):
    """The pre-registered prediction ``CalibrationResult`` carries, at a real held pose.

    ``climb_eps`` sizes the rising predicate's epsilon from the calibration record's
    scatter, and ADR-0019 renamed that field rather than redefining it, because every
    number on disk under the old name is the CLIP-loop estimate. Three arms now:

        single  ``sweep_render_scatter``        the pre-ADR-0017 estimator, unchanged
        clip    ``sweep_loop_scatter(...).clip`` the ADR-0017 arm, free from the same folds
        cue     ``sweep_loop_scatter(...).cue``  what ``climb_eps`` reads since ADR-0019

    The prediction is ``single > cue > clip``, because the cue readout averages
    ``cue_tail_steps`` renders where the clip readout averages ``clip_tail_steps``. It is
    PRINTED whichever way it lands and deliberately not gated: ``sweep_loop_scatter``'s
    own docstring already records Mac measurements where ``single/cue`` came out at 0.63,
    because the loop PHASE enters the cue arm and the renderer is not the only term. A
    hard assertion here would be inventing a result the box is being asked to produce.
    """

    def test_the_three_scatter_arms_and_the_bed_cross_term_are_printed(self):
        """The single most valuable number the first box run after ADR-0019 produces.

        If the cue arm comes out at or above the single arm, the averaging model behind
        ``climb_eps`` is wrong for a second time and the epsilon is again the wrong size.
        That is a finding, not a failure, and it belongs in the printout.

        What IS asserted is the shape of the measurement: the render bill each sweep
        pays, the length of what each returns, and the non-vacuity precondition -- that
        the renderer really did disagree with itself across repeats at a held pose. A
        scatter measured over identical renders would be zero for a reason that has
        nothing to do with the split.
        """
        config = AudioConfig()
        clip = _CLIP
        window = len(clip)
        hop = _hop()
        if _HELD_POSE is None or len(_HELD_RENDERS) < 2:
            self.fail(
                "setUpModule collected no held-pose renders, so the three scatter arms "
                "cannot be measured. THIS IS A NOT_RUN, WHICH IS RED."
            )
        held_l = max(int(ir.shape[1]) for ir in _HELD_RENDERS)
        clip_tail_steps = _expected_clip_tail_steps(window, held_l, hop)
        cue_tail_steps = _expected_cue_tail_steps(held_l, hop)

        single_render_at, single_used = _sequential_render_at(_HELD_RENDERS)
        single = sweep_render_scatter(
            _HELD_POSE, single_render_at, clip, SCATTER_REPEATS
        )
        loop_render_at, loop_used = _sequential_render_at(
            _HELD_RENDERS[len(single_used):]
        )
        loop = sweep_loop_scatter(
            _HELD_POSE, loop_render_at, clip, SCATTER_REPEATS, hop=hop
        )

        single_sd = render_scatter_of(single)
        cue_sd = render_scatter_of(loop.cue)
        clip_sd = render_scatter_of(loop.clip)
        distinct = len({ir.tobytes() for ir in _HELD_RENDERS})

        print("\n  --- ADR-0019: the three scatter arms at a held pose ---", flush=True)
        print("  pose ({:.2f}, {:.2f}, {:.2f})   L {}   cue_tail {}   clip_tail {}   "
              "repeats {}".format(
                  _HELD_POSE.x, _HELD_POSE.y, _HELD_POSE.z, held_l, cue_tail_steps,
                  clip_tail_steps, SCATTER_REPEATS), flush=True)
        print("  {} live renders of that pose; {} of them byte-distinct".format(
            len(_HELD_RENDERS), distinct), flush=True)
        print("  render_at calls: single {} (expected {}), loop {} (expected "
              "clip_tail + repeats = {})".format(
                  len(single_used), SCATTER_REPEATS, len(loop_used),
                  clip_tail_steps + SCATTER_REPEATS), flush=True)
        print("  single  mean {:.6g}   SD {:.6g}   SD/mean {:.3e}".format(
            sum(single) / len(single), single_sd,
            single_sd / max(sum(single) / len(single), 1e-30)), flush=True)
        print("  cue     mean {:.6g}   SD {:.6g}   SD/mean {:.3e}".format(
            sum(loop.cue) / len(loop.cue), cue_sd,
            cue_sd / max(sum(loop.cue) / len(loop.cue), 1e-30)), flush=True)
        print("  clip    mean {:.6g}   SD {:.6g}   SD/mean {:.3e}".format(
            sum(loop.clip) / len(loop.clip), clip_sd,
            clip_sd / max(sum(loop.clip) / len(loop.clip), 1e-30)), flush=True)
        print("  ratios: single/cue {:.4g}   cue/clip {:.4g}   single/clip {:.4g}"
              .format(single_sd / max(cue_sd, 1e-30), cue_sd / max(clip_sd, 1e-30),
                      single_sd / max(clip_sd, 1e-30)), flush=True)
        held = single_sd > cue_sd > clip_sd
        print("  PRE-REGISTERED single > cue > clip: {}".format(
            "HELD" if held else "DID NOT HOLD -- the averaging model behind climb_eps "
            "is wrong for a second time and the epsilon is again the wrong size"),
            flush=True)
        print("  climb_eps reads the CUE arm. eps would be sized from {:.6g}".format(
            cue_sd), flush=True)
        print("  cue samples: {}".format(
            " ".join("{:.6g}".format(value) for value in loop.cue)), flush=True)
        print("  clip samples: {}".format(
            " ".join("{:.6g}".format(value) for value in loop.clip)), flush=True)

        # THE BED CROSS-TERM, at both widths, on a FIXED IR so the only thing moving is
        # the readout's rotation. `calibration.py` names this residual: the sweep measures
        # the source alone while the runner reads `mix_bed(readout, bed)`, and the term
        # scales as 1/sqrt(n) in the window length, so it is sqrt(N/hop) larger at the cue
        # width. Unmeasured after the split until here. Printed, not gated -- gating an
        # unmeasured prediction is how a number becomes a claim.
        bed_cue = bed_signal(hop, config.bed_rms)
        bed_clip = bed_signal(window, config.bed_rms)
        bed_level = float(config.bed_rms)
        impulse = _HELD_RENDERS[0]
        state = open_tail(window=window, hop=hop, headroom=max(0, held_l - 1))
        for _ in range(clip_tail_steps + 1):
            state = advance_tail(state, ir=impulse, clip=clip, sounding=True)
        cue_cross: List[float] = []
        clip_cross: List[float] = []
        for _ in range(phase_folds(window=window, hop=hop)):
            state = advance_tail(state, ir=impulse, clip=clip, sounding=True)
            dry_cue = rms(cue_readout(state))
            wet_cue = rms(mix_bed(cue_readout(state), bed_cue))
            dry_clip = rms(clip_readout(state))
            wet_clip = rms(mix_bed(clip_readout(state), bed_clip))
            cue_cross.append(
                (wet_cue ** 2 - dry_cue ** 2 - bed_level ** 2)
                / max(dry_cue ** 2, 1e-30)
            )
            clip_cross.append(
                (wet_clip ** 2 - dry_clip ** 2 - bed_level ** 2)
                / max(dry_clip ** 2, 1e-30)
            )
        print("  --- the bed cross-term, both widths, fixed IR ---", flush=True)
        print("  cue  2<s,b>/rms(s)^2 by phase: {}".format(
            " ".join("{:+.3e}".format(value) for value in cue_cross)), flush=True)
        print("  clip 2<s,b>/rms(s)^2 by phase: {}".format(
            " ".join("{:+.3e}".format(value) for value in clip_cross)), flush=True)
        print("  spread (max - min): cue {:.3e}   clip {:.3e}   ratio {:.3f}   "
              "predicted sqrt(N/hop) = {:.3f}".format(
                  max(cue_cross) - min(cue_cross), max(clip_cross) - min(clip_cross),
                  (max(cue_cross) - min(cue_cross))
                  / max(max(clip_cross) - min(clip_cross), 1e-30),
                  math.sqrt(window / float(hop))), flush=True)

        self.assertEqual(len(single_used), SCATTER_REPEATS)
        # The settle is read off the FIRST render the loop arm was handed, exactly as
        # `sweep_loop_scatter` reads it off `state.clip_tail_steps` after folding that
        # render. Deriving it from the pool's widest IR instead would make this assertion
        # fail whenever the renderer returned two different widths at one pose, which is
        # a fact about the scene rather than about the sweep.
        self.assertEqual(
            len(loop_used),
            _expected_clip_tail_steps(window, int(loop_used[0].shape[1]), hop)
            + SCATTER_REPEATS,
        )
        self.assertEqual(len(single), SCATTER_REPEATS)
        self.assertEqual(len(loop.cue), SCATTER_REPEATS)
        self.assertEqual(len(loop.clip), SCATTER_REPEATS)
        self.assertGreater(
            distinct,
            1,
            "every held-pose render came back byte-identical, so all three scatter "
            "arms are measuring zero renderer disagreement and none of them can be "
            "compared. That is itself a finding about this box's renderer -- ticket 06 "
            "and detour-2 both assumed otherwise -- but it makes the prediction above "
            "unjudgeable here.",
        )
        for name, sd in (("single", single_sd), ("cue", cue_sd), ("clip", clip_sd)):
            self.assertTrue(
                math.isfinite(sd),
                "the {} scatter arm is not finite, so climb_eps would size an epsilon "
                "off a NaN and is_rising would compare False forever".format(name),
            )


class TestTheForcedFailureArms(unittest.TestCase):
    """ADR-0014's second half, twice. Each arm proves it CAN fire before it fires."""

    def test_a_truncating_buffer_makes_the_room_read_anechoic_in_the_cue_domain(self):
        """The implementation ``open_tail`` refuses to be, folded on the same real IR.

        ``advance_tail``'s docstring names truncation as "the dangerous one: it loses
        reverb silently and the signal stays plausible". Plausible is the word under test,
        and since ADR-0019 the lie has a sharper shape. The reverb a non-growing buffer
        throws away is exactly the samples past index ``window``, and those are exactly
        the samples that would have slid INTO the cue window on the next silent fold. So
        a truncating buffer does not merely lose a little energy: it reads
        ``cue_tail_steps == 1``, which is the anechoic answer, on a room whose honest cue
        tail is longer -- both numbers are asserted below off this box's own IR width.

        Both readouts are printed. The clip arm is the old, weak demonstration -- a few
        percent, late in the decay -- and it is kept so the two can be compared.
        """
        config = AudioConfig()
        clip = _CLIP
        impulse = _IRS[0]
        n_ir = int(impulse.shape[1])
        window = len(clip)
        hop = _hop()
        bed_cue = bed_signal(hop, config.bed_rms)
        clip_tail_steps = _expected_clip_tail_steps(window, n_ir, hop)
        cue_tail_steps = _expected_cue_tail_steps(n_ir, hop)
        energies = _phase_energies(clip, hop)
        sounding_folds = _folds_ending_on_the_loudest_phase(
            clip_tail_steps + 2, energies
        )

        self.assertGreaterEqual(
            n_ir,
            2,
            "a 1-sample IR writes nothing past the read window, so a truncating buffer "
            "would be identical to the honest one and this arm would prove nothing",
        )
        self.assertGreater(
            cue_tail_steps,
            1,
            "this room's cue tail is already 1 fold, which is what a truncating buffer "
            "produces, so the two are indistinguishable here",
        )

        honest = open_tail(window=window, hop=hop, headroom=max(0, n_ir - 1))
        cut = open_tail(window=window, hop=hop, headroom=0)
        for _ in range(sounding_folds):
            honest = advance_tail(honest, ir=impulse, clip=clip, sounding=True)
            cut = advance_tail(cut, ir=impulse, clip=clip, sounding=True)
            # The truncation itself: everything past the read window is thrown away, which
            # is precisely the `L - 1` samples `render_through_ir` also discards and the
            # accumulator exists to carry.
            cut = dataclasses.replace(
                cut, buffer=np.array(cut.buffer[:, :window], dtype=np.float32, order="C")
            )
        sounding_cue_gap = abs(
            rms(cue_readout(honest)) - rms(cue_readout(cut))
        ) / max(rms(cue_readout(honest)), 1e-30)
        sounding_clip_gap = abs(
            rms(clip_readout(honest)) - rms(clip_readout(cut))
        ) / max(rms(clip_readout(honest)), 1e-30)

        honest_cue: List[float] = []
        cut_cue: List[float] = []
        honest_clip: List[float] = []
        cut_clip: List[float] = []
        for _ in range(clip_tail_steps + 1):
            honest, _cue = heard_step(
                honest, ir=None, clip=clip, bed_cue=bed_cue, sounding=False
            )
            cut = advance_tail(cut, ir=None, clip=clip, sounding=False)
            cut = dataclasses.replace(
                cut, buffer=np.array(cut.buffer[:, :window], dtype=np.float32, order="C")
            )
            honest_cue.append(rms(cue_readout(honest)))
            cut_cue.append(rms(cue_readout(cut)))
            honest_clip.append(rms(clip_readout(honest)))
            cut_clip.append(rms(clip_readout(cut)))
        # Only where the honest readout still has something to be short OF: once it is
        # exactly zero both are, and a ratio there says nothing.
        cue_gaps = [
            (honest_level - cut_level) / honest_level
            for honest_level, cut_level in zip(honest_cue, cut_cue)
            if honest_level > 0.0
        ]
        clip_gaps = [
            (honest_level - cut_level) / honest_level
            for honest_level, cut_level in zip(honest_clip, cut_clip)
            if honest_level > 0.0
        ]
        honest_cue_zero = next(
            (i + 1 for i, level in enumerate(honest_cue) if level == 0.0), None
        )
        cut_cue_zero = next(
            (i + 1 for i, level in enumerate(cut_cue) if level == 0.0), None
        )
        honest_clip_zero = next(
            (i + 1 for i, level in enumerate(honest_clip) if level == 0.0), None
        )
        cut_clip_zero = next(
            (i + 1 for i, level in enumerate(cut_clip) if level == 0.0), None
        )

        print("\n  --- F29 FORCED FAILURE: the truncating buffer ---", flush=True)
        print("  IR {} samples, so a non-growing buffer drops {} reverb samples every "
              "sounding step".format(n_ir, n_ir - 1), flush=True)
        print("  honest post-offset CUE : {}".format(
            " ".join("{:.4g}".format(level) for level in honest_cue)), flush=True)
        print("  cut    post-offset CUE : {}".format(
            " ".join("{:.4g}".format(level) for level in cut_cue)), flush=True)
        print("  honest post-offset CLIP: {}".format(
            " ".join("{:.4g}".format(level) for level in honest_clip)), flush=True)
        print("  cut    post-offset CLIP: {}".format(
            " ".join("{:.4g}".format(level) for level in cut_clip)), flush=True)
        print("  relative gap WHILE SOUNDING   cue {:.4%}   clip {:.4%}   <- the lie, "
              "invisible".format(sounding_cue_gap, sounding_clip_gap), flush=True)
        print("  gap AT THE FIRST SILENT FOLD  cue {:.4%}   clip {:.4%}".format(
            cue_gaps[0] if cue_gaps else 0.0,
            clip_gaps[0] if clip_gaps else 0.0), flush=True)
        print("  WORST gap in the silent phase cue {:.4%}   clip {:.4%}".format(
            max(cue_gaps) if cue_gaps else 0.0,
            max(clip_gaps) if clip_gaps else 0.0), flush=True)
        print("  silent folds to exactly zero: CUE  honest {}  truncated {}   "
              "(cue_tail_steps {})".format(
                  honest_cue_zero, cut_cue_zero, cue_tail_steps), flush=True)
        print("  silent folds to exactly zero: CLIP honest {}  truncated {}   "
              "(clip_tail_steps {})".format(
                  honest_clip_zero, cut_clip_zero, clip_tail_steps), flush=True)
        print("  READ IT THIS WAY: the truncated buffer's CUE tail is {} fold, which is "
              "the ANECHOIC answer, on a room whose honest cue tail is {}.".format(
                  cut_cue_zero, honest_cue_zero), flush=True)

        self.assertEqual(
            honest_cue_zero,
            cue_tail_steps,
            "the honest fold's cue tail is not the length its own record claims",
        )
        self.assertEqual(
            cut_cue_zero,
            1,
            "the truncating buffer's cue readout was NOT exactly zero on the first "
            "silent fold. It has nothing past the read window to slide in, so this arm "
            "did not fire and proves nothing about the growth path.",
        )
        self.assertGreater(
            honest_cue_zero,
            cut_cue_zero,
            "truncation cost the cue tail nothing, so the loss this arm exists to "
            "demonstrate is not where advance_tail says it is",
        )
        self.assertGreater(
            cue_gaps[0],
            0.0,
            "the truncating buffer read IDENTICALLY to the honest one one fold after "
            "the offset step",
        )
        self.assertGreater(
            max(cue_gaps),
            sounding_cue_gap,
            "truncation cost no more at its worst in the silent phase than it did while "
            "the source was sounding",
        )
        self.assertIsNotNone(honest_clip_zero)
        self.assertIsNotNone(cut_clip_zero)
        self.assertLessEqual(cut_clip_zero, honest_clip_zero)

    def test_the_silent_phase_guard_refuses_an_sws_over_a_record_with_no_tail(self):
        """The record this box's own fold produces, then the same record crippled.

        ADR-0017 line 49 bars reporting an SWS before the accumulation buffer is in, and
        ``silent_phase_tally`` is where that bar lives. Both arms run through the SAME
        call: the real record publishes a rate, and one field short of it the call raises
        rather than counting the episode. A tally that could not tell them apart would
        make ``sws_status: measured`` decorative.

        ``cue_tail_steps`` rides on the record since ADR-0019 and is deliberately NOT one
        of ``tail_is_active``'s clauses -- every clause is evidence that the buffer folded
        a render, and the cue tail is evidence about the ROOM. That is asserted below
        rather than left as prose, because a fifth clause is exactly the scope this change
        makes newly reachable and deliberately does not take.
        """
        config = AudioConfig()
        clip = _CLIP
        impulse = _IRS[0]
        n_ir = int(impulse.shape[1])
        window = len(clip)
        hop = _hop()
        bed_cue = bed_signal(hop, config.bed_rms)
        plan = plan_window(
            t_anom=0,
            max_steps=RUN_MAX_STEPS,
            policy=WindowPolicy.FIXED_STEPS,
            sounding_steps=_expected_clip_tail_steps(window, n_ir, hop) + 2,
            budget_fraction=0.12,
            draw_steps_range=(30, 90),
            seed=PLACEMENT_SEED,
            episode_index=0,
        )
        offset_step = int(plan.offset_step)

        state = open_tail(window=window, hop=hop, headroom=max(0, n_ir - 1))
        steps: List[StepRecord] = []
        for step in range(
            offset_step + _expected_clip_tail_steps(window, n_ir, hop) + 1
        ):
            sounding = plan.is_sounding(step)
            state, cue = heard_step(
                state,
                ir=impulse if sounding else None,
                clip=clip,
                bed_cue=bed_cue,
                sounding=sounding,
            )
            # `measured_rms` is the CUE reading since ADR-0019, which is what the runner
            # writes and therefore what `post_offset_audible_steps` has to be handed.
            steps.append(
                StepRecord(step=step, measured_rms=rms(cue), source_playing=sounding)
            )

        record = SoundingWindowRecord(
            opens_at=int(plan.opens_at),
            offset_step=offset_step,
            policy=plan.policy.value,
            step_seconds=float(config.step_seconds),
            hop_samples=int(state.hop),
            analysis_window_samples=int(state.window),
            max_ir_samples=int(state.max_ir_samples),
            n_buffer_grows=int(state.n_grows),
            tail_steps=int(state.clip_tail_steps),
            cue_tail_steps=int(state.cue_tail_steps),
            ramp_steps=int(state.clip_ramp_steps),
            post_offset_audible_steps=post_offset_audible_steps(
                readings=[(row.step, row.measured_rms) for row in steps],
                offset_step=offset_step,
                bed_rms=config.bed_rms,
                tolerance=config.pre_onset_rms_tol,
            ),
        )
        audit = EpisodeAudit(
            episode_index=0,
            scene_id=None if _SCENE is None else _SCENE.scene_label,
            sounding_window=record,
            source_reached_step=offset_step + 1,
            funnel_stage=FunnelStage.SOURCE_REACHED,
            steps=tuple(steps),
        )

        print("\n  --- F29: the record this fold produces ---", flush=True)
        print("  {}".format(json.dumps(record.as_dict(), sort_keys=True)), flush=True)

        # THE HEALTHY ARM.
        self.assertTrue(
            tail_is_active(record),
            "tail_is_active rejected a record built from a real fold of a real IR",
        )
        tally = silent_phase_tally([audit])
        payload = tally.as_dict()
        print("  healthy arm  -> {}".format(
            json.dumps(payload, sort_keys=True)), flush=True)
        self.assertEqual(payload["sws_status"], "measured")
        self.assertEqual(payload["n_window_closed"], 1)
        self.assertEqual(payload["n_tail_active"], 1)
        self.assertEqual(tally.sws, 1.0)
        # It also has to survive the trip to disk, since that is the only form a reader
        # ever sees. `RunSummary.as_dict` is what `run()` hands `write_run_summary`.
        json.dumps(
            RunSummary(
                run_dir="unused",
                scene_label="unused",
                n_episodes=1,
                funnel={stage.name: 1 for stage in FunnelStage},
                silent_phase=tally,
            ).as_dict()
        )
        # Decision 3, asserted: the cue tail is NOT a fifth clause of the predicate.
        self.assertTrue(
            tail_is_active(dataclasses.replace(record, cue_tail_steps=None)),
            "tail_is_active refused a record whose only missing field is cue_tail_steps. "
            "The predicate is about the buffer having folded a render; the cue tail is "
            "about the room, and a scene whose room does not outlive a step must be "
            "MEASURED rather than silently disqualified.",
        )

        # FORCED FAILURE 1: the accumulator was configured and folded no render. This is
        # the shape a run with the window but without the buffer would write, and it is
        # the one ADR-0017 bars an SWS over.
        crippled = dataclasses.replace(record, max_ir_samples=0)
        self.assertFalse(tail_is_active(crippled))
        with self.assertRaises(TailNotActiveError) as caught:
            silent_phase_tally([dataclasses.replace(audit, sounding_window=crippled)])
        print("  forced failure 1 (max_ir_samples 0) -> {}: {}".format(
            type(caught.exception).__name__, str(caught.exception)[:160]), flush=True)

        # FORCED FAILURE 2: a pre-ADR-0017 record carries no window at all. The episode
        # is not eligible, and the answer must be NOT_RUN rather than 0.0 -- a reader who
        # takes a missing number for a zero publishes an SWS nobody measured.
        older = silent_phase_tally(
            [dataclasses.replace(audit, sounding_window=None)]
        ).as_dict()
        print("  forced failure 2 (no window on the record) -> {}".format(
            json.dumps(older, sort_keys=True)), flush=True)
        self.assertIsNone(older["sws"])
        self.assertEqual(older["sws_status"], "not_run")


class TestTheSwsReachesDisk(unittest.TestCase):
    """F20: ``run() -> summary.json`` is the only path an SWS reaches disk by.

    A Mac cannot walk it -- ``run()`` builds a ``World`` -- so every Mac test of the SWS
    stops at ``silent_phase_tally``. This walks it: one short real episode into a temp
    run directory, then the three artefacts read back off disk.

    Sized deliberately small (``t_anom`` pinned, an 8-step window, 40 steps) so the
    window CLOSES inside the episode. SWS's denominator counts episodes that ran past
    their own offset step, and the defaults (window 60, ``t_anom`` derived) would need a
    long episode to produce one.
    """

    def test_run_writes_a_measured_sws_and_the_windows_record_into_the_run_directory(
        self,
    ):
        if _CLIP_PATH is None:
            self.skipTest(
                "no ESC-50 recording staged at {}/{}.wav, and run() has no synthetic "
                "fallback on purpose. Stage it with `python -m earshot.audio.clips "
                "--out-dir data/anomaly_audio` and re-run. THIS IS A NOT_RUN, WHICH IS "
                "RED: the SWS-to-disk path is unverified on this box.".format(
                    AudioConfig().clip_dir, ANOMALY_CLASS
                )
            )
        run_dir = os.path.join(
            tempfile.mkdtemp(prefix="earshot-f29-"), "sounding-window"
        )
        cfg = RunConfig(
            run_dir=run_dir,
            scene=_SCENE_STEM,
            n_episodes=1,
            max_steps=RUN_MAX_STEPS,
            t_anom=RUN_T_ANOM,
            sounding_policy=WindowPolicy.FIXED_STEPS,
            sounding_steps=RUN_SOUNDING_STEPS,
            seed=PLACEMENT_SEED,
        )
        print("\n  --- F29: run() -> summary.json, the only path an SWS reaches disk "
              "by ---", flush=True)
        print("  run_dir: {}".format(run_dir), flush=True)
        try:
            summary = run(cfg)
        except EmptyDatasetError as exc:
            self.skipTest(
                "{} can place no anomaly episode ({}), which is a scene-yield fact "
                "rather than a defect in the window. THIS IS A NOT_RUN, WHICH IS "
                "RED.".format(_SCENE.scene_label, exc)
            )

        root, _episodes = run_paths(run_dir)
        with open(str(root / RUN_SUMMARY_NAME), "r") as handle:
            written = json.load(handle)
        _agent_path, audit_path = episode_paths(run_dir, 0)
        with open(str(audit_path), "r") as handle:
            audit = json.load(handle)
        with open(str(root / ENV_REPORT_NAME), "r") as handle:
            env_report = json.load(handle)

        silent = written["silent_phase"]
        window = audit["sounding_window"]
        print("  summary.json silent_phase: {}".format(
            json.dumps(silent, sort_keys=True)), flush=True)
        print("  audit.json sounding_window: {}".format(
            json.dumps(window, sort_keys=True)), flush=True)
        print("  audit.json source_reached_step: {}   steps run: {}".format(
            audit["source_reached_step"], len(audit["steps"])), flush=True)
        print("  env_report.json run_config: policy {} steps {} fraction {} draw "
              "{}".format(
                  env_report["run_config"]["sounding_policy"],
                  env_report["run_config"]["sounding_steps"],
                  env_report["run_config"]["sounding_budget_fraction"],
                  env_report["run_config"]["sounding_draw_steps"]), flush=True)
        print("  {}".format(summary.summary().replace("\n", "\n  ")), flush=True)

        self.assertIsNotNone(
            silent, "run() wrote a summary.json with no silent_phase block at all"
        )
        self.assertGreaterEqual(
            int(silent["n_window_closed"]),
            1,
            "the episode ended before its own offset step ({} steps run, offset at "
            "{}), so nothing was eligible and the SWS on disk is NOT_RUN. Raise "
            "RUN_MAX_STEPS.".format(len(audit["steps"]), window["offset_step"]),
        )
        self.assertEqual(silent["sws_status"], "measured")
        self.assertIsNotNone(silent["sws"])
        self.assertEqual(silent["n_tail_active"], silent["n_window_closed"])
        # The accumulator's own evidence, on disk, from the live renderer -- the fields
        # `tail_is_active` reads and the ones that would be missing if the buffer had
        # never met an IR.
        self.assertEqual(int(window["offset_step"]), RUN_T_ANOM + RUN_SOUNDING_STEPS)
        self.assertEqual(int(window["hop_samples"]), _hop())
        self.assertEqual(int(window["analysis_window_samples"]), len(_CLIP))
        self.assertGreater(
            int(window["max_ir_samples"]),
            0,
            "the episode's record says the accumulator folded no IR, so the tail was "
            "configured and never ran",
        )
        # ADR-0019's field, on disk, from the live renderer. `None` here would mean the
        # runner wrote a pre-split record, and smoke criterion 4 would then PASS without
        # asserting the silent phase's level at all -- a green that asserts nothing is
        # this tree's named failure.
        self.assertIsNotNone(
            window.get("cue_tail_steps"),
            "run() wrote a sounding_window with no cue_tail_steps. Smoke criterion 4 "
            "reads that field for its fence post and returns PASS-without-assertion "
            "when it is absent, so every episode this box produces would be judged on "
            "its trace half alone.",
        )
        self.assertEqual(
            int(window["cue_tail_steps"]),
            _expected_cue_tail_steps(int(window["max_ir_samples"]), _hop()),
            "the record's cue tail is not ceil((hop + L - 1)/hop) for the record's own "
            "hop and IR width",
        )
        self.assertEqual(
            int(window["tail_steps"]),
            _expected_clip_tail_steps(
                int(window["analysis_window_samples"]),
                int(window["max_ir_samples"]),
                _hop(),
            ),
            "the record's tail_steps is not ceil((N + L - 1)/hop). That field kept its "
            "name across ADR-0019 and changed ROLE -- it is the CLIP readout emptying, "
            "which is what CLAP reads and what tail_is_active gates on.",
        )
        self.assertGreater(
            int(window["cue_tail_steps"]),
            1,
            "this scene's room does not outlive one simulator step, so its silent phase "
            "is an honest hard cut and the run carries no reverb tail to measure. That "
            "is a scene fact worth knowing rather than a defect -- but on a scene the "
            "guard has already accepted, an IR of {} samples against a hop of {} makes "
            "it very unlikely, so check the record before believing "
            "it.".format(window["max_ir_samples"], _hop()),
        )
        self.assertEqual(
            env_report["run_config"]["sounding_policy"], WindowPolicy.FIXED_STEPS.value
        )
        self.assertEqual(
            int(env_report["run_config"]["sounding_steps"]), RUN_SOUNDING_STEPS
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
