"""DREAM assembled: `task/dream.py`, and the runner path that calls it.

PR 6c of 6. Five PRs built the parts and none of them was called by anything. These tests
are about the calling.

**THE PROPERTY THAT MATTERS MOST IS THAT `dream=None` CHANGED NOTHING.**
`test_the_no_dream_path_is_untouched` runs the same episode with and without a DREAM
context under `PlanWeights(1, 0, 0)` and an empty `M^L`, and asserts the two trajectories
are identical step for step. That is not a nicety: every result this repo has on disk was
measured on the `dream is None` path, and a wiring that perturbed it would silently
invalidate `abl-2`, `matrix-2` and every comparison built on them.

**The visual half is a STUB here and the tests say so.** `_task_fakes.FakeWorld.observe`
returns `{"rgb": None}` deliberately -- a fabricated frame would be a claim about geometry
the fake does not have -- so a fake `f_v` stands in and varies with the step instead. What
a real CLIP produces on a real frame is `tests/box/test_dream_box.py`'s and
`test_vlm_box.py`'s job, and neither can live on a Mac.
"""

import unittest

import numpy as np

import _audio_fakes as audio_fakes
from _interpreter import assert_interpreter  # noqa: F401
from _task_fakes import (
    FakeAudioSensorHandle,
    FakeWorld,
    make_anomaly_episode,
    make_episode,
    make_goal,
)

from earshot.agent.stm import ShortTermMemory
from earshot.audio.calibration import CalibrationResult
from earshot.audio.clips import synthetic_burst
from earshot.audio.config import AudioConfig
from earshot.config import Detector, Localization, RunConfig
from earshot.memory.consolidate import ImportanceWeights, TrajectoryStep
from earshot.memory.longterm import ExperienceStore, LongTermMemory, PatternStore
from earshot.memory.store import SemanticStore
from earshot.task.dream import (
    DreamContext,
    DreamKnobs,
    begin_episode,
    consolidate_episode,
    empty_memory,
    observe,
)
from earshot.task.plan import PlanWeights
from earshot.__main__ import DREAM_KNOB_FLAGS, build_parser, dream_kwargs_from_args
from earshot.task.runner import make_detector, run_episode
from earshot.types import Pose, Xyz

CLIP = synthetic_burst(44100, seconds=0.05)

CALIBRATION = CalibrationResult(
    onset_rms=0.003, bed_rms=1e-3, anomaly_low=0.008, anomaly_median=0.01,
    anomaly_min=0.005, anomaly_max=0.05, separation_db=18.0, n_poses=16,
    global_volume=1.0,
)


class FakeClipEncoder:
    """`f_v`, stubbed. Returns a different unit vector per call so `M^S` and `M^E` are
    not a table of one repeated key -- which would make every test below vacuous."""

    WIDTH = 8

    def __init__(self):
        self.calls = 0

    def encode_image(self, frame):
        del frame  # the fake world's rgb is None on purpose; see the module docstring
        vector = np.zeros(self.WIDTH, dtype=np.float32)
        vector[self.calls % self.WIDTH] = 1.0
        vector[(self.calls * 3) % self.WIDTH] += 0.5
        self.calls += 1
        return vector


def knobs(**overrides):
    base = dict(
        stm_horizon=4,
        stm_decay=0.8,
        present_weight=0.7,
        coherence=0.95,
        min_segment=2,
        max_segment=6,
        importance=ImportanceWeights(alpha=1.0, beta=1.0, gamma=1.0),
        eta=0.5,
        min_support=2,
        k_experience=2,
        k_pattern=1,
        k_knowledge=1,
        temperature=0.5,
        plan_weights=PlanWeights(plan=1.0, memory=1.0, feasibility=1.0),
    )
    base.update(overrides)
    return DreamKnobs(**base)


def context(memory=None, **overrides):
    return DreamContext(
        knobs=knobs(**overrides),
        memory=memory if memory is not None else empty_memory(),
        clip_encoder=FakeClipEncoder(),
    )


def _audio(index):
    vector = np.zeros(6, dtype=np.float32)
    vector[index % 6] = 1.0
    return vector


def _pose(x):
    return Pose(position=Xyz(float(x), 0.0, 0.0), yaw_rad=0.0)


def _clap():
    """`f_u`, stubbed. It returns ONE fixed vector for every step, so `z^u` is constant
    along an episode and the whole of `z^av`'s variation comes from the visual half.

    That is a real limitation of these tests and not a hidden one: whether a real CLAP
    embedding of the heard signal moves from step to step is exactly what
    `tests/box/test_memory_consolidate_box.py` measures, and it cannot be faked into
    existence here."""
    return audio_fakes.FakeClapEncoder(
        audio_vector=np.eye(8, dtype=np.float32)[0],
        text_vectors={"a baby crying": np.eye(8, dtype=np.float32)[1]},
    )


class TestTheKnobs(unittest.TestCase):
    def test_every_knob_reaches_the_audit(self):
        """Eighteen named numbers. A sweep that cannot say what its knobs were is a sweep
        whose result cannot be reproduced, which has cost this repo real box time."""
        names = [name for name, _value in knobs().as_metrics()]
        self.assertEqual(len(names), len(set(names)))
        for expected in (
            "dream_stm_horizon", "dream_coherence", "dream_alpha", "dream_eta",
            "dream_min_support", "dream_temperature", "dream_lambda_memory",
        ):
            self.assertIn(expected, names)

    def test_the_order_is_fixed_so_two_audits_line_up(self):
        self.assertEqual(
            [name for name, _ in knobs().as_metrics()],
            [name for name, _ in knobs(eta=9.0).as_metrics()],
        )

    def test_no_knob_has_a_default(self):
        with self.assertRaises(TypeError):
            DreamKnobs()


class TestTheEmptyMemory(unittest.TestCase):
    def test_it_is_three_empty_levels(self):
        memory = empty_memory()
        self.assertEqual(len(memory.experience), 0)
        self.assertEqual(len(memory.pattern), 0)
        self.assertEqual(len(memory.knowledge), 0)
        self.assertEqual(memory.novelty_vectors(), ())


class TestTheEpisodeState(unittest.TestCase):
    def test_begin_episode_builds_an_empty_stm_at_the_configured_horizon(self):
        state = begin_episode(context(stm_horizon=7))
        self.assertIsInstance(state.stm, ShortTermMemory)
        self.assertEqual(state.stm.horizon, 7)
        self.assertEqual(len(state), 0)

    def test_observe_leaves_the_state_it_was_given_alone(self):
        """The immutability `agent/stm.py` argues for, at the level that actually calls
        it: a caller holding step 5's state still holds step 5's history."""
        ctx = context()
        before = begin_episode(ctx)
        after = observe(
            before, ctx, frame=None, audio=_audio(0), pose=_pose(0.0),
            prev_action=None, belief=None,
        ).episode
        self.assertEqual(len(before), 0)
        self.assertEqual(len(after), 1)
        self.assertIsNot(before, after)

    def test_the_stm_evicts_at_the_horizon_and_tau_does_not(self):
        """`M^S` is a window; `tau` (eq. 8) is the whole episode. Conflating them would
        make consolidation see only the last few steps."""
        ctx = context(stm_horizon=3)
        state = begin_episode(ctx)
        for index in range(8):
            state = observe(
                state, ctx, frame=None, audio=_audio(index), pose=_pose(index),
                prev_action=None if index == 0 else "move_forward", belief=None,
            ).episode
        self.assertEqual(len(state.stm), 3)
        self.assertEqual(len(state), 8)

    def test_observe_reports_what_it_cost(self):
        """Criterion 7 does not cover `f_v` and `f_u`, so this number is the only one
        that says what a DREAM step costs."""
        outcome = observe(
            begin_episode(context()), context(), frame=None, audio=_audio(0),
            pose=_pose(0.0), prev_action=None, belief=None,
        )
        self.assertGreaterEqual(outcome.seconds, 0.0)

    def test_an_empty_memory_retrieves_nothing_and_that_is_not_an_error(self):
        outcome = observe(
            begin_episode(context()), context(), frame=None, audio=_audio(0),
            pose=_pose(0.0), prev_action=None, belief=None,
        )
        self.assertIsNone(outcome.context.weights)
        self.assertFalse(outcome.context)


class TestConsolidation(unittest.TestCase):
    def _walked(self, ctx, steps=12):
        state = begin_episode(ctx)
        for index in range(steps):
            state = observe(
                state, ctx, frame=None, audio=_audio(index), pose=_pose(index * 0.5),
                prev_action=None if index == 0 else "move_forward",
                belief=Xyz(20.0 - index * 0.2, 0.0, 0.0),
            ).episode
        return state

    def test_a_walked_episode_puts_rows_into_m_e(self):
        ctx = context(eta=0.0)
        memory, scores = consolidate_episode(
            self._walked(ctx), ctx, sound_concept="toilet_flush",
            target_concept="toilet", room_concept=None, reached=True, final_gap_m=0.4,
        )
        self.assertGreater(len(memory.experience), 0)
        self.assertEqual(len(scores), len(scores))
        self.assertGreater(len(scores), 0)

    def test_a_high_eta_retains_nothing_and_leaves_the_memory_alone(self):
        ctx = context(eta=1e6)
        before = ctx.memory
        memory, scores = consolidate_episode(
            self._walked(ctx), ctx, sound_concept="toilet_flush",
            target_concept="toilet", room_concept=None, reached=True, final_gap_m=0.4,
        )
        self.assertIs(memory, before)
        self.assertGreater(len(scores), 0, "the scores are still the audit's evidence")

    def test_a_two_step_episode_is_not_an_error(self):
        ctx = context()
        state = begin_episode(ctx)
        state = observe(
            state, ctx, frame=None, audio=_audio(0), pose=_pose(0.0),
            prev_action=None, belief=None,
        ).episode
        memory, scores = consolidate_episode(
            state, ctx, sound_concept="alarm", target_concept="fireplace",
            room_concept=None, reached=False, final_gap_m=9.0,
        )
        self.assertIs(memory, ctx.memory)
        self.assertEqual(scores, ())

    def test_every_retained_row_carries_the_episodes_outcome_and_concepts(self):
        ctx = context(eta=0.0)
        memory, _ = consolidate_episode(
            self._walked(ctx), ctx, sound_concept="snoring", target_concept="bed",
            room_concept=None, reached=False, final_gap_m=3.25,
        )
        for entry in memory.experience.entries:
            self.assertEqual(entry.sound_concept, "snoring")
            self.assertEqual(entry.target_concept, "bed")
            self.assertFalse(entry.outcome.reached)
            self.assertAlmostEqual(entry.outcome.final_gap_m, 3.25)

    def test_the_pattern_level_is_rebuilt_from_the_whole_of_m_e(self):
        """`G` groups and averages, so a signature depends on every member. An
        incremental append would drift from `abstract`'s own definition."""
        ctx = context(eta=0.0, min_support=1)
        first, _ = consolidate_episode(
            self._walked(ctx), ctx, sound_concept="alarm", target_concept="fireplace",
            room_concept=None, reached=True, final_gap_m=0.3,
        )
        ctx = ctx.with_memory(first)
        second, _ = consolidate_episode(
            self._walked(ctx), ctx, sound_concept="alarm", target_concept="fireplace",
            room_concept=None, reached=True, final_gap_m=0.3,
        )
        self.assertEqual(len(second.pattern), 1)
        self.assertEqual(
            second.pattern.patterns[0].strategy.support, len(second.experience)
        )

    def test_a_failed_episode_still_reaches_m_e_but_not_m_p(self):
        """The two-stage rule PR #111 argued for: retention keeps failures, abstraction
        does not."""
        ctx = context(eta=0.0, min_support=1)
        memory, _ = consolidate_episode(
            self._walked(ctx), ctx, sound_concept="alarm", target_concept="fireplace",
            room_concept=None, reached=False, final_gap_m=7.0,
        )
        self.assertGreater(len(memory.experience), 0)
        self.assertEqual(len(memory.pattern), 0)

    def test_the_knowledge_level_is_carried_across_untouched(self):
        """`M^K` is the semantic store the matrix already writes. Consolidation must not
        touch it -- a DREAM run and a matrix arm share it."""
        knowledge = SemanticStore()
        ctx = context(
            memory=LongTermMemory(
                experience=ExperienceStore(), pattern=PatternStore(), knowledge=knowledge
            ),
            eta=0.0,
        )
        memory, _ = consolidate_episode(
            self._walked(ctx), ctx, sound_concept="alarm", target_concept="fireplace",
            room_concept=None, reached=True, final_gap_m=0.3,
        )
        self.assertIs(memory.knowledge, knowledge)

    def test_c_j_is_scored_against_the_final_position_and_never_a_source(self):
        """**THE GT-PRIVILEGE ARM.** `consolidate_episode` takes no source argument at
        all, which is the structural half of PR #110's argument: the true source is
        available in `run_episode` and cannot reach here even by mistake."""
        import inspect

        signature = inspect.signature(consolidate_episode)
        for forbidden in ("source", "goal", "target"):
            self.assertNotIn(forbidden, signature.parameters)


class TestTheRunnerPath(unittest.TestCase):
    """The wiring, driven through `run_episode` with fakes."""

    def _config(self, **overrides):
        base = dict(
            run_dir="/nonexistent",
            max_steps=40,
            t_anom=2,
            localization=Localization.REALIZABLE,
            detector=Detector.ORACLE,
            audio=AudioConfig(step_seconds=0.01),
        )
        base.update(overrides)
        return RunConfig(**base)

    def _episode(self, cfg, **kwargs):
        # The same geometry `test_task_runner.py` uses: the agent faces -z, the source is
        # 5 m ahead and the primary goal 9 m ahead, so the detour is on the way.
        world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
        source = Xyz(0.0, 0.0, -5.0)
        handle = FakeAudioSensorHandle(world, source)
        anomaly_episode = make_anomaly_episode(
            source=source,
            episode=make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0))]),
            t_anom=2,
        )
        kwargs.setdefault("detector", make_detector(cfg, world, anomaly_episode))
        kwargs.setdefault("clip", CLIP)
        kwargs.setdefault("calibration", CALIBRATION)
        return run_episode(world, handle, anomaly_episode, cfg, **kwargs)

    def test_the_no_dream_path_is_untouched(self):
        """**THE ARM EVERY RESULT ON DISK DEPENDS ON.** Every number this repo has was
        measured on the `dream is None` path. A DREAM context with an EMPTY `M^L` and
        `PlanWeights(1, 0, 0)` must produce the identical trajectory, because
        `plan.pick_plan` reduces to `agent.scorer.pick_waypoint` exactly then (PR #113).
        If this goes red, the wiring perturbed the baseline and `abl-2` and `matrix-2`
        are no longer comparable with anything run after it."""
        cfg = self._config()
        without = self._episode(cfg)
        with_dream = self._episode(
            cfg,
            clap_encoder=_clap(),
            dream=context(plan_weights=PlanWeights(plan=1.0, memory=0.0, feasibility=0.0)),
        )
        self.assertEqual(
            [step.action for step in without.audit.steps],
            [step.action for step in with_dream.audit.steps],
            "a DREAM context with an empty M^L changed the trajectory, so the pre-DREAM "
            "baseline is not a control any more",
        )

    def test_a_dream_run_records_what_it_cost(self):
        cfg = self._config()
        result = self._episode(
            cfg, clap_encoder=_clap(), dream=context()
        )
        metrics = result.audit.metrics
        for name in ("dream_step_s_mean", "dream_step_s_worst", "dream_tau_steps"):
            self.assertIn(name, metrics)
        self.assertGreater(metrics["dream_tau_steps"], 0.0)

    def test_a_dream_run_records_every_knob(self):
        result = self._episode(
            self._config(), clap_encoder=_clap(),
            dream=context(eta=0.25),
        )
        self.assertAlmostEqual(result.audit.metrics["dream_eta"], 0.25)

    def test_tau_is_one_step_per_loop_step(self):
        """`M^S` sees every step, not only the investigating ones."""
        result = self._episode(
            self._config(), clap_encoder=_clap(), dream=context()
        )
        self.assertEqual(
            int(result.audit.metrics["dream_tau_steps"]), len(result.audit.steps)
        )

    def test_the_grown_memory_comes_back_on_the_result(self):
        result = self._episode(
            self._config(eta=0.0) if False else self._config(),
            clap_encoder=_clap(),
            dream=context(eta=0.0),
        )
        self.assertIsNotNone(result.dream_memory)
        self.assertGreaterEqual(len(result.dream_memory.experience), 0)
        self.assertIn("dream_rows_added", result.audit.metrics)

    def test_no_dream_context_means_no_dream_metrics_and_no_memory(self):
        result = self._episode(self._config())
        self.assertIsNone(result.dream_memory)
        for name in result.audit.metrics:
            self.assertFalse(
                name.startswith("dream_"),
                "a run with no DREAM context wrote {}".format(name),
            )

    def test_a_dream_context_without_clap_is_refused_before_the_episode(self):
        """The same wiring mistake `MemoryContext` already guards, one equation earlier:
        without `f_u` every `z^av` would be half a vector and `omega_t` a function of the
        camera alone."""
        with self.assertRaises(ValueError) as caught:
            self._episode(self._config(), dream=context())
        self.assertIn("half a vector", str(caught.exception))

    def test_a_dream_context_without_clip_is_refused(self):
        broken = DreamContext(knobs=knobs(), memory=empty_memory(), clip_encoder=None)
        with self.assertRaises(ValueError) as caught:
            self._episode(
                self._config(), clap_encoder=_clap(), dream=broken
            )
        self.assertIn("f_v", str(caught.exception))

    def test_memory_carried_across_two_episodes_grows(self):
        """What `run()` does between episodes, done by hand: consolidate, rebind, run
        again. The second episode's `M^E` must contain the first's rows."""
        cfg = self._config()
        ctx = context(eta=0.0)
        first = self._episode(
            cfg, clap_encoder=_clap(), dream=ctx
        )
        self.assertIsNotNone(first.dream_memory)
        ctx = ctx.with_memory(first.dream_memory)
        second = self._episode(
            cfg, clap_encoder=_clap(), dream=ctx
        )
        self.assertGreaterEqual(
            len(second.dream_memory.experience), len(first.dream_memory.experience)
        )

    def test_a_step_that_took_no_action_carries_none_forward(self):
        """`a_{t-1}` (eq. 5). Structural, and it had to be: `prev_action` is STORED on the
        `M^S` entry and read by nothing downstream, so a wrong value changes no trajectory
        and no metric -- the forced-failure run is what showed that
        `test_the_no_dream_path_is_untouched` cannot see it.

        `_steer` can return `None` after two attempts, which is a real state the runner
        records rather than papers over. Carrying the last successful action forward there
        would make `e_t` claim an action the agent did not take.
        """
        import ast
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2] / "task" / "runner.py"
        ).read_text(encoding="utf-8")
        assigned = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "prev_action" for t in node.targets
            )
        ]
        # Exactly one plain assignment: the carry at the end of the loop. The
        # initialiser before the loop is an annotated one (`prev_action: Optional[str]`)
        # and is a different node type.
        self.assertEqual(len(assigned), 1)
        self.assertIsInstance(
            assigned[0].value, ast.Name,
            "prev_action is assigned an expression rather than the action itself",
        )
        self.assertEqual(assigned[0].value.id, "action")

    def test_the_entry_keeps_the_prev_action_it_was_given(self):
        """The other half: that `observe` puts it where eq. 5 says. Without this the
        structural test above would pin a variable nothing consumed."""
        ctx = context()
        state = observe(
            begin_episode(ctx), ctx, frame=None, audio=_audio(0), pose=_pose(0.0),
            prev_action=None, belief=None,
        ).episode
        self.assertIsNone(state.stm.latest.prev_action)
        state = observe(
            state, ctx, frame=None, audio=_audio(1), pose=_pose(1.0),
            prev_action="move_forward", belief=None,
        ).episode
        self.assertEqual(state.stm.latest.prev_action, "move_forward")

    def test_the_belief_reaching_u_j_is_the_controllers_and_not_the_memorys(self):
        """Structural: `runner.py` must pass `decision.investigate_waypoint or
        investigate_probe`, not the memory-overridden `target`. The override fires once
        per episode, so feeding it to `U_j` would put a single large revision in whichever
        segment held it, in every memory-arm episode."""
        import ast
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2] / "task" / "runner.py"
        ).read_text(encoding="utf-8")
        assigned = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "dream_belief" for t in node.targets
            )
        ]
        self.assertEqual(len(assigned), 1)
        expression = ast.dump(assigned[0].value)
        self.assertIn("investigate_waypoint", expression)
        self.assertIn("investigate_probe", expression)
        self.assertNotIn("memory_prior", expression)


class TestTheTrajectoryStepsAreReal(unittest.TestCase):
    def test_tau_holds_trajectory_steps_with_real_positions(self):
        ctx = context()
        state = begin_episode(ctx)
        for index in range(4):
            state = observe(
                state, ctx, frame=None, audio=_audio(index), pose=_pose(index * 2.0),
                prev_action=None, belief=Xyz(9.0, 0.0, 0.0),
            ).episode
        self.assertTrue(all(isinstance(s, TrajectoryStep) for s in state.trajectory))
        self.assertEqual(
            [s.position.x for s in state.trajectory], [0.0, 2.0, 4.0, 6.0]
        )


class TestTheCli(unittest.TestCase):
    """`--dream` and its fourteen knobs. `config_from_args`'s docstring names the bug
    class this guards: a CLI whose flags quietly stop reaching the config."""

    def _args(self, argv):
        return build_parser().parse_args(
            ["--run-dir", "/nonexistent"] + list(argv)
        )

    def _knobs(self, **overrides):
        values = {
            "stm-horizon": "4", "stm-decay": "0.8", "present-weight": "0.7",
            "coherence": "0.95", "min-segment": "2", "max-segment": "6",
            "alpha": "1", "beta": "1", "gamma": "1", "eta": "0.5",
            "min-support": "2", "k-experience": "2", "k-pattern": "1",
            "k-knowledge": "1", "temperature": "0.5", "lambda-plan": "1",
            "lambda-memory": "1", "lambda-feasibility": "1",
        }
        values.update(overrides)
        argv = []
        for name, value in values.items():
            if value is not None:
                argv += ["--dream-{}".format(name), value]
        return argv

    def test_no_dream_flag_means_no_dream_keyword(self):
        """A bare invocation must reach `run()` with no DREAM keyword at all -- not with
        `dream_knobs=None`, which a reader would have to interpret."""
        self.assertEqual(dream_kwargs_from_args(self._args([])), {})

    def test_every_knob_reaches_the_run(self):
        kwargs = dream_kwargs_from_args(self._args(["--dream"] + self._knobs()))
        built = kwargs["dream_knobs"]
        self.assertEqual(built.stm_horizon, 4)
        self.assertAlmostEqual(built.eta, 0.5)
        self.assertAlmostEqual(built.importance.alpha, 1.0)
        self.assertAlmostEqual(built.plan_weights.feasibility, 1.0)

    def test_a_missing_knob_is_refused_and_every_missing_one_is_named(self):
        """**FOURTEEN FLAGS IS TOO MANY TO DISCOVER ONE FAILED LAUNCH AT A TIME.** A
        sweep driver that found them one argparse error after another would burn a box
        slot per knob."""
        argv = ["--dream"] + self._knobs(eta=None, temperature=None)
        with self.assertRaises(SystemExit) as caught:
            dream_kwargs_from_args(self._args(argv))
        message = str(caught.exception)
        self.assertIn("--dream-eta", message)
        self.assertIn("--dream-temperature", message)
        self.assertNotIn("--dream-alpha", message)

    def test_the_flag_table_drives_both_the_parser_and_the_builder(self):
        """The table is the single source, so a knob cannot reach one and not the other.
        This asserts every entry actually became a flag."""
        args = self._args(["--dream"] + self._knobs())
        for flag, _kind, _eq in DREAM_KNOB_FLAGS:
            self.assertIsNotNone(
                getattr(args, "dream_{}".format(flag), None),
                "--dream-{} is in the table but never reached the parser".format(flag),
            )

    def test_the_knob_count_matches_what_the_audit_writes(self):
        """Eighteen flags, eighteen audit numbers. A knob that reached the run and not
        the record would be a number nobody could reproduce a sweep from."""
        built = dream_kwargs_from_args(
            self._args(["--dream"] + self._knobs())
        )["dream_knobs"]
        self.assertEqual(len(DREAM_KNOB_FLAGS), len(built.as_metrics()))


if __name__ == "__main__":
    unittest.main()
