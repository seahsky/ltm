"""``RunConfig`` and the CLI that builds it — ADR-0013's configuration decision.

Two things are being held here, and neither is about a value being right.

**Every knob is reachable and recorded.** ``agent/config.py`` argues that a number which
gates a STOP and appears in no artefact is the class of thing this map keeps finding
after the fact. So a new ``RunConfig`` field has to be either settable from the CLI or
deliberately composed, and ``as_dict`` has to carry the sub-configs whole rather than as
a hand-maintained copy that drifts.

**``onset_rms`` is not configuration.** §2.3 derives it from the calibration sweep at run
start, and a flag for it would let an operator hand-set the one number the spec insists
is measured — "a threshold nudged until the smoke passes is a threshold that means
nothing", and this map's record has a matrix that ran to completion on exactly that kind
of number.
"""

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.__main__ import build_parser, config_from_args, memory_kwargs_from_args
from earshot.agent.config import ControllerConfig, DetectorConfig, PlannerConfig
from earshot.audio.config import AudioConfig, WindowPolicy
from earshot.config import (
    CastPolicy,
    ClimbRule,
    Detector,
    IrPolicy,
    LateralCue,
    Localization,
    RunConfig,
)
from earshot.memory.store import EpisodicStore, MemoryCondition, SemanticEntry, SemanticStore

# The fields that are composed rather than flagged: each is a module's own frozen config,
# defined beside the code it configures (ADR-0013). Pinned here so a new sub-config is a
# visible diff rather than a knob that quietly stopped being settable.
COMPOSED_FIELDS = {"audio", "planner", "controller", "detector_config"}


class TestRunConfig(unittest.TestCase):
    def test_it_composes_the_module_configs_at_their_own_defaults(self):
        config = RunConfig(run_dir="runs/x")
        self.assertEqual(config.audio, AudioConfig())
        self.assertEqual(config.planner, PlannerConfig())
        self.assertEqual(config.controller, ControllerConfig())
        self.assertEqual(config.detector_config, DetectorConfig())

    def test_the_default_arms_are_the_ones_the_smoke_runs(self):
        """§8: realizable localization and an oracle STOP, with the disclosure attached."""
        config = RunConfig(run_dir="runs/x")
        self.assertIs(config.localization, Localization.REALIZABLE)
        self.assertIs(config.detector, Detector.ORACLE)

    def test_the_default_arms_reproduce_todays_behaviour(self):
        """ADR-0018's four new arms: adding them must change NOTHING until one is set.

        Each default names the behaviour that ships today, so a run built with no
        arm flags at all is byte-identical to the pre-arm runner.
        """
        config = RunConfig(run_dir="runs/x")
        self.assertIs(config.climb_rule, ClimbRule.LIVE)
        self.assertIs(config.lateral_cue, LateralCue.LIVE)
        self.assertIs(config.cast_policy, CastPolicy.CAST)
        self.assertIs(config.ir_policy, IrPolicy.FULL)

    def test_it_is_frozen(self):
        config = RunConfig(run_dir="runs/x")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.t_anom = 5

    def test_as_dict_is_json_serialisable(self):
        """It lands inside ``env_report.json``, so an enum reaching it would raise there."""
        payload = json.loads(json.dumps(RunConfig(run_dir="runs/x").as_dict()))
        self.assertEqual(payload["localization"], "realizable")
        self.assertEqual(payload["detector"], "oracle")

    def test_the_four_new_arms_round_trip_through_as_dict_by_value(self):
        """Same shape as ``localization``/``detector``: hand-mapped to ``.value``.

        A sub-config would leave an Enum member as an Enum object and ``json.dumps``
        would raise before this ever reached ``env_report.json``.
        """
        payload = json.loads(json.dumps(RunConfig(run_dir="runs/x").as_dict()))
        self.assertEqual(payload["climb_rule"], "live")
        self.assertEqual(payload["lateral_cue"], "live")
        self.assertEqual(payload["cast_policy"], "cast")
        self.assertEqual(payload["ir_policy"], "full")

    def test_as_dict_carries_every_sub_config_field(self):
        """Through ``dataclasses.asdict``, so a new ``PlannerConfig`` field needs no edit here.

        A hand-written projection would be a partial copy of the configuration rather
        than the configuration, and the drift would be invisible in the run record.
        """
        payload = RunConfig(run_dir="runs/x").as_dict()
        for name, config in (
            ("audio", AudioConfig()),
            ("planner", PlannerConfig()),
            ("controller", ControllerConfig()),
            ("detector_config", DetectorConfig()),
        ):
            self.assertEqual(
                sorted(payload[name]),
                sorted(field.name for field in dataclasses.fields(config)),
            )

    def test_as_dict_carries_every_top_level_field(self):
        """The other half: a new run-level field must reach the record too."""
        payload = RunConfig(run_dir="runs/x").as_dict()
        self.assertEqual(
            sorted(payload),
            sorted(field.name for field in dataclasses.fields(RunConfig)),
        )

    def test_the_sounding_window_defaults_round_trip_as_json(self):
        """Why the policy is a TOP-LEVEL field and not on a sub-config.

        ``as_dict`` passes the four module configs through ``dataclasses.asdict``, which
        leaves an Enum member as an Enum OBJECT — ``json.dumps`` then raises and the run
        record cannot be written at all. Top-level enums are hand-mapped to ``.value``
        beside ``localization`` and ``detector``, which is the only shape that survives
        this round trip.
        """
        payload = json.loads(json.dumps(RunConfig(run_dir="runs/x").as_dict()))
        self.assertEqual(payload["sounding_policy"], "fixed_steps")
        self.assertEqual(payload["sounding_steps"], 60)
        self.assertAlmostEqual(payload["sounding_budget_fraction"], 0.12)
        self.assertEqual(payload["sounding_draw_steps"], [30, 90])

    def test_the_three_bounded_policies_agree_at_the_defaults(self):
        """60 steps, ``floor(0.12 * 500)`` and ``mean(30, 90)`` are all 60, on purpose.

        Switching policy at the defaults changes the VARIANCE of the duration and not its
        level, so the first policy comparison is not confounded by a level change riding
        along with it. The default itself is PROVISIONAL and has no sweep behind it —
        this pins the agreement, not the number.
        """
        config = RunConfig(run_dir="runs/x")
        self.assertEqual(config.sounding_steps, 60)
        self.assertEqual(int(config.sounding_budget_fraction * config.max_steps), 60)
        lo, hi = config.sounding_draw_steps
        self.assertEqual((lo + hi) // 2, 60)

    def test_onset_rms_is_not_a_field_anywhere_in_the_configuration(self):
        """§2.3 derives it; ``AudioConfig`` holds the sweep's INPUTS instead."""
        payload = RunConfig(run_dir="runs/x").as_dict()
        self.assertNotIn("onset_rms", payload)
        self.assertNotIn("onset_rms", payload["audio"])
        self.assertIn("bed_rms", payload["audio"])
        self.assertIn("audible_band_m", payload["audio"])


class TestTheCli(unittest.TestCase):
    def test_every_run_level_field_is_settable_or_deliberately_composed(self):
        """A knob that stops reaching the config is a run that ignored what it was told."""
        dests = {action.dest for action in build_parser()._actions}
        missing = [
            field.name
            for field in dataclasses.fields(RunConfig)
            if field.name not in COMPOSED_FIELDS and field.name not in dests
        ]
        self.assertEqual(
            missing,
            [],
            "RunConfig field(s) with no CLI flag and no entry in COMPOSED_FIELDS: "
            "{}".format(missing),
        )

    def test_the_composed_set_is_the_four_module_configs(self):
        """Widening it has to be a visible diff carrying a reason, not a quiet fifth."""
        names = {field.name for field in dataclasses.fields(RunConfig)}
        self.assertTrue(COMPOSED_FIELDS.issubset(names))
        self.assertEqual(COMPOSED_FIELDS, {"audio", "planner", "controller", "detector_config"})

    def test_no_flag_offers_to_set_the_onset_threshold(self):
        options = [option for action in build_parser()._actions for option in action.option_strings]
        self.assertEqual([option for option in options if "onset" in option], [])

    def _reject(self, argv):
        """argparse exits and prints its usage to stderr; the usage is not the subject."""
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(argv)

    def test_a_run_directory_is_required(self):
        """No default: the failure mode of one is a second run landing on top of the first."""
        self._reject([])

    def test_the_defaults_round_trip_to_the_config_defaults(self):
        config = config_from_args(build_parser().parse_args(["--run-dir", "runs/x"]))
        self.assertEqual(config, RunConfig(run_dir="runs/x"))

    def test_omitting_t_anom_leaves_it_unpinned_rather_than_at_a_constant(self):
        """The flag is a pin now, and its absence means "derive one per episode".

        A default step index is what put the smoke's anomaly on the last step of its
        episode: 30 was chosen against the 500-step budget, and the find ended at 30.
        ``None`` is the behaviour, so it is asserted rather than left to the defaults
        round-trip above, which would pass whatever the number was.
        """
        config = config_from_args(build_parser().parse_args(["--run-dir", "runs/x"]))
        self.assertIsNone(config.t_anom)
        self.assertIsNone(config.as_dict()["t_anom"])

    def test_every_flag_reaches_the_config(self):
        """Set every one to something that is not its default, and read them all back."""
        config = config_from_args(
            build_parser().parse_args(
                [
                    "--run-dir", "runs/y",
                    "--split", "train",
                    "--data-root", "/data",
                    "--scene", "TEEsavR23oF",
                    "--category", "toilet",
                    "--n-episodes", "3",
                    "--max-steps", "250",
                    "--t-anom", "40",
                    "--seed", "7",
                    "--localization", "oracle",
                    "--detector", "caption",
                    "--climb-rule", "off",
                    "--lateral-cue", "off",
                    "--cast-policy", "scan_only",
                    "--ir-policy", "anechoic",
                    "--anomaly-class", "glass_break",
                    "--anomaly-clip", "/tmp/x.wav",
                    "--clap",
                    "--min-source-sep-m", "5.5",
                    "--max-source-dy-m", "0.5",
                    "--audio-step-ceiling-s", "0.25",
                    "--sounding-policy", "drawn",
                    "--sounding-steps", "45",
                    "--sounding-budget-fraction", "0.2",
                    "--sounding-draw-steps", "10", "20",
                    "--overwrite",
                ]
            )
        )
        self.assertEqual(config.run_dir, "runs/y")
        self.assertEqual(config.split, "train")
        self.assertEqual(config.data_root, "/data")
        self.assertEqual(config.scene, "TEEsavR23oF")
        self.assertEqual(config.category, "toilet")
        self.assertEqual(config.n_episodes, 3)
        self.assertEqual(config.max_steps, 250)
        self.assertEqual(config.t_anom, 40)
        self.assertEqual(config.seed, 7)
        self.assertIs(config.localization, Localization.ORACLE)
        self.assertIs(config.detector, Detector.CAPTION)
        self.assertIs(config.climb_rule, ClimbRule.OFF)
        self.assertIs(config.lateral_cue, LateralCue.OFF)
        self.assertIs(config.cast_policy, CastPolicy.SCAN_ONLY)
        self.assertIs(config.ir_policy, IrPolicy.ANECHOIC)
        self.assertEqual(config.anomaly_class, "glass_break")
        self.assertEqual(config.anomaly_clip, "/tmp/x.wav")
        self.assertTrue(config.clap)
        self.assertAlmostEqual(config.min_source_sep_m, 5.5)
        self.assertAlmostEqual(config.max_source_dy_m, 0.5)
        self.assertAlmostEqual(config.audio_step_ceiling_s, 0.25)
        self.assertIs(config.sounding_policy, WindowPolicy.DRAWN)
        self.assertEqual(config.sounding_steps, 45)
        self.assertAlmostEqual(config.sounding_budget_fraction, 0.2)
        self.assertEqual(config.sounding_draw_steps, (10, 20))
        self.assertTrue(config.overwrite)

    def test_the_arms_are_enums_rather_than_booleans(self):
        """ADR-0013: a third option is addable without a flag explosion.

        The old tree read ``LTM_REALIZABLE_LOCALIZATION`` from the environment, which is
        the shape that cannot grow a third value.
        """
        self._reject(["--run-dir", "runs/x", "--localization", "sometimes"])
        self._reject(["--run-dir", "runs/x", "--sounding-policy", "sometimes"])
        self._reject(["--run-dir", "runs/x", "--climb-rule", "sometimes"])

    def test_the_drawn_range_arrives_as_a_tuple(self):
        """argparse's ``nargs=2`` yields a LIST, and ``RunConfig`` is compared by equality.

        ``test_the_defaults_round_trip_to_the_config_defaults`` above asserts the built
        config equals ``RunConfig(run_dir=...)``, which a list would fail for a reason
        that has nothing to do with the value. Frozen dataclasses also want a hashable
        field.
        """
        config = config_from_args(
            build_parser().parse_args(
                ["--run-dir", "runs/x", "--sounding-draw-steps", "10", "20"]
            )
        )
        self.assertEqual(config.sounding_draw_steps, (10, 20))
        self.assertIsInstance(config.sounding_draw_steps, tuple)

    def test_the_anomaly_class_is_restricted_to_the_locked_three(self):
        """``clips.ANOMALY_CLASSES`` — the classes ESC-50 is staged for.

        An unrecognised class would resolve to no staged file, and
        ``load_anomaly_clip`` would raise at the point of use rather than at the point of
        the typo.
        """
        self._reject(["--run-dir", "runs/x", "--anomaly-class", "thunder"])


class TestMemoryKwargsFromArgs(unittest.TestCase):
    """``memory_kwargs_from_args`` — the seam ``MemoryCondition``'s own docstring asks
    for: the matrix cell is not a ``RunConfig`` field, so this is a second, separate
    mapping from the CLI's ``--memory-*`` flags to ``run()``'s keywords."""

    def _args(self, *extra):
        return build_parser().parse_args(["--run-dir", "runs/x", *extra])

    def test_the_default_is_no_keywords_at_all(self):
        """Not `{"memory_condition": None}` -- an EMPTY dict, so `run(**kwargs)` reaches
        `run()` exactly as a caller with no memory flags always has."""
        self.assertEqual(memory_kwargs_from_args(self._args()), {})

    def test_a_condition_with_no_store_flag_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                memory_kwargs_from_args(
                    self._args("--memory-condition", "heard_seen")
                )

    def test_an_unrecognised_condition_is_rejected_by_the_parser_itself(self):
        self._reject(["--run-dir", "runs/x", "--memory-condition", "sometimes"])

    def test_a_dumped_store_round_trips_into_run_kwargs(self):
        """THE HEALTHY ARM, against a real file on disk -- the same file a prior-pass
        driver writes and a matrix sweep reads."""
        from earshot.task.memory_build import dump_stores

        semantic = SemanticStore(entries=(
            SemanticEntry(sound_class="alarm", room="bedroom", category="bed",
                          embedding=[1.0, 0.0], donor_scene="donor_scene"),
        ))
        episodic = EpisodicStore()
        with tempfile.TemporaryDirectory() as tmp:
            path = "{}/store.json".format(tmp)
            dump_stores(path, semantic, episodic)
            kwargs = memory_kwargs_from_args(
                self._args("--memory-condition", "heard_seen", "--memory-store", path,
                            "--memory-k", "3")
            )
        self.assertEqual(kwargs["memory_condition"], MemoryCondition.HEARD_SEEN)
        self.assertEqual(kwargs["memory_k"], 3)
        loaded_semantic, loaded_episodic = kwargs["memory_prior_stores"]
        self.assertEqual(
            [entry.sound_class for entry in loaded_semantic.entries], ["alarm"]
        )
        self.assertEqual(len(loaded_episodic), 0)

    def test_memory_k_defaults_to_five(self):
        from earshot.task.memory_build import dump_stores

        semantic = SemanticStore(entries=(
            SemanticEntry(sound_class="alarm", room="bedroom", category="bed",
                          embedding=[1.0, 0.0], donor_scene="donor_scene"),
        ))
        with tempfile.TemporaryDirectory() as tmp:
            path = "{}/store.json".format(tmp)
            dump_stores(path, semantic, EpisodicStore())
            kwargs = memory_kwargs_from_args(
                self._args("--memory-condition", "heard_seen", "--memory-store", path)
            )
        self.assertEqual(kwargs["memory_k"], 5)

    def _reject(self, argv):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(argv)


class TestImportingTheEntryPointDoesNotNeedTheSimulator(unittest.TestCase):
    """``__main__`` imports ``task/runner`` inside ``main()`` for this reason.

    ``task/runner.py`` reaches ``sim/world.py`` and therefore habitat-sim, so a
    module-level import would make the CLI's own tests uncollectable on a Mac — and this
    file is the proof, since it imported ``build_parser`` at the top.
    """

    def test_the_parser_and_the_mapping_are_importable_here(self):
        self.assertTrue(callable(build_parser))
        self.assertTrue(callable(config_from_args))

    def test_the_runner_is_imported_inside_main_rather_than_at_module_scope(self):
        import _tree

        path = _tree.PACKAGE_ROOT / "__main__.py"
        owners = [
            owner
            for owner, _lineno in _tree.module_imports_by_function(_tree.parse(path), "earshot")
            if owner != "<module>"
        ]
        self.assertIn("main", owners)


if __name__ == "__main__":
    unittest.main(verbosity=2)
