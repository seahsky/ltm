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
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.__main__ import build_parser, config_from_args
from earshot.agent.config import ControllerConfig, DetectorConfig, PlannerConfig
from earshot.audio.config import AudioConfig
from earshot.config import Detector, Localization, RunConfig

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

    def test_it_is_frozen(self):
        config = RunConfig(run_dir="runs/x")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.t_anom = 5

    def test_as_dict_is_json_serialisable(self):
        """It lands inside ``env_report.json``, so an enum reaching it would raise there."""
        payload = json.loads(json.dumps(RunConfig(run_dir="runs/x").as_dict()))
        self.assertEqual(payload["localization"], "realizable")
        self.assertEqual(payload["detector"], "oracle")

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
                    "--anomaly-class", "glass_break",
                    "--anomaly-clip", "/tmp/x.wav",
                    "--clap",
                    "--min-source-sep-m", "5.5",
                    "--max-source-dy-m", "0.5",
                    "--audio-step-ceiling-s", "0.25",
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
        self.assertEqual(config.anomaly_class, "glass_break")
        self.assertEqual(config.anomaly_clip, "/tmp/x.wav")
        self.assertTrue(config.clap)
        self.assertAlmostEqual(config.min_source_sep_m, 5.5)
        self.assertAlmostEqual(config.max_source_dy_m, 0.5)
        self.assertAlmostEqual(config.audio_step_ceiling_s, 0.25)
        self.assertTrue(config.overwrite)

    def test_the_arms_are_enums_rather_than_booleans(self):
        """ADR-0013: a third option is addable without a flag explosion.

        The old tree read ``LTM_REALIZABLE_LOCALIZATION`` from the environment, which is
        the shape that cannot grow a third value.
        """
        self._reject(["--run-dir", "runs/x", "--localization", "sometimes"])

    def test_the_anomaly_class_is_restricted_to_the_locked_three(self):
        """``clips.ANOMALY_CLASSES`` — the classes ESC-50 is staged for.

        An unrecognised class would resolve to no staged file, and
        ``load_anomaly_clip`` would raise at the point of use rather than at the point of
        the typo.
        """
        self._reject(["--run-dir", "runs/x", "--anomaly-class", "thunder"])


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
