#!/usr/bin/env python3
"""Off-box tests for ticket 06's render-cost probe.

    python3 test_rendercost_probe.py

Needs numpy (the probe imports it at module scope); needs no ``habitat_sim``.
On this Mac: `/opt/anaconda3/envs/habitat/bin/python`.

Ticket 06's first pass was verified against a throwaway stub that caught four
defects and was then not committed, so nothing held those fixes in place. This
file is that stub, kept — same bar as ticket 12's ``test_audio_guard.py``.

The warning ticket 16 fired applies here in full: **a green suite against fakes
licenses nothing about binding behaviour.** Ticket 12 passed 27 tests and then
raised on the first real spec. What these cover is the probe's own logic — the
seed, the guard seam, scene selection, walk sampling and the verdict rules —
which is where the four original defects lived. Whether ``pathfinder.seed`` and
``arm_audio_context`` behave as assumed against the real binary is a box fact,
and ticket 16 measured the guard half of it.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import rendercost_probe as rp  # noqa: E402


# ----------------------------------------------------------------------
# fakes — the simulator surface the probe actually touches, nothing more
# ----------------------------------------------------------------------


class FakePathfinder:
    """Deterministic navigable-point sampler with habitat's seed semantics."""

    def __init__(self, points=None):
        self.is_loaded = True
        self._points = points or [
            np.array([float(i), 0.0, float(i) * 0.5], dtype=np.float32)
            for i in range(1, 40)
        ]
        self._cursor = 0
        self.seed_calls = []

    def seed(self, value):
        self.seed_calls.append(value)
        # habitat's seed rewinds the sampler; the fake models exactly that, which
        # is the property the probe depends on and nothing more.
        self._cursor = value % len(self._points)

    def get_random_navigable_point(self):
        pt = self._points[self._cursor % len(self._points)]
        self._cursor += 1
        return pt.copy()

    def snap_point(self, p):
        return np.asarray(p, dtype=np.float32)

    def find_path(self, path):
        path.points = [path.requested_start, path.requested_end]
        path.geodesic_distance = float(
            np.linalg.norm(np.asarray(path.requested_end, dtype=np.float32)
                           - np.asarray(path.requested_start, dtype=np.float32)))
        return True


class SeedlessPathfinder(FakePathfinder):
    """A build whose PathFinder has no ``seed`` at all."""

    seed = None  # shadows the method: getattr(...) is not callable


class FakeShortestPath:
    def __init__(self):
        self.requested_start = None
        self.requested_end = None
        self.points = []
        self.geodesic_distance = 0.0


class FakeSim:
    def __init__(self, pathfinder=None):
        self.pathfinder = pathfinder or FakePathfinder()
        self.closed = False
        self.sensors = {}
        self.renders = 0

    def close(self):
        self.closed = True

    def get_agent(self, _i):
        return types.SimpleNamespace(_sensors=self.sensors)

    def get_sensor_observations(self):
        self.renders += 1
        return {uuid: np.zeros((2, 1024), dtype=np.float32)
                for uuid in self.sensors}


def install_fake_habitat_sim():
    """Put a habitat_sim stub in sys.modules for the duration of a test."""
    mod = types.ModuleType("habitat_sim")
    mod.ShortestPath = FakeShortestPath
    sys.modules["habitat_sim"] = mod
    return mod


class HabitatStubTest(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("habitat_sim")
        install_fake_habitat_sim()

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("habitat_sim", None)
        else:
            sys.modules["habitat_sim"] = self._saved


# ----------------------------------------------------------------------
# the seed — ticket 16's fix, ported
# ----------------------------------------------------------------------


class TestSeed(unittest.TestCase):
    def test_seed_is_applied_and_reported(self):
        sim = FakeSim()
        self.assertTrue(rp.seed_pathfinder(sim, 20260803))
        self.assertEqual(sim.pathfinder.seed_calls, [20260803])

    def test_no_seed_leaves_the_sampler_alone(self):
        sim = FakeSim()
        self.assertFalse(rp.seed_pathfinder(sim, None))
        self.assertEqual(sim.pathfinder.seed_calls, [])

    def test_absent_seed_method_reports_false_rather_than_raising(self):
        """A build without pathfinder.seed must say so, not claim it was seeded.

        The silent-failure class ticket 17 named: a run that reports seeded and
        is not turns an unexplained knob effect into a mystery two runs later.
        """
        sim = FakeSim(SeedlessPathfinder())
        self.assertFalse(rp.seed_pathfinder(sim, 20260803))

    def test_seeded_draw_repeats_across_simulators(self):
        """The whole point: two runs, same geometry."""
        picks = []
        for _ in range(2):
            sim = FakeSim()
            rp.seed_pathfinder(sim, 20260803)
            src, start, _d = rp.pick_source_and_start(sim, min_dist=1.0)
            picks.append((src.tolist(), start.tolist()))
        self.assertEqual(picks[0], picks[1])

    def test_unseeded_draw_does_not_repeat(self):
        """Guards the test above against a fake that repeats for other reasons."""
        pathfinder = FakePathfinder()
        sim = FakeSim(pathfinder)
        first = rp.pick_source_and_start(sim, min_dist=1.0)[0].tolist()
        second = rp.pick_source_and_start(sim, min_dist=1.0)[0].tolist()
        self.assertNotEqual(first, second)


# ----------------------------------------------------------------------
# the guard seam — a broken audio context must not produce a timing
# ----------------------------------------------------------------------


class FakeSensor:
    def __init__(self):
        self.source_transforms = []

    def setAudioSourceTransform(self, xyz):
        self.source_transforms.append(np.asarray(xyz, dtype=np.float32).tolist())


class TestGuardSeam(HabitatStubTest):
    def _patch_arm(self, fn):
        saved = {name: getattr(rp, name)
                 for name in ("build_sim", "attach_audio", "arm_audio_context",
                              "place")}
        sim = FakeSim()
        self.sensor = FakeSensor()
        self.placements = []
        rp.build_sim = lambda scene, with_camera: sim
        rp.attach_audio = lambda s, cfg, sr: (
            s.sensors.setdefault("audio_sensor", self.sensor), "audio_sensor")
        rp.arm_audio_context = fn
        rp.place = lambda s, sensor, listener, source: self.placements.append(
            (np.asarray(listener).tolist(), np.asarray(source).tolist()))
        for name, value in saved.items():
            self.addCleanup(lambda n=name, v=value: setattr(rp, n, v))
        return sim

    def test_healthy_context_is_recorded_on_the_geometry(self):
        report = types.SimpleNamespace(
            as_dict=lambda: {"n_vertices": 392364, "ir_peak_abs": 0.130,
                             "stdout_chars": 916, "stderr_chars": 0})
        self._patch_arm(lambda sensor, render: report)
        geom = rp.walk_geometry("scene.glb", 1.0, 0.5, 20, seed=1, guard=True)
        self.assertEqual(geom["audio_context"]["n_vertices"], 392364)
        self.assertTrue(geom["seed_applied"])

    def test_broken_context_propagates_instead_of_being_timed(self):
        """A scene whose audio is broken must contribute no number at all.

        Ticket 16 measured that a failed render still returns Success and prints
        only to fd 2. A broken context renders FAST, so swallowing this would
        bias the verdict toward "affordable" — the one direction we cannot
        afford to be wrong in.
        """
        def boom(sensor, render):
            raise rp.AudioContextError("scene mesh has 0 vertices")

        self._patch_arm(boom)
        with self.assertRaises(rp.AudioContextError):
            rp.walk_geometry("scene.glb", 1.0, 0.5, 20, seed=1, guard=True)

    def test_guard_off_skips_it_and_says_so(self):
        self._patch_arm(lambda sensor, render: self.fail("guard must not run"))
        geom = rp.walk_geometry("scene.glb", 1.0, 0.5, 20, seed=1, guard=False)
        self.assertIsNone(geom["audio_context"])

    def test_the_guard_reads_its_uuid_back_rather_than_choosing_one(self):
        """The box killed the obvious design here.

        The first instinct was to give the guard's sensor its own uuid so it
        could never be confused with a timing sensor. The box run found that
        `AudioSensorSpec` ships `uuid = "audio_sensor"` and assigning another
        does not take — the Python `_sensors` dict picks up the new name while
        the C++ suite keeps the old, and `get_sensor_observations()` dies on the
        cross-lookup. So the guard uses whatever uuid it is handed back.

        Isolation comes from the sim instead: the guard runs on the throwaway
        geometry sim, which is closed before any timing sim is built.
        """
        self.assertFalse(hasattr(rp, "GUARD_UUID"))

    def test_the_source_is_placed_before_the_guard_renders(self):
        """The guard asserts a non-silent IR, so an unplaced source fails it for
        a reason that has nothing to do with the audio context.

        Ticket 16 hit this and answered it by seeding a 1-8 m pair before arming.
        """
        seen = {}

        def arm(sensor, render):
            seen["placements_before_arm"] = len(self.placements)
            return types.SimpleNamespace(as_dict=lambda: {"n_vertices": 392364})

        self._patch_arm(arm)
        rp.walk_geometry("scene.glb", 1.0, 0.5, 20, seed=1, guard=True,
                         source_height=1.0)
        self.assertEqual(seen["placements_before_arm"], 1)

    def test_the_guard_source_carries_the_walk_source_height(self):
        """It must be the same elevated point the sweep will time against, not
        the raw navmesh point."""
        self._patch_arm(
            lambda sensor, render: types.SimpleNamespace(as_dict=lambda: {}))
        geom = rp.walk_geometry("scene.glb", 1.0, 0.5, 20, seed=1, guard=True,
                                source_height=1.25)
        _listener, source = self.placements[0]
        self.assertAlmostEqual(source[1], float(geom["source"][1]) + 1.25, places=5)


# ----------------------------------------------------------------------
# scene selection
# ----------------------------------------------------------------------


class TestFindScenes(unittest.TestCase):
    def setUp(self):
        rp.REPORT.clear()
        self._cwd = os.getcwd()
        import tempfile
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)
        self.addCleanup(lambda: os.chdir(self._cwd))

    def _touch(self, rel):
        os.makedirs(os.path.dirname(rel), exist_ok=True)
        open(rel, "w").close()

    def test_duplicate_trees_collapse_to_one_scene(self):
        """Ticket 05's suspected 9.3 GB duplicate, as a measurement bug.

        build_verdict keys scenes by basename, so two copies of one room read as
        "admissible in every scene" while only one room was ever measured.
        """
        self._touch("data/hm3d/scene_datasets/hm3d/val/00800-AAA/AAA.basis.glb")
        self._touch("data/hm3d/duplicate/hm3d/val/00800-AAA/AAA.basis.glb")
        self._touch("data/hm3d/scene_datasets/hm3d/val/00801-BBB/BBB.basis.glb")
        scenes = rp.find_scenes([], limit=2)
        self.assertEqual(sorted(os.path.basename(s) for s in scenes),
                         ["AAA.basis.glb", "BBB.basis.glb"])
        self.assertEqual(len(rp.REPORT["_duplicate_scenes_dropped"]), 1)

    def test_semantic_meshes_are_never_selected(self):
        self._touch("data/hm3d/scene_datasets/hm3d/val/00800-AAA/AAA.semantic.glb")
        self._touch("data/hm3d/scene_datasets/hm3d/val/00800-AAA/AAA.basis.glb")
        scenes = rp.find_scenes([], limit=4)
        self.assertEqual([os.path.basename(s) for s in scenes], ["AAA.basis.glb"])

    def test_explicit_scene_must_exist(self):
        with self.assertRaises(RuntimeError):
            rp.find_scenes(["nope.glb"], limit=1)


# ----------------------------------------------------------------------
# the walk — ticket 06's defect 2, held in place
# ----------------------------------------------------------------------


class TestResample(unittest.TestCase):
    def test_samples_span_the_whole_path(self):
        """Fixed spacing plus a truncation sampled 4 m of a 13.6 m path and
        scored every config FLAT. The far end is where the gradient lives."""
        pts = [np.array([0.0, 0.0, 0.0], dtype=np.float32),
               np.array([13.6, 0.0, 0.0], dtype=np.float32)]
        out = rp.resample_polyline(pts, 8)
        self.assertEqual(len(out), 8)
        self.assertAlmostEqual(float(out[0][0]), 0.0, places=3)
        self.assertAlmostEqual(float(out[-1][0]), 13.6, places=3)

    def test_degenerate_path_does_not_divide_by_zero(self):
        pts = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
        self.assertEqual(len(rp.resample_polyline(pts, 5)), 1)


# ----------------------------------------------------------------------
# the verdict — the pre-registered thresholds, and defect 3
# ----------------------------------------------------------------------


def _row(label, scene, ms, admissible):
    return {
        "label": label,
        "steady_state": {"median_ms": ms},
        "gradient_admissible": admissible,
        "gradient_rho": -0.9 if admissible else -0.1,
        "gradient_dynamic_range_db": 9.0 if admissible else 1.0,
    }


class TestVerdict(unittest.TestCase):
    def setUp(self):
        rp.REPORT.clear()

    def _run(self, per_scene):
        rp.REPORT["02_sweep"] = {"per_scene": per_scene, "ok": True}
        return rp.build_verdict()

    def test_admissible_in_one_scene_only_is_not_a_recommendation(self):
        """Defect 3: the cheapest admissible row across all scenes was taken as
        a win while the same config went flat in the other room."""
        v = self._run([
            {"scene": "a/AAA.glb", "results": [_row("cheap", "AAA", 10.0, True)]},
            {"scene": "b/BBB.glb", "results": [_row("cheap", "BBB", 10.0, False)]},
        ])
        self.assertEqual(v["verdict"], "INDETERMINATE")
        self.assertIn("some scenes but not all", v["reason"])

    def test_cost_is_quoted_at_the_worst_scene(self):
        v = self._run([
            {"scene": "a/AAA.glb", "results": [_row("cheap", "AAA", 10.0, True)]},
            {"scene": "b/BBB.glb", "results": [_row("cheap", "BBB", 44.0, True)]},
        ])
        self.assertEqual(v["verdict"], "LIVE_EVERY_STEP_HOLDS")
        self.assertEqual(v["best_admissible"]["steady_ms"], 44.0)

    def test_thresholds_are_the_pre_registered_ones(self):
        for ms, expected in ((50.0, "LIVE_EVERY_STEP_HOLDS"),
                             (50.1, "LIVE_EVERY_STEP_TOLERABLE"),
                             (150.0, "LIVE_EVERY_STEP_TOLERABLE"),
                             (150.1, "THROTTLE_REQUIRED")):
            with self.subTest(ms=ms):
                rp.REPORT.clear()
                v = self._run([{"scene": "a/AAA.glb",
                                "results": [_row("cheap", "AAA", ms, True)]}])
                self.assertEqual(v["verdict"], expected)

    def test_a_failed_scene_becomes_a_blocker(self):
        v = self._run([{"scene": "a/AAA.glb",
                        "error": "AudioContextError('0 vertices')"}])
        self.assertEqual(v["verdict"], "INDETERMINATE")
        self.assertTrue(any("AudioContextError" in b for b in v["blockers"]))

    def test_provenance_travels_with_the_verdict(self):
        rp.REPORT["_seed"] = 20260803
        rp.REPORT["_guard_enabled"] = True
        rp.REPORT["_duplicate_scenes_dropped"] = [{"dropped": "x", "kept": "y"}]
        v = self._run([{"scene": "a/AAA.glb", "seed_applied": True,
                        "audio_context": {"n_vertices": 392364},
                        "results": [_row("cheap", "AAA", 10.0, True)]}])
        prov = v["provenance"]
        self.assertEqual(prov["seed"], 20260803)
        self.assertEqual(prov["seeded_scenes"], 1)
        self.assertEqual(prov["guarded_scenes"], 1)
        self.assertEqual(prov["duplicate_scenes_dropped"], 1)


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------


class TestStats(unittest.TestCase):
    def test_spearman_is_minus_one_on_a_perfect_climb(self):
        energy = [-60.0, -50.0, -40.0, -30.0]
        dist = [8.0, 6.0, 4.0, 2.0]
        self.assertAlmostEqual(rp.spearman(energy, dist), -1.0, places=6)

    def test_spearman_handles_ties(self):
        self.assertIsNotNone(rp.spearman([1.0, 1.0, 2.0], [3.0, 2.0, 1.0]))

    def test_spearman_is_none_when_a_series_is_constant(self):
        self.assertIsNone(rp.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))

    def test_a_silent_ir_floors_ABOVE_the_minus_300_guard(self):
        """The reason `scorable` cannot filter on energy alone.

        The 1e-20 epsilon puts an all-zero IR at -200 dB, which clears the -300
        guard that was written for the empty-IR -inf. Pinned because the whole
        silent-sample rule rests on this number.
        """
        self.assertAlmostEqual(
            rp.energy_db(np.zeros((2, 128), dtype=np.float32)), -200.0, places=3)

    def test_empty_ir_is_minus_infinity(self):
        self.assertEqual(rp.energy_db(np.zeros((0,), dtype=np.float32)),
                         float("-inf"))


def _walk_row(i, dist, energy, nonzero=True):
    return {"i": i, "s": 0.01, "geodesic_m": dist, "energy_db": energy,
            "ir_samples": 1024, "ir_nonzero": nonzero}


class TestScorable(unittest.TestCase):
    def test_silent_renders_are_not_scored(self):
        """A silent step is a sentinel, not a measurement.

        The silent steps are the FAR ones, so leaving them in manufactures a
        steep gradient out of a dead field — a broken config scoring climbable.
        """
        rows = [_walk_row(0, 8.0, -200.0, nonzero=False),
                _walk_row(1, 4.0, -40.0),
                _walk_row(2, 2.0, -30.0)]
        self.assertEqual([r["i"] for r in rp.scorable(rows)], [1, 2])

    def test_unreachable_geodesic_is_not_scored(self):
        rows = [_walk_row(0, float("inf"), -40.0), _walk_row(1, 2.0, -30.0)]
        self.assertEqual([r["i"] for r in rp.scorable(rows)], [1])

    def test_empty_ir_sentinel_is_not_scored(self):
        rows = [_walk_row(0, 8.0, float("-inf")), _walk_row(1, 2.0, -30.0)]
        self.assertEqual([r["i"] for r in rp.scorable(rows)], [1])

    def test_a_dead_walk_scores_nothing_rather_than_scoring_well(self):
        rows = [_walk_row(i, 8.0 - i, -200.0, nonzero=False) for i in range(6)]
        self.assertEqual(rp.scorable(rows), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
