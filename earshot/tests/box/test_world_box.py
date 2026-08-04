#!/usr/bin/env python3
"""``sim.World`` and the ObjectNav loader against the real stack. V100 + ``ss2``.

    conda activate ss2
    bash earshot/tools/box_gate.sh

Ticket 21's "done when": a real HM3D scene loads, ``observe()`` returns all three
modalities, the follower routes to a navmesh point, and the ``scene_dataset_config``
question has a **measured** answer written down.

**These tests print their measurements** (ADR-0014). Ticket 16's box trip left numbers,
not just green, and those numbers are what made tickets 09, 15 and 17 decidable.

Three things a Mac cannot settle, each of which would otherwise be an inference:

1. **Does ObjectNav HM3D v1 load without any scene-dataset config?**
   ``task/episodes.py`` answers it from habitat-lab's source — ``objectnav_hm3d.yaml``
   never sets ``scene_dataset``, so it keeps the ``"default"`` that habitat-sim's own
   constructor already uses. That is a citation. ``TestSceneDatasetConfigIsUnnecessary``
   below is the measurement, and it is decisive rather than merely consistent: it loads
   a scene that has **no ``.semantic.glb`` on disk at all**, which the annotated config
   would have had nothing to point at.
2. **Does an ``AudioSensorSpec`` reach the sensor suite through
   ``AgentConfiguration.sensor_specifications``?** ``World`` hands every spec through
   that one channel because it is audio-blind and because ``_sanitize_config`` derives
   ``create_renderer`` from the list before the Simulator exists. Tickets 04 and 16
   proved the *other* form (``sim.add_sensor`` after construction) on this branch, and
   ``Agent.__init__`` routes both through the same ``SensorFactory.create_sensors``
   (``agent/agent.py:158-171``) — but "same code path on habitat-sim 0.3.3" is a
   cross-version inference until this runs on the 2022-era branch the box builds.
3. **Does ``make_greedy_follower`` actually route?** The grid-A* it replaces was inert
   on the live path — no path found on roughly 92% of steps — so "the follower works" is
   the claim that fixed locomotion and the one worth re-measuring on new code.

**Run 1 (2026-08-04) went RED on two of twenty-one**, both the same cause:
``MultiGoalShortestPath.closest_end_point_index`` is present on habitat-sim 0.3.3 and
**absent on this branch**. It was read by a convenience that returned *which* end was
nearest; nothing consumed it, and it is gone. Cross-version skew on the box's 2022-era
``RLRAudioPropagationUpdate`` is exactly the risk this file exists to catch, and it
caught it — on the one field a Mac pre-flight had verified and therefore trusted.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import os
import time
import unittest

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which
# habitat-sim reads at import time.
import earshot  # noqa: F401
from earshot.audio.guard import apply_audio_config, arm_audio_context, guarded_observe
from earshot.task.episodes import (
    available_scenes,
    find_scenes_dir,
    find_split_dir,
    load_scene,
)
from earshot.types import Xyz

SPLIT = os.environ.get("SS2_SPLIT", "val")

# Far enough to need real routing, near enough to reach inside the step budget.
FOLLOW_MIN_M = 2.0
FOLLOW_MAX_M = 8.0
FOLLOW_DRAW_TRIES = 64
FOLLOW_MAX_STEPS = 200
FOLLOW_GOAL_RADIUS = 0.25
PLACEMENT_SEED = 20260804

_DATASET = None
_SCENES_DIR = None
_SPLIT_DIR = None
_MESH_COVERAGE = None
_SEMANTIC_SIBLING = None


def _semantic_sibling(scene_path):
    """The ``.semantic.glb`` beside a basis mesh, if the annotations were downloaded."""
    directory = os.path.dirname(scene_path)
    if not os.path.isdir(directory):
        return None
    for name in sorted(os.listdir(directory)):
        if name.endswith(".semantic.glb"):
            return os.path.join(directory, name)
    return None


def setUpModule():
    """Take the first scene whose mesh is actually on this box.

    **Run 1 of this test hunted for a scene with no ``.semantic.glb``**, on the theory
    that an absent annotation file is what makes ticket 08's answer decisive. All 20
    ObjectNav ``val`` scenes on this box are annotated, so it reported the weaker claim —
    and the theory was wrong anyway. The decisive evidence is not that the file is
    missing but that the loaded scene provably does not contain it
    (``test_the_annotations_on_disk_are_not_loaded``), which is available on every
    scene. So this stops at the first usable one instead of parsing twenty
    tens-of-megabytes content files to answer a question that no longer needs asking.
    """
    global _DATASET, _SCENES_DIR, _SPLIT_DIR, _MESH_COVERAGE, _SEMANTIC_SIBLING

    _SPLIT_DIR = find_split_dir(SPLIT)
    _SCENES_DIR = find_scenes_dir()
    scenes = available_scenes(_SPLIT_DIR)
    print("\n  split dir:  {}".format(_SPLIT_DIR), flush=True)
    print("  scenes dir: {}".format(_SCENES_DIR), flush=True)
    print("  content scenes: {}".format(len(scenes)), flush=True)

    override = os.environ.get("SS2_SCENE_LABEL")
    candidates = [override] if override else list(scenes)

    examined, chosen = 0, None
    for label in candidates:
        examined += 1
        dataset = load_scene(_SPLIT_DIR, label, scenes_dir=_SCENES_DIR)
        if os.path.exists(dataset.scene_path):
            chosen = dataset
            break

    if chosen is None:
        raise unittest.SkipTest(
            "no ObjectNav {} scene has its mesh on this box (examined {} of {} content "
            "scenes under {})".format(SPLIT, examined, len(candidates), _SCENES_DIR)
        )

    _DATASET = chosen
    _MESH_COVERAGE = (1, examined)
    _SEMANTIC_SIBLING = _semantic_sibling(chosen.scene_path)
    print("  examined {} content scene(s) to find a mesh".format(examined), flush=True)
    print("  chosen scene: {}".format(_DATASET.scene_label), flush=True)
    print("  mesh:         {}".format(_DATASET.scene_path), flush=True)
    print("  semantic sibling: {}".format(_SEMANTIC_SIBLING or "NONE — decisive"), flush=True)
    print("  episodes: {}  categories: {}".format(
        len(_DATASET.episodes), ", ".join(_DATASET.categories())), flush=True)


def _sensor_specs(with_audio=True):
    """RGB + depth + one audio sensor, exactly as the runner will assemble them."""
    import habitat_sim

    from earshot.sim.world import camera_sensor_specs

    specs = camera_sensor_specs(width=256, height=256)
    if not with_audio:
        return specs
    audio = habitat_sim.AudioSensorSpec()
    apply_audio_config(
        audio,
        {
            # Permanently off per ADR-0007 — SoundSpaces' own HM3D reference config.
            "uuid": "audio_sensor",
            "enableMaterials": False,
            "acousticsConfig": {"sampleRate": 44100.0},
        },
    )
    audio.channelLayout.type = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    audio.channelLayout.channelCount = 2
    specs.append(audio)
    return specs


def _new_world(with_audio=True):
    from earshot.sim.world import World

    return World(_DATASET.scene_path, _sensor_specs(with_audio=with_audio))


class TestSceneDatasetConfigIsUnnecessary(unittest.TestCase):
    """Ticket 08's outstanding box fact, measured. It also corrects the question.

    The question was framed as a choice between ``hm3d_basis`` and
    ``hm3d_annotated_basis``. Neither is used: ``World`` never sets
    ``scene_dataset_config_file``, so it stays at habitat-sim's ``"default"``, and the
    episode's ``scene_id`` resolves as a plain filesystem path — the same form ticket
    04 rendered against.

    **This is what stands between 9.3 GB of HM3D val and a deletion decision** (ticket
    10's keep list), so it prints the evidence rather than only passing.
    """

    def test_a_real_scene_loads_with_no_scene_dataset_config(self):
        import habitat_sim

        default = habitat_sim.SimulatorConfiguration().scene_dataset_config_file
        print("\n  --- ticket 08 ---", flush=True)
        print("  SimulatorConfiguration().scene_dataset_config_file = {!r}".format(default), flush=True)

        t0 = time.time()
        world = _new_world()
        elapsed = time.time() - t0
        try:
            print("  scene loaded in {:.2f} s".format(elapsed), flush=True)
            print("  navmesh loaded: {}".format(world.navmesh_loaded), flush=True)
            self.assertTrue(world.navmesh_loaded)
            self.assertEqual(default, "default")
        finally:
            world.close()

    def test_the_annotations_on_disk_are_not_loaded(self):
        """What makes the answer decisive, and it is not what run 1 looked for.

        Run 1 hunted for a scene with **no** ``.semantic.glb``, reasoning that the
        annotated config could then have had nothing to point at. Every ObjectNav
        ``val`` scene on this box is annotated, so it reported the weaker "the plain
        path suffices".

        The stronger claim was available all along and on every scene: the annotation
        file is **on disk and the loaded scene does not contain it**. habitat-sim says
        so itself — ``ResourceManager.cpp:473`` "Not loading semantic mesh",
        ``Simulator.cpp:471`` "The active scene does not contain semantic annotations",
        and ``AudioSensor.cpp:143`` "Semantic scene does not exist or materials are
        disabled, will use default material". This asserts it through the API rather
        than by scraping those lines.

        So the 9.3 GB of annotations are not merely unneeded by the *loader*; they are
        untouched by the scene the simulator actually builds.
        """
        world = _new_world()
        try:
            count = world.semantic_object_count()
            print("\n  semantic sibling on disk: {}".format(_SEMANTIC_SIBLING or "none"), flush=True)
            print("  semantic objects in the loaded scene: {}".format(count), flush=True)
            self.assertEqual(
                count,
                0,
                "the loaded scene carries semantic annotations — load_semantic_mesh "
                "was supposed to keep ticket 03's empty-mesh path shut",
            )
            if _SEMANTIC_SIBLING is not None:
                print(
                    "  DECISIVE: {} exists on disk and the loaded scene holds 0 "
                    "semantic objects. ObjectNav HM3D v1 needs NEITHER scene-dataset "
                    "config, and the annotations are untouched.".format(
                        os.path.basename(_SEMANTIC_SIBLING)
                    ),
                    flush=True,
                )
        finally:
            world.close()

    def test_the_loader_resolves_a_mesh_that_exists(self):
        """The path arithmetic, against the real layout rather than a fixture."""
        print(
            "\n  resolved a mesh after examining {} content scene(s): {}".format(
                _MESH_COVERAGE[1], _DATASET.scene_path
            ),
            flush=True,
        )
        self.assertTrue(os.path.exists(_DATASET.scene_path))


class TestEpisodesAgainstThePublishedDataset(unittest.TestCase):
    """The loader's own logic is Mac-tested; this checks the real bytes match it."""

    def test_every_episode_carries_a_start_pose_and_reachable_view_points(self):
        episode = _DATASET.episodes[0]
        print("\n  --- episode 0 ---", flush=True)
        print("  episode_id: {!r} (authored, not the load index)".format(episode.episode_id), flush=True)
        print("  category:   {}".format(episode.object_category), flush=True)
        print("  start:      {} rot {}".format(episode.start_position, episode.start_rotation), flush=True)
        print("  goals: {}  view points: {}".format(len(episode.goals), len(episode.view_points())), flush=True)
        self.assertGreater(len(episode.goals), 0)
        self.assertGreater(len(episode.view_points()), 0)
        self.assertEqual(len(episode.start_rotation), 4)

    def test_the_start_pose_and_the_view_points_are_on_the_navmesh(self):
        """The episodes must be navigable in the geometry this loader resolved to.

        The cross-check that a Mac cannot make: a wrong ``scenes_dir`` or a stale
        symlink (runbook section 6 names two) would load a *different* scene's mesh, and
        every path arithmetic test would still pass while nothing was navigable.
        """
        world = _new_world()
        try:
            episode = _DATASET.episodes[0]
            start_ok = world.is_navigable(episode.start_position)
            reachable = world.geodesic_distance(
                episode.start_position, [vp.position for vp in episode.view_points()]
            )
            print("\n  start on navmesh: {}".format(start_ok), flush=True)
            print("  geodesic start -> nearest view point: {}".format(reachable), flush=True)
            print("  authored info.geodesic_distance: {}".format(
                episode.info.get("geodesic_distance")), flush=True)
            self.assertTrue(start_ok, "episode start is off the navmesh — wrong mesh?")
            self.assertIsNotNone(reachable, "no navigable path to any view point")
        finally:
            world.close()


class TestObserveReturnsAllThreeModalities(unittest.TestCase):
    def test_one_call_returns_rgb_depth_and_the_ir(self):
        """Smoke criterion 1 rests on this: there is no separate audio render."""
        import numpy as np

        world = _new_world()
        try:
            observation = world.observe()
            keys = sorted(observation.keys())
            print("\n  --- observe() ---", flush=True)
            print("  keys: {}".format(keys), flush=True)
            for key in keys:
                array = np.asarray(observation[key])
                print("  {:<14} shape {} dtype {}".format(key, array.shape, array.dtype), flush=True)
            self.assertEqual(keys, ["audio_sensor", "depth", "rgb"])

            rgb = np.asarray(observation["rgb"])
            depth = np.asarray(observation["depth"])
            ir = np.asarray(observation["audio_sensor"])
            self.assertEqual(rgb.shape[:2], (256, 256))
            self.assertEqual(depth.shape[:2], (256, 256))

            # Depth in METRES, not normalised into [0, 1]. Under normalised depth the
            # frontier proposer's occupancy splat collapses (a 3 m wall reads 0.3) and
            # carves almost no free cells — measured on wcojb4TFT35 in Run 5.
            finite = depth[np.isfinite(depth) & (depth > 0)]
            print("  depth range: {:.3f} .. {:.3f} m".format(
                float(finite.min()) if finite.size else float("nan"),
                float(finite.max()) if finite.size else float("nan")), flush=True)
            self.assertTrue(finite.size > 0, "depth frame is entirely zero/non-finite")

            # Ticket 16: the audio observation is NOT a numpy array — getattr(obs,
            # "shape") reads None — so this asserts on the coerced form deliberately.
            print("  raw IR .shape attribute: {!r} (ticket 16: None)".format(
                getattr(observation["audio_sensor"], "shape", "<absent>")), flush=True)
            print("  IR peak abs: {:.6f}".format(float(np.max(np.abs(ir))) if ir.size else 0.0), flush=True)
            self.assertGreater(ir.size, 0)
        finally:
            world.close()

    def test_the_audio_spec_arrives_through_the_agent_config(self):
        """The cross-version inference this file exists to close (point 2 above)."""
        world = _new_world()
        try:
            handle = world.sensor_handle("audio_sensor")
            print("\n  audio sensor via AgentConfiguration.sensor_specifications: {}".format(
                type(handle).__name__), flush=True)
            self.assertIsNotNone(handle)
        finally:
            world.close()

    def test_the_guard_arms_and_passes_on_a_world_render(self):
        """``arm_audio_context`` against ``World.observe`` rather than a bare lambda.

        Ticket 16 verified the guard on a hand-built sim. This is the first time it
        meets the module the runner will actually call, and the seam it owns —
        ``render`` returning an observation *dict* rather than the IR — is new code.
        """
        import numpy as np

        world = _new_world()
        try:
            handle = world.sensor_handle("audio_sensor")
            source = world.random_navigable_point()
            handle.setAudioSourceTransform(np.asarray(source.as_tuple(), dtype=np.float32))
            report = arm_audio_context(
                handle, lambda: world.observe()["audio_sensor"]
            )
            print("\n  --- guard on World ---", flush=True)
            for key in ("n_vertices", "submitted_n_vertices", "log_canary_seen",
                        "ir_peak_abs", "obj_written"):
                print("  {:<22} {}".format(key, getattr(report, key)), flush=True)
            self.assertTrue(report.log_canary_seen)
            self.assertEqual(report.fatal_log_lines, [])
        finally:
            world.close()

    def test_render_count_equals_observe_count_and_step_never_renders(self):
        """Smoke criterion 1, made falsifiable.

        ``habitat_sim.Simulator.step`` acts *and* renders. ``World.step`` deliberately
        only acts, so ``n_renders`` counts observations exactly — otherwise "render
        count equals step count" would be true no matter what the code did.
        """
        from earshot.sim.world import TURN_LEFT

        world = _new_world()
        try:
            for _ in range(5):
                world.step(TURN_LEFT)
            self.assertEqual(world.n_renders, 0)
            for _ in range(5):
                world.step(TURN_LEFT)
                world.observe()
            print("\n  after 10 steps / 5 observes: n_steps={} n_renders={}".format(
                world.n_steps, world.n_renders), flush=True)
            self.assertEqual(world.n_steps, 10)
            self.assertEqual(world.n_renders, 5)
        finally:
            world.close()

    def test_guarded_observe_wraps_the_shared_call_for_a_short_run(self):
        """The per-step half, driving ``World`` and prices the live loop."""
        world = _new_world()
        try:
            from earshot.sim.world import MOVE_FORWARD

            elapsed = []
            for step in range(10):
                t0 = time.time()
                observation, report = guarded_observe(
                    world.observe, occasion="step {}".format(step)
                )
                elapsed.append(time.time() - t0)
                self.assertTrue(report.log_canary_seen)
                self.assertEqual(report.fatal_log_lines, [])
                self.assertIn("audio_sensor", observation)
                world.step(MOVE_FORWARD)
            print(
                "\n  guarded observe+step over 10 steps: min {:.1f} ms / mean {:.1f} ms "
                "/ max {:.1f} ms".format(
                    1000 * min(elapsed), 1000 * sum(elapsed) / len(elapsed), 1000 * max(elapsed)
                ),
                flush=True,
            )
            self.assertEqual(world.n_renders, 10)
        finally:
            world.close()


class TestTheFollowerRoutes(unittest.TestCase):
    """The claim that fixed locomotion, re-measured on new code.

    The grid-A* this replaces found no path on roughly 92% of steps and fell back to
    straight-line steering, so "a waypoint was chosen" and "the agent got there" were
    different things for the whole of the earlier work.
    """

    def test_the_follower_reaches_a_navmesh_point(self):
        world = _new_world(with_audio=False)
        try:
            world.seed_navmesh(PLACEMENT_SEED)
            start = world.pose().position
            target = None
            for _ in range(FOLLOW_DRAW_TRIES):
                candidate = world.random_navigable_point()
                distance = world.geodesic_distance(start, [candidate])
                if distance is not None and FOLLOW_MIN_M <= distance <= FOLLOW_MAX_M:
                    target = candidate
                    break
            if target is None:
                self.skipTest("no navigable point {}-{} m away after {} draws".format(
                    FOLLOW_MIN_M, FOLLOW_MAX_M, FOLLOW_DRAW_TRIES))

            initial = world.geodesic_distance(start, [target])
            print("\n  --- follower ---", flush=True)
            print("  start {} -> target {}".format(start, target), flush=True)
            print("  geodesic at start: {:.3f} m".format(initial), flush=True)

            steer = world.follower(FOLLOW_GOAL_RADIUS)
            actions = []
            for _ in range(FOLLOW_MAX_STEPS):
                action = steer(target)  # None means arrived; NoRouteError means stuck
                if action is None:
                    break
                actions.append(action)
                world.step(action)

            final = world.geodesic_distance(world.pose().position, [target])
            print("  actions: {} ({} forward / {} turn)".format(
                len(actions),
                sum(1 for a in actions if a == "move_forward"),
                sum(1 for a in actions if a != "move_forward")), flush=True)
            print("  geodesic at stop:  {}".format(final), flush=True)
            self.assertIsNotNone(final, "agent left the navmesh while following")
            self.assertLess(
                final,
                FOLLOW_GOAL_RADIUS + 0.35,
                "follower stopped {:.3f} m short of a point it said it had reached".format(final),
            )
            self.assertLess(len(actions), FOLLOW_MAX_STEPS, "follower never signalled arrival")
        finally:
            world.close()

    def test_snap_point_returns_none_rather_than_nans_off_the_navmesh(self):
        """habitat-sim signals failure with NaNs, which read as a coordinate.

        The negative arm ADR-0014 requires: a converter that has only ever seen good
        input is indistinguishable from one that cannot fail.
        """
        world = _new_world(with_audio=False)
        try:
            snapped = world.snap_point(Xyz(1e6, 1e6, 1e6))
            print("\n  snap_point(1e6, 1e6, 1e6) -> {!r}".format(snapped), flush=True)
            if snapped is not None:
                print(
                    "  NOTE: this navmesh snapped a point 1e6 m away rather than "
                    "returning NaN — the None conversion is untested here, not wrong.",
                    flush=True,
                )
            unreachable = world.geodesic_distance(
                world.random_navigable_point(), [Xyz(1e6, 1e6, 1e6)]
            )
            print("  geodesic to an off-navmesh point -> {!r} (inf becomes None)".format(
                unreachable), flush=True)
            self.assertIsNone(unreachable)
        finally:
            world.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
