"""The ObjectNav loader, on real gzipped bytes. Our own logic, so it is a Mac test.

ADR-0014's rule is ownership, not dependency: the *subject* here is the parse and path
resolution this repo wrote, not habitat-sim's behaviour, so needing the real artefact
would be a seam defect rather than a reason to spend a box trip. The fixture is written
to disk with ``gzip`` and read back through the public entry point, so the test
exercises the same bytes-to-``Episode`` path a run does.

The schema constants are habitat-lab's, read from its source rather than inferred — the
citations are in ``task/episodes.py``'s docstring. What is asserted here is that this
loader honours them, which is the half a source citation cannot cover.
"""

import gzip
import json
import os
import shutil
import tempfile
import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401

from earshot.task.episodes import (
    DEFAULT_SCENE_PATH_PREFIX,
    EpisodeDataError,
    Xyz,
    available_scenes,
    find_scenes_dir,
    find_split_dir,
    goals_key,
    load_scene,
    parse_content,
    resolve_scene_path,
    scene_label,
)

SCENE = "TEEsavR23oF"
SCENE_ID = "hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"

# The rotation that pins the coefficient order. habitat-lab reads start_rotation as
# [x, y, z, w] (`utils/geometry_utils.py:55-60`), so a loader that quietly treated it as
# [w, x, y, z] would point the agent somewhere plausible and wrong. Deliberately
# asymmetric in every slot so a reorder cannot pass.
ROTATION = [0.1, 0.2, 0.3, 0.9273618]


def _view_point(x, y, z, iou=0.5):
    return {
        "agent_state": {"position": [x, y, z], "rotation": list(ROTATION)},
        "iou": iou,
    }


def _content(episodes=None, goals_by_category=None):
    """A content dict shaped like a published HM3D ObjectNav ``content/*.json.gz``."""
    return {
        "episodes": episodes
        if episodes is not None
        else [
            {
                "episode_id": "chair-cold-0",
                "scene_id": SCENE_ID,
                "start_position": [1.0, 0.1, 2.0],
                "start_rotation": list(ROTATION),
                "object_category": "chair",
                "goals": [],  # emptied by habitat-lab's dedup_goals
                "info": {"geodesic_distance": 7.5},
            },
            {
                "episode_id": "tv-warm-1",
                "scene_id": SCENE_ID,
                "start_position": [3.0, 0.1, 4.0],
                "start_rotation": list(ROTATION),
                "object_category": "tv_monitor",
                "goals": [],
                "info": {},
            },
        ],
        "goals_by_category": goals_by_category
        if goals_by_category is not None
        else {
            "TEEsavR23oF.basis.glb_chair": [
                {
                    "position": [5.0, 0.2, 6.0],
                    "object_id": "17",
                    "object_category": "chair",
                    "view_points": [_view_point(5.5, 0.1, 6.0), _view_point(4.5, 0.1, 6.0)],
                }
            ],
            "TEEsavR23oF.basis.glb_tv_monitor": [
                {
                    "position": [9.0, 1.2, 1.0],
                    "object_id": "31",
                    "object_category": "tv_monitor",
                    "view_points": [_view_point(9.0, 0.1, 2.0)],
                }
            ],
        },
    }


class TestPathResolution(unittest.TestCase):
    def test_scene_label_reduces_both_mesh_variants_to_one_key(self):
        """``.basis.glb`` and ``.glb`` must key the same scene.

        The label joins every per-scene analysis and names the content file, so a run on
        the basis mesh and a run on the plain mesh have to be comparable.
        """
        self.assertEqual(scene_label(SCENE_ID), SCENE)
        self.assertEqual(scene_label("hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.glb"), SCENE)
        self.assertEqual(scene_label("TEEsavR23oF.basis.glb"), SCENE)

    def test_resolve_scene_path_strips_the_published_prefix(self):
        """habitat-lab strips ``data/scene_datasets/`` then re-joins; so does this."""
        prefixed = DEFAULT_SCENE_PATH_PREFIX + SCENE_ID
        self.assertEqual(
            resolve_scene_path(prefixed, "data/hm3d/scene_datasets"),
            os.path.join("data/hm3d/scene_datasets", SCENE_ID),
        )

    def test_resolve_scene_path_leaves_an_unprefixed_id_alone(self):
        self.assertEqual(
            resolve_scene_path(SCENE_ID, "/box/data/hm3d/scene_datasets"),
            os.path.join("/box/data/hm3d/scene_datasets", SCENE_ID),
        )

    def test_goals_key_survives_a_multi_token_category(self):
        """``tv_monitor`` is why the key is built rather than split on ``_``."""
        self.assertEqual(goals_key(SCENE_ID, "tv_monitor"), "TEEsavR23oF.basis.glb_tv_monitor")
        self.assertEqual(goals_key(SCENE_ID, "chair"), "TEEsavR23oF.basis.glb_chair")


class TestDatasetDiscovery(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="earshot-episodes-")
        self.addCleanup(shutil.rmtree, self.root, True)

    def _make_split(self, layout, split="val", scenes=("TEEsavR23oF", "wcojb4TFT35")):
        content_dir = os.path.join(self.root, layout, split, "content")
        os.makedirs(content_dir)
        for scene in scenes:
            open(os.path.join(content_dir, scene + ".json.gz"), "wb").close()
        return os.path.join(self.root, layout, split)

    def test_the_canonical_box_layout_wins_over_the_legacy_one(self):
        """Both exist on some machines; the runbook's layout is the one that counts."""
        legacy = self._make_split("data/datasets/objectnav/hm3d/v1")
        canonical = self._make_split("data/hm3d/datasets/objectnav/hm3d/v1")
        self.assertTrue(os.path.isdir(legacy))
        self.assertEqual(find_split_dir("val", root=self.root), canonical)

    def test_the_legacy_layout_is_still_found_when_it_is_the_only_one(self):
        legacy = self._make_split("data/datasets/objectnav/hm3d/v1")
        self.assertEqual(find_split_dir("val", root=self.root), legacy)

    def test_a_missing_split_names_every_path_it_tried(self):
        """The old ``_default_episodes_path`` returned None and failed elsewhere."""
        with self.assertRaises(EpisodeDataError) as ctx:
            find_split_dir("val_mini", root=self.root)
        message = str(ctx.exception)
        self.assertIn("val_mini", message)
        self.assertIn("data/hm3d/datasets/objectnav/hm3d/v1", message)
        self.assertIn("data/datasets/objectnav/hm3d/v1", message)

    def test_scenes_dir_prefers_the_canonical_root(self):
        os.makedirs(os.path.join(self.root, "data/scene_datasets"))
        os.makedirs(os.path.join(self.root, "data/hm3d/scene_datasets"))
        self.assertEqual(
            find_scenes_dir(root=self.root),
            os.path.join(self.root, "data/hm3d/scene_datasets"),
        )

    def test_available_scenes_reads_the_content_directory_sorted(self):
        split_dir = self._make_split("data/hm3d/datasets/objectnav/hm3d/v1")
        self.assertEqual(available_scenes(split_dir), ("TEEsavR23oF", "wcojb4TFT35"))

    def test_available_scenes_ignores_anything_that_is_not_a_content_file(self):
        split_dir = self._make_split("data/hm3d/datasets/objectnav/hm3d/v1")
        open(os.path.join(split_dir, "content", "README.md"), "wb").close()
        self.assertEqual(available_scenes(split_dir), ("TEEsavR23oF", "wcojb4TFT35"))


class TestParse(unittest.TestCase):
    def test_the_authored_episode_id_survives(self):
        """The one deliberate divergence from habitat-lab, and it is a fix.

        ``object_nav_dataset.py:141`` overwrites every authored id with the load index.
        That renumbering is why the old analysis had to re-key onto
        ``(scene_id, target_category, visit_order)`` after silently dropping pairs, and
        why ``seed_only`` had to ride in ``episode.info``. Both are available here and
        neither is a lie.
        """
        episodes = parse_content(_content(), scenes_dir="/scenes")
        self.assertEqual([ep.episode_id for ep in episodes], ["chair-cold-0", "tv-warm-1"])
        self.assertEqual([ep.index for ep in episodes], [0, 1])

    def test_start_rotation_keeps_the_datasets_coefficient_order(self):
        """[x, y, z, w], verbatim. ``sim/world.py`` owns the reorder to (w, x, y, z)."""
        episode = parse_content(_content(), scenes_dir="/scenes")[0]
        self.assertEqual(list(episode.start_rotation), ROTATION)
        self.assertEqual(episode.start_position, Xyz(1.0, 0.1, 2.0))

    def test_goals_come_from_goals_by_category_when_the_episode_list_is_empty(self):
        """The published form: dedup_goals empties per-episode goals and hoists them."""
        episode = parse_content(_content(), scenes_dir="/scenes")[0]
        self.assertEqual(len(episode.goals), 1)
        self.assertEqual(episode.goals[0].position, Xyz(5.0, 0.2, 6.0))
        self.assertEqual(episode.goals[0].object_id, "17")
        self.assertEqual(len(episode.view_points()), 2)
        self.assertEqual(episode.view_points()[0].position, Xyz(5.5, 0.1, 6.0))
        self.assertAlmostEqual(episode.view_points()[0].iou, 0.5)

    def test_inline_goals_are_accepted_as_the_pre_dedup_form(self):
        """The smoke builders in the old tree wrote goals inline; both forms load."""
        content = _content(goals_by_category={})
        for episode in content["episodes"]:
            episode["goals"] = [
                {"position": [1.0, 2.0, 3.0], "view_points": [_view_point(1.0, 0.0, 3.0)]}
            ]
        parsed = parse_content(content, scenes_dir="/scenes")
        self.assertEqual(parsed[0].goals[0].position, Xyz(1.0, 2.0, 3.0))
        self.assertEqual(len(parsed[0].view_points()), 1)

    def test_goals_by_category_wins_over_a_stale_inline_list(self):
        """Both present is the pre-dedup file re-saved; the hoisted table is canonical."""
        content = _content()
        content["episodes"][0]["goals"] = [
            {"position": [-1.0, -1.0, -1.0], "view_points": []}
        ]
        episode = parse_content(content, scenes_dir="/scenes")[0]
        self.assertEqual(episode.goals[0].position, Xyz(5.0, 0.2, 6.0))

    def test_a_multi_token_category_resolves_its_own_goals(self):
        episode = parse_content(_content(), scenes_dir="/scenes")[1]
        self.assertEqual(episode.object_category, "tv_monitor")
        self.assertEqual(episode.goals[0].position, Xyz(9.0, 1.2, 1.0))

    def test_info_is_carried_and_copied(self):
        """``seed_only``, ``t_anom`` and the source position all ride here."""
        content = _content()
        episode = parse_content(content, scenes_dir="/scenes")[0]
        self.assertEqual(episode.info["geodesic_distance"], 7.5)
        content["episodes"][0]["info"]["geodesic_distance"] = 999.0
        self.assertEqual(episode.info["geodesic_distance"], 7.5)

    def test_the_scene_path_is_resolved_against_the_scenes_dir(self):
        episode = parse_content(_content(), scenes_dir="/box/data/hm3d/scene_datasets")[0]
        self.assertEqual(episode.scene_label, SCENE)
        self.assertEqual(
            episode.scene_path, "/box/data/hm3d/scene_datasets/" + SCENE_ID
        )


class TestParseRejectsBrokenData(unittest.TestCase):
    """Every field read here steers the agent, so a default is a silent wrong answer."""

    def test_a_missing_start_position_raises_rather_than_becoming_the_origin(self):
        content = _content()
        del content["episodes"][0]["start_position"]
        with self.assertRaises(EpisodeDataError) as ctx:
            parse_content(content, scenes_dir="/scenes")
        self.assertIn("start_position", str(ctx.exception))

    def test_a_three_element_rotation_raises(self):
        """Euler angles where coefficients belong is a plausible-looking corruption."""
        content = _content()
        content["episodes"][0]["start_rotation"] = [0.0, 1.0, 0.0]
        with self.assertRaises(EpisodeDataError) as ctx:
            parse_content(content, scenes_dir="/scenes")
        self.assertIn("[x, y, z, w]", str(ctx.exception))

    def test_a_category_with_no_goal_entry_raises_and_says_which_key_it_wanted(self):
        content = _content(goals_by_category={})
        with self.assertRaises(EpisodeDataError) as ctx:
            parse_content(content, scenes_dir="/scenes")
        self.assertIn("TEEsavR23oF.basis.glb_chair", str(ctx.exception))

    def test_an_empty_episode_list_raises(self):
        with self.assertRaises(EpisodeDataError):
            parse_content(_content(episodes=[]), scenes_dir="/scenes")

    def test_a_view_point_without_a_position_raises(self):
        content = _content()
        content["goals_by_category"]["TEEsavR23oF.basis.glb_chair"][0]["view_points"][0][
            "agent_state"
        ] = {}
        with self.assertRaises(EpisodeDataError) as ctx:
            parse_content(content, scenes_dir="/scenes")
        self.assertIn("view_point[0]", str(ctx.exception))


class TestLoadSceneOnRealBytes(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="earshot-episodes-")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.split_dir = os.path.join(
            self.root, "data/hm3d/datasets/objectnav/hm3d/v1/val"
        )
        os.makedirs(os.path.join(self.split_dir, "content"))
        self._write(SCENE, _content())

    def _write(self, scene, content):
        path = os.path.join(self.split_dir, "content", scene + ".json.gz")
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(content, handle)
        return path

    def test_a_real_gzipped_content_file_round_trips(self):
        dataset = load_scene(self.split_dir, SCENE, scenes_dir="/scenes")
        self.assertEqual(dataset.scene_label, SCENE)
        self.assertEqual(dataset.scene_path, "/scenes/" + SCENE_ID)
        self.assertEqual(len(dataset.episodes), 2)
        self.assertEqual(dataset.categories(), ("chair", "tv_monitor"))

    def test_the_category_filter_selects_and_the_dataset_agrees(self):
        dataset = load_scene(self.split_dir, SCENE, scenes_dir="/scenes", category="chair")
        self.assertEqual(len(dataset.episodes), 1)
        self.assertEqual(dataset.episodes[0].object_category, "chair")
        self.assertEqual(dataset.filter_category("chair"), dataset.episodes)

    def test_an_absent_category_raises_and_lists_what_is_there(self):
        """Empty-list-on-no-match reads downstream as "this scene has no episodes"."""
        with self.assertRaises(EpisodeDataError) as ctx:
            load_scene(self.split_dir, SCENE, scenes_dir="/scenes", category="sofa")
        self.assertIn("chair", str(ctx.exception))
        self.assertIn("tv_monitor", str(ctx.exception))

    def test_an_unknown_scene_raises_and_lists_the_available_ones(self):
        with self.assertRaises(EpisodeDataError) as ctx:
            load_scene(self.split_dir, "notascene", scenes_dir="/scenes")
        self.assertIn(SCENE, str(ctx.exception))

    def test_a_content_file_spanning_two_meshes_raises(self):
        """One content file is one scene; a mixed file would load the wrong geometry."""
        content = _content()
        content["episodes"][1]["scene_id"] = "hm3d/val/00801-wcojb4TFT35/wcojb4TFT35.basis.glb"
        content["goals_by_category"]["wcojb4TFT35.basis.glb_tv_monitor"] = content[
            "goals_by_category"
        ]["TEEsavR23oF.basis.glb_tv_monitor"]
        self._write("mixed", content)
        with self.assertRaises(EpisodeDataError) as ctx:
            load_scene(self.split_dir, "mixed", scenes_dir="/scenes")
        self.assertIn("one content file is one scene", str(ctx.exception))


class TestNoSceneDatasetConfig(unittest.TestCase):
    """Ticket 08's box fact, pinned as an invariant rather than left as a comment.

    Stock habitat-lab's ObjectNav HM3D benchmark never sets ``scene_dataset``, so it
    keeps the ``"default"`` that habitat-sim's own constructor already uses, and the
    episode's ``scene_id`` resolves as a plain filesystem path. The annotated config the
    old tree reached for bought exactly one thing — a semantic sensor — and that sensor
    is gone (ADR-0007, ticket 03, and the measured all-zeros behind every earlier
    result).

    So no module in this tree may *use* a scene-dataset config. Written as a test
    because the failure mode is someone reintroducing it as a convenience while chasing
    semantics, which is how the 9.3 GB stayed on the keep list in the first place.

    **AST-shaped, not grep-shaped, and that was learned here.** The first version of
    this class scanned raw lines and went red on ``task/episodes.py``'s own docstring —
    the citation chain that establishes the config is unnecessary read as a use of it.
    That is ticket 19's "a grep verifies presence not truth" arriving in the one place
    the map had not applied it. Docstrings and comments are excluded structurally, so
    documenting the decision cannot break the test that enforces it.
    """

    def test_no_module_sets_a_scene_dataset_config_file(self):
        offenders = []
        for path in _tree.agent_python_files():
            tree = _tree.parse(path)
            for lineno, attr in _tree.attribute_names(tree):
                if attr == "scene_dataset_config_file":
                    offenders.append(
                        "{}:{} — .{}".format(_tree.relative_path(path), lineno, attr)
                    )
        self.assertEqual(
            offenders,
            [],
            "ObjectNav HM3D v1 resolves scene_id as a plain path and habitat-sim's "
            "default is already 'default'; setting the field puts the semantic "
            "annotations back on the critical path:\n  " + "\n  ".join(offenders),
        )

    def test_no_module_carries_a_scene_dataset_config_path(self):
        """Neither the plain basis config nor the annotated one, as a live string."""
        offenders = []
        for path in _tree.agent_python_files():
            for lineno, value in _tree.code_string_constants(_tree.parse(path)):
                if "scene_dataset_config" in value or "hm3d_annotated_basis" in value:
                    offenders.append(
                        "{}:{} — {!r}".format(_tree.relative_path(path), lineno, value)
                    )
        self.assertEqual(offenders, [], "\n  ".join([""] + offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)
