"""The episode builder: ADR-0010's floor rule, the xz decoupling, and what is NOT here.

Every rule below is a rule about the *task*, so a violation does not crash anything — it
produces an episode whose anomaly response is degenerate and whose numbers still look
reasonable. That is the class of failure this map keeps finding after the fact, which is
why the builder raises and why these cases are stated one at a time.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401
from _task_fakes import make_episode, make_goal

from earshot.task.dataset import (
    PlacementError,
    build_anomaly_episodes,
    goal_table,
    place_anomaly_source,
    primary_anchor,
)
from earshot.task.episodes import EpisodeDataset
from earshot.types import Xyz


def dataset(episodes):
    return EpisodeDataset(
        scene_label="FAKE", scene_path="/nonexistent/FAKE.basis.glb", episodes=tuple(episodes)
    )


def table(**categories):
    """``{category: (goals,)}`` from ``{category: [positions]}``."""
    return {
        name: tuple(make_goal(position, category=name) for position in positions)
        for name, positions in categories.items()
    }


class TestGoalTable(unittest.TestCase):
    def test_it_gathers_every_category_in_the_scene(self):
        """The source is drawn from categories the primary episode knows nothing about.

        The published content file hoists one goal list per category and the loader
        resolves it onto each episode, so every category present in the scene appears on
        some episode — which is what makes a different-category source available without
        a second read of the file.
        """
        built = goal_table(
            dataset(
                [
                    make_episode(episode_id="0", category="chair"),
                    make_episode(
                        episode_id="1",
                        category="sofa",
                        goals=[make_goal(Xyz(4.0, 0.0, 0.0))],
                    ),
                ]
            )
        )
        self.assertEqual(sorted(built), ["chair", "sofa"])

    def test_instances_are_deduplicated_across_episodes_of_a_category(self):
        """Every episode of a category carries the same goal list.

        A naive concatenation would offer the same object as a candidate once per
        episode, which changes nothing about the pick and everything about a count.
        """
        goals = [make_goal(Xyz(4.0, 0.0, 0.0)), make_goal(Xyz(-4.0, 0.0, 0.0))]
        built = goal_table(
            dataset(
                [
                    make_episode(episode_id="0", category="chair", goals=goals),
                    make_episode(episode_id="1", category="chair", goals=goals),
                ]
            )
        )
        self.assertEqual(len(built["chair"]), 2)


class TestPrimaryAnchor(unittest.TestCase):
    def test_it_is_the_goal_view_point_nearest_the_start(self):
        """"The primary goal" is not one point when a category has several instances.

        The nearest-to-start instance is the one the episode is realistically about — the
        goal a working agent reaches — so it is the goal the detour has to be decoupled
        from, and the floor rule is measured against it.
        """
        episode = make_episode(
            start=Xyz(0.0, 0.0, 0.0),
            goals=[make_goal(Xyz(0.0, 0.0, -12.0)), make_goal(Xyz(0.0, 0.0, -3.0))],
        )
        self.assertEqual(primary_anchor(episode), Xyz(0.0, 0.0, -3.0))

    def test_an_episode_with_no_goals_raises(self):
        with self.assertRaises(PlacementError):
            primary_anchor(make_episode(goals=[]))


class TestPlacement(unittest.TestCase):
    def test_a_different_category_wins_over_a_nearer_same_category_instance(self):
        """The carried preference order, and the regime it exists for.

        ``anomaly_object`` differing from the find-target is what makes the detour
        genuinely decoupled — the agent is not investigating a second copy of the thing
        it was already looking for.
        """
        episode = make_episode(category="chair", goals=[make_goal(Xyz(0.0, 0.0, -9.0))])
        placement = place_anomaly_source(
            episode,
            table(
                chair=[Xyz(0.0, 0.0, -9.0), Xyz(3.5, 0.0, -9.0)],
                sofa=[Xyz(8.0, 0.0, -9.0)],
            ),
        )
        self.assertEqual(placement.anomaly_object, "sofa")
        self.assertFalse(placement.same_category)

    def test_among_one_category_the_nearest_qualifier_wins(self):
        """Proximity correlates with being on the same navmesh component.

        The farthest-first pick repeatedly landed on disconnected islands — an infinite
        geodesic, and a NaN in the soft-SPL that followed it.
        """
        episode = make_episode(goals=[make_goal(Xyz(0.0, 0.0, 0.0))])
        placement = place_anomaly_source(
            episode, table(sofa=[Xyz(20.0, 0.0, 0.0), Xyz(4.0, 0.0, 0.0)])
        )
        self.assertEqual(placement.position, Xyz(4.0, 0.0, 0.0))
        self.assertAlmostEqual(placement.separation_m, 4.0)

    def test_a_same_category_instance_is_the_fallback_and_says_so(self):
        """Recorded rather than silent, because the runner pays for it.

        The oracle detector's table is keyed by object name, so a same-category source
        merges with the primary goal's entry: ``detects("chair")`` is then true at either,
        and the visual confirm can fire at the wrong instance.
        """
        episode = make_episode(category="chair", goals=[make_goal(Xyz(0.0, 0.0, 0.0))])
        placement = place_anomaly_source(
            episode, table(chair=[Xyz(0.0, 0.0, 0.0), Xyz(5.0, 0.0, 0.0)])
        )
        self.assertEqual(placement.anomaly_object, "chair")
        self.assertTrue(placement.same_category)

    def test_the_separation_is_measured_against_every_primary_instance(self):
        """The disclosed strengthening over the old builder.

        The agent succeeds at ANY instance of the category, so a source 4 m from
        instance A and 0.5 m from instance B is not decoupled from the goal — it is a
        second route to it. The old builder measured against the one instance it had
        chosen as the cold start.
        """
        episode = make_episode(
            category="chair",
            goals=[make_goal(Xyz(0.0, 0.0, 0.0)), make_goal(Xyz(8.0, 0.0, 0.0))],
        )
        with self.assertRaises(PlacementError):
            place_anomaly_source(episode, table(sofa=[Xyz(7.5, 0.0, 0.0)]))

    def test_the_floor_rule_is_applied_before_the_nearest_first_tie_break(self):
        """ADR-0010, and the ordering is the whole point.

        An xz-near cross-floor candidate would otherwise WIN the nearest-first pick for
        being near — which is exactly what happened in ``TEEsavR23oF``, where a bed
        upstairs at y≈3.16 beat a chair downstairs 3.56 m away in xz.
        """
        episode = make_episode(goals=[make_goal(Xyz(0.0, 0.0, 0.0))])
        placement = place_anomaly_source(
            episode,
            table(sofa=[Xyz(3.5, 3.2, 0.0), Xyz(9.0, 0.0, 0.0)]),
        )
        self.assertEqual(placement.position, Xyz(9.0, 0.0, 0.0))
        self.assertLess(abs(placement.height_difference_m), 1.0)

    def test_a_scene_that_cannot_decouple_raises_rather_than_placing_at_the_goal(self):
        """A source at the goal is the exact degeneracy this module exists to prevent."""
        episode = make_episode(goals=[make_goal(Xyz(0.0, 0.0, 0.0))])
        with self.assertRaises(PlacementError) as caught:
            place_anomaly_source(episode, table(sofa=[Xyz(1.0, 0.0, 0.0)]))
        self.assertIn("too near", str(caught.exception))

    def test_the_placement_records_the_height_the_adr_does_not_constrain(self):
        """The controller pays for the start-to-source drop even when the ADR is satisfied.

        A source on the goal's floor is still a stair-climb if the agent starts a storey
        below it, and only the recorded number would say so afterwards.
        """
        episode = make_episode(
            start=Xyz(0.0, -3.0, 0.0), goals=[make_goal(Xyz(0.0, 0.0, 0.0))]
        )
        placement = place_anomaly_source(episode, table(sofa=[Xyz(5.0, 0.4, 0.0)]))
        self.assertAlmostEqual(placement.height_difference_m, 0.4)
        self.assertAlmostEqual(placement.height_difference_to_start_m, 3.4)


class TestBuild(unittest.TestCase):
    def test_it_builds_what_it_can_and_reports_what_it_cannot(self):
        """Placement attrition is a scene property and is counted, never swallowed.

        Distinct from §2.5's audibility attrition, which is NOT screened here at all and
        shows up as the funnel's stage 3.
        """
        buildable = make_episode(
            episode_id="ok", category="sofa", goals=[make_goal(Xyz(0.0, 0.0, 0.0))]
        )
        # Its two goal instances cover both objects in the scene, so every candidate is
        # within the bar of one of them and nothing can carry the source.
        stranded = make_episode(
            episode_id="stranded",
            category="chair",
            goals=[make_goal(Xyz(0.0, 0.0, 0.0)), make_goal(Xyz(6.0, 0.0, 0.0))],
        )
        scene = dataset([buildable, stranded])
        build = build_anomaly_episodes(scene, anomaly_class="alarm", t_anom=30)
        self.assertEqual([e.episode.episode_id for e in build.episodes], ["ok"])
        self.assertEqual([episode_id for episode_id, _ in build.skipped], ["stranded"])
        self.assertIn("stranded", build.summary())

    def test_a_scene_that_builds_nothing_raises(self):
        """A build that produced nothing and said so only in a list is a run that starts
        and immediately does nothing."""
        scene = dataset([make_episode(goals=[make_goal(Xyz(0.0, 0.0, 0.0))])])
        with self.assertRaises(PlacementError):
            build_anomaly_episodes(scene, anomaly_class="alarm", t_anom=30)

    def test_the_category_filter_is_on_the_primary_goal_only(self):
        """The source is still drawn from every category, which is what keeps a
        different-category source available under a single-category run."""
        scene = dataset(
            [
                make_episode(episode_id="0", category="chair", goals=[make_goal(Xyz(0.0, 0.0, 0.0))]),
                make_episode(episode_id="1", category="sofa", goals=[make_goal(Xyz(6.0, 0.0, 0.0))]),
            ]
        )
        build = build_anomaly_episodes(
            scene, anomaly_class="alarm", t_anom=30, category="chair"
        )
        self.assertEqual(len(build.episodes), 1)
        self.assertEqual(build.episodes[0].primary_category, "chair")
        self.assertEqual(build.episodes[0].source.anomaly_object, "sofa")

    def test_n_episodes_caps_the_build(self):
        goals = [make_goal(Xyz(0.0, 0.0, 0.0))]
        scene = dataset(
            [
                make_episode(episode_id="0", category="chair", goals=goals),
                make_episode(episode_id="1", category="chair", goals=goals),
                make_episode(episode_id="2", category="sofa", goals=[make_goal(Xyz(6.0, 0.0, 0.0))]),
            ]
        )
        build = build_anomaly_episodes(
            scene, anomaly_class="alarm", t_anom=30, n_episodes=1
        )
        self.assertEqual(len(build.episodes), 1)

    def test_the_anomaly_class_and_t_anom_reach_every_episode(self):
        scene = dataset(
            [
                make_episode(episode_id="0", category="chair", goals=[make_goal(Xyz(0.0, 0.0, 0.0))]),
                make_episode(episode_id="1", category="sofa", goals=[make_goal(Xyz(6.0, 0.0, 0.0))]),
            ]
        )
        build = build_anomaly_episodes(scene, anomaly_class="glass_break", t_anom=17)
        for anomaly_episode in build.episodes:
            self.assertEqual(anomaly_episode.anomaly_class, "glass_break")
            self.assertEqual(anomaly_episode.t_anom, 17)

    def test_every_source_names_an_object(self):
        """The map's standing hand-off to this ticket.

        The realizable arm's arrival is peak-or-plateau PLUS visual confirm, and there is
        nothing to visually confirm about a sentinel — so an unnamed source could only
        leave INVESTIGATE through the step-budget abort, visible as ``investigate_aborted``
        rather than as a crash. Every candidate is a real goal object, so every placement
        names one.
        """
        scene = dataset(
            [
                make_episode(episode_id="0", category="chair", goals=[make_goal(Xyz(0.0, 0.0, 0.0))]),
                make_episode(episode_id="1", category="sofa", goals=[make_goal(Xyz(6.0, 0.0, 0.0))]),
            ]
        )
        build = build_anomaly_episodes(scene, anomaly_class="alarm", t_anom=30)
        for anomaly_episode in build.episodes:
            self.assertTrue(anomaly_episode.source.anomaly_object)


class TestAudibilityIsNotScreened(unittest.TestCase):
    """§2.5, structurally: pre-screening would reintroduce offline rendering.

    Armed rather than documented. The tempting shortcut — "only place a source the agent
    could actually hear" — needs a render, and a builder that grew one would put the
    grid's offline pass back in through the back door and hide the attrition §6 wants
    visible at stage 3.
    """

    def test_the_builder_imports_nothing_from_the_audio_layer(self):
        import _tree

        path = _tree.PACKAGE_ROOT / "task" / "dataset.py"
        targets = [edge.target for edge in _tree.intra_package_edges(path, _tree.parse(path))]
        self.assertEqual(
            [target for target in targets if target.startswith("audio")],
            [],
            "task/dataset.py reached into audio/: §2.5 says audibility is not screened at "
            "build time, and the only way to screen it is to render",
        )

    def test_it_does_not_import_the_simulator_either(self):
        """Navigability comes from the source being a real goal view point, not a snap."""
        import _tree

        path = _tree.PACKAGE_ROOT / "task" / "dataset.py"
        self.assertFalse(_tree.imports_module(_tree.parse(path), "habitat_sim"))
        targets = [edge.target for edge in _tree.intra_package_edges(path, _tree.parse(path))]
        self.assertEqual([target for target in targets if target.startswith("sim")], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
