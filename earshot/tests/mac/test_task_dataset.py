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
    ARRIVAL_RADIUS_M,
    FORWARD_STEP_M,
    T_ANOM_FLOOR_STEPS,
    PlacementError,
    build_anomaly_episodes,
    derive_t_anom,
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


class TestTheFloorRuleCoversTheStartToo(unittest.TestCase):
    """The break ticket 26's first full run walked into, measured on the box.

    HM3D ObjectNav episodes routinely start a storey from their goal — the smoke's own
    episode 0 begins at y +2.064 with its nearest bed view point at y -0.536, a **2.6 m**
    gap and an authored geodesic of 5.98 m via stairs. The floor rule measured the source
    against ``primary_anchor`` alone, so a source at the goal's level passed
    (``|anchor - source|`` 0.000, ``source_dy_m`` 0.000) while sitting a full storey below
    where the agent begins.

    The episode was then legal by the builder's own test and unwinnable in practice: the
    onset fired at step 30 with the agent still upstairs, and a greedy energy climb cannot
    take stairs. It spent its entire 120-step budget on the wrong floor — 62 forwards, **0
    collisions**, 15.5 m of clean walking — while the measured RMS *fell* from 0.0407 to
    0.0121, chasing sound leaking through a stairwell. ``source_is_visible`` was false at
    every one of 153 steps. Smoke criterion 5 was unreachable by construction.

    **Both anchors, because ``t_anom`` makes the agent's floor unknowable at build time.**
    The anomaly fires mid-episode, so the agent may be on the start's floor or the goal's.
    Requiring the source within ``max_dy_m`` of *both* is the only placement that is
    climbable either way — and it has the right side effect: in a cross-floor episode the
    two anchors are further apart than the rule allows, no candidate can satisfy both, and
    the episode is skipped with a reason rather than run as a silent null.
    """

    def _cross_floor(self):
        """Start upstairs, primary goal and every candidate downstairs. The smoke's case."""
        return make_episode(
            category="bed",
            start=Xyz(0.0, 2.064, 0.0),
            goals=[make_goal(Xyz(0.0, -0.536, -3.0))],
        )

    def test_a_source_a_storey_below_the_start_is_rejected(self):
        with self.assertRaises(PlacementError):
            place_anomaly_source(
                self._cross_floor(), table(toilet=[Xyz(0.0, -0.536, -8.0)])
            )

    def test_the_message_names_the_start_so_the_cause_is_readable(self):
        """A skip reason that only said "another floor" would point at the goal."""
        with self.assertRaises(PlacementError) as caught:
            place_anomaly_source(
                self._cross_floor(), table(toilet=[Xyz(0.0, -0.536, -8.0)])
            )
        self.assertIn("start", str(caught.exception).lower())

    def test_a_source_on_the_starts_floor_is_still_rejected_when_the_goal_is_not(self):
        """Both anchors, not either. The agent may have descended by ``t_anom``."""
        with self.assertRaises(PlacementError):
            place_anomaly_source(
                self._cross_floor(), table(toilet=[Xyz(0.0, 2.064, -8.0)])
            )

    def test_a_same_floor_episode_is_unaffected(self):
        """The regression guard: the rule only tightens where the start disagrees."""
        episode = make_episode(
            category="bed", start=Xyz(0.0, 0.0, 0.0), goals=[make_goal(Xyz(0.0, 0.0, -3.0))]
        )
        placement = place_anomaly_source(episode, table(toilet=[Xyz(0.0, 0.0, -8.0)]))
        self.assertEqual(placement.anomaly_object, "toilet")

    def test_a_step_rather_than_a_storey_still_qualifies(self):
        """``max_dy_m`` is a floor rule, not a flatness rule — 0.4 m is a threshold."""
        episode = make_episode(
            category="bed", start=Xyz(0.0, 0.4, 0.0), goals=[make_goal(Xyz(0.0, 0.0, -3.0))]
        )
        placement = place_anomaly_source(episode, table(toilet=[Xyz(0.0, 0.0, -8.0)]))
        self.assertEqual(placement.anomaly_object, "toilet")

    def test_a_cross_floor_episode_is_skipped_rather_than_built(self):
        """The consequence at the build layer: recorded attrition, not a silent null.

        A same-floor episode rides along so the build does not raise for having produced
        nothing at all, which is a different failure with a different message.
        """
        upstairs = make_episode(
            episode_id="cross",
            category="bed",
            start=Xyz(0.0, 2.064, 0.0),
            goals=[make_goal(Xyz(0.0, -0.536, -3.0))],
        )
        flat = make_episode(
            episode_id="flat",
            category="toilet",
            start=Xyz(0.0, -0.536, -20.0),
            goals=[make_goal(Xyz(0.0, -0.536, -20.0))],
        )
        build = build_anomaly_episodes(
            dataset([upstairs, flat]), anomaly_class="alarm", t_anom=30
        )
        # The label leads with the candidate's build position: HM3D authors
        # `episode_id` as "0" on every episode, and matrix-1 wrote 65 skip rows for
        # one scene that were indistinguishable for exactly that reason.
        self.assertIn("cand0000 id=cross", [episode_id for episode_id, _ in build.skipped])
        self.assertNotIn(
            "cross", [built.episode.episode_id for built in build.episodes]
        )


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

    def test_the_placement_records_both_heights_and_the_rule_now_bounds_them(self):
        """**Supersedes "the height the ADR does not constrain".**

        This test used to construct a start a storey below the goal, place a source at the
        goal's level, and assert that the placement *succeeded* while recording a 3.4 m
        start-to-source drop — the position being that the number was worth recording but
        not worth enforcing, since "only the recorded number would say so afterwards".

        The box says recording was not enough. ``height_difference_to_start_m`` never
        reached the run's metrics (the audit surfaced ``source_dy_m``, which is the
        *anchor* difference and read 0.000), so nothing flagged the smoke's episode 0 as
        the 2.6 m stair-climb it was, and it ran as a silent null: 120 steps of clean
        climbing on the wrong floor, criterion 5 unreachable by construction.

        So the rule now constrains it, and the field stays — bounded rather than
        unbounded, which is the new fact worth asserting.
        """
        episode = make_episode(
            start=Xyz(0.0, -0.4, 0.0), goals=[make_goal(Xyz(0.0, 0.0, 0.0))]
        )
        placement = place_anomaly_source(episode, table(sofa=[Xyz(5.0, 0.4, 0.0)]))
        self.assertAlmostEqual(placement.height_difference_m, 0.4)
        self.assertAlmostEqual(placement.height_difference_to_start_m, 0.8)

    def test_no_qualifying_placement_can_exceed_the_rule_from_either_anchor(self):
        """What the two-anchor rule buys, as the property rather than a case.

        Tighter than it first looks, and tighter than I first asserted: constraining the
        source against *both* anchors bounds the start-to-source drop by ``max_dy_m``
        itself, not by twice it. A source 0.9 m above an anchor with the start 0.9 m below
        it is 1.8 m from the start and is correctly rejected — the slack does not compose.
        """
        episode = make_episode(
            start=Xyz(0.0, -0.4, 0.0), goals=[make_goal(Xyz(0.0, 0.0, 0.0))]
        )
        placement = place_anomaly_source(episode, table(sofa=[Xyz(5.0, 0.4, 0.0)]))
        self.assertLessEqual(abs(placement.height_difference_m), 1.0)
        self.assertLessEqual(abs(placement.height_difference_to_start_m), 1.0)

    def test_the_slack_does_not_compose_across_the_two_anchors(self):
        """The rejected half of the property above, so it is checked rather than argued."""
        episode = make_episode(
            start=Xyz(0.0, -0.9, 0.0), goals=[make_goal(Xyz(0.0, 0.0, 0.0))]
        )
        with self.assertRaises(PlacementError):
            place_anomaly_source(episode, table(sofa=[Xyz(5.0, 0.9, 0.0)]))


class TestTheClassChoosesTheObject(unittest.TestCase):
    """Rule 4's first key, and the change to the task that it is.

    Before 2026-09-02 this module read nothing about the sound class: the source went to
    whatever object cleared the separation rules, so every episode this repo ran -- `abl-1`
    included -- placed an alarm wherever the geometry put it. A semantic memory that learns
    "an alarm is heard at a bed" has nothing to predict in a world like that, which is why
    ADR-0018's heard axis could not have measured anything.

    It is a PREFERENCE, not a filter, and both arms are here: the anchor wins when it
    qualifies, and when nothing of that category qualifies the ranking falls through to
    exactly the old behaviour and the record says so.
    """

    def test_the_anchor_category_beats_a_nearer_instance_of_another_category(self):
        episode = make_episode(category="chair", goals=[make_goal(Xyz(0.0, 0.0, -9.0))])
        placement = place_anomaly_source(
            episode,
            table(sofa=[Xyz(4.0, 0.0, -9.0)], bed=[Xyz(12.0, 0.0, -9.0)]),
            anchor_category="bed",
        )
        self.assertEqual(placement.anomaly_object, "bed")
        self.assertTrue(placement.at_class_anchor)

    def test_the_anchor_beats_the_decoupling_preference_too(self):
        """The anchor outranks "a different category from the primary goal".

        The decoupling preference is a soft tiebreak; the class rule is what the memory
        learns. `same_category` is still recorded, so an episode that paid for it is
        visible rather than merely allowed.
        """
        episode = make_episode(category="bed", goals=[make_goal(Xyz(0.0, 0.0, -9.0))])
        placement = place_anomaly_source(
            episode,
            table(bed=[Xyz(0.0, 0.0, -9.0), Xyz(6.0, 0.0, -9.0)], sofa=[Xyz(8.0, 0.0, -9.0)]),
            anchor_category="bed",
        )
        self.assertEqual(placement.anomaly_object, "bed")
        self.assertTrue(placement.at_class_anchor)
        self.assertTrue(placement.same_category)

    def test_among_the_anchors_the_nearest_qualifier_still_wins(self):
        episode = make_episode(goals=[make_goal(Xyz(0.0, 0.0, 0.0))])
        placement = place_anomaly_source(
            episode,
            table(bed=[Xyz(20.0, 0.0, 0.0), Xyz(4.0, 0.0, 0.0)]),
            anchor_category="bed",
        )
        self.assertEqual(placement.position, Xyz(4.0, 0.0, 0.0))

    def test_a_scene_with_no_qualifying_anchor_falls_back_and_records_it(self):
        """The forced-failure arm. Yield cannot drop, and the record says why.

        An episode whose source is NOT at the class's anchor is one the memory prior could
        not have got right. Splitting the readout on `at_class_anchor` is the difference
        between "the memory was wrong" and "this episode did not follow the rule".
        """
        episode = make_episode(category="chair", goals=[make_goal(Xyz(0.0, 0.0, -9.0))])
        placement = place_anomaly_source(
            episode, table(sofa=[Xyz(4.0, 0.0, -9.0)]), anchor_category="bed"
        )
        self.assertEqual(placement.anomaly_object, "sofa")
        self.assertFalse(placement.at_class_anchor)

    def test_an_anchor_that_fails_the_separation_rule_is_not_rescued_by_being_the_anchor(self):
        """The preference reorders SURVIVORS. It does not readmit a rejected candidate.

        A bed 0.5 m from the goal is still too near, and an anchor rule that overrode
        ADR-0010's geometry would place sources on top of the thing the agent is finding.
        """
        episode = make_episode(category="chair", goals=[make_goal(Xyz(0.0, 0.0, -9.0))])
        placement = place_anomaly_source(
            episode,
            table(bed=[Xyz(0.5, 0.0, -9.0)], sofa=[Xyz(8.0, 0.0, -9.0)]),
            anchor_category="bed",
        )
        self.assertEqual(placement.anomaly_object, "sofa")
        self.assertFalse(placement.at_class_anchor)

    def test_no_anchor_category_reproduces_the_old_ordering_exactly(self):
        """Every caller that does not know its class gets the pre-2026-09-02 behaviour."""
        episode = make_episode(category="chair", goals=[make_goal(Xyz(0.0, 0.0, -9.0))])
        args = (episode, table(sofa=[Xyz(4.0, 0.0, -9.0)], bed=[Xyz(12.0, 0.0, -9.0)]))
        old = place_anomaly_source(*args)
        self.assertEqual(old.anomaly_object, "sofa")
        self.assertFalse(old.at_class_anchor)
        # And a class whose anchor is absent from the scene behaves identically.
        absent = place_anomaly_source(*args, anchor_category="toilet")
        self.assertEqual(absent.position, old.position)
        self.assertFalse(absent.at_class_anchor)

    def test_the_flag_reaches_the_serialized_record(self):
        episode = make_episode(goals=[make_goal(Xyz(0.0, 0.0, 0.0))])
        placement = place_anomaly_source(
            episode, table(bed=[Xyz(4.0, 0.0, 0.0)]), anchor_category="bed"
        )
        self.assertIs(placement.as_dict()["at_class_anchor"], True)


class TestTheSourceIsKeptOffTheAgentToo(unittest.TestCase):
    """The mirror of the 3 m goal keep-out, and it was missing until `detour-1`.

    That run's episode 18 placed a source 0.75 m from the agent's start — inside the
    arrival radius before the anomaly had sounded — and it counted as a completed
    anomaly-response loop: INVESTIGATE at step 5, RESUME at step 7. One of eight
    successes was a source at the agent's feet, so 8/20 read honestly is 7/20.

    Both arms, per ADR-0014: the near candidate is REJECTED and the far one is still
    taken. A rule only ever seen accepting is a rule that has never run.
    """

    @staticmethod
    def _episode(start=Xyz(0.0, 0.0, 0.0)):
        """Primary goal 10 m out along +x, so the goal keep-out cannot be what fires."""
        return make_episode(category="chair", start=start,
                            goals=[make_goal(Xyz(10.0, 0.0, 0.0), category="chair")])

    def test_a_source_at_the_agents_feet_is_rejected(self):
        episode = self._episode()
        near = table(chair=[Xyz(10.0, 0.0, 0.0)], sofa=[Xyz(1.0, 0.0, 0.0)])
        with self.assertRaises(PlacementError) as caught:
            place_anomaly_source(episode, near)
        self.assertIn("at the start", str(caught.exception))
        self.assertIn("1 at the start", str(caught.exception))

    def test_a_source_beyond_the_bar_is_still_taken(self):
        """The healthy arm: same geometry, the candidate moved past the bar."""
        episode = self._episode()
        far = table(chair=[Xyz(10.0, 0.0, 0.0)], sofa=[Xyz(0.0, 0.0, 5.0)])
        placement = place_anomaly_source(episode, far)
        self.assertEqual(placement.position, Xyz(0.0, 0.0, 5.0))

    def test_the_near_candidate_loses_to_the_far_one_rather_than_failing_the_episode(self):
        """Nearest-first would otherwise actively prefer the degenerate candidate."""
        episode = self._episode()
        both = table(chair=[Xyz(10.0, 0.0, 0.0)],
                     sofa=[Xyz(1.0, 0.0, 0.0), Xyz(0.0, 0.0, 5.0)])
        self.assertEqual(place_anomaly_source(episode, both).position, Xyz(0.0, 0.0, 5.0))

    def test_the_bar_is_measured_from_the_start_not_from_the_goal(self):
        """Move the START and the same candidate flips from legal to degenerate, with
        the goal untouched — which is what makes this a different rule from `too_near`."""
        candidate = table(chair=[Xyz(10.0, 0.0, 0.0)], sofa=[Xyz(0.0, 0.0, 5.0)])
        self.assertTrue(place_anomaly_source(self._episode(), candidate))
        moved = self._episode(start=Xyz(0.0, 0.0, 4.0))  # now 1 m from the candidate
        with self.assertRaises(PlacementError) as caught:
            place_anomaly_source(moved, candidate)
        self.assertIn("1 at the start", str(caught.exception))

    def test_the_bar_is_configurable_and_zero_restores_the_old_behaviour(self):
        """Every yield measured before this rule is an overestimate, and reproducing one
        needs the rule off rather than a correction applied to the number."""
        episode = self._episode()
        near = table(chair=[Xyz(10.0, 0.0, 0.0)], sofa=[Xyz(1.0, 0.0, 0.0)])
        placement = place_anomaly_source(episode, near, min_start_sep_m=0.0)
        self.assertEqual(placement.position, Xyz(1.0, 0.0, 0.0))

    def test_it_is_counted_apart_from_too_near(self):
        """On top of the goal and on top of the agent are different degeneracies; a
        report that pooled them would name the wrong rule to revisit."""
        episode = self._episode()
        mixed = table(chair=[Xyz(10.0, 0.0, 0.0)],   # the goal itself: 0 m from itself
                      sofa=[Xyz(9.0, 0.0, 0.0)],      # 1 m from the goal
                      bed=[Xyz(1.0, 0.0, 0.0)])       # 1 m from the start
        with self.assertRaises(PlacementError) as caught:
            place_anomaly_source(episode, mixed)
        message = str(caught.exception)
        self.assertIn("2 too near", message)
        self.assertIn("1 at the start", message)

    def test_the_yield_report_can_read_the_new_rule_back_out(self):
        """The skip reason is prose that `yield_report` parses; a rule it cannot parse
        shows up as `unattributed` and the per-rule totals under-count by that much."""
        from earshot.tools.yield_report import aggregate

        episode = self._episode()
        near = table(chair=[Xyz(10.0, 0.0, 0.0)], sofa=[Xyz(1.0, 0.0, 0.0)])
        try:
            place_anomaly_source(episode, near)
        except PlacementError as exc:
            reason = str(exc)
        agg = aggregate([{"scene": "FAKE", "n_episodes": 0,
                          "skipped": [{"episode_id": "0", "reason": reason}]}])
        self.assertEqual(agg["rules"]["at_the_start"], 1)
        self.assertEqual(agg["unattributed_skips"], 0)


class TestTheConfigAndTheBuilderAgree(unittest.TestCase):
    """ADR-0013 puts `config` at ("audio.config", "agent.config", "types"), so it cannot
    import `task.dataset` and the builder's three numbers are spelled twice. That is the
    drift trap ticket 24 named; this is the mechanism `test_report_artifacts` already
    uses for the notifier's copy of `summary.json`, applied to all three."""

    def test_every_placement_default_matches_the_builder(self):
        import inspect

        from earshot.config import RunConfig

        signature = inspect.signature(place_anomaly_source)
        cfg = RunConfig(run_dir="x")
        for config_field, parameter in (("min_source_sep_m", "min_sep_m"),
                                        ("max_source_dy_m", "max_dy_m"),
                                        ("min_source_start_sep_m", "min_start_sep_m")):
            self.assertEqual(
                getattr(cfg, config_field), signature.parameters[parameter].default,
                "RunConfig.{} and place_anomaly_source({}=) have drifted".format(
                    config_field, parameter))


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
        self.assertEqual(
            [episode_id for episode_id, _ in build.skipped], ["cand0001 id=stranded"]
        )
        self.assertIn("stranded", build.summary())

    def test_a_scene_that_builds_nothing_raises(self):
        """A build that produced nothing and said so only in a list is a run that starts
        and immediately does nothing."""
        scene = dataset([make_episode(goals=[make_goal(Xyz(0.0, 0.0, 0.0))])])
        with self.assertRaises(PlacementError):
            build_anomaly_episodes(scene, anomaly_class="alarm", t_anom=30)

    def test_the_empty_build_is_carried_on_the_raise_so_it_can_be_written_down(self):
        """A 0% yield is the most informative point a denominator has, and yield-1 lost
        it: `mL8ThkuaVTM` placed none of 99 candidates and left no record, so the yield
        report aggregated the scenes that yielded something and called it the yield of
        all of them. The message formats five reasons; the caller needs all of them."""
        from earshot.task.dataset import EmptyDatasetError

        goals = [make_goal(Xyz(0.0, 0.0, 0.0))]
        scene = dataset([make_episode(episode_id=str(i), category="chair", goals=goals)
                         for i in range(9)])
        with self.assertRaises(EmptyDatasetError) as caught:
            build_anomaly_episodes(scene, anomaly_class="alarm", t_anom=30)
        error = caught.exception
        self.assertEqual(error.scene_label, scene.scene_label)
        self.assertEqual(error.build.episodes, ())
        self.assertEqual(len(error.build.skipped), 9,
                         "all nine, not the five the message formats")
        self.assertTrue(all(why for _, why in error.build.skipped))

    def test_it_is_still_a_placement_error_so_existing_handlers_hold(self):
        from earshot.task.dataset import EmptyDatasetError

        self.assertTrue(issubclass(EmptyDatasetError, PlacementError))

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


class TestDeriveTAnom(unittest.TestCase):
    """When the anomaly starts, derived from the episode rather than fixed for the scene.

    The bug this replaces was invisible in every artefact the run produced. ``t_anom``
    was 30 and the smoke's second box episode reached its bed at step 30, so the source
    started sounding on the last step of the episode: the onset fired, the funnel read
    ONSET_FIRED, and criterion 5 failed with nothing anywhere saying the interrupt had
    arrived too late to be one. Every number in the record was correct.
    """

    def _episode(self, reach_m, **kwargs):
        """One episode whose nearest goal view point is ``reach_m`` from the start."""
        return make_episode(
            start=Xyz(0.0, 0.0, 0.0), goals=[make_goal(Xyz(0.0, 0.0, -reach_m))], **kwargs
        )

    def test_the_anomaly_starts_before_the_find_can_possibly_end(self):
        """The property the derivation exists for, over the range HM3D episodes span.

        ``earliest_end`` is not an estimate: the agent cannot be within the oracle radius
        of the goal before it has walked the rest of the straight line, and no step covers
        more than one forward stride. So a find cannot end earlier than this, and the
        onset must land strictly inside it.
        """
        for reach_m in (3.0, 5.0, 7.5, 12.0, 25.0, 40.0):
            with self.subTest(reach_m=reach_m):
                earliest_end = (reach_m - ARRIVAL_RADIUS_M) / FORWARD_STEP_M
                self.assertLess(derive_t_anom(self._episode(reach_m)), earliest_end)

    def test_it_is_never_zero_so_the_pre_onset_invariant_has_readings(self):
        """§3.1's first invariant is checked per pre-onset step, so ``t_anom = 0`` leaves
        it unexercised — and ``assert_provenance`` raises rather than passing quietly."""
        for reach_m in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            with self.subTest(reach_m=reach_m):
                self.assertGreaterEqual(derive_t_anom(self._episode(reach_m)), 1)

    def test_a_goal_within_arms_reach_falls_back_to_the_floor(self):
        """The one case where the guarantee above does not hold, stated rather than hidden.

        A goal the agent is already standing on has no search to interrupt. The floor wins,
        the find can end before the source sounds, and §2.5 says that shows up as a funnel
        stage rather than as a screened-out episode.
        """
        self.assertEqual(derive_t_anom(self._episode(0.5)), T_ANOM_FLOOR_STEPS)

    def test_it_measures_to_the_nearest_instance_not_the_first(self):
        """A category has several instances and the agent succeeds at any of them, so the
        find is as long as the *shortest* route to one — the same argument that put every
        view point in the separation bar rather than one."""
        far_first = make_episode(
            start=Xyz(0.0, 0.0, 0.0),
            goals=[make_goal(Xyz(0.0, 0.0, -30.0)), make_goal(Xyz(0.0, 0.0, -6.0))],
        )
        self.assertEqual(derive_t_anom(far_first), derive_t_anom(self._episode(6.0)))

    def test_a_longer_find_moves_the_onset_later(self):
        """It tracks the episode. A constant could not, which is the whole point."""
        derived = [derive_t_anom(self._episode(reach)) for reach in (4.0, 9.0, 20.0)]
        self.assertEqual(derived, sorted(derived))
        self.assertLess(derived[0], derived[-1])

    def test_an_episode_with_no_view_points_raises(self):
        """Rather than deriving 0 from a missing distance, which would read as "the source
        sounds from step 0" — the `anommxv` break this map invalidated a matrix over."""
        bare = make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0), view_points=[])])
        with self.assertRaises(PlacementError):
            derive_t_anom(bare)

    def test_the_builder_derives_one_per_episode_when_none_is_pinned(self):
        scene = dataset(
            [
                self._episode(6.0, episode_id="near", category="chair"),
                make_episode(
                    episode_id="far",
                    category="sofa",
                    goals=[make_goal(Xyz(0.0, 0.0, -24.0))],
                ),
            ]
        )
        build = build_anomaly_episodes(scene, anomaly_class="alarm")
        derived = {e.episode.episode_id: e.t_anom for e in build.episodes}
        self.assertEqual(len(derived), 2)
        self.assertLess(derived["near"], derived["far"])

    def test_a_pinned_t_anom_is_used_verbatim(self):
        """An experiment holding the onset fixed across episodes is what the flag is for,
        and a derivation that quietly overrode it would make the flag a lie."""
        scene = dataset(
            [
                self._episode(6.0, episode_id="near", category="chair"),
                make_episode(
                    episode_id="far",
                    category="sofa",
                    goals=[make_goal(Xyz(0.0, 0.0, -24.0))],
                ),
            ]
        )
        build = build_anomaly_episodes(scene, anomaly_class="alarm", t_anom=30)
        self.assertEqual([e.t_anom for e in build.episodes], [30, 30])


class TestTheDerivationsConstantsMatchTheirSources(unittest.TestCase):
    """``FORWARD_STEP_M`` and ``ARRIVAL_RADIUS_M`` are copies, so they can drift.

    They are copies because they have to be: this module may not import ``sim`` (the test
    below this one holds that), and ``sim/world.py`` imports torch at module scope, so a
    Mac cannot load it to ask. Read the defaults out of their own source with ``ast``
    instead — the same answer ``test_box_call_arity.py`` reached for the same reason.

    A drift here is silent and one-directional: a larger real step size or a smaller real
    radius makes the derived ``t_anom`` too late again, which is the bug it replaced.
    """

    def _default(self, relative, class_name, func_name, argument):
        import ast

        import _tree

        tree = _tree.parse(_tree.PACKAGE_ROOT / relative)
        scope = tree
        if class_name is not None:
            scope = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
        if func_name is None:
            target = next(
                node.value
                for node in scope.body
                if isinstance(node, ast.AnnAssign)
                and getattr(node.target, "id", None) == argument
            )
            return ast.literal_eval(target)
        function = next(
            node
            for node in ast.walk(scope)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        )
        args = function.args
        names = [arg.arg for arg in args.args] + [arg.arg for arg in args.kwonlyargs]
        defaults = list(args.defaults) + list(args.kw_defaults)
        return ast.literal_eval(defaults[names.index(argument) - (len(names) - len(defaults))])

    def test_the_forward_step_matches_the_action_spec(self):
        self.assertEqual(
            FORWARD_STEP_M,
            self._default("sim/world.py", None, "__init__", "step_size_m"),
            "the simulator's forward stride changed; a longer one means the derived "
            "t_anom is no longer inside the find",
        )

    def test_the_arrival_radius_matches_the_detector(self):
        self.assertEqual(
            ARRIVAL_RADIUS_M,
            self._default("agent/config.py", "DetectorConfig", None, "oracle_radius_m"),
            "the oracle STOP radius changed; the derivation subtracts it as the part of "
            "the route that is never walked",
        )


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
