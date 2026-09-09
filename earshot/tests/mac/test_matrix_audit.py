"""`tools/matrix_audit.py`: the review's four read-only questions, and the coverage gate.

Everything here is the pure half — synthetic provenance blocks and synthetic audit rows,
because the loaders are thin wrappers over `report.artifacts` and the analysis must be
decidable without a finished sweep on disk. The gate is exercised through its real CLI
(`main`) against a real `store.json` written by `dump_stores`, in both arms (ADR-0014):
the healthy pass, and the red exit a silently-dropped scene must produce.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from _interpreter import assert_interpreter  # noqa: F401

from earshot.memory.store import EpisodicEntry, EpisodicStore, SemanticStore
from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_episode
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.task.memory_build import dump_stores
from earshot.tools.matrix_audit import (
    _print_coverage,
    abstain_table,
    anchors_by_scene,
    discordance_where_identical,
    gate_missing,
    graded_confidences,
    load_assignment,
    main,
    prior_distribution,
    scene_of,
    seen_axis_divergence,
    store_coverage,
)
from earshot.types import Xyz


def _row(
    reached=False,
    source=(1.0, 2.0, 3.0),
    category=None,
    miss=None,
    instances=None,
    distance=None,
    confidence=None,
):
    return {
        "reached": reached,
        "source": source,
        "condition": "heard_seen",
        "category": category,
        "miss": miss,
        "instances": instances,
        "distance_m": distance,
        "confidence": confidence,
    }


class TestStoreCoverage(unittest.TestCase):
    def test_a_pass_provenance_store_accounts_for_every_scene(self):
        provenance = {
            "scenes_requested": ["A", "B", "C"],
            "scenes_complete": ["A"],
            "scenes_incomplete": [{"scene": "B", "complete": False}],
            "scenes_failed": [{"scene": "C", "error": "no mesh"}],
        }
        coverage = store_coverage(provenance, {"A": 4})
        self.assertEqual(coverage["unaccounted"], [])
        self.assertTrue(coverage["records_incomplete"])
        self.assertEqual(coverage["incomplete"], ["B"])
        self.assertEqual(coverage["failed"], ["C"])

    def test_a_legacy_store_reports_its_silent_incompletes_as_unaccounted(self):
        """matrix-1's own store shape: no `scenes_incomplete` key, so a loaded scene
        whose tour left a leg unreached is in NO list. That is the D3 silent case, and
        `unaccounted` is where it must surface."""
        provenance = {
            "scenes_requested": ["A", "B"],
            "scenes_complete": ["A"],
            "scenes_failed": [],
        }
        coverage = store_coverage(provenance, {"A": 4})
        self.assertFalse(coverage["records_incomplete"])
        self.assertEqual(coverage["unaccounted"], ["B"])

    def test_the_recorded_reason_travels_with_the_scene_name(self):
        """`scenes_incomplete` exists so a scene that did not tour says why. A coverage
        report holding only the name sends the reader to the log for a fact the store
        already has -- which is what `prior-2` cost."""
        provenance = {
            "scenes_requested": ["A", "B"],
            "scenes_complete": [],
            "scenes_incomplete": [
                {"scene": "A", "ok": True, "complete": False,
                 "rooms_reached": ["bathroom"], "n_observations": 1, "error": None},
            ],
            "scenes_failed": [{"scene": "B", "ok": False, "error": "no mesh"}],
        }
        coverage = store_coverage(provenance, {"A": 1})
        self.assertEqual(coverage["detail"]["A"]["rooms_reached"], ["bathroom"])
        self.assertEqual(coverage["detail"]["B"]["error"], "no mesh")

    def test_a_zero_leg_tour_is_told_apart_from_a_partial_one(self):
        """The two ways `prior-2` lost a scene, which read identically before this: a
        partial tour (`p53SfW6mjZe`, 2 of 3) against a tour that planned nothing
        (`qyAac8rV8Zk`, 0 of 0). And within the second, no candidates at all against
        candidates the navmesh refused."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_coverage(
                store_coverage(
                    {
                        "scenes_requested": ["partial", "islanded", "barren"],
                        "scenes_complete": [],
                        "scenes_incomplete": [
                            {"scene": "partial", "rooms_reached": ["bathroom", "bedroom"],
                             "n_observations": 2, "unreachable": [], "error": None},
                            {"scene": "islanded", "rooms_reached": [],
                             "n_observations": 0, "error": None,
                             "unreachable": [{"room": "bathroom", "category": "toilet",
                                              "reason": "no route"}]},
                            {"scene": "barren", "rooms_reached": [],
                             "n_observations": 0, "unreachable": [], "error": None},
                        ],
                        "scenes_failed": [],
                    },
                    {},
                ),
                print,
            )
        out = buffer.getvalue()
        self.assertIn("rooms reached: bathroom, bedroom", out)
        self.assertIn("1 candidate(s) unroutable (bathroom)", out)
        self.assertIn("no candidate stop offered", out)

    def test_a_complete_tour_that_dropped_a_room_says_which_room(self):
        """`prior-7`: the floor test took a room out of two scenes, both reached every
        leg they still had, and both went green. A report that prints only failures said
        nothing at all -- so the one fact distinguishing a scene with two rooms from a
        scene whose third was dropped never reached disk."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_coverage(
                store_coverage(
                    {
                        "scenes_requested": ["shortened"],
                        "scenes_complete": ["shortened"],
                        "scenes_incomplete": [],
                        "scenes_failed": [],
                        "scenes_toured": [
                            {"scene": "shortened", "ok": True, "complete": True,
                             "rooms_reached": ["bathroom", "living room"],
                             "n_observations": 2, "abandoned": [], "error": None,
                             "unreachable": [
                                 {"room": "bedroom", "category": "bed",
                                  "reason": "on another floor (2.24 m of height "
                                            "from the start)"},
                             ]},
                        ],
                    },
                    {"shortened": 2},
                ),
                print,
            )
        out = buffer.getvalue()
        self.assertIn("COMPLETE, SHORT A ROOM: shortened", out)
        self.assertIn("1 candidate(s) unroutable (bedroom)", out)
        self.assertIn("on another floor", out)

    def test_a_complete_tour_that_dropped_nothing_is_not_reported(self):
        """The other arm, and the reason the line is conditional: a tour that kept every
        candidate the scene offered must print no drop line, or the section cries wolf on
        every green scene in a 19-scene pass and the real one stops being visible."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_coverage(
                store_coverage(
                    {
                        "scenes_requested": ["whole"],
                        "scenes_complete": ["whole"],
                        "scenes_incomplete": [],
                        "scenes_failed": [],
                        "scenes_toured": [
                            {"scene": "whole", "ok": True, "complete": True,
                             "rooms_reached": ["bathroom"], "n_observations": 1,
                             "unreachable": [], "abandoned": [], "error": None},
                        ],
                    },
                    {"whole": 1},
                ),
                print,
            )
        self.assertNotIn("SHORT A ROOM", buffer.getvalue())

    def test_a_store_without_scenes_toured_still_reads_its_failures(self):
        """`scenes_toured` is additive: every store written before it keeps reporting
        exactly what it did, and claims no drop it cannot see."""
        coverage = store_coverage(
            {
                "scenes_requested": ["A", "B"],
                "scenes_complete": ["A"],
                "scenes_incomplete": [{"scene": "B", "rooms_reached": ["den"]}],
                "scenes_failed": [],
            },
            {"A": 3},
        )
        self.assertEqual(coverage["complete_with_drops"], [])
        self.assertEqual(coverage["detail"]["B"]["rooms_reached"], ["den"])

    def test_an_abandoned_leg_says_where_it_stalled_and_why(self):
        """`prior-5` moved both scenes into the same bucket -- a leg that WAS planned and
        that the follower never arrived at -- and the store could say it was missed but
        not why. The gap tells the faults apart: stalled just outside the goal radius is
        an arrival-threshold problem, stalled far away is a navigation failure."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_coverage(
                store_coverage(
                    {
                        "scenes_requested": ["S"],
                        "scenes_complete": [],
                        "scenes_incomplete": [{
                            "scene": "S", "rooms_reached": ["bathroom"],
                            "n_observations": 1, "unreachable": [], "error": None,
                            "abandoned": [{
                                "room": "bedroom", "category": "bed",
                                "final_gap_m": 1.42, "steps": 200,
                                "reason": "budget of 200 steps exhausted",
                            }],
                        }],
                        "scenes_failed": [],
                    },
                    {},
                ),
                print,
            )
        out = buffer.getvalue()
        self.assertIn("bedroom PLANNED but not reached", out)
        self.assertIn("stalled 1.42 m out", out)
        self.assertIn("budget of 200 steps exhausted", out)

    def test_an_abandoned_leg_with_no_gap_says_so_rather_than_printing_a_zero(self):
        """`final_gap_m` is None when the navmesh could not measure it, which is not the
        same fact as arriving -- and a 0.00 there would read as a perfect arrival."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_coverage(
                store_coverage(
                    {
                        "scenes_requested": ["S"],
                        "scenes_complete": [],
                        "scenes_incomplete": [{
                            "scene": "S", "rooms_reached": [], "n_observations": 0,
                            "unreachable": [], "error": None,
                            "abandoned": [{
                                "room": "bedroom", "category": "bed",
                                "final_gap_m": None, "steps": 3,
                                "reason": "follower refused the target: NoRoute",
                            }],
                        }],
                        "scenes_failed": [],
                    },
                    {},
                ),
                print,
            )
        out = buffer.getvalue()
        self.assertIn("gap unknown", out)
        self.assertNotIn("0.00 m out", out)

    def test_a_legacy_store_has_no_detail_and_fabricates_none(self):
        coverage = store_coverage(
            {"scenes_requested": ["A"], "scenes_complete": [], "scenes_failed": []},
            {},
        )
        self.assertEqual(coverage["detail"], {})

    def test_a_complete_scene_with_no_rows_is_a_named_disagreement(self):
        provenance = {
            "scenes_requested": ["A"],
            "scenes_complete": ["A"],
            "scenes_incomplete": [],
            "scenes_failed": [],
        }
        coverage = store_coverage(provenance, {})
        self.assertEqual(coverage["complete_without_rows"], ["A"])

    def test_scene_of_reads_both_list_shapes(self):
        self.assertEqual(scene_of("A"), "A")
        self.assertEqual(scene_of({"scene": "B", "error": "x"}), "B")


class TestGateMissing(unittest.TestCase):
    def test_full_coverage_is_green(self):
        coverage = store_coverage(
            {
                "scenes_requested": ["A", "B"],
                "scenes_complete": ["A", "B"],
                "scenes_incomplete": [],
                "scenes_failed": [],
            },
            {"A": 1, "B": 1},
        )
        self.assertEqual(gate_missing(coverage), [])

    def test_every_kind_of_absence_is_red(self):
        """Incomplete, failed, and silently-unaccounted scenes all block the gate —
        however a scene went missing, the cells must not run over it."""
        coverage = store_coverage(
            {
                "scenes_requested": ["A", "B", "C", "D"],
                "scenes_complete": ["A"],
                "scenes_incomplete": [{"scene": "B"}],
                "scenes_failed": [{"scene": "C"}],
            },
            {"A": 1},
        )
        self.assertEqual(gate_missing(coverage), ["B", "C", "D"])


class TestSeenAxisDivergence(unittest.TestCase):
    def test_instances_differing_at_a_shared_category_is_the_live_signature(self):
        seen = {"S": {0: _row(category="bed", instances=1.0, distance=2.0)}}
        unseen = {"S": {0: _row(category="bed", instances=3.0, distance=2.0)}}
        d = seen_axis_divergence(seen, unseen)
        self.assertEqual(d["pairs"], 1)
        self.assertEqual(d["same_category"], 1)
        self.assertEqual(d["instances_differ_same_category"], 1)
        self.assertEqual(d["instances_differ_scenes"], ["S"])
        self.assertEqual(d["category_differs"], 0)

    def test_a_category_flip_is_counted_apart_not_attributed(self):
        seen = {"S": {0: _row(category="bed", instances=1.0)}}
        unseen = {"S": {0: _row(category="chair", instances=1.0)}}
        d = seen_axis_divergence(seen, unseen)
        self.assertEqual(d["category_differs"], 1)
        self.assertEqual(d["same_category"], 0)
        self.assertEqual(d["instances_differ_same_category"], 0)

    def test_a_source_mismatch_is_not_a_pair(self):
        """`episode_diff`'s own discipline: two audits that disagree on where the sound
        was are two different tasks, and comparing their priors would be meaningless."""
        seen = {"S": {0: _row(source=(0.0, 0.0, 0.0), category="bed")}}
        unseen = {"S": {0: _row(source=(9.0, 9.0, 9.0), category="bed")}}
        d = seen_axis_divergence(seen, unseen)
        self.assertEqual(d["pairs"], 0)


class TestPriorDistribution(unittest.TestCase):
    def test_categories_misses_and_never_consulted_are_three_separate_tallies(self):
        arm = {
            "S": {
                0: _row(category="bed"),
                1: _row(category="bed"),
                2: _row(miss="no_prediction"),
                3: _row(),
            }
        }
        dist = prior_distribution(arm)
        self.assertEqual(dist["episodes"], 4)
        self.assertEqual(dist["categories"], {"bed": 2})
        self.assertEqual(dist["misses"], {"no_prediction": 1})
        self.assertEqual(dist["never_consulted"], 1)


class TestDiscordanceWhereIdentical(unittest.TestCase):
    def test_zero_row_scenes_measure_the_apparatus_and_others_do_not(self):
        seen = {
            "empty": {0: _row(reached=True), 1: _row(reached=False)},
            "toured": {0: _row(reached=True)},
        }
        unseen = {
            "empty": {0: _row(reached=False), 1: _row(reached=False)},
            "toured": {0: _row(reached=False)},
        }
        split = discordance_where_identical(seen, unseen, {"toured": 5})
        self.assertEqual(split["zero_row_scenes"]["pairs"], 2)
        self.assertEqual(split["zero_row_scenes"]["discordant"], 1)
        self.assertEqual(split["zero_row_scenes"]["scenes"], {"empty": 1})
        self.assertEqual(split["scenes_with_rows"]["pairs"], 1)
        self.assertEqual(split["scenes_with_rows"]["discordant"], 1)


class TestTheGradingKey(unittest.TestCase):
    def test_the_assignment_tsv_is_read_back_as_scene_to_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.tsv"
            path.write_text("sceneA\ttoilet_flush\n\nsceneB\tsnoring\n", encoding="utf-8")
            self.assertEqual(
                load_assignment(str(path)),
                {"sceneA": "toilet_flush", "sceneB": "snoring"},
            )

    def test_a_malformed_line_raises_rather_than_being_skipped(self):
        """A grading key that silently drops a scene grades that scene against nothing,
        and every episode in it would fall into `no_confidence` looking innocent."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignment.tsv"
            path.write_text("sceneA\ttoilet_flush\nsceneB\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_assignment(str(path))

    def test_the_anchor_is_the_category_a_correct_recall_names(self):
        anchors = anchors_by_scene(
            {"A": "toilet_flush", "B": "snoring", "C": "keyboard_typing"}
        )
        self.assertEqual(anchors, {"A": "toilet", "B": "bed", "C": "chair"})

    def test_a_class_that_anchors_nowhere_has_no_right_answer(self):
        self.assertEqual(anchors_by_scene({"A": "glass_break"}), {"A": None})


class TestGradedConfidences(unittest.TestCase):
    def test_the_voted_category_is_graded_against_the_scenes_own_class(self):
        arm = {
            "A": {
                0: _row(category="toilet", confidence=0.9),
                1: _row(category="bed", confidence=0.4),
            }
        }
        graded = graded_confidences(arm, {"A": "toilet"})
        self.assertEqual(graded["correct"], [0.9])
        self.assertEqual(graded["wrong"], [0.4])

    def test_a_miss_carries_no_confidence_and_is_counted_not_dropped(self):
        """The section's blind spot, and the reason it is printed: `unreachable` is a
        real recall the audit records no score for, so no threshold can be tried on it."""
        arm = {"A": {0: _row(miss="unreachable"), 1: _row()}}
        graded = graded_confidences(arm, {"A": "toilet"})
        self.assertEqual(graded["correct"], [])
        self.assertEqual(graded["wrong"], [])
        self.assertEqual(graded["ungraded"]["no_confidence"], 2)

    def test_an_unanchored_class_is_excluded_rather_than_counted_wrong(self):
        arm = {"A": {0: _row(category="chair", confidence=0.7)}}
        graded = graded_confidences(arm, {"A": None})
        self.assertEqual(graded["wrong"], [])
        self.assertEqual(graded["ungraded"]["unknown_anchor"], 1)


class TestAbstainTable(unittest.TestCase):
    def test_a_separable_pair_buys_every_wrong_recall_for_free(self):
        table = abstain_table(correct=[0.8, 0.9], wrong=[0.1, 0.2])
        self.assertTrue(table["separable"])
        self.assertEqual(table["free"]["threshold"], 0.8)
        self.assertEqual(table["free"]["wrong_dropped"], 2)
        self.assertEqual(table["free"]["correct_lost"], 0)
        self.assertAlmostEqual(table["best"]["j"], 1.0)

    def test_an_overlapping_pair_trades_one_for_the_other(self):
        """The forced-failure arm of the same question: interleaved scores, so the free
        floor buys one wrong recall and nothing above it is free."""
        table = abstain_table(correct=[0.2, 0.6, 0.8], wrong=[0.1, 0.5, 0.7])
        self.assertFalse(table["separable"])
        self.assertEqual(table["free"]["threshold"], 0.2)
        self.assertEqual(table["free"]["wrong_dropped"], 1)
        self.assertEqual(table["free"]["correct_lost"], 0)
        self.assertAlmostEqual(table["best"]["j"], 1.0 / 3.0)

    def test_the_tie_break_takes_the_least_aggressive_floor(self):
        """Three thresholds reach J = 1/3 on this pair (0.2, 0.6, 0.8) and the reported
        one must be 0.2, which loses nothing. Computing J in floating point put
        `1 - 2/3` one ulp above `2/3 - 1/3` and handed the tie to 0.8 instead, so the
        tool recommended discarding two correct recalls to buy the same J."""
        table = abstain_table(correct=[0.2, 0.6, 0.8], wrong=[0.1, 0.5, 0.7])
        self.assertEqual(table["best"]["threshold"], 0.2)
        self.assertEqual(table["best"]["correct_lost"], 0)

    def test_one_empty_side_reports_no_floor_rather_than_a_zero(self):
        """Every `not_heard` cell of a three-class bank is wrong by construction, so an
        arm read alone can have an empty correct side. A 0.0 there would read as a
        measured floor."""
        table = abstain_table(correct=[], wrong=[0.3, 0.4])
        self.assertIsNone(table["free"])
        self.assertIsNone(table["best"])
        self.assertIsNone(table["separable"])
        self.assertEqual(table["n_wrong"], 2)


class TestTheCoverageGateCli(unittest.TestCase):
    """The gate through its real CLI against a real `dump_stores` file — the enforcement
    `matrix_sweep.sh` calls between the prior pass and the cells."""

    @staticmethod
    def _store(tmp, complete):
        path = Path(tmp) / "store.json"
        dump_stores(
            str(path),
            SemanticStore(),
            EpisodicStore(
                entries=tuple(
                    EpisodicEntry(
                        scene=scene, room="bedroom", category="bed",
                        point=Xyz(0.0, 0.0, 0.0),
                    )
                    for scene in complete
                )
            ),
            provenance={
                "scenes_requested": list(complete),
                "scenes_complete": list(complete),
                "scenes_incomplete": [],
                "scenes_failed": [],
            },
        )
        return str(path)

    def test_full_coverage_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, complete=["A", "B"])
            self.assertEqual(main(["--store", store, "--gate-scenes", "A B"]), 0)

    def test_a_store_on_its_own_reports_coverage_and_exits_clean(self):
        """`--store` with no run_dir is how a prior pass is read BEFORE any cell has run
        over it, which is the whole point of the gate. It printed section A and then died
        on `Path(None)`, so the tool looked broken right after it had answered."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, complete=["A"])
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = main(["--store", store])
        self.assertEqual(code, 0)
        self.assertIn("A. THE STORE", buffer.getvalue())
        self.assertIn("sections B-E are not run", buffer.getvalue())

    def test_a_missing_scene_exits_two(self):
        """The forced-failure arm: the sweep must stop before the cells, because a cell
        run over an uncovered scene has seen == unseen by construction."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, complete=["A"])
            self.assertEqual(main(["--store", store, "--gate-scenes", "A B"]), 2)


class TestSectionEThroughTheCli(unittest.TestCase):
    """Section E over a real sweep layout, written by the real writer and read back by
    the real loader — the grading key, the confidence off `metrics`, and the verdict
    line. ADR-0014: the capability is exercised, not proxied by calling the pure
    functions with hand-built dicts.
    """

    SCENE = "sceneA"

    @classmethod
    def _sweep(cls, tmp, *, with_assignment):
        root = Path(tmp)
        dump_stores(
            str(root / "prior" / "store.json"),
            SemanticStore(),
            EpisodicStore(entries=(EpisodicEntry(
                scene=cls.SCENE, room="bathroom", category="toilet",
                point=Xyz(0.0, 0.0, 0.0),
            ),)),
            provenance={
                "scenes_requested": [cls.SCENE],
                "scenes_complete": [cls.SCENE],
                "scenes_incomplete": [],
                "scenes_failed": [],
            },
        )
        scene_dir = root / "heard_seen" / cls.SCENE
        # `toilet_flush` anchors at `toilet`, so episode 0 recalled right and episode 1
        # recalled wrong — and the wrong one scored lower, which is the separable case.
        for index, (category, confidence) in enumerate(
            (("toilet", 0.91), ("bed", 0.42))
        ):
            write_episode(
                str(scene_dir), index, AgentReport(),
                EpisodeAudit(
                    episode_index=index,
                    scene_id=cls.SCENE,
                    memory_condition="heard_seen",
                    memory_prior_category=category,
                    source_xyz=Xyz(2.0, 0.1, -4.0),
                    funnel_stage=FunnelStage.SOURCE_REACHED,
                    metrics={"memory_prior_confidence": confidence},
                ),
            )
        # A third episode whose prior missed: no confidence recorded, so section E must
        # count it as a blind spot rather than grade it.
        write_episode(
            str(scene_dir), 2, AgentReport(),
            EpisodeAudit(
                episode_index=2, scene_id=cls.SCENE, memory_condition="heard_seen",
                memory_prior_miss="unreachable", source_xyz=Xyz(2.0, 0.1, -4.0),
                funnel_stage=FunnelStage.SOURCE_REACHED,
            ),
        )
        if with_assignment:
            (root / "assignment.tsv").write_text(
                "{}\ttoilet_flush\n".format(cls.SCENE), encoding="utf-8"
            )
        return str(root)

    def _run(self, run_dir):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main([run_dir, "--arms", "heard_seen"])
        return code, buffer.getvalue()

    def test_the_grading_key_turns_the_audits_into_a_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run(self._sweep(tmp, with_assignment=True))
        self.assertEqual(code, 0)
        self.assertIn("heard_seen: 1 correct, 1 wrong, 1 with no confidence", out)
        self.assertIn("free floor 0.9100", out)
        self.assertIn("SEPARABLE", out)

    def test_without_the_key_the_section_says_so_rather_than_guessing(self):
        """The forced-failure arm. A sweep with no assignment.tsv cannot know which
        category a correct recall would have named, and inventing one would grade every
        episode against the wrong answer while printing a confident floor."""
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run(self._sweep(tmp, with_assignment=False))
        self.assertEqual(code, 0)
        self.assertIn("SKIPPED: no grading key", out)
        self.assertNotIn("free floor", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
