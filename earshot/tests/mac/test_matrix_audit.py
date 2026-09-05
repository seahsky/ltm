"""`tools/matrix_audit.py`: the review's four read-only questions, and the coverage gate.

Everything here is the pure half — synthetic provenance blocks and synthetic audit rows,
because the loaders are thin wrappers over `report.artifacts` and the analysis must be
decidable without a finished sweep on disk. The gate is exercised through its real CLI
(`main`) against a real `store.json` written by `dump_stores`, in both arms (ADR-0014):
the healthy pass, and the red exit a silently-dropped scene must produce.
"""

import tempfile
import unittest
from pathlib import Path

from _interpreter import assert_interpreter  # noqa: F401

from earshot.memory.store import EpisodicEntry, EpisodicStore, SemanticStore
from earshot.task.memory_build import dump_stores
from earshot.tools.matrix_audit import (
    discordance_where_identical,
    gate_missing,
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
):
    return {
        "reached": reached,
        "source": source,
        "condition": "heard_seen",
        "category": category,
        "miss": miss,
        "instances": instances,
        "distance_m": distance,
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

    def test_a_missing_scene_exits_two(self):
        """The forced-failure arm: the sweep must stop before the cells, because a cell
        run over an uncovered scene has seen == unseen by construction."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, complete=["A"])
            self.assertEqual(main(["--store", store, "--gate-scenes", "A B"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
