"""The yield arithmetic, against injected records — the runs need a GPU, the maths does not.

`aggregate()` turns `summary.json` files into the denominator every experiment matrix is
planned against: how much of HM3D can pose the task at all. Ticket 19's third row, applied
to a number rather than to a gate — given records that say a scene refused every episode,
does the report say so, or does it quietly average the refusal away.
"""

import json
import pathlib
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.yield_report import aggregate, format_report, load_summaries

REAL_REASON = (
    "no object in 4ok3usBNeis is >= 3.00 m (xz) from every 'bed' goal AND within 1.00 m "
    "in y of BOTH the primary anchor and the episode start (rejected: 11 too near, 4 on "
    "another floor, 0 with no view point). The scene cannot express a decoupled anomaly "
    "response for this episode."
)


def summary(scene, built, skips=()):
    return {
        "run_dir": "runs/x/" + scene,
        "scene": scene,
        "n_episodes": built,
        "n_skipped": len(skips),
        "funnel": {},
        "skipped": [{"episode_id": str(i), "reason": r} for i, r in enumerate(skips)],
    }


class TestAggregate(unittest.TestCase):
    def test_the_headline_is_built_over_offered(self):
        agg = aggregate([summary("a", 3, [REAL_REASON]), summary("b", 1, [REAL_REASON] * 3)])
        self.assertEqual((agg["built"], agg["skipped"], agg["offered"]), (4, 4, 8))
        self.assertAlmostEqual(agg["yield"], 0.5)

    def test_the_real_builder_reason_parses_into_rules(self):
        """Pinned against the exact string the box produced, not a paraphrase of it."""
        agg = aggregate([summary("a", 0, [REAL_REASON])])
        self.assertEqual(agg["rules"]["too_near"], 11)
        self.assertEqual(agg["rules"]["on_another_floor"], 4)
        self.assertEqual(agg["rules"]["no_view_point"], 0)
        self.assertEqual(agg["unattributed_skips"], 0)

    def test_a_scene_that_built_nothing_reads_as_zero_not_as_absent(self):
        agg = aggregate([summary("a", 4), summary("b", 0, [REAL_REASON] * 5)])
        by_scene = {r["scene"]: r for r in agg["per_scene"]}
        self.assertEqual(by_scene["b"]["yield"], 0.0)
        self.assertAlmostEqual(agg["yield"], 4 / 9)

    def test_no_episodes_offered_is_none_not_zero(self):
        """A yield of zero and no data are different claims; only one is a measurement."""
        agg = aggregate([summary("a", 0)])
        self.assertIsNone(agg["yield"])
        self.assertIsNone(agg["per_scene"][0]["yield"])

    def test_an_unparsed_reason_is_counted_rather_than_dropped(self):
        """If the builder's wording changes, the loss must show as a gap, not shrink."""
        agg = aggregate([summary("a", 0, ["the scene said no, in words nobody parses"])])
        self.assertEqual(agg["skipped"], 1)
        self.assertEqual(agg["unattributed_skips"], 1)
        self.assertEqual(agg["rules"], {})

    def test_empty_input_does_not_claim_anything(self):
        agg = aggregate([])
        self.assertIsNone(agg["yield"])
        self.assertEqual(agg["n_scenes"], 0)


class TestFormat(unittest.TestCase):
    def test_the_report_names_the_scenes_and_the_total(self):
        text = format_report(aggregate([summary("aaa", 2, [REAL_REASON]),
                                        summary("bbb", 0, [REAL_REASON])]))
        self.assertIn("aaa", text)
        self.assertIn("bbb", text)
        self.assertIn("TOTAL (2 scenes)", text)
        self.assertIn("too_near", text)

    def test_an_unmeasured_yield_prints_as_n_a_not_as_zero(self):
        self.assertIn("n/a", format_report(aggregate([summary("aaa", 0)])))


class TestLoad(unittest.TestCase):
    def test_it_finds_one_summary_per_scene_directory(self):
        with tempfile.TemporaryDirectory() as td:
            for scene in ("aaa", "bbb"):
                d = pathlib.Path(td) / scene
                d.mkdir()
                (d / "summary.json").write_text(json.dumps(summary(scene, 1)),
                                                encoding="utf-8")
            found = load_summaries(td)
        self.assertEqual(sorted(s["scene"] for s in found), ["aaa", "bbb"])

    def test_an_empty_sweep_directory_yields_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(load_summaries(td), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
