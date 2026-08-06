"""The flip arithmetic, against injected records — the runs need a GPU, the maths does not.

`compare()` answers the question two real runs raised and neither could settle: how much
of an episode outcome is the agent and how much is the renderer. `yield-1` and `detour-1`
both reported 8/20 source-reached and did not agree on which eight.

The trap this pins is that the AGGREGATE can be perfectly stable while the membership
churns underneath it, which is exactly what those two runs did. A comparison that only
watched the rate would have called that reproducible.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.flip_report import compare, format_report


def run(name, rays, reached):
    """`reached` maps episode index -> did the loop close."""
    return {"run": name, "rays": rays, "reached": dict(reached)}


class TestTheFlipRate(unittest.TestCase):
    def test_identical_repeats_have_no_flips(self):
        same = {0: True, 1: False, 2: True}
        arm = compare([run("a", 500, same), run("b", 500, same)])["arms"][0]
        self.assertEqual(arm["n_flipped"], 0)
        self.assertEqual(arm["flip_rate"], 0.0)

    def test_a_stable_aggregate_with_churning_membership_is_caught(self):
        """The yield-1/detour-1 shape: same count, different episodes. A report that
        watched only the rate would have called this reproducible."""
        first = {0: True, 1: True, 2: False, 3: False}
        second = {0: True, 1: False, 2: True, 3: False}
        arm = compare([run("a", 500, first), run("b", 500, second)])["arms"][0]
        self.assertEqual(arm["reached"], [2, 2], "the aggregate is identical")
        self.assertEqual(arm["flipped"], [1, 2])
        self.assertEqual(arm["flip_rate"], 0.5)

    def test_unanimity_not_majority(self):
        """With three repeats, 2-of-3 agreement is still an episode whose outcome
        depends on the roll."""
        arm = compare([run("a", 500, {0: True}), run("b", 500, {0: True}),
                       run("c", 500, {0: False})])["arms"][0]
        self.assertEqual(arm["flipped"], [0])

    def test_one_run_reports_no_evidence_rather_than_perfect_stability(self):
        """A single run cannot disagree with itself, and 0% would read as stability."""
        arm = compare([run("a", 500, {0: True, 1: False})])["arms"][0]
        self.assertIsNone(arm["flip_rate"])
        self.assertIsNone(arm["n_flipped"])
        self.assertIn("one run only", format_report({"arms": [arm]}))

    def test_arms_are_grouped_by_ray_count(self):
        comparison = compare([
            run("a", 500, {0: True, 1: True}),
            run("b", 500, {0: False, 1: True}),
            run("c", 2500, {0: True, 1: True}),
            run("d", 2500, {0: True, 1: True}),
        ])
        by_rays = {row["rays"]: row for row in comparison["arms"]}
        self.assertEqual(by_rays[500]["flip_rate"], 0.5)
        self.assertEqual(by_rays[2500]["flip_rate"], 0.0)

    def test_an_episode_missing_from_one_repeat_is_reported_not_dropped(self):
        """A repeat that built a different episode set is not a repeat, and silently
        intersecting would turn that into a smaller and quieter flip rate."""
        arm = compare([run("a", 500, {0: True, 1: False}),
                       run("b", 500, {0: True})])["arms"][0]
        self.assertEqual(arm["ragged"], [1])
        self.assertEqual(arm["n_compared"], 1)
        self.assertIn("MISSING from some repeat", format_report({"arms": [arm]}))

    def test_a_run_that_set_no_ray_count_is_its_own_arm(self):
        """`None` in the record means the run took the preset default rather than
        choosing one — a different fact from '500 was chosen' only in intent, and intent
        is what a comparison across arms reads."""
        comparison = compare([run("a", None, {0: True}), run("b", 500, {0: True})])
        self.assertEqual([row["rays"] for row in comparison["arms"]], [500, None])
        self.assertIn("preset", format_report(comparison))


class TestTheReportIsReadable(unittest.TestCase):
    def test_it_names_the_episodes_that_flipped(self):
        text = format_report(compare([
            run("a", 500, {0: True, 1: True, 2: False}),
            run("b", 500, {0: True, 1: False, 2: False}),
        ]))
        self.assertIn("not unanimous", text)
        self.assertIn("1", text)

    def test_it_shows_the_aggregate_beside_the_flip_rate(self):
        """Both numbers, because the first being stable is what makes the second
        surprising."""
        text = format_report(compare([
            run("a", 500, {0: True, 1: False}),
            run("b", 500, {0: False, 1: True}),
        ]))
        self.assertIn("1 1", text)   # reached per run: identical
        self.assertIn("100%", text)  # and yet every episode flipped


if __name__ == "__main__":
    unittest.main(verbosity=2)
