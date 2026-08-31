"""``audio/window.py`` -- the boundary the whole of ADR-0017 is written against.

A failure here is not a wrong number, it is a wrong TASK. If the offset step is off by
one the source sounds for a step it should not and the agent can home on a cue the
episode says was already gone; if the drawn duration is not a pure function of
``(seed, episode_index)`` then ``tools/episode_diff.py`` pairs two DIFFERENT tasks and
the only test this apparatus has that can resolve a delta of a dozen episodes silently
stops meaning anything.

The other thing pinned here is the CONTROL ARM. ``WindowPolicy.CONTINUOUS`` reproduces
the pre-ADR-0017 ``playing = step >= t_anom`` exactly, so a windowed run's funnel delta
can be measured against the arm where the window is absent (ADR-0014, and the
hermeticity gate that once called a pre-existing failure a leak for want of one).
"""

import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.window import SoundingWindow, WindowPolicy, plan_window
from earshot.config import RunConfig

# The defaults `audio/window.py` documents as provisional, WRITTEN OUT rather than
# imported: the agreement below is between four numbers, and reading all four off one
# object would make it circular. `test_the_local_constants_are_the_shipped_defaults`
# is what stops that from becoming a private arithmetic identity -- it pins each one
# against `RunConfig`'s real default, so a change in `earshot/config.py` reaches here.
# `T_ANOM` is not among them: it is derived per episode (`task.dataset.derive_t_anom`)
# and `RunConfig.t_anom` is None unless a run pins one.
T_ANOM = 40
MAX_STEPS = 500
SOUNDING_STEPS = 60
BUDGET_FRACTION = 0.12
DRAW_RANGE = (30, 90)


def plan(policy, **overrides):
    """One call site, so a test names only what it changes."""
    kwargs = dict(
        t_anom=T_ANOM,
        max_steps=MAX_STEPS,
        policy=policy,
        sounding_steps=SOUNDING_STEPS,
        budget_fraction=BUDGET_FRACTION,
        draw_steps_range=DRAW_RANGE,
        seed=20260817,
        episode_index=0,
    )
    kwargs.update(overrides)
    return plan_window(**kwargs)


class TestTheWindowBoundary(unittest.TestCase):
    def test_the_offset_step_is_the_first_silent_step_not_the_last_sounding_one(self):
        """"Closes at" is ambiguous in prose and this is the one place it is settled.

        Off by one here and the source emits on a step the record says it did not, which
        is the `anommxv` shape: a funnel that counts and a signal that disagrees with it.
        """
        window = plan(WindowPolicy.FIXED_STEPS)
        self.assertEqual(window.offset_step, T_ANOM + SOUNDING_STEPS)
        self.assertTrue(window.is_sounding(window.offset_step - 1))
        self.assertFalse(window.is_sounding(window.offset_step))
        self.assertFalse(window.is_silent(window.offset_step - 1))
        self.assertTrue(window.is_silent(window.offset_step))
        self.assertEqual(window.duration_steps, SOUNDING_STEPS)

    def test_a_continuous_window_never_closes_and_is_the_arm_the_window_is_measured_against(self):
        """THE CONTROL ARM: same input, previous mechanism (`playing = step >= t_anom`).

        Kept as a policy rather than deleted because any funnel delta a windowed run
        reports crosses two changes at once -- the offset step and the accumulating
        renderer -- and this repo's rule is that a claim that X broke because of a change
        needs the arm where the change is absent.
        """
        window = plan(WindowPolicy.CONTINUOUS)
        self.assertIsNone(window.offset_step)
        self.assertIsNone(window.duration_steps)
        for step in (T_ANOM, T_ANOM + 1, MAX_STEPS, 10 * MAX_STEPS):
            self.assertTrue(window.is_sounding(step))
            self.assertFalse(window.is_silent(step))

    def test_nothing_sounds_before_the_window_opens(self):
        """§3.1's pre-onset invariant needs this under EVERY policy, not just the old one."""
        for policy in WindowPolicy:
            window = plan(policy)
            self.assertFalse(window.is_sounding(T_ANOM - 1), policy)
            self.assertFalse(window.is_sounding(0), policy)
            self.assertTrue(window.is_sounding(T_ANOM), policy)
            # ...and a pre-t_anom step is not "silent phase" either: those steps belong
            # to the provenance check, and pooling them with the tail's steps is the
            # confusion `test_task_runner.py`'s `not row.source_playing` filter has.
            self.assertFalse(window.is_silent(T_ANOM - 1), policy)

    def test_the_local_constants_are_the_shipped_defaults(self):
        """The four numbers above are ``RunConfig``'s, and this is what says so.

        Without it the agreement test below is arithmetic on two literals declared in
        the same file -- ``(30 + 90) // 2 == 60`` reduces to ``60 == 60`` -- and a
        change to the real defaults reaches ``test_config.py`` and nothing here, so the
        claim "the three policies agree AT THEIR DEFAULTS" would be about numbers this
        build no longer ships.
        """
        shipped = RunConfig(run_dir="/nonexistent")
        self.assertEqual(shipped.max_steps, MAX_STEPS)
        self.assertEqual(shipped.sounding_steps, SOUNDING_STEPS)
        self.assertEqual(shipped.sounding_budget_fraction, BUDGET_FRACTION)
        self.assertEqual(tuple(shipped.sounding_draw_steps), DRAW_RANGE)
        self.assertIs(shipped.sounding_policy, WindowPolicy.FIXED_STEPS)

    def test_the_three_policies_agree_at_their_defaults(self):
        """Why the first policy comparison is a VARIANCE comparison, not a level one.

        60, floor(0.12 * 500) and mean(30, 90) are the same number on purpose. If they
        disagreed, switching policy would move the duration's level as well as its
        spread and the comparison would be confounded before it started.

        The constants are the shipped ones -- ``test_the_local_constants_are_the_shipped_
        defaults`` binds them -- so this is a statement about the build and not about
        five literals in this file.
        """
        fixed = plan(WindowPolicy.FIXED_STEPS)
        fraction = plan(WindowPolicy.BUDGET_FRACTION)
        self.assertEqual(fixed.offset_step, fraction.offset_step)
        self.assertEqual(fraction.duration_steps, 60)
        self.assertEqual((DRAW_RANGE[0] + DRAW_RANGE[1]) // 2, SOUNDING_STEPS)
        for index in range(24):
            drawn = plan(WindowPolicy.DRAWN, episode_index=index)
            self.assertGreaterEqual(drawn.duration_steps, DRAW_RANGE[0])
            self.assertLessEqual(drawn.duration_steps, DRAW_RANGE[1])

    def test_the_offset_step_is_not_clamped_to_the_step_budget(self):
        """An episode that ends before its window closes has NO silent phase.

        That is a funnel fact about that episode -- and it is exactly what SWS's
        denominator is built on. Clamping would manufacture a silent phase in an episode
        that never had one, and the record would then say the source stopped when it did
        not.
        """
        window = plan(WindowPolicy.FIXED_STEPS, sounding_steps=MAX_STEPS + 200)
        self.assertGreater(window.offset_step, MAX_STEPS)
        self.assertTrue(window.is_sounding(MAX_STEPS - 1))
        self.assertFalse(window.is_silent(MAX_STEPS - 1))

    def test_the_window_has_no_serialiser_of_its_own(self):
        """ONE serialiser for the window, and it is ``report.audit``'s.

        There were two. This type carried an ``as_dict`` that never reached disk while
        ``SoundingWindowRecord`` carried the one that does, and they disagreed — the
        record has the accumulator's measurements and the window's had a
        ``duration_steps`` the record does not. Two dicts for one concept is how a reader
        ends up comparing a key that exists in only one of them, and ADR-0013 forbids
        ``report`` importing this module, so nothing could ever have checked them against
        each other in production code.

        What the runner reads instead is pinned here: the boundaries, and the policy's
        ``.value`` (the audit is JSON, and an Enum member in it is a TypeError at write
        time).
        """
        window = plan(WindowPolicy.DRAWN)
        self.assertFalse(
            hasattr(window, "as_dict"),
            "a second serialiser is back; report/audit.SoundingWindowRecord is the one",
        )
        self.assertIsInstance(window.policy.value, str)
        self.assertEqual(window.policy.value, "drawn")
        self.assertEqual(window.opens_at, T_ANOM)
        self.assertEqual(window.duration_steps, window.offset_step - window.opens_at)
        self.assertIsNone(plan(WindowPolicy.CONTINUOUS).offset_step)
        self.assertIsNone(plan(WindowPolicy.CONTINUOUS).duration_steps)


class TestTheDraw(unittest.TestCase):
    def test_the_drawn_duration_is_a_function_of_the_seed_and_the_index_alone(self):
        """A global draw costs two things this project has already paid for once.

        A red run that cannot be reproduced is not evidence (``RunConfig.seed``'s own
        comment). And ``tools/episode_diff.py`` pairs the SAME episode index across two
        sweeps, so a duration that depended on anything else -- how many navmesh poses
        happened to be drawn first, say -- would put a different task on each side of the
        pair and break the only test that can resolve a delta of a dozen episodes.
        """
        first = plan(WindowPolicy.DRAWN, episode_index=4)
        again = plan(WindowPolicy.DRAWN, episode_index=4)
        self.assertEqual(first, again)

        # THE FORCED FAILURE ARM: exhaust the global stream between the two calls. An
        # implementation reaching for `np.random` rather than a local generator answers
        # differently here and identically above.
        np.random.default_rng(0).integers(0, 100, size=1000)
        np.random.seed(1)
        np.random.random(4096)
        self.assertEqual(plan(WindowPolicy.DRAWN, episode_index=4), first)

        # ...and it is not a constant dressed up as a draw.
        durations = {
            plan(WindowPolicy.DRAWN, episode_index=i).duration_steps for i in range(16)
        }
        self.assertGreater(len(durations), 1)
        self.assertNotEqual(
            plan(WindowPolicy.DRAWN, episode_index=4, seed=1).duration_steps,
            first.duration_steps,
        )

    def test_a_different_index_is_a_different_episode(self):
        """Every episode drawing the same duration would be FIXED_STEPS with extra steps."""
        drawn = [
            plan(WindowPolicy.DRAWN, episode_index=i).duration_steps for i in range(40)
        ]
        self.assertGreater(len(set(drawn)), 10)


class TestRefusals(unittest.TestCase):
    def test_a_window_that_closes_before_it_opens_raises(self):
        """A zero-length window is not a silent episode, it is a broken one: the record
        would carry an offset step at or before t_anom and §3.1's ordering would be
        unreadable."""
        with self.assertRaises(ValueError) as caught:
            plan(WindowPolicy.FIXED_STEPS, sounding_steps=0)
        self.assertIn("0", str(caught.exception))
        with self.assertRaises(ValueError):
            plan(WindowPolicy.FIXED_STEPS, sounding_steps=-5)

    def test_an_out_of_range_budget_fraction_raises_and_an_unused_one_does_not(self):
        """Only the CHOSEN policy's parameters are validated: a run that never draws must
        not be blocked by a range nobody reads, and the unused values are still recorded.
        """
        with self.assertRaises(ValueError) as caught:
            plan(WindowPolicy.BUDGET_FRACTION, budget_fraction=0.0)
        self.assertIn("0.0", str(caught.exception))
        with self.assertRaises(ValueError):
            plan(WindowPolicy.BUDGET_FRACTION, budget_fraction=1.5)
        self.assertEqual(
            plan(WindowPolicy.FIXED_STEPS, budget_fraction=0.0).duration_steps,
            SOUNDING_STEPS,
        )
        self.assertEqual(
            plan(WindowPolicy.FIXED_STEPS, draw_steps_range=(90, 30)).duration_steps,
            SOUNDING_STEPS,
        )

    def test_a_bad_draw_range_raises_only_when_it_is_drawn_from(self):
        with self.assertRaises(ValueError) as caught:
            plan(WindowPolicy.DRAWN, draw_steps_range=(90, 30))
        self.assertIn("90", str(caught.exception))
        with self.assertRaises(ValueError):
            plan(WindowPolicy.DRAWN, draw_steps_range=(0, 30))

    def test_a_negative_t_anom_or_an_empty_budget_raises_under_every_policy(self):
        for policy in WindowPolicy:
            with self.assertRaises(ValueError):
                plan(policy, t_anom=-1)
            with self.assertRaises(ValueError):
                plan(policy, max_steps=0)

    def test_the_window_is_frozen(self):
        """The runner owns the episode's one mutable slot; a window edited mid-episode is
        a window the record cannot describe."""
        window = plan(WindowPolicy.FIXED_STEPS)
        with self.assertRaises(Exception):
            window.offset_step = 3  # type: ignore[misc]
        self.assertIsInstance(
            SoundingWindow(opens_at=0, offset_step=1, policy=WindowPolicy.FIXED_STEPS),
            SoundingWindow,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
