"""``audio/normality.py`` — the abstain contract, and the taxonomy that has to line up.

The abstain contract is what makes an **unmeasured** labeller safe to ship. ADR-0002's
$0 room-classifier accuracy gate carries across the captioner substitution and has not
been run, so nothing here licenses the labeller's *accuracy*; what it licenses is that a
labeller which does not know says so, and that "does not know" leaves the context-free
verdict alone.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.clap import ANOMALY_CLASSES, SOUNDING_CLASSES
from earshot.audio.normality import (
    ROOM_KEYWORDS,
    ROOM_PRIOR,
    CaptionerRoomLabeler,
    NullRoomLabeler,
    RoomLabeler,
    is_anomalous_here,
    known_rooms,
    resolve_room_from_caption,
    room_conditioned_anomaly,
)


class StubCaptioner:
    def __init__(self, caption="a photo of a kitchen", raises=False):
        self.caption_text = caption
        self.raises = raises
        self.n_calls = 0

    def caption(self, image):
        self.n_calls += 1
        if self.raises:
            raise RuntimeError("the VLM fell over")
        return self.caption_text


class TestTheRoomArmCanOnlySayTrueOrAbstain(unittest.TestCase):
    """A standing fact about the wiring, asserted so it becomes a decision when it ends.

    `runner.py` hands `is_anomalous_here` the TESTIMONY class -- what was heard -- which
    is the right input for "is that sound unexpected here". Today it changes nothing:
    `ROOM_PRIOR`'s normal sets name `running_water` and `appliance_hum`, and neither is
    in `ANOMALY_CLASSES` or in `SOUNDING_CLASSES`. So for every class a run can actually
    place, `room_conditioned_anomaly` returns True or abstains, and the room arm can
    never veto an interrupt.

    That is worth pinning rather than assuming. If a sounding class is ever added to a
    room's normal set, this fails, and the runner line stops being a no-op -- which is
    the moment someone has to decide whether a `pouring_water` in a bathroom should
    still divert the agent.
    """

    def test_no_class_a_run_can_place_is_normal_in_any_room(self):
        placeable = set(ANOMALY_CLASSES) | set(SOUNDING_CLASSES)
        normal_anywhere = set().union(*ROOM_PRIOR.values())
        self.assertEqual(
            placeable & normal_anywhere,
            set(),
            "a placeable class became normal somewhere; the runner's room arm is no "
            "longer a no-op and the wiring is now a decision",
        )

    def test_so_the_verdict_is_true_wherever_the_room_is_known(self):
        for name in sorted(set(ANOMALY_CLASSES) | set(SOUNDING_CLASSES)):
            for room in ROOM_PRIOR:
                self.assertTrue(room_conditioned_anomaly(name, room), (name, room))

    def test_and_the_prior_does_hold_a_normal_sound_so_this_is_not_vacuous(self):
        """The other arm: the table CAN say normal, just not about a placeable class."""
        self.assertTrue(any(ROOM_PRIOR.values()))
        self.assertFalse(room_conditioned_anomaly("running_water", "bathroom"))


class TestTaxonomy(unittest.TestCase):
    def test_the_prior_and_the_keywords_describe_the_same_rooms(self):
        """A caption-resolved room the prior has never heard of abstains forever, and
        it would look exactly like a room that carries no normality knowledge."""
        self.assertEqual(sorted(ROOM_PRIOR), sorted(ROOM_KEYWORDS))
        self.assertEqual(list(known_rooms()), sorted(ROOM_PRIOR))

    def test_an_empty_normal_set_is_a_claim_not_an_absence(self):
        """"bedroom knows of no normal sound" differs from "bedroom is not in the
        table": the first conditions, the second abstains."""
        self.assertIn("bedroom", ROOM_PRIOR)
        self.assertEqual(ROOM_PRIOR["bedroom"], frozenset())
        self.assertIsNone(room_conditioned_anomaly("running_water", "garage"))
        self.assertTrue(room_conditioned_anomaly("running_water", "bedroom"))


class TestCaptionResolution(unittest.TestCase):
    def test_the_earliest_mention_wins(self):
        """"a bedroom with a door to the hallway" is about the bedroom."""
        self.assertEqual(
            resolve_room_from_caption("a bedroom with a door to the hallway"), "bedroom"
        )
        self.assertEqual(
            resolve_room_from_caption("a hallway leading to a bedroom"), "hallway"
        )

    def test_no_room_named_is_none(self):
        self.assertIsNone(resolve_room_from_caption("a photo of a chair"))
        self.assertIsNone(resolve_room_from_caption(""))
        self.assertIsNone(resolve_room_from_caption(None))

    def test_matching_is_case_insensitive(self):
        self.assertEqual(resolve_room_from_caption("A KITCHEN"), "kitchen")


class TestLabellers(unittest.TestCase):
    def test_the_null_labeller_always_abstains(self):
        """What the smoke runs: with one sound that is the anomaly by construction,
        a room label would put an unmeasured component on the critical path."""
        self.assertIsNone(NullRoomLabeler().label(object()))

    def test_both_labellers_satisfy_the_seam(self):
        self.assertIsInstance(NullRoomLabeler(), RoomLabeler)
        self.assertIsInstance(CaptionerRoomLabeler(StubCaptioner()), RoomLabeler)

    def test_the_captioner_labeller_reads_the_caption(self):
        captioner = StubCaptioner("a tidy bathroom with a sink")
        labeller = CaptionerRoomLabeler(captioner)
        self.assertEqual(labeller.label(object()), "bathroom")
        self.assertEqual(captioner.n_calls, 1)
        self.assertEqual(labeller.n_labelled, 1)
        self.assertEqual(labeller.last_caption, "a tidy bathroom with a sink")

    def test_a_captioner_that_raises_abstains_rather_than_ending_the_episode(self):
        """The one place in this layer that swallows an exception, and why: the label
        refines an optional verdict, so a VLM hiccup at step 200 is not fatal."""
        labeller = CaptionerRoomLabeler(StubCaptioner(raises=True))
        self.assertIsNone(labeller.label(object()))
        self.assertEqual(labeller.n_abstained, 1)
        self.assertIsNone(labeller.last_caption)

    def test_a_caption_naming_no_room_abstains(self):
        labeller = CaptionerRoomLabeler(StubCaptioner("a close-up of a potted plant"))
        self.assertIsNone(labeller.label(object()))
        self.assertEqual(labeller.n_abstained, 1)


class TestVerdict(unittest.TestCase):
    def test_the_same_sound_is_normal_in_one_room_and_anomalous_in_another(self):
        """ADR-0002's behavioural test, which ADR-0012 keeps unchanged."""
        self.assertFalse(room_conditioned_anomaly("running_water", "bathroom"))
        self.assertTrue(room_conditioned_anomaly("running_water", "living_room"))

    def test_it_abstains_without_a_class_or_a_room(self):
        self.assertIsNone(room_conditioned_anomaly(None, "kitchen"))
        self.assertIsNone(room_conditioned_anomaly("alarm", None))

    def test_the_room_verdict_replaces_the_context_free_one(self):
        self.assertFalse(is_anomalous_here(True, "running_water", "bathroom"))
        self.assertTrue(is_anomalous_here(False, "running_water", "living_room"))

    def test_an_abstain_leaves_the_context_free_verdict_alone(self):
        self.assertTrue(is_anomalous_here(True, "alarm", None))
        self.assertFalse(is_anomalous_here(False, "alarm", None))
        self.assertTrue(is_anomalous_here(True, "alarm", "garage"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
