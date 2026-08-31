"""Anchor accuracy, and the affinity cut the first gate run forgot to apply.

Two corrections to `clapsmoke-3` are under test here, and both were mistakes of measurement
rather than of code.

**Class top-1 was the wrong number.** The agent navigates to an OBJECT, so a class confused
for a sibling of the same anchor costs it nothing. In that run every one of `snoring`'s 60
misses landed on `breathing` and every one of `clock_tick`'s 53 landed on `clock_alarm` --
both bed classes -- so recalls of 0.500 and 0.558 understated what the agent would have done.

**The prune applied one of ADR-0018's two cuts.** A weak-affinity class is disqualified
whatever its recall, because the semantic store cannot learn an association that is not
there. `coughing` scored a perfect 1.000 and is still disqualified: people cough on every one
of the six objects. The run kept it and five other weak classes.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.separation import GateRow, anchor_top1_of, prune, summarise
from earshot.audio.vocabulary import CANDIDATE_VOCABULARY, ROOM_OF_ANCHOR

# Two bed classes and one toilet class: a sibling confusion and a cross-anchor one.
ANCHORS = {"snoring": "bed", "breathing": "bed", "toilet_flush": "toilet"}
AFFINITIES = {"snoring": "strong", "breathing": "moderate", "toilet_flush": "strong"}
CLASSES = tuple(ANCHORS)


def row(true_class, winner, *, in_vocabulary=True, distance=3.0, recording=0, normal=0.0):
    scores = {name: 0.1 for name in CLASSES}
    if winner is not None:
        scores[winner] = 0.5
    return GateRow(
        true_class=true_class,
        in_vocabulary=in_vocabulary,
        distance_m=distance,
        scene="TESTSCENE",
        recording_index=recording,
        scores=scores,
        normal_cosine=normal,
    )


def absent(name, *, normal=0.3):
    scores = {item: 0.1 for item in CLASSES}
    return GateRow(
        true_class=name,
        in_vocabulary=False,
        distance_m=3.0,
        scene="TESTSCENE",
        recording_index=0,
        scores=scores,
        normal_cosine=normal,
    )


class TestAnchorTop1(unittest.TestCase):
    def test_a_sibling_confusion_is_the_right_anchor(self):
        """snoring heard as breathing still sends the agent to the bed."""
        true_anchor, predicted = anchor_top1_of(row("snoring", "breathing"), ANCHORS)
        self.assertEqual((true_anchor, predicted), ("bed", "bed"))

    def test_a_cross_anchor_confusion_is_the_wrong_anchor(self):
        true_anchor, predicted = anchor_top1_of(row("snoring", "toilet_flush"), ANCHORS)
        self.assertEqual((true_anchor, predicted), ("bed", "toilet"))

    def test_a_class_missing_from_the_anchor_map_raises(self):
        """Defaulting it to its own name would score a free hit on the task's headline."""
        with self.assertRaises(KeyError):
            anchor_top1_of(row("snoring", "breathing"), {"snoring": "bed"})

    def test_an_absent_row_has_no_true_anchor(self):
        with self.assertRaises(ValueError):
            anchor_top1_of(absent("chainsaw"), ANCHORS)


class TestSummariseWithAnchors(unittest.TestCase):
    def _rows(self):
        rows = []
        # snoring: every miss goes to its bed sibling. Class recall 0.5, anchor accuracy 1.0.
        for index in range(10):
            rows.append(row("snoring", "snoring" if index < 5 else "breathing", recording=index))
        # toilet_flush is never confused at all.
        for index in range(10):
            rows.append(row("toilet_flush", "toilet_flush", recording=index))
        rows.append(absent("chainsaw"))
        return rows

    def test_anchor_accuracy_exceeds_class_accuracy_on_sibling_confusion(self):
        report = summarise(self._rows(), affinities=AFFINITIES, anchors=ANCHORS)
        self.assertAlmostEqual(report.top1_accuracy, 0.75, places=6)
        self.assertAlmostEqual(report.anchor_top1_accuracy, 1.0, places=6)

    def test_the_bed_anchor_is_perfect_despite_a_half_recall_class(self):
        report = summarise(self._rows(), affinities=AFFINITIES, anchors=ANCHORS)
        by_anchor = {item.anchor: item for item in report.per_anchor}
        self.assertAlmostEqual(by_anchor["bed"].accuracy, 1.0, places=6)
        by_class = {item.name: item for item in report.per_class}
        self.assertAlmostEqual(by_class["snoring"].recall, 0.5, places=6)

    def test_anchor_accuracy_is_none_when_no_map_is_given(self):
        """Absent and 0.0 are different facts and must not read the same."""
        report = summarise(self._rows(), affinities=AFFINITIES)
        self.assertIsNone(report.anchor_top1_accuracy)
        self.assertEqual(report.per_anchor, ())

    def test_a_cross_anchor_confusion_lowers_the_anchor_number(self):
        rows = [row("snoring", "toilet_flush", recording=index) for index in range(10)]
        rows += [row("toilet_flush", "toilet_flush", recording=index) for index in range(10)]
        rows.append(absent("chainsaw"))
        report = summarise(rows, affinities=AFFINITIES, anchors=ANCHORS)
        self.assertAlmostEqual(report.anchor_top1_accuracy, 0.5, places=6)


class TestAffinityCut(unittest.TestCase):
    def _report(self):
        rows = []
        for index in range(10):
            rows.append(row("snoring", "snoring", recording=index))
            rows.append(row("breathing", "breathing", recording=index))
            rows.append(row("toilet_flush", "toilet_flush", recording=index))
        rows.append(absent("chainsaw"))
        return summarise(
            rows,
            affinities={"snoring": "strong", "breathing": "weak", "toilet_flush": "moderate"},
            anchors=ANCHORS,
        )

    def test_a_perfect_weak_class_is_still_cut(self):
        """`coughing` scored 1.000 in clapsmoke-3 and is disqualified anyway."""
        kept, cut = prune(
            self._report(),
            min_recall=0.5,
            recall_level="anchor",
            min_n=8,
            allowed_affinities=("strong", "moderate"),
        )
        self.assertIn("breathing", cut)
        self.assertIn("snoring", kept)
        self.assertIn("toilet_flush", kept)

    def test_omitting_the_affinity_cut_keeps_the_weak_class(self):
        """The old behaviour, preserved so a caller must ASK for the cut and say so."""
        kept, _cut = prune(
            self._report(), min_recall=0.5, recall_level="anchor", min_n=8
        )
        self.assertIn("breathing", kept)


class TestTheCutIsMadeAtTheAnchor(unittest.TestCase):
    """ADR-0018 Q9: the separation cut reads ANCHOR recall, because that is what the task pays.

    `clapgate-1` measured `pouring_water` at 0.354 class recall and 1.000 anchor recall: every
    one of its misses landed on another bathroom class, so the agent walked to the right room
    every time. Cutting it on class recall priced a cost the task never pays and took a quarter
    of the bathroom vocabulary with it.
    """

    def _report(self):
        """`snoring` is the pouring_water case: 0.4 class recall, 1.0 anchor recall."""
        rows = []
        for index in range(10):
            rows.append(row("snoring", "snoring" if index < 4 else "breathing", recording=index))
            rows.append(row("breathing", "breathing", recording=index))
            rows.append(row("toilet_flush", "toilet_flush", recording=index))
        rows.append(absent("chainsaw"))
        return summarise(rows, affinities=AFFINITIES, anchors=ANCHORS)

    def test_a_sibling_confused_class_survives_the_anchor_bar(self):
        kept, cut = prune(self._report(), min_recall=0.5, recall_level="anchor", min_n=8)
        self.assertIn("snoring", kept)
        self.assertNotIn("snoring", cut)

    def test_the_same_class_is_cut_by_the_class_bar(self):
        """Both bars stay scoreable so the looser one can be checked against the stricter."""
        kept, cut = prune(self._report(), min_recall=0.5, recall_level="class", min_n=8)
        self.assertIn("snoring", cut)
        self.assertNotIn("snoring", kept)

    def test_the_anchor_bar_on_a_report_with_no_anchor_map_raises(self):
        """NOT_RUN is red: a recall the report never measured must not read as a pass."""
        report = summarise(
            [row("snoring", "snoring", recording=index) for index in range(10)]
            + [absent("chainsaw")],
            affinities=AFFINITIES,
        )
        self.assertIsNone(report.per_class[0].anchor_recall)
        with self.assertRaises(ValueError):
            prune(report, min_recall=0.5, recall_level="anchor")

    def test_an_unknown_recall_level_raises_rather_than_defaulting(self):
        with self.assertRaises(ValueError):
            prune(self._report(), min_recall=0.5, recall_level="ANCHOR")

    def test_anchor_recall_matches_the_per_anchor_arithmetic(self):
        by_class = {item.name: item for item in self._report().per_class}
        self.assertAlmostEqual(by_class["snoring"].recall, 0.4, places=6)
        self.assertAlmostEqual(by_class["snoring"].anchor_recall, 1.0, places=6)


class TestTheVocabularyCanStillBeSplit(unittest.TestCase):
    def test_at_least_two_anchors_carry_two_or_more_eligible_classes(self):
        """The heard/not-heard split needs >= 2 classes at an anchor to be splittable there.

        This guards the vocabulary TABLE, not a gate result: if a future edit drops the
        candidate set to one class per anchor, the split becomes impossible and every anchor
        lands in one column only, confounding the columns with object difficulty.
        """
        eligible = [
            entry
            for entry in CANDIDATE_VOCABULARY
            if entry.room_affinity in ("strong", "moderate")
        ]
        counts = {}
        for entry in eligible:
            room = ROOM_OF_ANCHOR[entry.anchor_object]
            counts[room] = counts.get(room, 0) + 1
        splittable = [anchor for anchor, count in counts.items() if count >= 2]
        self.assertGreaterEqual(
            len(splittable),
            3,
            "only {} room(s) carry 2+ non-weak classes: {}. The room taxonomy was adopted "
            "for exactly this -- under the object taxonomy only bed and toilet were "
            "splittable, and a 2-way semantic space is too thin to build the matrix "
            "on.".format(len(splittable), counts),
        )


if __name__ == "__main__":
    unittest.main()
