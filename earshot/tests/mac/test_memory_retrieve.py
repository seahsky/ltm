"""DREAM §III.F (eq. 19-24), and the one sentence in the paper that is a testable claim.

PR 5 of 6 building DREAM as `ICRA2027_Memory` specifies it.

§III.F: "Under familiar navigation conditions, concrete episodic experience can provide
direct guidance. As the current condition becomes less similar to previous experience,
higher-level pattern and semantic-spatial memories provide more transferable priors."

That is the central hypothesis, and `TestTheOmegaShift` asserts it as a property rather
than quoting it as a comment. Everything else here exists to make that measurement
trustworthy:

  * `f_q` blending in the right direction, or `q_t` asks about where the agent has been;
  * a level that retrieved NOTHING getting weight 0 and not a real softmax share, or an
    empty `M^P` quietly draws weight away from the levels that answered;
  * `M^K` queried with the AUDIO half of the fused query, tied here to the halves-in-order
    property `test_agent_stm.py` pins on `f_fuse`;
  * eq. 24 summing only the two levels that share a space, which is the second place the
    paper's arithmetic does not close (PR #111 found the first).

No torch, no simulator.
"""

import math
import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.stm import ShortTermMemory, StmEntry, fuse
from earshot.memory.longterm import (
    ExperienceEntry,
    ExperienceStore,
    LongTermMemory,
    Outcome,
    PatternStore,
    abstract,
    trajectory_descriptor,
)
from earshot.memory.retrieve import (
    ExperienceHit,
    KnowledgeHit,
    PatternHit,
    RetrievedContext,
    Weights,
    query,
    retrieve,
)
from earshot.memory.store import SemanticEntry, SemanticStore
from earshot.types import Pose, Xyz

VISUAL_WIDTH = 4
AUDIO_WIDTH = 3


def _vec(*values):
    return np.asarray(values, dtype=np.float32)


def _fused(visual_axis, audio_axis):
    """A fused vector with all its visual mass on one axis and all its audio mass on
    another, so the two halves can be steered independently."""
    visual = np.zeros(VISUAL_WIDTH, dtype=np.float32)
    visual[visual_axis] = 1.0
    audio = np.zeros(AUDIO_WIDTH, dtype=np.float32)
    audio[audio_axis] = 1.0
    return fuse(visual, audio)


def _experience(context, sound="toilet_flush", obj="toilet", reached=True, gap=0.3):
    return ExperienceEntry(
        context=np.asarray(context, dtype=np.float32),
        trajectory=trajectory_descriptor(
            [Xyz(0.0, 0.0, 0.0), Xyz(1.0, 0.0, 0.0), Xyz(2.0, 0.0, 0.0)]
        ),
        target_concept=obj,
        outcome=Outcome(reached=reached, final_gap_m=gap),
        sound_concept=sound,
        room_concept=None,
    )


def _knowledge(*rows):
    return SemanticStore(entries=tuple(
        SemanticEntry(
            sound_class=sound, room=room, category=category,
            embedding=np.asarray(embedding, dtype=np.float32), donor_scene="scene_a",
        )
        for sound, room, category, embedding in rows
    ))


def _memory(experiences=(), knowledge=None, min_support=1):
    store = ExperienceStore().extend(experiences)
    return LongTermMemory(
        experience=store,
        pattern=abstract(store, min_support=min_support),
        knowledge=knowledge if knowledge is not None else SemanticStore(),
    )


class TestFq(unittest.TestCase):
    """`q_t = f_q(z^av_t, M^S_t)` (eq. 19). The paper names `f_q` and does not define it."""

    def test_an_empty_stm_gives_the_observation_alone(self):
        """`None` context is the ordinary state at t=0. Blending with a zero vector
        instead would silently halve the observation's contribution for the first K steps
        of every episode."""
        observation = _vec(3.0, 4.0)
        np.testing.assert_allclose(
            query(observation, None, present_weight=0.5), _vec(0.6, 0.8), atol=1e-6
        )

    def test_the_query_is_unit_norm(self):
        result = query(_vec(1.0, 0.0), _vec(0.0, 5.0), present_weight=0.5)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)

    def test_a_present_weight_of_one_ablates_the_stm(self):
        """The arm that makes `M^S` removable from eq. 19 with nothing else changing."""
        np.testing.assert_allclose(
            query(_vec(1.0, 0.0), _vec(0.0, 1.0), present_weight=1.0),
            _vec(1.0, 0.0), atol=1e-6,
        )

    def test_a_present_weight_of_zero_is_the_context_alone(self):
        np.testing.assert_allclose(
            query(_vec(1.0, 0.0), _vec(0.0, 1.0), present_weight=0.0),
            _vec(0.0, 1.0), atol=1e-6,
        )

    def test_a_larger_weight_leans_on_the_present(self):
        """The direction arm. A blend that ran backwards would make `q_t` ask about where
        the agent has been rather than where it is."""
        present = _vec(1.0, 0.0)
        leaning = query(present, _vec(0.0, 1.0), present_weight=0.9)
        even = query(present, _vec(0.0, 1.0), present_weight=0.5)
        self.assertGreater(float(np.dot(leaning, present)), float(np.dot(even, present)))

    def test_the_halves_are_normalised_before_the_blend(self):
        """Otherwise the weight that decides the balance is not the only thing deciding
        it: a half arriving at 1000x would carry the query whatever the weight said.

        BOTH halves are scaled, one at a time. An earlier version scaled only the context,
        and the forced-failure run showed that an implementation which normalised the
        context but not the observation passed it -- half a test for a two-sided property.
        """
        reference = query(_vec(1.0, 0.0), _vec(0.0, 1.0), present_weight=0.5)
        loud_context = query(_vec(1.0, 0.0), _vec(0.0, 1000.0), present_weight=0.5)
        loud_observation = query(_vec(1000.0, 0.0), _vec(0.0, 1.0), present_weight=0.5)
        np.testing.assert_allclose(reference, loud_context, atol=1e-6)
        np.testing.assert_allclose(reference, loud_observation, atol=1e-6)

    def test_a_weight_outside_zero_to_one_raises(self):
        for weight in (-0.1, 1.1):
            with self.assertRaises(ValueError) as caught:
                query(_vec(1.0, 0.0), _vec(0.0, 1.0), present_weight=weight)
            self.assertIn("[0, 1]", str(caught.exception))

    def test_a_zero_norm_half_raises(self):
        with self.assertRaises(ValueError):
            query(np.zeros(2, dtype=np.float32), None, present_weight=1.0)
        with self.assertRaises(ValueError):
            query(_vec(1.0, 0.0), np.zeros(2, dtype=np.float32), present_weight=0.5)

    def test_present_weight_has_no_default(self):
        with self.assertRaises(TypeError):
            query(_vec(1.0, 0.0), None)

    def test_it_composes_with_the_real_short_term_memory(self):
        """`f_q`'s second argument is `M^S_t`, and this is the call that actually makes
        it one — `memory` cannot import `agent`, so the context arrives as the array
        `ShortTermMemory.context` returns."""
        stm = ShortTermMemory(horizon=3)
        for index in range(3):
            stm = stm.push(StmEntry(
                visual=_vec(1.0, float(index)), audio=_vec(0.0, 1.0),
                pose=Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0), prev_action=None,
            ))
        result = query(stm.latest.fused, stm.context(decay=0.8), present_weight=0.7)
        self.assertEqual(result.shape, stm.latest.fused.shape)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)


class TestRetrieveFromEachLevel(unittest.TestCase):
    """Eq. 20-22. The paper names `Retrieve` and does not define it."""

    def _kwargs(self, **overrides):
        kwargs = dict(
            k_experience=2, k_pattern=2, k_knowledge=1, temperature=1.0
        )
        kwargs.update(overrides)
        return kwargs

    def test_the_nearest_experience_comes_first(self):
        memory = _memory([
            _experience(_fused(0, 0)), _experience(_fused(1, 1)), _experience(_fused(2, 2))
        ])
        context = retrieve(_fused(1, 1), memory, **self._kwargs())
        np.testing.assert_allclose(
            context.experience[0].entry.context, _fused(1, 1), atol=1e-6
        )
        self.assertGreater(context.experience[0].score, context.experience[1].score)

    def test_k_bounds_how_many_come_back(self):
        memory = _memory([_experience(_fused(i, 0)) for i in range(4)])
        self.assertEqual(len(retrieve(_fused(0, 0), memory, **self._kwargs(
            k_experience=1)).experience), 1)
        self.assertEqual(len(retrieve(_fused(0, 0), memory, **self._kwargs(
            k_experience=3)).experience), 3)

    def test_a_k_below_one_raises_on_every_level(self):
        memory = _memory([_experience(_fused(0, 0))])
        for name in ("k_experience", "k_pattern", "k_knowledge"):
            with self.assertRaises(ValueError) as caught:
                retrieve(_fused(0, 0), memory, **self._kwargs(**{name: 0}))
            self.assertIn("k >= 1", str(caught.exception))

    def test_a_zero_norm_key_is_skipped_and_not_scored_zero(self):
        """`store.py::_vote` and `consolidate.novelty` skip for the same reason: a row
        with no direction is not evidence, and scoring it 0.0 lets it outrank a genuinely
        dissimilar row."""
        memory = _memory([
            _experience(np.zeros(VISUAL_WIDTH + AUDIO_WIDTH, dtype=np.float32)),
            _experience(-_fused(0, 0)),
        ])
        context = retrieve(_fused(0, 0), memory, **self._kwargs())
        self.assertEqual(len(context.experience), 1)
        self.assertLess(context.experience[0].score, 0.0)

    def test_a_key_of_the_wrong_width_raises(self):
        """A silent slice here is how a wrong encoder reaches a memory unnoticed."""
        memory = _memory([_experience(_fused(0, 0))])
        with self.assertRaises(ValueError) as caught:
            retrieve(_vec(1.0, 0.0), memory, **self._kwargs())
        self.assertIn("two embedding spaces", str(caught.exception))

    def test_an_empty_memory_retrieves_nothing_and_weighs_nothing(self):
        """`None` weights, not three zeros and not a uniform split over nothing. A caller
        that blended on `None` would be blending with a memory that does not exist."""
        context = retrieve(_fused(0, 0), _memory(), **self._kwargs())
        self.assertEqual(context.experience, ())
        self.assertIsNone(context.weights)
        self.assertIsNone(context.fused_context())
        self.assertFalse(context)

    def test_the_pattern_level_is_retrieved_from_its_signatures(self):
        memory = _memory(
            [
                _experience(_fused(0, 0), sound="alarm", obj="fireplace"),
                _experience(_fused(0, 0), sound="alarm", obj="fireplace"),
                _experience(_fused(1, 1), sound="snoring", obj="bed"),
                _experience(_fused(1, 1), sound="snoring", obj="bed"),
            ],
            min_support=2,
        )
        context = retrieve(_fused(1, 1), memory, **self._kwargs())
        self.assertEqual(len(memory.pattern), 2)
        self.assertEqual(context.pattern[0].pattern.sound_concept, "snoring")


class TestTheKnowledgeQueryIsTheAudioHalf(unittest.TestCase):
    """Eq. 22, and the property `f_fuse` was built to make possible."""

    def _memory(self):
        # Two classes whose CLAP embeddings sit on different audio axes.
        knowledge = _knowledge(
            ("toilet_flush", "bathroom", "toilet", [1.0, 0.0, 0.0]),
            ("snoring", "bedroom", "bed", [0.0, 1.0, 0.0]),
        )
        return _memory([_experience(_fused(0, 0))], knowledge=knowledge)

    def test_the_vote_follows_the_audio_half_and_ignores_the_visual_one(self):
        """**THE ARM THAT TIES THIS MODULE TO `f_fuse`.** The split is the LAST
        `knowledge.dim` components, which rests on `fuse` concatenating [visual, audio] in
        that order. If the slice ever took the FIRST components instead, these two queries
        — identical in their audio halves, opposite in their visual ones — would stop
        agreeing."""
        memory = self._memory()
        first = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
        )
        second = retrieve(
            _fused(3, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
        )
        self.assertEqual(first.knowledge.category, "toilet")
        self.assertEqual(second.knowledge.category, "toilet")

    def test_changing_the_audio_half_changes_the_vote(self):
        """The control for the test above: if the vote never moved, it would agree for
        the wrong reason."""
        memory = self._memory()
        other = retrieve(
            _fused(0, 1), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
        )
        self.assertEqual(other.knowledge.category, "bed")

    def test_an_empty_knowledge_store_is_no_hit_rather_than_a_raise(self):
        memory = _memory([_experience(_fused(0, 0))])
        context = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
        )
        self.assertIsNone(context.knowledge)
        self.assertEqual(context.weights.knowledge, 0.0)

    def test_a_knowledge_store_as_wide_as_the_query_raises(self):
        """The audio half of a fused query is strictly narrower than the whole of it. A
        store whose keys are the full width was not built by the same pair of encoders."""
        knowledge = _knowledge(
            ("toilet_flush", "bathroom", "toilet", [1.0] * (VISUAL_WIDTH + AUDIO_WIDTH)),
        )
        memory = _memory([_experience(_fused(0, 0))], knowledge=knowledge)
        with self.assertRaises(ValueError) as caught:
            retrieve(
                _fused(0, 0), memory,
                k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
            )
        self.assertIn("strictly narrower", str(caught.exception))


class TestTheOmegaShift(unittest.TestCase):
    """**THE PAPER'S CENTRAL CLAIM, AS A PROPERTY.**

    §III.F: "Under familiar navigation conditions, concrete episodic experience can
    provide direct guidance. As the current condition becomes less similar to previous
    experience, higher-level pattern and semantic-spatial memories provide more
    transferable priors."
    """

    def _memory(self):
        """TWO WELL-SEPARATED EXPERIENCES, so `M^P`'s signature is their centroid and is
        a genuinely different vector from either.

        The first version of this fixture used two experiences 0.995 apart, and every
        assertion below failed or passed for the wrong reason: a pattern that sits on top
        of its own members is indistinguishable from them, `omega_t` is near-uniform, and
        the level weights carry no information at all. That is a real property and
        `test_a_pattern_over_near_identical_experiences_carries_no_information` now holds
        it, because a sweep whose `G` produced tight clusters would measure exactly that
        and look like a null result.
        """
        experiences = [
            _experience(_fused(0, 0), sound="alarm", obj="fireplace"),
            _experience(_fused(1, 1), sound="alarm", obj="fireplace"),
        ]
        return _memory(experiences, min_support=2)

    def _unfamiliar(self):
        """Equally similar to BOTH stored experiences and well away from each.

        `A`, `B` and `C` are mutually orthogonal, so `unit(A + B + 0.882 C)` sits at
        cosine 0.6 from each of `A` and `B` and at 0.849 from their centroid. It is
        therefore less like every stored experience than the familiar query is, AND more
        like the abstraction -- which is the condition §III.F's second sentence describes.
        """
        raw = _fused(0, 0) + _fused(1, 1) + 0.882 * _fused(2, 2)
        return raw / np.linalg.norm(raw)

    def _weights(self, query_vector):
        return retrieve(
            query_vector, self._memory(),
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=0.1,
        ).weights

    def test_a_familiar_condition_puts_the_weight_on_episodic_experience(self):
        familiar = self._weights(_fused(0, 0))
        self.assertGreater(
            familiar.experience, familiar.pattern,
            "a query that exactly matches a stored experience did not weight the "
            "episodic level highest, so eq. 23's first sentence does not hold",
        )

    def test_weight_moves_off_episodic_as_the_condition_becomes_less_familiar(self):
        """The second sentence, which is the transferable-prior claim the whole paper
        rests on: as similarity to `M^E` falls, `omega^E` falls with it.

        **AND WHY IT HOLDS, WHICH IS NOT OBVIOUS.** A softmax is shift-invariant, so
        weighting by raw similarity cannot respond to "everything got less familiar" on
        its own. What makes the direction come out right is geometric: `M^P`'s signature
        is a CENTROID, and a centroid is nearer a point away from the cluster than any
        individual member is. The abstraction level wins as the query becomes unfamiliar
        because abstractions generalise, and that is a fact about means rather than a
        tuned bias. Nothing here is trained and nothing is hand-weighted per level.
        """
        familiar = self._weights(_fused(0, 0))
        unfamiliar = self._weights(self._unfamiliar())
        self.assertLess(
            unfamiliar.experience, familiar.experience,
            "omega^E did not fall as the query moved away from every stored experience, "
            "so the retrieval is not dynamic in the direction eq. 23 claims",
        )
        self.assertGreater(unfamiliar.pattern, familiar.pattern)
        self.assertGreater(
            unfamiliar.pattern, unfamiliar.experience,
            "the unfamiliar query did not end up weighted toward the transferable level, "
            "which is the whole of eq. 23's second sentence",
        )

    def test_the_unfamiliar_query_really_is_less_like_every_stored_experience(self):
        """The control for the test above. Without it a green there could mean the
        fixture moved TOWARD the pattern rather than away from the experiences."""
        memory = self._memory()
        unfamiliar = self._unfamiliar()
        for key in memory.experience.keys:
            self.assertLess(float(np.dot(unfamiliar, key)), 0.7)
            self.assertAlmostEqual(float(np.dot(_fused(0, 0), memory.experience.keys[0])),
                                   1.0, places=5)

    def test_a_pattern_over_near_identical_experiences_carries_no_information(self):
        """**THE DEGENERATE CASE, ASSERTED RATHER THAN LEFT TO BE REDISCOVERED.** When
        `G` abstracts a tight cluster, the signature sits on top of its own members, every
        level scores the same, and `omega_t` is near-uniform however familiar the query
        is. A sweep whose patterns look like this would measure a null and it would not be
        the hypothesis failing -- it would be `M^P` having nothing to say."""
        tight = _memory(
            [
                _experience(_fused(0, 0), sound="alarm", obj="fireplace"),
                _experience(
                    fuse(_vec(1.0, 0.05, 0.0, 0.0), _vec(1.0, 0.0, 0.0)),
                    sound="alarm", obj="fireplace",
                ),
            ],
            min_support=2,
        )
        weights = retrieve(
            _fused(0, 0), tight,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=0.1,
        ).weights
        self.assertLess(
            abs(weights.experience - weights.pattern), 0.2,
            "this fixture was meant to be degenerate and is not, so the warning it "
            "documents no longer describes anything",
        )

    def test_the_weights_are_a_distribution(self):
        weights = self._weights(_fused(0, 0))
        self.assertAlmostEqual(sum(weights.as_tuple()), 1.0, places=6)
        self.assertTrue(all(value >= 0.0 for value in weights.as_tuple()))

    def test_a_level_that_retrieved_nothing_gets_exactly_zero(self):
        """**NOT A LOGIT OF 0.0.** That is a real number and takes real softmax mass, so
        an empty `M^P` would quietly draw weight away from the levels that answered, and
        `fused_context` would then be scaled by a weight belonging to nothing."""
        memory = _memory([_experience(_fused(0, 0))], min_support=99)
        self.assertEqual(len(memory.pattern), 0)
        weights = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
        ).weights
        self.assertEqual(weights.pattern, 0.0)
        self.assertEqual(weights.knowledge, 0.0)
        self.assertAlmostEqual(weights.experience, 1.0, places=6)

    def test_a_high_temperature_is_the_ablation_of_the_whole_mechanism(self):
        """DREAM with dynamic retrieval switched off and every other part intact: the
        arm any claim about `omega_t` has to be measured against."""
        memory = self._memory()
        weights = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1000.0,
        ).weights
        self.assertAlmostEqual(weights.experience, 0.5, places=3)
        self.assertAlmostEqual(weights.pattern, 0.5, places=3)

    def test_a_low_temperature_approaches_an_argmax(self):
        weights = self._weights(_fused(0, 0))
        sharp = retrieve(
            _fused(0, 0), self._memory(),
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=0.005,
        ).weights
        self.assertGreater(sharp.experience, weights.experience)
        self.assertGreater(sharp.experience, 0.99)

    def test_temperature_must_be_positive(self):
        """At 0 the softmax divides by zero; below it the weights INVERT and the level
        that matched WORST would be weighted highest, which is eq. 23 run backwards."""
        for temperature in (0.0, -1.0):
            with self.assertRaises(ValueError) as caught:
                retrieve(
                    _fused(0, 0), self._memory(),
                    k_experience=1, k_pattern=1, k_knowledge=1, temperature=temperature,
                )
            self.assertIn("run backwards", str(caught.exception))

    def test_temperature_has_no_default(self):
        with self.assertRaises(TypeError):
            retrieve(
                _fused(0, 0), self._memory(),
                k_experience=1, k_pattern=1, k_knowledge=1,
            )


class TestEq24(unittest.TestCase):
    """`K_t` (eq. 24), which does not close as written. See the module docstring."""

    def test_the_fused_context_is_unit_norm_and_in_the_query_space(self):
        memory = _memory([_experience(_fused(0, 0)), _experience(_fused(1, 1))])
        context = retrieve(
            _fused(0, 0), memory,
            k_experience=2, k_pattern=1, k_knowledge=1, temperature=1.0,
        )
        blended = context.fused_context()
        self.assertEqual(blended.shape, (VISUAL_WIDTH + AUDIO_WIDTH,))
        self.assertAlmostEqual(float(np.linalg.norm(blended)), 1.0, places=5)

    def test_a_level_with_more_rows_does_not_outweigh_one_with_fewer_by_count(self):
        """Each level is summarised by the MEAN of its retrieved keys before weighting.
        Weighting is `omega_t`'s job and nothing else's.

        **BUILT DIRECTLY RATHER THAN THROUGH `retrieve`, AND THAT IS THE POINT.** An
        end-to-end version of this test cannot see the difference: `M^P`'s signature is
        the CENTROID of the `M^E` rows it abstracts, so both levels always point the same
        way and a sum and a mean normalise to the same direction. The forced-failure run
        is what showed that -- swapping `.mean` for `.sum` left the whole suite green. The
        property only becomes observable when the two levels point differently, which is
        exactly what the hand-built context below arranges.
        """
        entry_a = _experience(_fused(0, 0))
        entry_c = _experience(_fused(1, 1))
        pattern = abstract(
            ExperienceStore().extend([
                _experience(_fused(2, 2), sound="snoring", obj="bed"),
                _experience(_fused(2, 2), sound="snoring", obj="bed"),
            ]),
            min_support=2,
        ).patterns[0]
        context = RetrievedContext(
            experience=(
                ExperienceHit(entry=entry_a, score=1.0),
                ExperienceHit(entry=entry_c, score=1.0),
            ),
            pattern=(PatternHit(pattern=pattern, score=1.0),),
            knowledge=None,
            weights=Weights(experience=0.5, pattern=0.5, knowledge=0.0),
        )
        expected = (
            0.5 * (entry_a.context + entry_c.context) / 2.0
            + 0.5 * pattern.strategy.signature
        )
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(context.fused_context(), expected, atol=1e-6)
        summed = 0.5 * (entry_a.context + entry_c.context) + 0.5 * pattern.strategy.signature
        summed = summed / np.linalg.norm(summed)
        self.assertFalse(
            np.allclose(expected, summed, atol=1e-6),
            "this fixture cannot tell a mean from a sum, so it proves nothing",
        )

    def test_the_knowledge_level_is_not_in_the_sum(self):
        """**THE SECOND PLACE THE PAPER'S ARITHMETIC DOES NOT CLOSE.** PR #111 found the
        first, in eq. 12. `K^K` is not a vector at all — retrieving from a sound-to-object
        table returns a CONCEPT — so it cannot be added to two fused vectors. It rides
        beside the sum with its own weight, and §III.G's `S_mem` names only experiences
        and patterns, which is the paper agreeing."""
        knowledge = _knowledge(
            ("toilet_flush", "bathroom", "toilet", [1.0, 0.0, 0.0]),
        )
        memory = _memory([_experience(_fused(0, 0))], knowledge=knowledge)
        context = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
        )
        self.assertIsInstance(context.knowledge, KnowledgeHit)
        self.assertGreater(context.weights.knowledge, 0.0)
        self.assertEqual(
            context.fused_context().shape, (VISUAL_WIDTH + AUDIO_WIDTH,),
            "an M^K key of a different width reached eq. 24's sum",
        )

    def test_a_knowledge_only_retrieval_has_no_fused_context(self):
        """The knowledge level contributes a concept and there is no vector to return,
        even though `omega_t` is defined and the retrieval did happen."""
        knowledge = _knowledge(
            ("toilet_flush", "bathroom", "toilet", [1.0, 0.0, 0.0]),
        )
        memory = LongTermMemory(
            experience=ExperienceStore(), pattern=PatternStore(), knowledge=knowledge
        )
        context = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
        )
        self.assertIsNotNone(context.weights)
        self.assertTrue(context)
        self.assertAlmostEqual(context.weights.knowledge, 1.0, places=6)
        self.assertIsNone(context.fused_context())

    def test_the_weighted_sum_leans_toward_the_level_that_matched(self):
        """Eq. 24's whole point: change `omega_t` and `K_t` moves with it."""
        experiences = [
            _experience(_fused(0, 0), sound="alarm", obj="fireplace"),
            _experience(_fused(1, 1), sound="alarm", obj="fireplace"),
        ]
        memory = _memory(experiences, min_support=2)
        sharp = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=0.01,
        )
        flat = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1000.0,
        )
        nearest = memory.experience.keys[0]
        self.assertGreater(
            float(np.dot(sharp.fused_context(), nearest)),
            float(np.dot(flat.fused_context(), nearest)),
            "sharpening omega_t did not move K_t toward the level it weighted up",
        )


class TestTheTypes(unittest.TestCase):
    def test_weights_report_as_a_triple_in_the_papers_order(self):
        weights = Weights(experience=0.5, pattern=0.3, knowledge=0.2)
        self.assertEqual(weights.as_tuple(), (0.5, 0.3, 0.2))

    def test_an_empty_context_is_falsey_and_a_filled_one_is_not(self):
        self.assertFalse(RetrievedContext(
            experience=(), pattern=(), knowledge=None, weights=None
        ))
        self.assertTrue(RetrievedContext(
            experience=(), pattern=(), knowledge=KnowledgeHit(category="toilet", score=0.9),
            weights=Weights(experience=0.0, pattern=0.0, knowledge=1.0),
        ))

    def test_scores_are_cosines_and_can_be_negative(self):
        """`min_support=99` empties `M^P` on purpose, so the one weight under test is not
        split with a pattern built from the same single experience."""
        memory = _memory([_experience(-_fused(0, 0))], min_support=99)
        self.assertEqual(len(memory.pattern), 0)
        context = retrieve(
            _fused(0, 0), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
        )
        self.assertAlmostEqual(context.experience[0].score, -1.0, places=5)
        self.assertAlmostEqual(context.weights.experience, 1.0, places=6)

    def test_the_query_must_have_a_direction(self):
        with self.assertRaises(ValueError):
            retrieve(
                np.zeros(VISUAL_WIDTH + AUDIO_WIDTH, dtype=np.float32),
                _memory([_experience(_fused(0, 0))]),
                k_experience=1, k_pattern=1, k_knowledge=1, temperature=1.0,
            )


class TestTheWholePath(unittest.TestCase):
    """Eq. 19 through 24 in one call chain, over memory the other modules built."""

    def test_an_observation_becomes_a_weighted_context(self):
        stm = ShortTermMemory(horizon=4)
        for index in range(4):
            visual = np.zeros(VISUAL_WIDTH, dtype=np.float32)
            visual[index % VISUAL_WIDTH] = 1.0
            audio = np.zeros(AUDIO_WIDTH, dtype=np.float32)
            audio[0] = 1.0
            stm = stm.push(StmEntry(
                visual=visual, audio=audio,
                pose=Pose(position=Xyz(float(index), 0.0, 0.0), yaw_rad=0.0),
                prev_action=None if index == 0 else "move_forward",
            ))
        memory = _memory(
            [
                _experience(_fused(0, 0), sound="alarm", obj="fireplace"),
                _experience(_fused(1, 0), sound="alarm", obj="fireplace"),
            ],
            knowledge=_knowledge(("alarm", "living_room", "fireplace", [1.0, 0.0, 0.0])),
            min_support=2,
        )
        q = query(stm.latest.fused, stm.context(decay=0.8), present_weight=0.7)
        context = retrieve(
            q, memory, k_experience=2, k_pattern=1, k_knowledge=1, temperature=0.5
        )
        self.assertTrue(context)
        self.assertAlmostEqual(sum(context.weights.as_tuple()), 1.0, places=6)
        self.assertEqual(context.knowledge.category, "fireplace")
        self.assertEqual(
            context.fused_context().shape, (VISUAL_WIDTH + AUDIO_WIDTH,)
        )
        self.assertFalse(math.isnan(context.experience[0].score))


if __name__ == "__main__":
    unittest.main()
