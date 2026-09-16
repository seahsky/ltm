"""HOW FAR `omega_t` CAN REACH — the algebra `dream-1` made worth writing down.

`dream-1` came back a null (92 vs 96 of 282, exact McNemar p = 0.6835) and the readout
said why in three numbers: `omega^E` median 1.0000, `omega^P` median 0.0000, `omega^K`
MAX 0.0000. Two of the three memory levels were empty all night. The obvious next move
was to wire `M^K` to the prior pass's semantic store and give eq. 23 a third logit.

**THAT MOVE WOULD HAVE CHANGED NOTHING, AND THIS FILE IS WHY.** `omega_t` reaches the
agent through exactly one path — `pick_plan` -> `score_plan` -> `memory_consistency` —
and that function renormalises `omega^E` and `omega^P` between themselves. A softmax over
three logits renormalised across the first two IS the softmax over those two, so the
knowledge level's share cancels exactly. `fused_context` is the only other consumer of
`omega` in the tree and nothing outside `memory/retrieve.py` calls it.

So the tests here pin what `omega` can and cannot move, because the alternative is
rediscovering it after another night on the box:

  * a knowledge hit cannot change `S_mem` at all;
  * with only `M^E` answering, `omega^E`'s VALUE cannot change `S_mem` either -- one
    level renormalised against itself is 1.0 whatever the softmax said;
  * with BOTH levels answering it bites, which is the arm that stops the two above from
    passing on a `memory_consistency` that ignores `omega` entirely.

The consequence for the experiment: `omega_t` influences behaviour only on episodes where
`M^P` is non-empty. `dream-1` had at most ONE pattern in any episode and none in the
median one, so the paper's central mechanism was inert on most of the run by arithmetic
rather than by weak effect.

No torch, no simulator.
"""

import pathlib
import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.proposers import SOURCE_FRONTIER, Candidate
from earshot.memory.longterm import (
    ExperienceEntry,
    ExperienceStore,
    Outcome,
    abstract,
)
from earshot.memory.retrieve import (
    ExperienceHit,
    KnowledgeHit,
    PatternHit,
    RetrievedContext,
    Weights,
)
from earshot.task.plan import memory_consistency
from earshot.types import Xyz

TEMPERATURE = 0.5
S_EXPERIENCE = 0.8
S_PATTERN = 0.6
S_KNOWLEDGE = 0.9


def softmax(*logits, temperature=TEMPERATURE):
    """Computed here rather than imported from `retrieve._softmax`.

    An independent derivation, so a defect in the shipped softmax cannot make these
    assertions agree with it and both be wrong.
    """
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    exponentiated = np.exp(scaled - scaled.max())
    return tuple(float(value) for value in exponentiated / exponentiated.sum())


def candidate(distance_m=2.0, geodesic_m=2.0):
    return Candidate(
        candidate_id=0,
        position=Xyz(float(distance_m), 0.0, 0.0),
        source=SOURCE_FRONTIER,
        distance_m=float(distance_m),
        bearing_rad=0.0,
        raw_score=0.5,
        geodesic_m=float(geodesic_m),
    )


def traj(path_length, straightness):
    return np.asarray([path_length, 0.0, straightness, 0.0], dtype=np.float32)


def entry(path_length=2.0, straightness=1.0):
    return ExperienceEntry(
        context=np.asarray([1.0, 0.0], dtype=np.float32),
        trajectory=traj(path_length, straightness),
        target_concept="fireplace",
        outcome=Outcome(reached=True, final_gap_m=0.2),
        sound_concept="alarm",
        room_concept=None,
    )


def pattern(path_length, straightness):
    store = ExperienceStore().extend([
        entry(path_length, straightness), entry(path_length, straightness)
    ])
    return abstract(store, min_support=2).patterns[0]


# The two levels must DISAGREE about the candidate, or a weighting between them has
# nothing to weight and every test below passes on a constant.
NEAR_EXPERIENCE = entry(path_length=2.0, straightness=1.0)
FAR_PATTERN = pattern(path_length=9.0, straightness=0.05)


def context(weights, *, knowledge=None):
    return RetrievedContext(
        experience=(ExperienceHit(entry=NEAR_EXPERIENCE, score=S_EXPERIENCE),),
        pattern=(PatternHit(pattern=FAR_PATTERN, score=S_PATTERN),),
        knowledge=knowledge,
        weights=weights,
    )


class TestTheKnowledgeLevelCannotMoveTheScore(unittest.TestCase):
    """`omega^K` cancels out of `S_mem` exactly. Seeding `M^K` is not a lever."""

    def test_a_third_live_level_leaves_s_mem_unchanged(self):
        two = softmax(S_EXPERIENCE, S_PATTERN)
        three = softmax(S_EXPERIENCE, S_PATTERN, S_KNOWLEDGE)

        without = memory_consistency(
            candidate(), context(Weights(experience=two[0], pattern=two[1],
                                         knowledge=0.0))
        )
        with_knowledge = memory_consistency(
            candidate(),
            context(
                Weights(experience=three[0], pattern=three[1], knowledge=three[2]),
                knowledge=KnowledgeHit(category="fireplace", score=S_KNOWLEDGE),
            ),
        )

        self.assertAlmostEqual(without[0], with_knowledge[0], places=12)
        self.assertTrue(without[1] and with_knowledge[1])
        print(
            "omega without M^K {:.4f}/{:.4f} -> S_mem {:.6f}\n"
            "omega with    M^K {:.4f}/{:.4f}/{:.4f} -> S_mem {:.6f}\n"
            "  the knowledge share ({:.1f}% of the mass) changed the score by {:.2e}"
            .format(
                two[0], two[1], without[0],
                three[0], three[1], three[2], with_knowledge[0],
                100.0 * three[2], abs(without[0] - with_knowledge[0]),
            )
        )

    def test_it_holds_however_much_mass_the_knowledge_level_takes(self):
        """Not an accident of one score. Swept across a knowledge logit that dominates."""
        two = softmax(S_EXPERIENCE, S_PATTERN)
        baseline = memory_consistency(
            candidate(), context(Weights(experience=two[0], pattern=two[1],
                                         knowledge=0.0))
        )[0]
        for logit in (0.0, 0.5, 1.0, 5.0, 50.0):
            three = softmax(S_EXPERIENCE, S_PATTERN, logit)
            score = memory_consistency(
                candidate(),
                context(
                    Weights(experience=three[0], pattern=three[1], knowledge=three[2]),
                    knowledge=KnowledgeHit(category="fireplace", score=logit),
                ),
            )[0]
            self.assertAlmostEqual(baseline, score, places=10, msg=str(logit))
            print("  omega^K {:.6f} -> S_mem {:.6f} (baseline {:.6f})".format(
                three[2], score, baseline))


class TestOmegaOnlyBitesWhenBothLevelsAnswer(unittest.TestCase):
    def test_with_only_M_E_the_weight_value_is_irrelevant(self):
        """One level renormalised against itself is 1.0 whatever the softmax said.

        This is the median `dream-1` episode: `M^P` empty, so `S_mem` is the experience
        agreement and `omega` is arithmetically absent from it.
        """
        scores = []
        for share in (0.34, 0.5, 0.9, 1.0):
            only_experience = RetrievedContext(
                experience=(ExperienceHit(entry=NEAR_EXPERIENCE,
                                          score=S_EXPERIENCE),),
                pattern=(),
                knowledge=None,
                weights=Weights(experience=share, pattern=1.0 - share,
                                knowledge=0.0),
            )
            scores.append(memory_consistency(candidate(), only_experience)[0])
        self.assertEqual(len(set(round(value, 12) for value in scores)), 1)
        print("omega^E in {} all give S_mem {:.6f}".format(
            (0.34, 0.5, 0.9, 1.0), scores[0]))

    def test_with_both_levels_the_weight_value_changes_the_score(self):
        """THE FORCED-FAILURE ARM.

        Without this, the two tests above would pass on a `memory_consistency` that
        ignored `omega` completely, and the file would prove nothing about `omega` at
        all -- only that this function is constant.
        """
        experience_heavy = memory_consistency(
            candidate(), context(Weights(experience=0.9, pattern=0.1, knowledge=0.0))
        )[0]
        pattern_heavy = memory_consistency(
            candidate(), context(Weights(experience=0.1, pattern=0.9, knowledge=0.0))
        )[0]

        self.assertNotAlmostEqual(experience_heavy, pattern_heavy, places=3)
        print("omega^E 0.9 -> S_mem {:.6f};  omega^E 0.1 -> S_mem {:.6f};  "
              "spread {:.6f}".format(
                  experience_heavy, pattern_heavy,
                  abs(experience_heavy - pattern_heavy)))

    def test_an_empty_pattern_level_reports_informed_on_the_experience_alone(self):
        """`informed` is about whether memory SPOKE, not about whether omega mattered.

        The distinction matters for reading `dream_informed_steps`: 263 of `dream-1`'s
        282 episodes were informed, and on most of them omega still could not act.
        """
        only_experience = RetrievedContext(
            experience=(ExperienceHit(entry=NEAR_EXPERIENCE, score=S_EXPERIENCE),),
            pattern=(),
            knowledge=None,
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        score, informed = memory_consistency(candidate(), only_experience)
        self.assertTrue(informed)
        self.assertGreater(score, 0.0)
        print("informed={} with M^P empty, S_mem {:.6f}".format(informed, score))


class TestEqTwentyFourReachesNoCaller(unittest.TestCase):
    """`fused_context` is implemented and nothing consumes it.

    Not a complaint about the implementation, which is correct and tested. It is a fact
    about the SYSTEM that a reader of `retrieve.py` would otherwise get wrong: eq. 24's
    fused vector does not feed the planner, so the knowledge concept and the weighted
    context blend reach the agent through no path at all.

    If you have just wired it into the runner, DELETE THIS TEST in the same commit --
    it has done its job.
    """

    def test_no_shipped_module_outside_retrieve_calls_it(self):
        root = pathlib.Path(__file__).resolve().parents[3]
        callers = []
        for path in sorted(root.glob("earshot/**/*.py")):
            relative = path.relative_to(root)
            if relative.parts[1] == "tests" or relative.name == "retrieve.py":
                continue
            if "fused_context" in path.read_text(encoding="utf-8"):
                callers.append(str(relative))
        self.assertEqual(
            callers, [],
            "fused_context now has caller(s) {} -- eq. 24 reaches the agent. Delete this "
            "test and update this module's docstring, which says it does not.".format(
                callers
            ),
        )
        print("eq. 24's fused_context: 0 caller(s) outside memory/retrieve.py")

    def test_the_search_would_find_a_caller_if_there_were_one(self):
        """THE CONTROL. A glob that matches nothing passes the test above vacuously.

        It earned its place on the first run: the walk was rooted at `parents[2]`, which
        is `earshot/`, so `glob("earshot/**/*.py")` matched no file and the test above
        passed over an empty search. This is the arm that said so.
        """
        root = pathlib.Path(__file__).resolve().parents[3]
        found = [
            str(path.relative_to(root))
            for path in sorted(root.glob("earshot/**/*.py"))
            if "memory_consistency" in path.read_text(encoding="utf-8")
            and path.relative_to(root).parts[1] != "tests"
        ]
        self.assertIn("earshot/task/plan.py", found)
        print("the same search finds memory_consistency in {}".format(found))


if __name__ == "__main__":
    unittest.main()
