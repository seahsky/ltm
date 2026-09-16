"""`M^E` across invocations — the round trip, and what it must refuse.

`dream-1`'s readout measured the defect this file removes: all 19 scenes started from an
empty `M^E` because the sweep invokes `run()` once per scene. A memory that does not
survive the invocation cannot accumulate the two successes at one concept triple that
`abstract` needs, and `test_dream_omega_reach.py` shows `omega_t` is arithmetically inert
until `M^P` is non-empty.

The round trip is asserted on a memory built by the REAL constructors, not on a
hand-written dict: a serialiser tested against its own idea of the shape is a serialiser
that agrees with itself. And every refusal has its arm — a missing file, a wrong format
version, a truncated row — because a loader that silently returns an empty memory
reproduces the exact defect it was written to fix, and nothing downstream can tell the
difference.

No torch, no simulator.
"""

import json
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.memory.longterm import (
    ExperienceEntry,
    ExperienceStore,
    LongTermMemory,
    Outcome,
    abstract,
)
from earshot.memory.store import SemanticEntry, SemanticStore
from earshot.task.dream_store import (
    DREAM_MEMORY_FORMAT_VERSION,
    DreamMemoryError,
    dump_memory,
    load_memory,
    memory_summary,
    read_provenance,
)

MIN_SUPPORT = 2


def entry(*, sound="alarm", obj="fireplace", room=None, reached=True, gap=0.2,
          seed=0.0):
    return ExperienceEntry(
        context=np.asarray([1.0 + seed, 0.5 - seed, 0.25], dtype=np.float32),
        trajectory=np.asarray([4.0 + seed, 3.0, 0.75, 0.4], dtype=np.float32),
        target_concept=obj,
        outcome=Outcome(reached=reached, final_gap_m=gap),
        sound_concept=sound,
        room_concept=room,
    )


def memory(entries, *, knowledge=None):
    store = ExperienceStore().extend(entries)
    return LongTermMemory(
        experience=store,
        pattern=abstract(store, min_support=MIN_SUPPORT),
        knowledge=SemanticStore() if knowledge is None else knowledge,
    )


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def path(self, name="memory.json"):
        return str(pathlib.Path(self.root) / name)


class TestTheRoundTrip(Fixture):
    def test_every_field_of_every_row_survives(self):
        original = memory([
            entry(seed=0.0, room=None),
            entry(seed=0.1, sound="glass_break", obj="sink", room="kitchen",
                  reached=False, gap=7.25),
        ])
        path = self.path()

        dump_memory(path, original)
        restored = load_memory(path, min_support=MIN_SUPPORT)

        self.assertEqual(len(restored.experience), 2)
        for before, after in zip(original.experience.entries,
                                 restored.experience.entries):
            np.testing.assert_allclose(before.context, after.context, rtol=1e-6)
            np.testing.assert_allclose(before.trajectory, after.trajectory, rtol=1e-6)
            self.assertEqual(before.target_concept, after.target_concept)
            self.assertEqual(before.sound_concept, after.sound_concept)
            self.assertEqual(before.room_concept, after.room_concept)
            self.assertEqual(before.outcome.reached, after.outcome.reached)
            self.assertAlmostEqual(before.outcome.final_gap_m,
                                   after.outcome.final_gap_m, places=6)
        print("round trip: {}".format(memory_summary(restored)))

    def test_an_abstaining_room_comes_back_as_none_and_not_as_a_room(self):
        """`None` is the ORDINARY value — `NullRoomLabeler` always abstains.

        A round trip that turned it into "" or "unknown" would give `abstract` a third
        grouping key that looked like a room, and every pattern would be keyed on a
        label no labeller produced.
        """
        dump_memory(self.path(), memory([entry(room=None)]))
        restored = load_memory(self.path(), min_support=MIN_SUPPORT)

        self.assertIsNone(restored.experience.entries[0].room_concept)
        raw = json.loads(pathlib.Path(self.path()).read_text(encoding="utf-8"))
        self.assertIsNone(raw["experience"][0]["room_concept"])
        print("room_concept survives as null, not as a string")

    def test_M_P_is_rebuilt_from_M_E_and_is_not_read_from_the_file(self):
        """The pattern store has no independent state, so the file carries none.

        Two rows sharing a triple make a pattern at `min_support=2`; the same file read
        at `min_support=3` makes none. If `M^P` were persisted, the second read would
        return a pattern the caller's own knob forbids.
        """
        dump_memory(self.path(), memory([entry(seed=0.0), entry(seed=0.1)]))

        at_two = load_memory(self.path(), min_support=2)
        at_three = load_memory(self.path(), min_support=3)

        self.assertEqual(len(at_two.pattern), 1)
        self.assertEqual(len(at_three.pattern), 0)
        self.assertEqual(len(at_two.experience), len(at_three.experience))
        raw = json.loads(pathlib.Path(self.path()).read_text(encoding="utf-8"))
        self.assertNotIn("pattern", raw)
        print("min_support 2 -> {} pattern(s); 3 -> {}; the file holds no pattern key"
              .format(len(at_two.pattern), len(at_three.pattern)))

    def test_the_knowledge_level_is_an_input_and_never_a_round_trip(self):
        """Nothing in the DREAM path writes `M^K`, so the file has nothing to say."""
        seeded = SemanticStore(entries=(
            SemanticEntry(
                sound_class="alarm", room="kitchen", category="sink",
                embedding=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                donor_scene="sceneA",
            ),
        ))
        dump_memory(self.path(), memory([entry()], knowledge=seeded))

        default = load_memory(self.path(), min_support=MIN_SUPPORT)
        given = load_memory(self.path(), min_support=MIN_SUPPORT, knowledge=seeded)

        self.assertEqual(len(default.knowledge.entries), 0)
        self.assertEqual(len(given.knowledge.entries), 1)
        raw = json.loads(pathlib.Path(self.path()).read_text(encoding="utf-8"))
        self.assertNotIn("knowledge", raw)
        print("M^K: {} without a store, {} with one".format(
            len(default.knowledge.entries), len(given.knowledge.entries)))

    def test_an_empty_memory_round_trips_as_empty(self):
        dump_memory(self.path(), memory([]))
        restored = load_memory(self.path(), min_support=MIN_SUPPORT)
        self.assertEqual(len(restored.experience), 0)
        self.assertEqual(len(restored.pattern), 0)
        print("empty in, empty out: {}".format(memory_summary(restored)))


class TestTheChainAccumulates(Fixture):
    def test_three_scenes_in_sequence_build_one_memory(self):
        """The whole point, in the shape the driver runs it.

        Scene 1 writes, scene 2 restores and writes, scene 3 restores and writes. The
        assertion that matters is the last one: a pattern exists at the end that no
        single scene could have produced, because its two supporting experiences came
        from different scenes.
        """
        path = self.path()
        carried = memory([])
        for index, scene in enumerate(("sceneA", "sceneB", "sceneC")):
            if index:
                carried = load_memory(path, min_support=MIN_SUPPORT)
            grown = ExperienceStore().extend(
                list(carried.experience.entries) + [entry(seed=index * 0.1)]
            )
            carried = LongTermMemory(
                experience=grown,
                pattern=abstract(grown, min_support=MIN_SUPPORT),
                knowledge=SemanticStore(),
            )
            dump_memory(path, carried,
                        scenes=read_provenance(path) + (scene,))

        final = load_memory(path, min_support=MIN_SUPPORT)
        self.assertEqual(len(final.experience), 3)
        self.assertEqual(len(final.pattern), 1)
        self.assertEqual(read_provenance(path), ("sceneA", "sceneB", "sceneC"))
        print("3 scenes -> {} and provenance {}".format(
            memory_summary(final), read_provenance(path)))

    def test_one_scene_alone_makes_no_pattern_at_the_same_min_support(self):
        """THE CONTROL. Without the chain the same three episodes make nothing.

        This is `dream-1`: each scene's memory in isolation. If it also produced a
        pattern, the test above would prove nothing about carrying the memory over.
        """
        path = self.path("alone.json")
        dump_memory(path, memory([entry(seed=0.0)]))
        alone = load_memory(path, min_support=MIN_SUPPORT)

        self.assertEqual(len(alone.experience), 1)
        self.assertEqual(len(alone.pattern), 0)
        print("one scene alone: {}".format(memory_summary(alone)))

    def test_provenance_is_empty_before_the_first_write_and_never_raises(self):
        """The first scene of a chain has no provenance and that is not an error."""
        self.assertEqual(read_provenance(self.path("absent.json")), ())
        print("no file -> provenance ()")


class TestWhatTheLoaderRefuses(Fixture):
    def test_a_missing_file_raises_rather_than_returning_an_empty_memory(self):
        """THE REFUSAL THAT MATTERS MOST.

        An empty memory from a missing path is byte-identical downstream to the
        per-scene reset this module exists to remove, and no artefact afterwards could
        tell the two apart.
        """
        with self.assertRaises(DreamMemoryError) as caught:
            load_memory(self.path("nothing.json"), min_support=MIN_SUPPORT)
        self.assertIn("is NOT an empty memory", str(caught.exception))
        print(str(caught.exception))

    def test_a_wrong_format_version_is_refused_and_not_read_anyway(self):
        path = self.path()
        dump_memory(path, memory([entry()]))
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        payload["format_version"] = DREAM_MEMORY_FORMAT_VERSION + 1
        pathlib.Path(path).write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaises(DreamMemoryError) as caught:
            load_memory(path, min_support=MIN_SUPPORT)
        self.assertIn("format version", str(caught.exception))
        print(str(caught.exception))

    def test_a_truncated_row_names_the_row_and_the_missing_fields(self):
        """A short row is a broken write, never a row to be defaulted.

        A zeroed `h^av` would be a retrieval key pointing nowhere, and `ExperienceEntry`
        would accept it — the refusal has to be here.
        """
        path = self.path()
        dump_memory(path, memory([entry(), entry(seed=0.1)]))
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        del payload["experience"][1]["context"]
        del payload["experience"][1]["reached"]
        pathlib.Path(path).write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaises(DreamMemoryError) as caught:
            load_memory(path, min_support=MIN_SUPPORT)
        message = str(caught.exception)
        self.assertIn("row 1", message)
        self.assertIn("context", message)
        self.assertIn("reached", message)
        print(message)

    def test_unreadable_json_is_refused_with_the_path_named(self):
        path = self.path()
        pathlib.Path(path).write_text("{not json", encoding="utf-8")
        with self.assertRaises(DreamMemoryError) as caught:
            load_memory(path, min_support=MIN_SUPPORT)
        self.assertIn(path, str(caught.exception))
        print(str(caught.exception))

    def test_a_good_file_loads_so_the_refusals_are_not_the_only_outcome(self):
        """THE HEALTHY ARM. Four refusals above and nothing that proves it ever reads."""
        path = self.path()
        dump_memory(path, memory([entry(), entry(seed=0.1)]))
        restored = load_memory(path, min_support=MIN_SUPPORT)
        self.assertEqual(len(restored.experience), 2)
        print("the same loader reads a good file: {}".format(memory_summary(restored)))


class TestTheWriteIsAtomic(Fixture):
    def test_a_second_write_replaces_the_first_and_leaves_no_partial_file(self):
        """Overwriting is deliberate here; a leftover `.partial` is not."""
        path = self.path()
        dump_memory(path, memory([entry()]))
        dump_memory(path, memory([entry(), entry(seed=0.1), entry(seed=0.2)]))

        restored = load_memory(path, min_support=MIN_SUPPORT)
        self.assertEqual(len(restored.experience), 3)
        leftovers = sorted(
            child.name for child in pathlib.Path(self.root).iterdir()
            if child.name.endswith(".partial")
        )
        self.assertEqual(leftovers, [])
        print("overwrote to {} row(s), no .partial left behind".format(
            len(restored.experience)))

    def test_it_creates_the_parent_directory(self):
        path = str(pathlib.Path(self.root) / "made" / "up" / "memory.json")
        dump_memory(path, memory([entry()]))
        self.assertTrue(pathlib.Path(path).is_file())
        print("wrote into a directory that did not exist")


if __name__ == "__main__":
    unittest.main()
