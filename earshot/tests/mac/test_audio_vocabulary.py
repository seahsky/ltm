"""The sounding class vocabulary's invariants, and the fence around the sound-object mapping.

`CONTEXT.md` says the **sound-object mapping** is placement ground truth and analyst-only:
handing `anchor_object` to the agent turns the unseen-and-heard cell of the generalization
matrix into a measurement of the author's table rather than the agent's semantic store.
ADR-0018 records the decision; this file is the enforcement.

The fence covers two prefixes now, not one: `agent/` (the controller) and `memory/`
(ADR-0018's semantic store). Both are named in `VOCABULARY_FENCED_PREFIXES` below, and
`memory/` gets an extra AST-level test the import check alone cannot give it — see
`test_no_memory_module_names_the_placement_table`.

ADR-0013's layer graph already forbids `agent/` importing `audio/` at all, so the naive leak
cannot compile. The leak that CAN happen is the wiring layer reading the anchor and passing
it in, which no layer rule catches. So the fence here is an ALLOWLIST of call sites, in the
shape `test_layering.py` uses: a new caller is a decision someone makes on purpose, not one
that happens because the anchor was convenient to reach.
"""

import ast
import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.vocabulary import (
    ABSENT_CLASSES,
    AFFINITY_GRADES,
    CANDIDATE_VOCABULARY,
    HM3D_GOAL_CATEGORIES,
    ROOM_OF_ANCHOR,
    ROOMS,
    SoundClass,
    anchor_object,
    by_affinity,
    class_names,
    prompt_of,
    prompts,
    room_of,
)

# Who may read the placement mapping. The dataset builder places sources with it, the gate
# reports affinity beside recall, and the analysis tools read it off finished runs. Nothing
# in `agent/` is here and nothing in `agent/` may be added.
# `_tree.relative_path` is relative to `earshot/`, NOT to the repo root. Spelling these
# with an `earshot/` prefix made the sibling scan below match nothing and pass vacuously,
# which is the failure mode CLAUDE.md calls red. Both tests now assert they visited files.
ANCHOR_CALLERS_ALLOWED = (
    "audio/vocabulary.py",
    "task/dataset.py",
    "task/clap_gate.py",
)

# The prefixes that may not reach the placement table by any route. `memory/` joins
# `agent/` because the semantic store's whole claim is that it LEARNED the association;
# a store that can read ROOM_OF_ANCHOR measures the author's table instead.
VOCABULARY_FENCED_PREFIXES = ("agent/", "memory/")

# The same three strings, however they are reached: the exact name, and the two ways of
# naming it that the AST walker's helpers still see through a hand-edit --
# `code_string_constants` catches `getattr(vocabulary, "room_of")`, and
# `attribute_names` catches `vocabulary.room_of(...)` written as a real attribute.
PLACEMENT_TABLE_NAMES = ("ROOM_OF_ANCHOR", "anchor_object", "room_of")


class TestVocabularyTable(unittest.TestCase):
    def test_every_anchor_is_an_hm3d_goal_category(self):
        """A source can only be placed at an object the ObjectNav episode JSON carries."""
        for entry in CANDIDATE_VOCABULARY:
            self.assertIn(
                entry.anchor_object,
                HM3D_GOAL_CATEGORIES,
                "{} anchors at {!r}, which the builder cannot find".format(
                    entry.name, entry.anchor_object
                ),
            )

    def test_every_anchor_resolves_to_a_room(self):
        """An object with no room cannot carry a class: there is nothing to learn at it.

        This is what retired `plant`. A houseplant has no characteristic sound, and grouping
        did not rescue it -- `greenery` scored the identical 0.383 the object taxonomy gave
        `plant`, because the map was one-to-one. The classes were misassigned, not mis-grouped.
        """
        for entry in CANDIDATE_VOCABULARY:
            self.assertIn(entry.anchor_object, ROOM_OF_ANCHOR, entry.name)
            self.assertIn(room_of(entry.name), ROOMS, entry.name)

    def test_every_affinity_is_a_declared_grade(self):
        for entry in CANDIDATE_VOCABULARY:
            self.assertIn(entry.room_affinity, AFFINITY_GRADES, entry.name)

    def test_class_names_are_unique(self):
        names = list(class_names())
        self.assertEqual(
            len(names), len(set(names)), "duplicate class name(s): {}".format(names)
        )

    def test_absent_classes_are_disjoint_from_the_vocabulary(self):
        """The forced-failure arm is vacuous if an absent class is in the prompt bank.

        This is the `anommxv` failure in its structural form: a gate that rejected 0 of 8
        looks identical to a gate whose negatives were never negative.
        """
        overlap = sorted(set(ABSENT_CLASSES) & set(class_names()))
        self.assertEqual(overlap, [], "absent class(es) also in the vocabulary: {}".format(overlap))

    def test_the_prompt_bank_exposes_no_anchor(self):
        """The agent may hold the prompts. It may not hold the TABLE.

        A prompt that happens to name its object ("a toilet flushing") is NOT a leak, and
        this test deliberately does not forbid it. Language already relates a flush to a
        toilet, and an agent reading only the class name is exactly the not-heard column's
        intended baseline: with no LTM entry, all it has left is the language prior. What
        the heard column then measures is what EXPERIENCE adds over that prior, which is a
        sharper question than "does the agent know what a toilet is".

        What must never reach the agent is the resolved mapping itself, so the invariant is
        structural: `prompts()` is name-to-text and carries no anchor, in its keys or its
        values, as a dict a controller could invert.
        """
        bank = prompts()
        self.assertEqual(sorted(bank), sorted(class_names()))
        for entry in CANDIDATE_VOCABULARY:
            self.assertNotIn(
                entry.anchor_object,
                bank,
                "{!r} appears as a KEY of the prompt bank — that is the mapping, "
                "inverted".format(entry.anchor_object),
            )
        self.assertTrue(
            all(isinstance(value, str) for value in bank.values()),
            "the prompt bank must be flat text; a nested value could carry the anchor",
        )

    def test_every_grade_has_at_least_one_class(self):
        """A grade with no members makes the strong-versus-weak breakdown unreadable."""
        for grade in AFFINITY_GRADES:
            self.assertTrue(by_affinity(grade), "no class at affinity {!r}".format(grade))

    def test_prompt_of_and_anchor_object_raise_on_an_unknown_class(self):
        """Raising, not defaulting — a plausible default is this map's recurring failure."""
        with self.assertRaises(KeyError):
            prompt_of("not_a_class")
        with self.assertRaises(KeyError):
            anchor_object("not_a_class")

    def test_a_bad_row_raises_at_construction(self):
        with self.assertRaises(ValueError):
            SoundClass("x", "x", "a sound", "microwave", "strong")
        with self.assertRaises(ValueError):
            SoundClass("x", "x", "a sound", "toilet", "very strong")
        # An HM3D category that resolves to no room. `plant` is the real case.
        with self.assertRaises(ValueError):
            SoundClass("x", "x", "a sound", "plant", "strong")

    def test_the_vocabulary_is_generous_enough_to_prune(self):
        """The candidate set exists to be CUT. Too small and the gate has nothing to say."""
        self.assertGreaterEqual(
            len(CANDIDATE_VOCABULARY),
            12,
            "a candidate set this small cannot support a heard/not-heard class split "
            "after the gate has pruned it",
        )


class TestAnchorFence(unittest.TestCase):
    """The sound-object mapping must not reach the agent, by any route."""

    def test_no_agent_module_imports_the_vocabulary(self):
        """Belt to the layer graph's braces, and greppable by the name of the rule.

        Generalised over `VOCABULARY_FENCED_PREFIXES`, with the vacuousness guard kept
        PER PREFIX: a typo in `"memory/"` must fail loudly by scanning nothing for that
        prefix, rather than being masked by `agent/` alone having files to scan.
        """
        offenders = []
        for prefix in VOCABULARY_FENCED_PREFIXES:
            scanned = 0
            for path in _tree.agent_python_files():
                if not _tree.relative_path(path).startswith(prefix):
                    continue
                scanned += 1
                if _tree.imports_module(_tree.parse(path), "earshot.audio.vocabulary"):
                    offenders.append(_tree.relative_path(path))
            self.assertGreater(
                scanned, 0, "the fence scanned no {} module — vacuous".format(prefix)
            )
        self.assertEqual(
            offenders,
            [],
            "fenced module(s) importing the vocabulary: {}".format(offenders),
        )

    def test_no_memory_module_names_the_placement_table(self):
        """The AST-level half: `memory/` may not even NAME the table, call or no call.

        `test_no_agent_module_imports_the_vocabulary` above catches the import; this
        catches the harder-to-see case of the string or attribute name reaching a
        `memory/` file some other way (`getattr(vocabulary, "room_of")`,
        `earshot.audio.vocabulary.ROOM_OF_ANCHOR` copied inline) — the same two helpers
        `test_analyst_only.py` uses for the `sourceIsVisible` fence.
        """
        offenders = []
        scanned = 0
        for path in _tree.agent_python_files():
            relative = _tree.relative_path(path)
            if not relative.startswith("memory/"):
                continue
            scanned += 1
            tree = _tree.parse(path)
            for lineno, name in _tree.code_string_constants(tree):
                if name in PLACEMENT_TABLE_NAMES:
                    offenders.append("{}:{} — {!r}".format(relative, lineno, name))
            for lineno, name in _tree.attribute_names(tree):
                if name in PLACEMENT_TABLE_NAMES:
                    offenders.append("{}:{} — .{}".format(relative, lineno, name))
        self.assertGreater(scanned, 0, "the fence scanned no memory/ module — vacuous")
        self.assertEqual(
            offenders,
            [],
            "memory/ module(s) naming the placement table {}:\n{}".format(
                PLACEMENT_TABLE_NAMES, "\n".join(offenders)
            ),
        )

    def test_anchor_object_is_called_only_from_the_allowlist(self):
        """A new caller is a decision. Add it here with a reason, or do not add it."""
        offenders = []
        scanned = 0
        for path in _tree.agent_python_files():
            relative = _tree.relative_path(path)
            if relative in ANCHOR_CALLERS_ALLOWED:
                continue
            scanned += 1
            tree = _tree.parse(path)
            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                if name == "anchor_object":
                    offenders.append("{}:{}".format(relative, getattr(node, "lineno", 0)))
        self.assertGreater(scanned, 0, "the fence scanned no module outside the allowlist — vacuous")
        self.assertEqual(
            offenders,
            [],
            "sound-object mapping read outside the allowlist {}:\n{}".format(
                list(ANCHOR_CALLERS_ALLOWED), "\n".join(offenders)
            ),
        )


if __name__ == "__main__":
    unittest.main()
