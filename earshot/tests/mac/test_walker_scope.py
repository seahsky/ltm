"""The walker's scope is pinned, so widening it fails a test before it lands.

ADR-0014 asks for exactly this: "a test asserting the exemption set equals exactly
``{reference}``". It turns out to be three, and the pin is worth more for saying so
than it would have been for repeating the ADR — see ``_tree.NON_AGENT_ROOTS`` for the
two ``tests/`` cases that would have made the suite red on day one.

The direction of the default is the whole design. An allowlist was rejected because a
new ``earshot/experimental/`` would then be silently unchecked until someone remembered
to add it; with a denylist it is checked immediately and the person adding it has to
say, in a diff, why it should not be.
"""

import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401


class TestWalkerScope(unittest.TestCase):
    def test_the_non_agent_roots_are_exactly_these_three(self):
        self.assertEqual(
            set(_tree.NON_AGENT_ROOTS),
            {"reference", "tools", "tests"},
            "widening the walker's scope is a decision — record the reason beside the "
            "entry in _tree.NON_AGENT_ROOTS and update this pin deliberately",
        )

    def test_every_exclusion_carries_a_reason(self):
        for root, reason in _tree.NON_AGENT_ROOTS.items():
            self.assertTrue(reason.strip(), "{} is excluded with no reason given".format(root))

    def test_the_walk_actually_reaches_the_agent_tree(self):
        """A denylist that excluded everything would pass all three invariants silently.

        This is the vacuous-green check on the walker itself: the files the tree is
        known to contain today must be in scope.
        """
        found = {_tree.relative_path(p) for p in _tree.agent_python_files()}
        for expected in (
            "__init__.py",
            "types.py",
            "metrics.py",
            "audio/guard.py",
            # Ticket 21. `sim/world.py` most of all: it is the subject of the
            # one-importer invariant, so a walker that stopped reaching it would make
            # that test pass by finding nothing to check.
            "sim/world.py",
            "task/episodes.py",
            # Ticket 23, for the same reason one layer over: `test_analyst_only.py`
            # asserts that nothing outside the recording path names `sourceIsVisible`,
            # and `agent/controller.py` is the module that must not. A walker that
            # stopped reaching it would make that invariant pass by scanning nothing.
            "agent/controller.py",
        ):
            self.assertIn(expected, found)

    def test_the_excluded_roots_are_genuinely_excluded(self):
        found = {_tree.relative_path(p) for p in _tree.agent_python_files()}
        for rel in found:
            self.assertNotIn(rel.split("/", 1)[0], _tree.NON_AGENT_ROOTS, rel)

    def test_the_excluded_roots_all_exist(self):
        """An exclusion for a directory that is not there is an inert pin.

        Same class as ticket 17's constraint on a package that is never installed: it
        reports success while enforcing nothing, and nothing else would notice.
        """
        for root in _tree.NON_AGENT_ROOTS:
            self.assertTrue(
                (_tree.PACKAGE_ROOT / root).is_dir(),
                "{} is excluded from the walk but does not exist".format(root),
            )

    def test_reference_is_excluded_but_still_on_disk(self):
        """It is excluded three times by three mechanisms; this is the test one.

        Import is ``reference/__init__.py`` raising, lint is ruff's exclude, and this
        is the walker's. The duplication between the last two is accepted rather than
        engineered away — this one fails loudly if it drifts and ruff's does not, and
        reading ``ruff.toml`` from a 3.9 test would need a ``tomli`` dependency to
        check a one-line string.
        """
        self.assertTrue((_tree.PACKAGE_ROOT / "reference" / "memory" / "README.md").exists())
        self.assertIn("reference", _tree.NON_AGENT_ROOTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
