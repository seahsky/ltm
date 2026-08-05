"""``import torch`` must precede ``import habitat_sim``, in the same scope, every time.

MEASURED on the box 2026-08-05 by
``.scratch/ss2-clean-room/probes/import_order_ladder.sh`` — six one-process cases, of
which the only green import is ``import torch, habitat_sim``:

    0 interpreter  GREEN      3 numpy        RED (free(): invalid pointer)
    1 bare         RED        4 torch        GREEN
    2 pinned       RED        5 earshot      RED

The failure is ``free(): invalid pointer``, exit 134. That is an abort, not an
exception: nothing is raised, nothing can catch it, and the process dies with no
Python-level diagnostic at all. ``env_check``'s honest "habitat_sim did not import"
branch is unreachable in this failure mode, which is why the constraint has to be held
here rather than reported at runtime.

Why a structural test and not a comment: until this file, the tree satisfied the
constraint **by accident**. ``assert_env()`` imports torch two probes before anything
reaches ``sim/world.py``, so every run through ``__main__`` happened to be fine while a
REPL, a box script, or a box test calling one probe aborted. Nothing stated the
dependency, so nothing would have noticed it being removed — and an ordering that only
holds because of the order two unrelated functions happen to be listed in is the
"written down and quietly stopped being true" shape ADR-0013's invariants exist to
stop. ``ruff`` cannot help: an unused ``import torch`` is exactly what ``F401`` exists
to delete, so the ``noqa`` on it is load-bearing and this test is what says why.

Scope note: this asserts our *source*, not the box. Whether torch-first still suffices
after an env rebuild is a box question, and ``tests/box/`` owns it. What a Mac can hold
is that nobody silently drops the line.
"""

import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401

ORDERING_DEPENDENCY = "torch"
SIMULATOR_PACKAGE = "habitat_sim"


class TestTorchPrecedesHabitatSim(unittest.TestCase):
    def test_every_habitat_import_has_torch_before_it_in_the_same_scope(self):
        """Same scope, because a torch import in another function is not a guarantee.

        ``env_check.py`` is the case that forces this: it imports torch inside
        ``probe_torch_min_version`` and habitat-sim inside
        ``probe_habitat_sim_audio_enum_member``. Reading that as satisfied would encode
        an ordering dependency between two functions with no call edge — and
        ``tests/box/`` runs the probes one at a time.
        """
        violations = []
        for relative in sorted(_tree.SIMULATOR_IMPORT_ALLOWED):
            tree = _tree.parse(_tree.PACKAGE_ROOT / relative)
            torch_first = {}
            for owner, lineno in _tree.module_imports_by_function(
                tree, ORDERING_DEPENDENCY
            ):
                if owner not in torch_first or lineno < torch_first[owner]:
                    torch_first[owner] = lineno
            for owner, lineno in _tree.module_imports_by_function(
                tree, SIMULATOR_PACKAGE
            ):
                guard = torch_first.get(owner)
                if guard is None or guard >= lineno:
                    violations.append(
                        "{}:{} imports {} in {} with no preceding `import {}`".format(
                            relative, lineno, SIMULATOR_PACKAGE, owner,
                            ORDERING_DEPENDENCY,
                        )
                    )
        self.assertEqual(
            violations,
            [],
            "bare `import habitat_sim` aborts the box interpreter with `free(): invalid "
            "pointer` (exit 134, nothing raised). Put `import torch  # noqa: F401` "
            "immediately before it, in the same scope: {}".format(violations),
        )

    def test_the_allowlist_it_walks_is_not_empty(self):
        """A path glob that matched nothing would pass the test above without meaning it."""
        self.assertGreaterEqual(len(_tree.SIMULATOR_IMPORT_ALLOWED), 2)

    def test_every_allowlisted_file_actually_imports_habitat_sim(self):
        """Otherwise a file could drop its habitat import and still satisfy the walk.

        The ordering test is vacuous for a file with nothing to order. This is the
        companion that keeps the denominator honest — the same reason
        ``test_suite_hygiene`` asserts the suite is not empty.
        """
        without = []
        for relative in sorted(_tree.SIMULATOR_IMPORT_ALLOWED):
            tree = _tree.parse(_tree.PACKAGE_ROOT / relative)
            if not _tree.module_imports_by_function(tree, SIMULATOR_PACKAGE):
                without.append(relative)
        self.assertEqual(
            without,
            [],
            "allowlisted to import habitat_sim and does not: {}. Either the import "
            "moved (and the ordering guard moved with it) or the entry is stale".format(
                without
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
