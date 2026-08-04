"""The interpreter refusal cannot be opted out of by forgetting.

``unittest discover earshot/tests/mac`` sets ``top_level_dir`` to the start directory
and imports each ``test_*.py`` as a **top-level** module, so ``tests/mac/__init__.py``
is never executed and a refusal placed there would never run — verified against
CPython's ``TestLoader._find_tests``, not assumed. The refusal therefore lives in
``_interpreter.py``, which raises at import, and every test module imports it.

That leaves one gap: a new test file that simply does not import it would run happily
on 3.14. This closes it, structurally, in the same shape as the other invariants.
"""

import ast
import pathlib
import unittest

from _interpreter import assert_interpreter  # noqa: F401

SUITE_DIR = pathlib.Path(__file__).resolve().parent
GUARD_MODULE = "_interpreter"


class TestSuiteHygiene(unittest.TestCase):
    def test_every_test_module_imports_the_interpreter_guard(self):
        missing = []
        for path in sorted(SUITE_DIR.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = any(
                (isinstance(node, ast.ImportFrom) and node.module == GUARD_MODULE)
                or (
                    isinstance(node, ast.Import)
                    and any(a.name == GUARD_MODULE for a in node.names)
                )
                for node in ast.walk(tree)
            )
            if not imported:
                missing.append(path.name)
        self.assertEqual(
            missing,
            [],
            "these would run on any interpreter: {}. Add `from _interpreter import "
            "assert_interpreter  # noqa: F401`.".format(missing),
        )

    def test_the_suite_is_not_empty(self):
        """A glob that matched nothing would pass the test above without meaning it."""
        self.assertGreaterEqual(len(list(SUITE_DIR.glob("test_*.py"))), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
