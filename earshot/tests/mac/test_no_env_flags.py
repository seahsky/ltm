"""Invariant 3 — ADR-0008's "no flag surface", made checkable rather than intended.

The old tree read ``LTM_REALIZABLE_LOCALIZATION`` from the environment at the runner,
and ``memory_bridge.py`` carries a dozen more ``LTM_*`` reads. The clean room carries
behaviour, not flags: the two surviving experimental arms are enums on ``RunConfig``,
built from ``argparse``, so a third option is addable without a flag explosion.

Two agent modules are allowed to touch ``os.environ``, and in both the environment is
the *subject* rather than the configuration: ``audio/guard.py`` (setting
``HABITAT_SIM_LOG`` IS ``pin_habitat_logging``) and ``env_check.py`` (reading the
resolved environment is its whole job).
"""

import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401


class TestNoEnvFlags(unittest.TestCase):
    def test_environment_is_read_only_where_it_is_the_subject(self):
        violations = []
        for path in _tree.agent_python_files():
            rel = _tree.relative_path(path)
            if rel in _tree.ENV_ACCESS_ALLOWED:
                continue
            for lineno, what in _tree.environ_accesses(_tree.parse(path)):
                violations.append("{}:{} — {}".format(rel, lineno, what))
        self.assertEqual(
            violations,
            [],
            "ADR-0008 removed the flag surface; configuration is RunConfig, from "
            "argparse. Offenders:\n  " + "\n  ".join(violations),
        )

    def test_the_allowlist_is_the_two_modules_the_adr_names(self):
        """Widening the allowlist has to be a visible diff, not a quiet third entry."""
        self.assertEqual(
            _tree.ENV_ACCESS_ALLOWED,
            frozenset({"audio/guard.py", "env_check.py"}),
        )

    def test_the_guards_access_is_the_pin_and_nothing_else(self):
        """The exemption is load-bearing, so check it is still spent on what earned it.

        ``pin_habitat_logging`` is the only reason ``guard.py`` is on the allowlist. If
        the module grows a second environment read, the exemption stops being about the
        pin and this is where that shows up.
        """
        path = _tree.PACKAGE_ROOT / "audio" / "guard.py"
        accesses = _tree.environ_accesses(_tree.parse(path))
        self.assertEqual(
            [what for _, what in accesses],
            ["os.environ"],
            "guard.py's exemption covers the HABITAT_SIM_LOG pin only; found {}".format(
                accesses
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
