"""§3.3's oracle boundary, as a structural check rather than a docstring.

``sourceIsVisible()`` is computed from the **ground-truth source position**. It is free,
it is the best diagnostic there is for why a gradient climb stalled, and §3.2 records it
at every step — but feeding it to the decision rule plants a hidden oracle inside the
realizable arm, which is the one thing ADR-0011 exists to avoid.

Nothing would catch that at runtime. The realizable arm would still climb, its report
would still validate against §5.1's schema, and the arm would silently stop being
realizable. ``report/agent.py``'s frozen field list (ticket 24) stops the *value*
leaving in the testimony; this stops the controller *reading* it in the first place.

**Armed before its subject exists**, in the shape ticket 21 left ``test_layering.py``:
that one asserted the ``habitat_sim`` importers were a *subset* of ``{sim/world.py}``
while the file was still to be written, so it fired the moment a second file reached for
the simulator. ``agent/`` is ticket 23's, so the scan below has nothing to scan yet —
which is exactly why the second test exists: it pins that the name being searched for is
still the name the sensor exposes, so a rename cannot turn this file into a check that
passes by looking for nothing.
"""

import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401

# The two spellings: the binding's method and the wrapper's. A string constant counts
# too — `getattr(sensor, "sourceIsVisible")` is the same reach with the AST hidden.
ORACLE_NAMES = ("sourceIsVisible", "source_is_visible")

# Where the oracle legitimately appears. `audio/sensor.py` and `audio/guard.py` expose
# and record it; `report/` and `task/` carry it into the audit record, which is the
# privileged artefact it belongs in (§5.2). `agent/` is on nobody's list.
ORACLE_ALLOWED_PREFIXES = ("audio/", "report/", "task/")


def _oracle_reaches(path):
    tree = _tree.parse(path)
    hits = [
        (lineno, name)
        for lineno, name in _tree.attribute_names(tree)
        if name in ORACLE_NAMES
    ]
    hits += [
        (lineno, value)
        for lineno, value in _tree.code_string_constants(tree)
        if value in ORACLE_NAMES
    ]
    return hits


class TestTheControllerCannotSeeTheOracle(unittest.TestCase):
    def test_no_module_outside_the_recording_path_reads_source_visibility(self):
        violations = []
        for path in _tree.agent_python_files():
            rel = _tree.relative_path(path)
            if rel.startswith(ORACLE_ALLOWED_PREFIXES):
                continue
            for lineno, name in _oracle_reaches(path):
                violations.append("{}:{} — {}".format(rel, lineno, name))
        self.assertEqual(
            violations,
            [],
            "§3.3: sourceIsVisible() is computed from the ground-truth source position "
            "and is analyst-only. A controller that reads it is an oracle arm wearing "
            "the realizable arm's report schema, and nothing downstream would say so. "
            "Offenders:\n  " + "\n  ".join(violations),
        )

    def test_the_name_this_searches_for_is_still_the_one_the_sensor_exposes(self):
        """Otherwise a rename turns the test above into a search for nothing.

        Docstrings are excluded from ``code_string_constants``, so the citations in
        ``sensor.py``'s prose do not count — this is the real method call.
        """
        found = _oracle_reaches(_tree.PACKAGE_ROOT / "audio" / "sensor.py")
        self.assertTrue(
            found,
            "audio/sensor.py names neither {} — either the sensor stopped exposing it "
            "or it was renamed, and the analyst-only check above is now vacuous".format(
                " nor ".join(ORACLE_NAMES)
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
