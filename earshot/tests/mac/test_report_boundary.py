"""Invariant 2 — the agent's testimony cannot reach ground truth, by type.

**Armed by ticket 24.** ``report/agent.py`` and ``report/audit.py`` now exist, so the
``skipTest`` scaffolding is gone: a subject that disappears must fail here, not skip
green. The mechanism assertions below stay, because getting them wrong is a red build
on the box rather than on this Mac.

The leak ticket 18's requirement 10 wants closed is not hypothetical — it is the
current code. ``build_report`` (``anomaly_controller.py:302-316``) emits
``"source_xyz": ev.get("source_xyz")``, returns an untyped ``Dict[str, Any]``, and
mutates the state it was handed; ticket 10's "port near-verbatim" would have carried
all three in. "The controller cannot see ground truth" is not available as the rule,
because the oracle arm's controller legitimately holds ``source_xyz`` as its waypoint
while the task spec requires an identical schema in both arms. So the boundary is drawn
at the **report type**: ``AgentReport`` is frozen with exactly §5.1's nine fields, so
nothing privileged can appear in it whatever the controller holds.
"""

from __future__ import annotations

import dataclasses
import typing
import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401

# Task spec §5.1, verbatim. Written down here rather than imported so that ticket 24
# has something to be checked *against* — a test that reads the field list off the
# dataclass it is testing asserts only that the dataclass equals itself.
AGENT_REPORT_FIELDS = frozenset(
    {
        "primary_completed",
        "heard_at_step",
        "room",
        "anomaly_class",
        "stopped_at_pose",
        "visual_confirm_object",
        "investigate_aborted",
        "resumed",
        "n_benign_ignored",
    }
)

# The privileged set §5.2 puts in the audit record and nowhere else. Every name here
# must stay out of the testimony — and, since ticket 24 built the audit, every name here
# must also genuinely be ON it. A misspelled entry would pass the exclusion check while
# checking nothing, which is the inert-pin class this map keeps finding: ticket 17's
# constraint on a package nothing installs, ticket 20's exclusion of an absent directory.
PRIVILEGED_FIELDS = frozenset({"source_xyz", "dist_at_stop", "source_is_visible_history"})


@dataclasses.dataclass(frozen=True)
class _PostponedUnion:
    """A stand-in with the annotation shape ADR-0013 says 3.9 cannot resolve.

    Legal here only because of this module's ``from __future__ import annotations``:
    the annotation is stored as the source string and never evaluated at class
    creation. Evaluating it is what raises, which is the point.
    """

    heard_at_step: int | None = None
    room: str | None = None


class TestTheHarnessReadsDataclassFields(unittest.TestCase):
    """The mechanism, asserted on a fixture so it is real before ticket 24 lands."""

    def test_get_type_hints_raises_on_a_postponed_pep604_union(self):
        """Python 3.9 evaluates the annotation string, and ``int | None`` is a TypeError.

        This is why the disjointness check below reads ``__dataclass_fields__``. A
        harness written against ``get_type_hints()`` passes on any modern interpreter
        and dies on the box — the exact version-skew failure the interpreter refusal
        exists to prevent, arriving through a different door.
        """
        with self.assertRaises(TypeError):
            typing.get_type_hints(_PostponedUnion)

    def test_dataclass_fields_gives_the_names_without_resolving_them(self):
        self.assertEqual(
            set(_PostponedUnion.__dataclass_fields__),
            {"heard_at_step", "room"},
        )


class TestReportBoundary(unittest.TestCase):
    """The real assertion, armed: no skip path, so a missing subject is a red."""

    def _load(self, module_name, symbol):
        import importlib

        module = importlib.import_module("{}.report.{}".format(_tree.PACKAGE_NAME, module_name))
        return getattr(module, symbol)

    def test_agent_report_is_exactly_the_nine_fields(self):
        agent_report = self._load("agent", "AgentReport")
        self.assertEqual(set(agent_report.__dataclass_fields__), AGENT_REPORT_FIELDS)

    def test_agent_report_is_frozen(self):
        agent_report = self._load("agent", "AgentReport")
        self.assertTrue(agent_report.__dataclass_params__.frozen)

    def test_testimony_and_audit_field_names_are_disjoint(self):
        agent_report = self._load("agent", "AgentReport")
        audit = self._load("audit", "EpisodeAudit")
        overlap = set(agent_report.__dataclass_fields__) & set(audit.__dataclass_fields__)
        self.assertEqual(overlap, set(), "privileged names leaked into the testimony")

    def test_no_privileged_name_appears_in_the_testimony(self):
        agent_report = self._load("agent", "AgentReport")
        leaked = PRIVILEGED_FIELDS & set(agent_report.__dataclass_fields__)
        self.assertEqual(leaked, set())

    def test_every_privileged_name_is_genuinely_on_the_audit(self):
        """The pin above, armed. A name that is on neither type checks nothing.

        Reads fields *and* properties, because ``source_is_valid_history``'s real home
        is a property derived from the per-step rows — storing it twice would be a
        drift trap, and a field-only check would call the correct design a violation.
        """
        audit = self._load("audit", "EpisodeAudit")
        available = set(audit.__dataclass_fields__) | {
            name for name in dir(audit) if isinstance(getattr(audit, name, None), property)
        }
        self.assertEqual(
            PRIVILEGED_FIELDS - available,
            set(),
            "these are pinned as privileged but appear on neither report type, so the "
            "exclusion check above is enforcing nothing for them",
        )

    def test_report_agent_imports_nothing_that_can_supply_ground_truth(self):
        """A reviewer handed ``ep0000.agent.json`` must be unable to see the answer key.

        The type is the boundary, and this is the second half of it: a module that
        imports ``sim`` or ``task`` could construct a privileged value even with a
        frozen nine-field schema.
        """
        path = _tree.PACKAGE_ROOT / "report" / "agent.py"
        tree = _tree.parse(path)
        reached = {
            edge.target.split(".", 1)[0]
            for edge in _tree.intra_package_edges(path, tree)
        }
        self.assertEqual(
            reached & {"sim", "task", "agent"},
            set(),
            "report/agent.py may reach types and audio.guard only",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
