"""Invariant 2 — the agent's testimony cannot reach ground truth, by type.

``report/agent.py`` and ``report/audit.py`` land in ticket 24, so the disjointness
assertion below is conditional today. What is **not** deferred is the mechanism it has
to use, and that is asserted here and now, because getting it wrong is a red build on
the box rather than on this Mac.

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

# The privileged set §5.2 puts in the audit record and nowhere else. Not exhaustive —
# ticket 24 owns the full list — but every name here must stay out of the testimony.
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
    """The real assertion. Skips until ticket 24 builds the two modules."""

    def _load(self, module_name, symbol):
        path = _tree.PACKAGE_ROOT / "report" / (module_name + ".py")
        if not path.exists():
            self.skipTest("report/{}.py lands in ticket 24".format(module_name))
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

    def test_report_agent_imports_nothing_that_can_supply_ground_truth(self):
        """A reviewer handed ``ep0000.agent.json`` must be unable to see the answer key.

        The type is the boundary, and this is the second half of it: a module that
        imports ``sim`` or ``task`` could construct a privileged value even with a
        frozen nine-field schema.
        """
        path = _tree.PACKAGE_ROOT / "report" / "agent.py"
        if not path.exists():
            self.skipTest("report/agent.py lands in ticket 24")
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
