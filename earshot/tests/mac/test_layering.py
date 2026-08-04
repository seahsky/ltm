"""Invariant 1 — ADR-0013's layer graph, and the one file that may touch the simulator.

Structural, not fake-based: this parses the real ``earshot/`` source, so it cannot go
vacuous the way a suite of doubles can. It is the enforcement ADR-0013 chose *instead
of* documenting the graph, on this repo's record — ticket 14's self-update gotcha,
ticket 17's inert pin and ticket 13's version-blind skip were all written down and then
quietly stopped being true.

``import habitat_sim`` in a second file is meant to be a decision someone makes on
purpose, not one that happens because a depth frame was convenient to reach.
"""

import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401


class TestLayerGraph(unittest.TestCase):
    def test_every_agent_module_is_in_the_graph(self):
        """A new top-level package is checked by default — that is the denylist shape.

        Failing here means either the module belongs in ADR-0013's graph (add the edge
        and say why) or it is not agent code (add it to ``NON_AGENT_ROOTS`` with its
        reason). Both are decisions; neither is a default.
        """
        unknown = []
        for path in _tree.agent_python_files():
            if _tree.layer_key(path) not in _tree.LAYER_IMPORTS:
                unknown.append(_tree.relative_path(path))
        self.assertEqual(
            unknown,
            [],
            "module(s) governed by no layer in ADR-0013's graph: {}".format(unknown),
        )

    def test_only_the_graphs_edges_exist(self):
        violations = []
        for path in _tree.agent_python_files():
            key = _tree.layer_key(path)
            allowed = _tree.LAYER_IMPORTS.get(key)
            if allowed is None:
                continue  # reported by the test above
            for edge in _tree.intra_package_edges(path, _tree.parse(path)):
                if not _tree.edge_allowed(edge.target, allowed):
                    violations.append(
                        "{}:{} — {} (layer {!r} may import {})".format(
                            _tree.relative_path(path),
                            edge.lineno,
                            edge.raw,
                            key,
                            list(allowed) or "nothing intra-package",
                        )
                    )
        self.assertEqual(violations, [], "\n".join([""] + violations))

    def test_audio_never_imports_sim(self):
        """One of the two backwards edges ticket 18 rejected, named so it is greppable.

        ``audio/`` reaching into ``sim/`` for the simulator handle is the convenient
        move that would put ``import habitat_sim`` in a second file by accident.
        """
        self._assert_no_edge_from("audio", "sim")

    def test_agent_never_imports_sim(self):
        """The other one: ``agent/`` reaching into ``sim/`` for a depth frame."""
        self._assert_no_edge_from("agent", "sim")

    def _assert_no_edge_from(self, layer, forbidden):
        offenders = []
        for path in _tree.agent_python_files():
            if _tree.layer_key(path) != layer:
                continue
            for edge in _tree.intra_package_edges(path, _tree.parse(path)):
                if edge.target == forbidden or edge.target.startswith(forbidden + "."):
                    offenders.append(
                        "{}:{} — {}".format(
                            _tree.relative_path(path), edge.lineno, edge.raw
                        )
                    )
        self.assertEqual(
            offenders,
            [],
            "{}/ must not import {}/ (ADR-0013): {}".format(layer, forbidden, offenders),
        )


class TestSimulatorImportIsUnique(unittest.TestCase):
    def test_habitat_sim_is_imported_in_exactly_the_designated_modules(self):
        """``import habitat_sim`` appears in exactly the pinned set.

        An **equality** since ticket 21 landed ``sim/world.py``. It was a subset until
        then, which caught a second importer but would also have passed over the module
        going missing — and ADR-0013's claim is that the simulator has a designated
        door, not at most one. Both directions fail: an unlisted file reaching for the
        simulator, and a listed one ceasing to be the file that owns it (renamed,
        split, or its import moved behind a lazy helper somewhere else).

        Ticket 24 made it two. See ``_tree.SIMULATOR_IMPORT_ALLOWED`` for why the audio
        enum probe cannot be reached any other way — and for the dynamic-import dodge
        that was rejected rather than taken.
        """
        importers = sorted(
            _tree.relative_path(path)
            for path in _tree.agent_python_files()
            if _tree.imports_module(_tree.parse(path), "habitat_sim")
        )
        self.assertEqual(
            importers,
            sorted(_tree.SIMULATOR_IMPORT_ALLOWED),
            "only {} may import habitat_sim (ADR-0013); found {}".format(
                sorted(_tree.SIMULATOR_IMPORT_ALLOWED), importers
            ),
        )

    def test_the_allowlist_is_the_two_the_adr_licenses(self):
        """Widening it has to be a visible diff carrying a reason, not a quiet third."""
        self.assertEqual(
            set(_tree.SIMULATOR_IMPORT_ALLOWED),
            {"sim/world.py", "env_check.py"},
        )
        for module, reason in _tree.SIMULATOR_IMPORT_ALLOWED.items():
            self.assertTrue(reason.strip(), "{} is exempt with no reason given".format(module))

    def test_env_checks_exemption_is_still_spent_on_the_enum_probe(self):
        """The same discipline ``test_no_env_flags`` applies to ``guard.py``'s pin.

        Named function rather than a count. A **top-level** ``import habitat_sim`` here
        would be the real regression: it would fire on every import of the module,
        including from this Mac, and would turn a probe that reports NOT_RUN into an
        ``ImportError`` at the entry point of a tree that is otherwise fine.
        """
        path = _tree.PACKAGE_ROOT / "env_check.py"
        owners = sorted(name for name, _ in _tree.module_imports_by_function(
            _tree.parse(path), "habitat_sim"
        ))
        self.assertEqual(
            owners,
            ["probe_habitat_sim_audio_enum_member"],
            "env_check's habitat_sim exemption covers the enum member probe only; "
            "found it in {}".format(owners),
        )

    def test_nothing_imports_habitat_lab(self):
        """habitat-lab is deliberately not installed — the runner drives habitat_sim.

        A stray ``import habitat`` would not fail on the Mac (nothing here imports the
        tree) and would fail on the box only at the moment it ran.
        """
        importers = sorted(
            _tree.relative_path(path)
            for path in _tree.agent_python_files()
            if _tree.imports_module(_tree.parse(path), "habitat")
        )
        self.assertEqual(importers, [], "habitat-lab is not a dependency: {}".format(importers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
