"""Do ``tests/box/`` calls into the tree match the signatures they will meet? Statically.

    python -m unittest discover earshot/tests/mac

**Written because a box trip was spent on a missing argument.** ``tests/box/`` is the only
code in this tree that nothing on a Mac imports, runs, or reads — the box suite needs
``habitat_sim``, so it is invisible to CI, invisible to the layering walker, and its calls
into ``earshot`` are checked by nothing until the box runs them. ``World(scene_path)`` sat
in ``test_investigate_route_box.py`` through a green Mac suite, a green lint and a merge,
and cost a round trip on the V100 to find that ``World.__init__`` takes ``sensor_specs``
too.

This is ADR-0014's **structural** layer applied to the box suite: an ``ast`` read of the
real subject, which is the one Mac layer that does not go vacuous when the fake drifts. It
checks arity only — that a call passes at least the required positional arguments and no
unknown keywords — because that is what a static read can know for certain and it is
exactly the class of error that costs a trip. Types, ordering and semantics remain the
box's to find.

Deliberately not a lint rule: ``ruff`` is scoped to ``F``+``E9`` on this tree (ticket 19),
and cross-module signature resolution is not something it does.

**The signatures are read with ``ast``, not with ``inspect``.** The first version imported
``World`` and asked ``inspect.signature`` — and could not, because ``sim/world.py`` imports
``torch`` at module scope (the torch-before-habitat_sim ordering PR #32 made explicit) and
the ``earshot-mac`` env has no torch. Which is the same lesson one layer up: the box suite's
subjects are exactly the ones a Mac cannot load, so a check that needs to import them is a
check that cannot run. Ticket 23 hit this and answered it the same way, reading
``camera_sensor_specs``' defaults out of the source because ``agent/`` may not import it.
"""

import ast
import pathlib
import unittest

from _interpreter import assert_interpreter  # noqa: F401
from _tree import parse, relative_path

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOX_DIR = ROOT / "tests" / "box"

# name at the call site -> (defining file, class or None, function). A named list rather
# than a sweep on purpose: a sweep would silently skip whatever it could not resolve and
# read as broader coverage than it has.
CHECKED = {
    "World": ("sim/world.py", "World", "__init__"),
    "camera_sensor_specs": ("sim/world.py", None, "camera_sensor_specs"),
    "apply_audio_config": ("audio/guard.py", None, "apply_audio_config"),
}


def _definition(name):
    """The ``ast`` node for a CHECKED target, found in its own source."""
    relative, class_name, func_name = CHECKED[name]
    tree = parse(ROOT / relative)
    scope = tree
    if class_name is not None:
        scope = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    return next(
        node
        for node in ast.walk(scope)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == func_name
    )


def _required_positional(name):
    """Positional parameters the callable will not supply for itself."""
    args = _definition(name).args
    positional = [a.arg for a in list(args.posonlyargs) + list(args.args)]
    if positional and positional[0] == "self":
        positional = positional[1:]
    defaulted = len(args.defaults)
    return positional[: len(positional) - defaulted] if defaulted else positional


def _accepts(name):
    args = _definition(name).args
    names = {
        a.arg
        for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    }
    return names - {"self"}, args.kwarg is not None


def _calls_in(path):
    """``(name, node)`` for every direct call to a CHECKED name in one box file."""
    tree = parse(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in CHECKED:
                yield node.func.id, node


class TestBoxTestsCallRealSignatures(unittest.TestCase):
    def test_there_are_box_files_to_check(self):
        """An empty sweep passes vacuously, which is the failure this file exists about."""
        self.assertTrue(list(BOX_DIR.glob("test_*_box.py")))

    def test_at_least_one_checked_call_is_actually_found(self):
        """A rename in CHECKED would otherwise turn this whole module into a no-op."""
        found = [
            name for path in BOX_DIR.glob("test_*_box.py") for name, _ in _calls_in(path)
        ]
        self.assertTrue(found, "no CHECKED call found in any box test — the names drifted")

    def test_every_call_supplies_the_required_positional_arguments(self):
        problems = []
        for path in sorted(BOX_DIR.glob("test_*_box.py")):
            for name, node in _calls_in(path):
                required = _required_positional(name)
                supplied = len(node.args) + sum(
                    1 for kw in node.keywords if kw.arg in required
                )
                if any(kw.arg is None for kw in node.keywords):
                    continue  # `**kwargs` at the call site; a static read cannot count it
                if supplied < len(required):
                    problems.append(
                        "{}:{} {}() needs {} ({}), got {}".format(
                            relative_path(path), node.lineno, name,
                            len(required), ", ".join(required), supplied,
                        )
                    )
        self.assertEqual(
            problems,
            [],
            "a box test calls a signature it will not meet — this is a wasted V100 "
            "trip, found statically:\n  " + "\n  ".join(problems),
        )

    def test_no_call_passes_a_keyword_the_target_does_not_take(self):
        problems = []
        for path in sorted(BOX_DIR.glob("test_*_box.py")):
            for name, node in _calls_in(path):
                names, var_keyword = _accepts(name)
                if var_keyword:
                    continue
                for kw in node.keywords:
                    if kw.arg is not None and kw.arg not in names:
                        problems.append(
                            "{}:{} {}() has no parameter {!r}".format(
                                relative_path(path), node.lineno, name, kw.arg
                            )
                        )
        self.assertEqual(problems, [], "\n  " + "\n  ".join(problems))


class TestTheCheckCatchesThePlantItWasWrittenFor(unittest.TestCase):
    """The bug that cost the trip, as a fixture — so the check cannot go quietly inert."""

    def _arity_of(self, source):
        tree = ast.parse(source)
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "World"
        )
        return len(call.args), len(_required_positional("World"))

    def test_the_one_argument_call_is_short(self):
        supplied, required = self._arity_of("World(scene_path)")
        self.assertLess(supplied, required)

    def test_the_two_argument_call_is_not(self):
        supplied, required = self._arity_of("World(scene_path, camera_sensor_specs())")
        self.assertGreaterEqual(supplied, required)


if __name__ == "__main__":
    unittest.main(verbosity=2)
