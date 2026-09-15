"""Every name a box test imports from `earshot/` must exist. Checked WITHOUT importing.

**THIS EXISTS BECAUSE FIVE BOX TESTS SHIPPED WITH AN IMPORT THAT COULD NEVER RESOLVE.**
PRs #109-#115 each added a box test beginning

    from earshot.task.dataset import available_scenes, find_scenes_dir, find_split_dir, load_scene

and those four functions live in `earshot/task/episodes.py`. `earshot.task.dataset` is a
real module and the import is syntactically perfect, so:

  * the mac suite never noticed -- it does not import `tests/box/`, by design, because
    those modules need habitat-sim;
  * `ruff` never noticed -- it lints names and formatting and does not resolve imports
    across modules;
  * the PR bodies never noticed -- four of the five were copied from the first.

`test_agent_stm_box.py` carried it from the day #109 merged and stayed red on the box for
five PRs, because the box gate was not run in between. The gate then failed on FIVE
modules at once, having executed none of them, and the four measurements those PRs were
written to produce -- the coherence spread, `h^traj`'s variance, `omega_t`'s movement, and
what a DREAM step costs -- were not taken.

**THE CHECK IS STRUCTURAL, AND IT HAS TO BE.** Importing a box test on a Mac is what this
tree cannot do: several of them import `earshot.sim.world` or habitat-sim at module scope.
So this parses each box module with `ast`, collects every `from earshot.<module> import
<names>`, parses the TARGET module, and asserts each name is bound there at module level.
No `earshot` submodule is imported and no simulator is touched.

**What it cannot catch, said plainly.** A name bound by a star-import or a conditional
`try: import x except: x = None` in the target, a name that exists but is the wrong THING,
and anything about the module BODY of a box test past its imports. This is the cheapest
check that would have caught the actual defect, not a substitute for running the gate.
"""

import ast
import unittest

from _interpreter import assert_interpreter  # noqa: F401

import _tree

BOX_DIR = _tree.PACKAGE_ROOT / "tests" / "box"


def _box_modules():
    return sorted(path for path in BOX_DIR.glob("test_*.py"))


def _earshot_imports(tree):
    """Every `from earshot.<dotted> import a, b` in one module, as (dotted, names, line)."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        if not node.module or not (
            node.module == "earshot" or node.module.startswith("earshot.")
        ):
            continue
        names = [alias.name for alias in node.names]
        found.append((node.module, names, node.lineno))
    return found


def _module_path(dotted):
    """`earshot.task.episodes` -> the file, or `None` if there is no such module."""
    relative = dotted.split(".")[1:]
    if not relative:
        return _tree.PACKAGE_ROOT / "__init__.py"
    direct = _tree.PACKAGE_ROOT.joinpath(*relative).with_suffix(".py")
    if direct.is_file():
        return direct
    package = _tree.PACKAGE_ROOT.joinpath(*relative) / "__init__.py"
    return package if package.is_file() else None


def _bound_names(tree):
    """Every name bound at MODULE level: defs, classes, assignments, and re-exports."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` and friends still bind for an importer.
            for inner in ast.walk(node):
                if isinstance(inner, (ast.FunctionDef, ast.ClassDef)):
                    names.add(inner.name)
                elif isinstance(inner, (ast.Import, ast.ImportFrom)):
                    for alias in inner.names:
                        names.add(alias.asname or alias.name.split(".")[0])
    return names


class TestEveryBoxImportResolves(unittest.TestCase):
    def test_the_walk_finds_the_box_suite(self):
        """The control. A glob that returned nothing would make the check below pass over
        an empty loop, which is how a gate stops gating without going red -- the same
        failure `test_memory_store.py`'s fence control exists for."""
        found = {path.name for path in _box_modules()}
        self.assertGreater(len(found), 8)
        self.assertIn("test_world_box.py", found)
        self.assertIn("test_dream_box.py", found)

    def test_every_imported_module_exists(self):
        offenders = []
        for path in _box_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for dotted, _names, lineno in _earshot_imports(tree):
                if _module_path(dotted) is None:
                    offenders.append("{}:{} no module {}".format(
                        path.name, lineno, dotted))
        self.assertEqual(offenders, [], "\n".join([""] + offenders))

    def test_every_imported_name_is_bound_in_its_module(self):
        """**THE ONE THAT WOULD HAVE CAUGHT IT.** `available_scenes` is not in
        `earshot.task.dataset`; it is in `earshot.task.episodes`. Both modules exist, so
        the module check above passes and only this fails."""
        offenders = []
        for path in _box_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for dotted, names, lineno in _earshot_imports(tree):
                target = _module_path(dotted)
                if target is None:
                    continue  # reported by the test above
                bound = _bound_names(
                    ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
                )
                for name in names:
                    if name == "*" or name in bound:
                        continue
                    # A submodule is a legal `from package import name` target too.
                    if _module_path("{}.{}".format(dotted, name)) is not None:
                        continue
                    offenders.append("{}:{} {} has no {}".format(
                        path.name, lineno, dotted, name))
        self.assertEqual(offenders, [], "\n".join([""] + offenders))

    def test_the_resolver_can_actually_fail(self):
        """The forced-failure arm. A resolver that bound every name it was asked about
        would report green over the real defect, and the two tests above would be
        decorative -- which is exactly what they were before the box run."""
        module = ast.parse("def real(): pass\n")
        bound = _bound_names(module)
        self.assertIn("real", bound)
        self.assertNotIn("available_scenes", bound)
        self.assertIsNone(_module_path("earshot.task.no_such_module"))
        self.assertIsNotNone(_module_path("earshot.task.episodes"))

    def test_the_exact_defect_is_recognised(self):
        """Pinned as a regression: the literal line that shipped five times, checked
        against the real tree rather than against a fixture."""
        wrong = ast.parse(
            "from earshot.task.dataset import available_scenes\n"
        )
        dotted, names, _line = _earshot_imports(wrong)[0]
        target = _module_path(dotted)
        self.assertIsNotNone(target, "earshot.task.dataset is a real module")
        bound = _bound_names(
            ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
        )
        self.assertNotIn(
            names[0], bound,
            "available_scenes is now in task/dataset.py; this regression test is stale",
        )
        right = _module_path("earshot.task.episodes")
        self.assertIn(
            names[0],
            _bound_names(ast.parse(right.read_text(encoding="utf-8"), filename=str(right))),
        )


class TestTheBoxSuiteParses(unittest.TestCase):
    """Cheaper than the above and catches a different thing: a syntax error in a box test
    also fails the whole gate at load time, with no test having run."""

    def test_every_box_module_is_valid_python(self):
        for path in _box_modules():
            with self.subTest(module=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
