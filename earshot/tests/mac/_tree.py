"""The shared structural walker. One tree walk, three invariants on top of it.

The structural tests are a **fourth verification layer**, not ordinary Mac tests, on
one ground: they are the only Mac layer that reads the **real subject**. Everything
else here runs our logic against fakes and could in principle pass while the binary is
broken; an ``ast`` parse of ``earshot/`` cannot go vacuous that way. They are also the
enforcement ADR-0008 and ADR-0013 chose *instead of* flags and documentation, on this
repo's record of things that were written down and then quietly stopped being true.

**Denylist-shaped, so new top-level code is checked by default.** An allowlist was
rejected for the direction of its default: a new ``earshot/experimental/`` would be
silently unchecked until someone remembered to add it, which is the wrong failure mode
for this repo. ``test_walker_scope.py`` pins the excluded set, so widening it fails a
test before it lands.
"""

import ast
import os
import pathlib
from typing import Dict, Iterator, List, NamedTuple, Tuple

from _interpreter import assert_interpreter  # noqa: F401  (raises on the wrong Python)

# earshot/tests/mac/_tree.py -> earshot/
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE_NAME = PACKAGE_ROOT.name

# ----------------------------------------------------------------------
# scope
# ----------------------------------------------------------------------

# The top-level entries of `earshot/` the three invariants do NOT apply to, each with
# the reason it is out of scope. ADR-0014 named only `reference`; there are three, and
# the two it missed are load-bearing rather than cosmetic — both invariants fire on the
# test tree itself:
#
#   - `tests/box/` MUST `import habitat_sim`. It drives the real artefact, which is
#     ADR-0014's own definition of a box test, so the one-importer rule cannot cover it.
#   - `tests/mac/test_audio_guard.py` MUST touch `os.environ`. It tests the logging
#     pin, and `TestLoggingPin` saves and restores `HABITAT_SIM_LOG` around the call.
#
# Left unpinned, ADR-0014's `{reference}` would have produced a red suite on day one.
NON_AGENT_ROOTS: Dict[str, str] = {
    # ~3,400 LOC vendored deliberately broken (ticket 10). `memory_bridge.py` reads
    # `LTM_*` from the environment throughout so the env-flag rule fires on it, and its
    # modules import faiss, sentence-transformers and each other so the layering rule
    # fails on it. Excluded from import, test and lint by three separate mechanisms.
    "reference": (
        "vendored and unreachable from the package. Two tenants now: memory/ is inert and "
        "deliberately broken (reference/memory/README.md), and savnce/ is a pinned "
        "submodule that only its own conda env runs (reference/SAVNCE.md, ADR-0015)."
    ),
    # Operator-facing and not part of the agent (ADR-0013). `notify_email.py` reads
    # RESEND_API_KEY and friends: credentials, not agent configuration, and the flag
    # surface ADR-0008 removed was about behaviour.
    "tools": "operator tooling, not the agent — bash plus the carried notify trio",
    # The suite asserting the invariants cannot be its own subject, and the box half
    # exists precisely to touch what the agent may not.
    "tests": "the verification surface itself, mac and box",
}

# The module that owns the simulator's lifecycle (ADR-0013). Everything else reaches the
# simulator through an injected callable, which is what keeps the Mac surface most of the
# tree instead of a corner of it.
SIMULATOR_MODULE = "sim/world.py"

# Where `import habitat_sim` is permitted. Two entries, and the second is a tension
# inside ADR-0013 that its own prose already resolved: the ADR states the one-importer
# rule, and then — dissolving ticket 17's ordering contradiction — states that "`env_check`
# is free to import habitat-sim for its enum probe". Ticket 24 hit both halves at once.
#
# The exemption is narrow and cannot be met any other way. The audio enum MEMBER probe's
# subject IS habitat-sim's build: `AudioSensorSpec` is bound even in non-audio builds
# (habitat-sim #2340), so only resolving the member distinguishes them. It cannot arrive
# injected, because `env_check` runs BEFORE a `World` exists — on an environment that may
# not be able to build one, which is the case it exists to catch — and `env_check` sits at
# layer `()` so it cannot import `sim` to ask.
#
# The alternative was `importlib.import_module("habitat_sim")`, which the AST walker does
# not see. Rejected outright: dodging a structural test with a dynamic import is the
# "written down and quietly stopped being true" pattern these invariants exist to stop.
SIMULATOR_IMPORT_ALLOWED: Dict[str, str] = {
    SIMULATOR_MODULE: "the simulator lifecycle: Simulator(cfg), observe, navmesh, follower",
    "env_check.py": "the audio enum MEMBER probe — a capability that has no injectable form",
}

# `os.environ` is allowed in exactly two agent modules. ADR-0008 removed the flag
# surface — the old tree read `LTM_REALIZABLE_LOCALIZATION` at the runner — and both
# survivors have the environment as their *subject* rather than as configuration.
ENV_ACCESS_ALLOWED = frozenset(
    {
        # the HABITAT_SIM_LOG pin: setting the variable IS the function
        "audio/guard.py",
        # ticket 17's assertion: reading the resolved environment is its whole job
        "env_check.py",
    }
)

# ----------------------------------------------------------------------
# ADR-0013's layer graph
# ----------------------------------------------------------------------
#
#     types, metrics, audio.guard, vlm   -> nothing                 (leaves)
#     sim                                -> audio.guard, types
#     audio                              -> audio.guard, vlm, types  (NOT sim)
#     agent                              -> vlm, types               (NOT sim)
#     report                             -> audio.guard, types
#     task                               -> everything               (the wiring layer)
#
# The load-bearing edge is the one that is ABSENT. Both backwards dependencies rejected
# during ticket 18's grilling — `agent/` reaching into `audio/` for a depth frame,
# `audio/` reaching into `agent/` for a room label — are one convenient import away,
# and this is what stops them.
#
# Values are dotted module prefixes relative to `earshot`. A target is allowed if it
# equals a prefix or begins with `prefix.`. The empty prefix matches everything.
_EVERYTHING: Tuple[str, ...] = ("",)

LAYER_IMPORTS: Dict[str, Tuple[str, ...]] = {
    # leaves — nothing intra-package at all
    "types": (),
    "metrics": (),
    "vlm": (),
    "audio.guard": (),
    # the simulator lifecycle
    "sim": ("audio.guard", "types"),
    # audio owns the sensor and never reaches for the simulator
    "audio": ("audio", "vlm", "types"),
    "agent": ("agent", "vlm", "types"),
    "report": ("report", "audio.guard", "types"),
    # the only wiring layers
    "task": _EVERYTHING,
    "__main__": _EVERYTHING,
    # the root: one call to pin_habitat_logging() and nothing else
    "__init__": ("audio.guard",),
    # RunConfig composes the per-module frozen sub-configs
    "config": ("audio.config", "agent.config", "types"),
    # ticket 17's assertion answers to the environment, not to the tree
    "env_check": (),
}


class Edge(NamedTuple):
    """One intra-package import, resolved to a dotted path relative to ``earshot``."""

    target: str
    lineno: int
    raw: str


def relative_path(path: pathlib.Path) -> str:
    """``earshot/audio/guard.py`` -> ``audio/guard.py``."""
    return path.relative_to(PACKAGE_ROOT).as_posix()


def iter_python_files() -> Iterator[pathlib.Path]:
    """Every ``.py`` under ``earshot/``, sorted. Nothing excluded."""
    for dirpath, dirnames, filenames in os.walk(PACKAGE_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield pathlib.Path(dirpath) / name


def agent_python_files() -> List[pathlib.Path]:
    """The agent package: every ``.py`` outside the pinned non-agent roots.

    The subject of all three invariants. Adding a top-level directory puts it in scope
    automatically — that is the whole point of the denylist shape.
    """
    return [
        path
        for path in iter_python_files()
        if relative_path(path).split("/", 1)[0] not in NON_AGENT_ROOTS
    ]


def parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def module_name(path: pathlib.Path) -> str:
    """Dotted path relative to ``earshot``. The root ``__init__.py`` is ``""``."""
    rel = relative_path(path)
    if rel == "__init__.py":
        return ""
    if rel.endswith("/__init__.py"):
        rel = rel[: -len("/__init__.py")]
    elif rel.endswith(".py"):
        rel = rel[: -len(".py")]
    return rel.replace("/", ".")


def package_of(path: pathlib.Path) -> str:
    """The package a relative import inside this file resolves against."""
    rel = relative_path(path)
    if rel.endswith("/__init__.py") or rel == "__init__.py":
        return module_name(path)
    name = module_name(path)
    return name.rsplit(".", 1)[0] if "." in name else ""


def layer_key(path: pathlib.Path) -> str:
    """Which entry of ``LAYER_IMPORTS`` governs this file.

    ``audio/guard.py`` is its own layer — a stdlib-only leaf, unlike the rest of
    ``audio/`` — so it is matched before the package it lives in. The root
    ``__init__.py`` and ``__main__.py`` are matched by name.
    """
    rel = relative_path(path)
    if rel == "__init__.py":
        return "__init__"
    name = module_name(path)
    if name in LAYER_IMPORTS:
        return name
    return name.split(".", 1)[0]


def intra_package_edges(path: pathlib.Path, tree: ast.Module) -> List[Edge]:
    """Imports resolving inside ``earshot``, absolute and relative alike.

    External imports (``os``, ``numpy``, ``habitat_sim``) are not layer edges and are
    not returned — the graph constrains the tree's shape, not its dependencies.
    """
    edges: List[Edge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE_NAME or alias.name.startswith(PACKAGE_NAME + "."):
                    edges.append(
                        Edge(
                            alias.name[len(PACKAGE_NAME) :].lstrip("."),
                            node.lineno,
                            "import " + alias.name,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = [p for p in package_of(path).split(".") if p]
                for _ in range(node.level - 1):
                    if parts:
                        parts.pop()
                if base:
                    parts.extend(base.split("."))
                edges.append(
                    Edge(
                        ".".join(parts),
                        node.lineno,
                        "from {}{} import ...".format("." * node.level, base),
                    )
                )
            elif base == PACKAGE_NAME or base.startswith(PACKAGE_NAME + "."):
                edges.append(
                    Edge(
                        base[len(PACKAGE_NAME) :].lstrip("."),
                        node.lineno,
                        "from {} import ...".format(base),
                    )
                )
    return edges


def edge_allowed(target: str, allowed: Tuple[str, ...]) -> bool:
    """Does ``target`` sit under any allowed prefix? The empty prefix matches all."""
    for prefix in allowed:
        if prefix == "" or target == prefix or target.startswith(prefix + "."):
            return True
    return False


def imports_module(tree: ast.Module, name: str) -> bool:
    """Does this file import the named top-level module, in any form?"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == name or a.name.startswith(name + ".") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (
                not node.level
                and node.module
                and (node.module == name or node.module.startswith(name + "."))
            ):
                return True
    return False


def code_string_constants(tree: ast.Module) -> List[Tuple[int, str]]:
    """Every string literal except docstrings, with line numbers.

    A structural invariant that greps raw text cannot tell a *use* from the *citation
    explaining why it is not used* — ticket 19's "a grep verifies presence not truth",
    met head-on the first time a test was written that way. Docstrings are excluded by
    identity rather than by heuristic, and comments never enter the AST at all, so what
    is left is what the module actually does.
    """
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    ]


def attribute_names(tree: ast.Module) -> List[Tuple[int, str]]:
    """Every ``x.attr`` reach, with line numbers. Attribute name only, no receiver."""
    return [
        (node.lineno, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    ]


def enclosing_function_by_lineno(tree: ast.Module) -> Dict[int, str]:
    """``{lineno: enclosing function name}``. Lines outside any ``def`` are absent.

    Shared by the two exemption checks, which ask the same question of different
    subjects: is this reach still spent on what earned it? A count says only that the
    exemption did not grow.
    """
    owner: Dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                lineno = getattr(child, "lineno", None)
                # Nested defs: the innermost walk wins because it is visited later only
                # if it is reached later, so take the first (outermost) claim and let
                # the inner one refine it.
                if lineno is not None:
                    owner[lineno] = node.name
    return owner


def environ_accesses_by_function(tree: ast.Module) -> List[Tuple[str, str]]:
    """Every environment reach paired with the function that contains it.

    Stronger than counting accesses, which is what this replaced. A count says the
    exemption did not grow; naming the enclosing function says it is still spent on
    what earned it — ``guard.py`` legitimately both *sets* the pin and *asserts* it, and
    a bare count could not tell that second read from a new configuration flag.

    ``"<module>"`` for a top-level access, since a module-level environment read is a
    different and worse thing than one inside a named function.
    """
    owner = enclosing_function_by_lineno(tree)
    return [
        (owner.get(lineno, "<module>"), what) for lineno, what in environ_accesses(tree)
    ]


def module_imports_by_function(tree: ast.Module, name: str) -> List[Tuple[str, int]]:
    """Every ``import <name>`` reach paired with the function that contains it.

    ``"<module>"`` for a top-level import. The distinction is load-bearing for
    ``env_check.py``: a *top-level* ``import habitat_sim`` there would run on every
    import of the module, including from a Mac, which is precisely what the exemption
    does not cover.
    """
    owner = enclosing_function_by_lineno(tree)
    hits: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        matched = False
        if isinstance(node, ast.Import):
            matched = any(a.name == name or a.name.startswith(name + ".") for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            matched = bool(
                not node.level
                and node.module
                and (node.module == name or node.module.startswith(name + "."))
            )
        if matched:
            hits.append((owner.get(node.lineno, "<module>"), node.lineno))
    return hits


def environ_accesses(tree: ast.Module) -> List[Tuple[int, str]]:
    """Every ``os.environ`` / ``os.getenv`` / ``environ[...]`` reach, with line numbers.

    Matched on the attribute rather than on the import, because
    ``from os import environ`` and ``os.environ`` are the same flag surface and only
    one of them mentions ``os`` at the use site.
    """
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv", "putenv"):
            hits.append((node.lineno, "os." + node.attr))
        elif isinstance(node, ast.Name) and node.id in ("environ", "getenv"):
            hits.append((node.lineno, node.id))
    return hits
