"""Vendored, inert, and deliberately un-importable.

Nothing under ``earshot.reference`` is live code. It is an archive of the memory
stack kept for the follow-on effort, vendored **broken** on purpose (ticket 10): it
imports ``faiss`` and ``sentence-transformers``, and ``memory_bridge.py``'s interface
is built against the deleted ``episode_runner`` and the env-flag surface ADR-0008
removed. If it can be imported by accident, vendoring it was a mistake.

This raises rather than being absent, because absence does not work. PEP 420
namespace packages make ``earshot.reference.memory.ltm`` import cleanly from a regular
parent package on Python 3.3+, so an omitted ``__init__.py`` is no barrier at all —
verified, not assumed. Today the only thing stopping that import is ``faiss`` not
being installed, which is luck, and which flips the day someone installs faiss to work
on the memory follow-on.
"""

raise ImportError(
    "earshot.reference is vendored, inert code and is not importable. It is an "
    "archive for the memory follow-on effort, not part of the running tree — see "
    "earshot/reference/memory/README.md for the write path, the read path, and the "
    "levers already closed."
)
