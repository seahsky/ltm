"""Vendored, inert, and deliberately un-importable. See ``../__init__.py``.

Both levels raise. The parent alone would be enough for ``import
earshot.reference.memory``, but a direct ``import earshot.reference.memory.ltm``
walks both packages and the failure should name the level a confused session is
actually standing on.
"""

raise ImportError(
    "earshot.reference.memory is vendored, inert code and is not importable. Read "
    "earshot/reference/memory/README.md first: it records what the write and read "
    "paths were, which levers are already closed, and where the measured bottleneck "
    "is. Reviving any of it is a new effort, not an import."
)
