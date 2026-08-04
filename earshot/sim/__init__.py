"""The simulator lifecycle. ``world.py`` is the only module in the tree that may
``import habitat_sim`` (ADR-0013), and ``tests/mac/test_layering.py`` enforces it.

Deliberately empty otherwise: importing this package must not import habitat-sim, so
that ``tests/mac`` can walk the tree and ``ruff`` can lint it on a machine where the
simulator does not exist.
"""
