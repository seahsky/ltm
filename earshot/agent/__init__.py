"""The agent: occupancy, proposers, the reachability invariant, scorer, detector, controller.

ADR-0008's candidate-pool frontier explorer. Deliberately does **not** import ``sim``
(ADR-0013): ``snap_point``, the geodesic query, the follower and the oracle's distance
function all arrive as injected callables, which is what makes this whole layer
Mac-testable and keeps ``import habitat_sim`` in exactly one file.

Kept empty on purpose, in the shape ``audio/__init__.py`` set: ``earshot/__init__.py``
runs on every entry into the package, so anything imported here is paid for by every
process that touches the tree — including ``ruff`` and the structural tests.
"""
