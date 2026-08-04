"""The output layer: two types, one writer.

Split on whether the information is agent-estimable (task spec §5). ``agent.py`` is the
testimony and ``audit.py`` is the answer key, and ADR-0013 draws the boundary at the
**type** rather than at the controller — because the oracle arm's controller
legitimately holds ``source_xyz`` as its waypoint while the spec requires an identical
schema in both arms.

``artifacts.py`` is the only module in the whole tree that writes anything.
"""
