"""The two geometric types the whole tree shares. A leaf: imports nothing.

Frame conventions, both inherited from habitat-sim:

- **y is up.** Horizontal distance is therefore over ``(x, z)`` and never over all
  three axes. That is not a stylistic preference — the old tree's detector snap gate
  used a 3D distance while every consumer used only ``(x, z)``, so it rejected correct
  detections of elevated objects (fixed in commits ``3307f19`` / ``7fbf370``).
  ``horizontal_distance_to`` is what stops that coming back, and ADR-0010's
  ``|Δy| < 1.0 m`` floor rule is ``height_difference_to``.
- **Yaw is a scalar heading in radians**, stored instead of a quaternion because
  everything downstream — the lateral cue, the report's ``stopped_at_pose`` — wants a
  number. ``sim/world.py`` owns the conversion; nothing here knows habitat-sim exists.

**No bearing helper lives here on purpose.** Ticket 09 found that the lateral sign
silently inverted from world frame to agent frame under live rendering with no code
change, so the convention is *measured* by ``tests/box/`` and owned by
``audio/lateral.py``. A plausible ``atan2`` in this leaf would be a frame convention
asserted by nobody, which is the class of quietly-wrong the map keeps catching.

Python 3.9: annotations are postponed, so nothing here evaluates a PEP 604 union.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

__all__ = ["Xyz", "Pose"]


@dataclass(frozen=True)
class Xyz:
    """A point in the scene's world frame, metres, y up."""

    x: float
    y: float
    z: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "Xyz":
        """Build from anything 3-long — a list, a tuple, a numpy row.

        Coerced through ``float`` so a numpy scalar cannot leak into a frozen
        dataclass that is later compared or serialised.
        """
        x, y, z = (float(v) for v in values)
        return cls(x, y, z)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def horizontal_distance_to(self, other: "Xyz") -> float:
        """Distance over the floor plane, ignoring height.

        The one every consumer actually wants: goals, waypoints and sound sources are
        compared on where they stand, not on how high they sit.
        """
        return math.hypot(self.x - other.x, self.z - other.z)

    def height_difference_to(self, other: "Xyz") -> float:
        """Signed height gap. ADR-0010's floor test is ``abs(...) < 1.0``."""
        return self.y - other.y


@dataclass(frozen=True)
class Pose:
    """Where the agent stands and which way it faces."""

    position: Xyz
    yaw_rad: float

    def as_dict(self) -> Dict[str, float]:
        """The serialised form. ``stopped_at_pose`` in the agent's report is this."""
        return {
            "x": self.position.x,
            "y": self.position.y,
            "z": self.position.z,
            "yaw_rad": self.yaw_rad,
        }
