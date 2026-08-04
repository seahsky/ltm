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

**``NoRouteError`` lives here, and it moved for a structural reason.** Ticket 21 defined
it in ``sim/world.py``, where the follower raises it — but a caller that wants to catch
it has to be able to *name* the type, and only ``sim/world.py`` may import the simulator.
``task/runner.py`` catching it from there would drag ``import habitat_sim`` into the one
module the whole Mac suite needs to be able to import, and the alternative — catching
``RuntimeError`` and sniffing the class name — is the kind of quietly-wrong this tree
keeps removing. It is a leaf type about routing, not about habitat, so it belongs beside
the geometry. ``sim.world`` re-exports it, so every existing reference still resolves.

Python 3.9: annotations are postponed, so nothing here evaluates a PEP 604 union.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

__all__ = ["Xyz", "Pose", "NoRouteError"]


class NoRouteError(RuntimeError):
    """The follower could not route to the target from where the agent stands.

    Its own type because the alternative reading — ``None``, meaning *arrived* — is the
    exact confusion that made the old tree's navigation unfalsifiable. The grid-A* it
    replaced found no path on roughly 92% of steps and silently fell back to
    straight-line steering, so "a waypoint was chosen" and "the agent got there" came
    apart with nothing in the code marking where. A caller that wants to re-propose
    catches this; a caller that does not gets a loud failure instead of a slow drift.

    Raised by ``sim.world.World.follower``; caught by ``task.runner._steer``.
    """


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

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> "Pose":
        """The inverse of ``as_dict``. Ticket 24: ``read_episode`` needs the round trip.

        It lives here rather than in ``report/artifacts.py`` so the pose's flat
        ``{x, y, z, yaw_rad}`` layout is known in exactly one module. A reader that
        re-derived the key names would be a second copy of the convention, which is
        the shape of every silent drift this map has caught.
        """
        return cls(
            position=Xyz(float(data["x"]), float(data["y"]), float(data["z"])),
            yaw_rad=float(data["yaw_rad"]),
        )
