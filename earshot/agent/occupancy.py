"""The agent's frame algebra, and the depth-derived occupancy grid.

Two things live here, and the pairing is the point.

**The frame.** Habitat's agent has ``y`` up, forward ``-z`` and right ``+x`` at zero
yaw, and ``sim.world.yaw_from_quaternion`` returns the rotation angle about ``+y``. So
in world coordinates ``forward(yaw) = (-sin yaw, -cos yaw)`` and
``right(yaw) = (cos yaw, -sin yaw)`` over ``(x, z)``.

**The old tree's splat used ``(+sin, +cos)`` as forward, which is 180 degrees out**, and
that is a defect this rewrite fixes rather than carries. Read from source, not from a
run: ``habitat_env.py:620`` extracts exactly the yaw above, ``episode_runner.py:1439``
hands it to ``FrontierPlanner.update`` unmodified, and ``frontier_planner.py:541-557``
marches along ``(sin theta, cos theta)``. The whole depth cone therefore landed
point-reflected through the agent, so the map was a 180-degree rotation of the room.
Two knock-ons, both internally consistent and both wrong against the simulator: the
candidate bearing (``episode_runner.py:2371``, same convention) made a frontier straight
ahead read as ``|bearing| ~ pi``, so the scorer's bearing-alignment term systematically
preferred candidates *behind* the agent; and the detector's back-projection
(``goal_detector.py:196`` with ``episode_runner.py:2985``) placed every detection behind
the agent and 0.88 m too low.

The predicted symptoms are the recorded ones — A* found no path on roughly 92% of steps,
``n_waypoint_unreachable`` 60-99 per episode, ``min_d2g`` stuck around 8 m,
``n_detector_localized`` 0 across a whole matrix, and a back-projection 0.76 m below the
navmesh. That is corroboration, not proof: nothing here re-runs the old tree, and its
verdicts are not reopened. What matters forward is that the clean room does not inherit
it.

**So the algebra lives in one module, and a test compares it with the tree's other
frame consumer.** ``audio/lateral.bearing_lateral_sign`` derives the same right axis
independently; ``agent/`` may not import ``audio/`` (ADR-0013), so the agreement is
asserted by ``tests/mac/test_agent_frame.py``, which may import both. That is the
missing piece in ticket 09's finding: the convention inverted underneath code that did
not change, because it was written down in two places and checked in none.

**The grid.** A top-down occupancy grid over the world ``(x, z)`` plane, re-centred on
the agent at every reset. ``integrate_depth`` returns a **new** grid: the array copy is
40 KB against a 27 ms/step audio budget, and a frozen grid means a stale reference
cannot silently be the previous episode's map — the same discipline ``audio/onset.py``
applies to ``OnsetState``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

import numpy as np

from earshot.agent.config import PlannerConfig
from earshot.types import Pose, Xyz

__all__ = [
    "CELL_UNKNOWN",
    "CELL_FREE",
    "CELL_OCCUPIED",
    "Intrinsics",
    "OccupancyGrid",
    "intrinsics_from_hfov",
    "forward_xz",
    "right_xz",
    "heading_to",
    "bearing_rel",
    "wrap_pi",
    "new_grid",
    "integrate_depth",
    "cell_counts",
    "camera_to_world",
]

# Cell states, uint8 for compactness. OCCUPIED is sticky: a wall is never erased by a
# floor ray that overshoots it.
CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_OCCUPIED = 2


# ----------------------------------------------------------------------
# the frame
# ----------------------------------------------------------------------


def wrap_pi(angle: float) -> float:
    """Fold an angle into ``(-pi, pi]``."""
    wrapped = math.fmod(float(angle) + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def forward_xz(yaw_rad: float) -> Tuple[float, float]:
    """The agent's forward direction over ``(x, z)``: ``(-sin yaw, -cos yaw)``.

    A unit vector. At yaw 0 it is ``-z``, which is habitat's local forward rotated by
    the identity — see the module docstring for why this is not the ``(+sin, +cos)`` the
    old tree used.
    """
    return (-math.sin(yaw_rad), -math.cos(yaw_rad))


def right_xz(yaw_rad: float) -> Tuple[float, float]:
    """The agent's right direction over ``(x, z)``: ``(cos yaw, -sin yaw)``.

    Identical to the axis ``audio/lateral.bearing_lateral_sign`` derives, and
    ``tests/mac/test_agent_frame.py`` asserts the two agree rather than trusting that
    two modules wrote the same arithmetic down twice.
    """
    return (math.cos(yaw_rad), -math.sin(yaw_rad))


def heading_to(dx: float, dz: float) -> float:
    """The yaw that would point the agent along ``(dx, dz)``.

    The inverse of :func:`forward_xz`: solving ``(-sin h, -cos h) = (dx, dz) / |d|``
    gives ``h = atan2(-dx, -dz)``.
    """
    return math.atan2(-float(dx), -float(dz))


def bearing_rel(yaw_rad: float, dx: float, dz: float) -> float:
    """Where a target sits relative to the agent's heading. **Positive is left.**

    Zero is straight ahead. The sign convention is the one the greedy turn rule wants:
    a positive rotation about ``+y`` swings forward from ``-z`` toward ``-x``, which is
    the agent's left, and habitat's ``turn_left`` is that positive rotation. So
    ``bearing_rel > 0`` means turn left to face the target.

    Only the magnitude reaches the scorer's alignment term, which is exactly why the old
    tree's 180-degree error was invisible there: ``|bearing|`` stayed a plausible number
    and simply ranked the pool backwards.
    """
    return wrap_pi(heading_to(dx, dz) - float(yaw_rad))


# ----------------------------------------------------------------------
# the camera
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for one camera, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """Intrinsics from image size and horizontal field of view.

    ``fy == fx`` because habitat's pixels are square: ``CameraSensorSpec.hfov`` fixes
    the horizontal field and the vertical follows from the aspect ratio, so one focal
    length serves both axes. The old comment claimed this "assumes a square sensor"; it
    does not — square *pixels* are enough, and 640x480 is not square.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive, got {}x{}".format(width, height))
    focal = (float(width) / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    return Intrinsics(
        fx=focal,
        fy=focal,
        cx=float(width) / 2.0,
        cy=float(height) / 2.0,
        width=int(width),
        height=int(height),
    )


def camera_to_world(pose: Pose, eye_height_m: float) -> np.ndarray:
    """The 4x4 transform from camera coordinates to world coordinates.

    Camera coordinates are habitat's: ``x`` right, ``y`` up, and the optical axis along
    ``-z``, so a point at distance ``d`` straight ahead is ``(0, 0, -d)``. The rotation
    is the yaw about ``+y``; the translation is the **sensor's** position, which is the
    agent's own position lifted by ``eye_height_m``.

    Both halves are corrections to ``episode_runner._agent_pose_matrix``, which used the
    agent's base position as the camera origin (0.88 m low) and was consumed by a
    back-projection that put ``+z`` forward (180 degrees out).
    """
    yaw = float(pose.yaw_rad)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    # Columns are the world-frame images of the camera axes: right, up, and +z (which is
    # backward, since the optical axis is -z).
    transform[:3, 0] = (cos_y, 0.0, -sin_y)
    transform[:3, 1] = (0.0, 1.0, 0.0)
    transform[:3, 2] = (sin_y, 0.0, cos_y)
    transform[:3, 3] = (
        pose.position.x,
        pose.position.y + float(eye_height_m),
        pose.position.z,
    )
    return transform


# ----------------------------------------------------------------------
# the grid
# ----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class OccupancyGrid:
    """A top-down uint8 grid over the world ``(x, z)`` plane. Immutable by contract.

    ``eq=False`` because the generated ``__eq__`` would compare ``cells`` with ``==``
    and hand a boolean array to ``bool()``. Tests compare ``cells`` explicitly, which is
    what they mean anyway.
    """

    resolution_m: float
    origin_xy: Tuple[float, float]
    cells: np.ndarray

    @property
    def n(self) -> int:
        return int(self.cells.shape[0])

    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        """``(row, col)`` for a world point. Rows index ``z``, columns index ``x``.

        **Floors, where the old grid truncated** (``frontier_planner.py:79-83`` used
        ``int()``). An undisclosed behavioural change and a fix: ``int()`` rounds toward
        zero, so a point in the half-open cell just *outside* the low edge — quotient in
        ``(-1, 0)`` — aliased to row or column 0. The consequence is not a rounding
        nicety: ``integrate_depth`` would write a ray endpoint that left the grid into the
        grid's edge cell, and ``state_at`` would then report a neighbour's state as that
        cell's own.
        """
        ox, oz = self.origin_xy
        return (
            int(math.floor((float(z) - oz) / self.resolution_m)),
            int(math.floor((float(x) - ox) / self.resolution_m)),
        )

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """The world ``(x, z)`` at the centre of a cell."""
        ox, oz = self.origin_xy
        return (
            ox + (float(col) + 0.5) * self.resolution_m,
            oz + (float(row) + 0.5) * self.resolution_m,
        )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n and 0 <= col < self.n

    def state_at(self, row: int, col: int) -> int:
        """The cell state, or ``CELL_UNKNOWN`` out of bounds.

        Out of bounds reads as unknown rather than raising: a ray that leaves the 20 m
        window is not an error, and the alternative is every caller bounds-checking.
        """
        if not self.in_bounds(row, col):
            return CELL_UNKNOWN
        return int(self.cells[row, col])


def new_grid(centre: Xyz, cfg: PlannerConfig) -> OccupancyGrid:
    """An all-unknown grid centred on ``centre``.

    Re-centring per episode is load-bearing rather than cosmetic: the old grid was
    pinned to the world origin with a 20 m span while HM3D agent starts are routinely
    15-20 m away, so every lookup fell out of bounds and the occupancy data was silently
    discarded.
    """
    n = int(round(cfg.grid_size_m / cfg.grid_res_m))
    half = cfg.grid_size_m / 2.0
    return OccupancyGrid(
        resolution_m=float(cfg.grid_res_m),
        origin_xy=(centre.x - half, centre.z - half),
        cells=np.full((n, n), CELL_UNKNOWN, dtype=np.uint8),
    )


def integrate_depth(
    grid: OccupancyGrid, depth: Any, pose: Pose, cfg: PlannerConfig
) -> OccupancyGrid:
    """Splat one depth frame into the grid and return the updated grid.

    Per sampled pixel: back-project to a camera-frame offset, march ``CELL_FREE`` along
    the ray's ground range, then mark the endpoint on a **height gate** — endpoints
    within ``obstacle_min_h`` of the floor are floor (free, walkable, which is what
    fills doorways) and higher ones are obstacles. This is the Run-5 densification; a
    single eye-height scanline mostly hit walls and furniture, carved almost no free
    space, and left the agent with no navigable subgoal.

    The height gate is the old arithmetic with two cancelling errors removed rather than
    preserved. It read ``world_h = agent_y + Y_camera`` against a floor fixed at
    ``agent_y - camera_height``, which was a camera 0.88 m too low measured against a
    floor 0.88 m too deep. Here the floor is the agent's own ``y`` — habitat seats the
    body node on the navmesh — so the height above it is ``eye_height + Y_camera``
    directly, and the episode's floor is not frozen at reset, which is what a
    multi-floor scene needs.

    Depth arrives raw and metric (ticket 21: habitat-lab's ``min_depth`` / ``max_depth``
    / ``normalize_depth`` do not exist on ``CameraSensorSpec``), so the only range
    handling is the clamp to ``max_depth_m``. Under normalised depth a 3 m wall would
    read 0.3, the height gate would mark nearly every endpoint occupied, and the map
    would carve almost nothing — the trap that collapsed Run 5's occupancy.
    """
    frame = np.asarray(depth)
    if frame.size == 0:
        return grid
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.ndim != 2:
        raise ValueError(
            "the occupancy splat needs an (H, W) or (H, W, 1) depth frame, got {}".format(
                frame.shape
            )
        )

    height, width = int(frame.shape[0]), int(frame.shape[1])
    intr = intrinsics_from_hfov(width, height, cfg.forward_fov_deg)
    res = grid.resolution_m
    ax, az = pose.position.x, pose.position.z
    yaw = float(pose.yaw_rad)
    eye = float(cfg.eye_height_m)

    cells = grid.cells.copy()
    row_step = max(1, height // int(cfg.splat_samples))
    col_step = max(1, width // int(cfg.splat_samples))

    def mark(row: int, col: int, state: int) -> None:
        if 0 <= row < cells.shape[0] and 0 <= col < cells.shape[1]:
            if state == CELL_OCCUPIED or cells[row, col] != CELL_OCCUPIED:
                cells[row, col] = state

    for v in range(0, height, row_step):
        depth_row = frame[v]
        up_term = (intr.cy - v) / intr.fy
        for u in range(0, width, col_step):
            d = float(depth_row[u])
            if not math.isfinite(d) or d <= 0.05:
                continue
            d = min(d, float(cfg.max_depth_m))
            x_cam = d * (u - intr.cx) / intr.fx
            ground_range = math.hypot(x_cam, d)
            if ground_range < res:
                continue
            # The ray's heading is the agent's yaw MINUS the pixel's horizontal angle:
            # a positive rotation about +y turns left, and a pixel to the right is to
            # the right. Derived in the module docstring's frame; the old tree added it.
            ray_yaw = yaw - math.atan2(x_cam, d)
            fx_dir, fz_dir = forward_xz(ray_yaw)

            n_free = int((ground_range - res) / res)
            for s in range(n_free):
                reach = (s + 0.5) * res
                row, col = grid.world_to_grid(ax + fx_dir * reach, az + fz_dir * reach)
                mark(row, col, CELL_FREE)

            row, col = grid.world_to_grid(
                ax + fx_dir * ground_range, az + fz_dir * ground_range
            )
            height_above_floor = eye + d * up_term
            mark(
                row,
                col,
                CELL_OCCUPIED
                if height_above_floor > float(cfg.obstacle_min_h)
                else CELL_FREE,
            )

    return replace(grid, cells=cells)


def cell_counts(grid: OccupancyGrid) -> Dict[str, int]:
    """Free / occupied / unknown census, which sums to ``n * n``.

    Makes a run interpretable rather than trusted: if ``cells_free`` is still tiny the
    densification did not take, and that is the failure mode Run 5 spent a matrix on.
    The frontier count is ``proposers.frontier_cells``, which is a query about
    exploration rather than about the grid.
    """
    return {
        "cells_free": int(np.count_nonzero(grid.cells == CELL_FREE)),
        "cells_occupied": int(np.count_nonzero(grid.cells == CELL_OCCUPIED)),
        "cells_unknown": int(np.count_nonzero(grid.cells == CELL_UNKNOWN)),
    }
