"""The agent layer's frozen sub-configs (ADR-0013), composed into ``RunConfig``.

No environment flag anywhere: ADR-0008 removed the flag surface, and
``test_no_env_flags.py`` holds it. That has one visible consequence in this file —
ticket 23 names ``DETECTOR_SNAP_FLOOR_EPS`` as a thing to keep, and what carries is the
**gate**, as ``DetectorConfig.snap_floor_eps_m``. The old spelling survives in the
comment so the behaviour is greppable back to the commit that added it
(``3307f19`` / ``7fbf370``), not as a variable anything reads.

**Three dataclasses, where ticket 23 named two.** ``DetectorConfig`` exists because the
detector has four numbers that decide whether a detection is believed, and the
alternative is constructor defaults — which no run record captures. A number that
gates a STOP and is invisible in ``env_report.json`` is the class of thing this map
keeps finding after the fact.

Every constant carries a **provenance tag** (ADR-0014): ``box`` and ``source`` are
measured, ``fake`` and ``runtime`` are not, and an unmeasured constant is set
generously rather than tightly.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PlannerConfig", "ControllerConfig", "DetectorConfig"]


@dataclass(frozen=True)
class PlannerConfig:
    """Occupancy grid, depth splat, decision cadence, and candidate geometry."""

    # provenance: source — carried from `FrontierPlanner.__init__`. Emit a fresh
    # candidate set every N steps; between decisions the runner keeps steering to the
    # waypoint it already has.
    decision_period: int = 10

    # provenance: source — K, the pool size the scorer picks one from.
    n_candidates: int = 4

    # provenance: source — position spread below which the agent counts as stuck and
    # re-proposes early, over `stuck_window` steps.
    stuck_radius_m: float = 0.1
    stuck_window: int = 8

    # provenance: box — MUST equal `sim.world.camera_sensor_specs(hfov=)`, which is
    # habitat-lab's published ObjectNav HM3D value. The splat back-projects depth with
    # intrinsics derived from this number, so a disagreement silently scales the map.
    # Ticket 25 wires both from `RunConfig`; there is no runtime check because `agent/`
    # cannot see the sensor spec (ADR-0013).
    forward_fov_deg: float = 79.0

    # provenance: box — MUST equal `camera_sensor_specs(eye_height=)` for the same
    # reason. It is also the floor offset the height gate below measures against: the
    # agent's own y is floor level (habitat seats the body node on the navmesh), so a
    # back-projected endpoint's height above the floor is `eye_height + Y_camera`.
    eye_height_m: float = 0.88

    # provenance: source — depth beyond this is clamped rather than dropped, so a long
    # corridor still carves free space out to the range the splat trusts. Habitat-sim
    # depth arrives raw and metric (ticket 21), so this is the only range limit.
    max_depth_m: float = 5.0

    # provenance: source — a back-projected endpoint counts as an obstacle only if it
    # rises this far above the floor. Lower endpoints are floor, marked FREE, which is
    # what fills doorways and openings; without the gate an eye-height scanline marks
    # walls and furniture only and the map carves almost nothing (Run 5).
    obstacle_min_h: float = 0.3

    # provenance: source — grid side and cell size. 20 m square at 0.1 m is the old
    # tree's, re-centred on the agent at every reset (a grid pinned to the world origin
    # was silently out of bounds for HM3D starts 15-20 m away).
    grid_size_m: float = 20.0
    grid_res_m: float = 0.1

    # provenance: source — the depth frame is subsampled to about this many rows and
    # columns, so roughly 800 rays per step rather than 300k pixels. Ticket 06's audio
    # budget is 27 ms/step, so this loop is not the thing to optimise, but a full-frame
    # splat would be.
    splat_samples: int = 28

    # provenance: source — the compass fan's radius, and how far past it the occupancy
    # scan looks when scoring a fan direction.
    compass_dist_m: float = 1.5
    compass_scan_extra_m: float = 0.5

    # provenance: source — greedy clustering radius for frontier cells, in cells. 3
    # cells at 0.1 m/cell groups a 0.3 m neighbourhood into one candidate.
    cluster_radius_cells: int = 3

    # provenance: source — ADR-0010's floor rule, reused as the one snap the reachability
    # filter refuses. A candidate carries the agent's own y, so a navmesh snap that moves
    # it this far vertically has landed on another storey: a different room reached by
    # stairs rather than a correction. There is deliberately no *horizontal* snap cap —
    # see `reachability.py` for why a waypoint and a detection answer different questions.
    same_floor_m: float = 1.0


@dataclass(frozen=True)
class ControllerConfig:
    """The interrupt-resume machine's two numbers. No `enabled` flag (ADR-0008)."""

    # provenance: fake — task spec §9's first "left to the builder" number, set by ticket
    # 26 against a measurement rather than an estimate. The detour's step sub-budget; on
    # overflow the controller aborts and resumes the primary task rather than spending the
    # whole episode climbing.
    #
    # **40 was never enough, and the abort hid it.** The abort left `investigated` False,
    # SEARCH's guard read only that, and the detour was re-entered — so the budget was
    # per-attempt with nothing bounding the aggregate, and the first box episode spent
    # about 210 of its 250 steps on six of them. The fake stall-turn climb in
    # `test_task_runner` reaches a source 5.4 m away, starting 180 degrees backwards, in
    # **59 steps** — it passed under a 40-step budget only because a second attempt
    # started from a better pose.
    #
    # 120 is that 59 with room for a longer detour or a worse start: about 11 steps per
    # metre in the fake (0.25 m/step, plus the turn-and-re-probe a reversal costs), so it
    # covers roughly 11 m. Against `max_steps` 250 it leaves 130 steps for the primary,
    # and because the abort is now terminal it is the whole detour's cost — strictly less
    # than the 240 the first box run actually spent. The cost of it being too small is a
    # silently truncated investigation; too large costs primary steps the smoke does not
    # require to succeed (§8).
    investigate_max_steps: int = 120

    # provenance: MEASURED, and the measurement is `detour-3`. How many multiples of the
    # renderer's own scatter a rise must clear before the climb calls it a rise. The
    # scatter itself is measured per episode (`calibration.render_scatter`); this is the
    # only part left to choose.
    #
    # **1.0 cost 4 of 7 source-reaches.** `detour-3` ran at one sigma and returned 3/20
    # against 7/20, with median walked distance collapsing 9.75 m -> 4.00 m: an agent
    # that turns instead of moving. The reason is in the same run's slopes — the reached
    # arm climbs 2.18e-2 per metre, which at 0.25 m/step is ~5.5e-3 of rise per step,
    # against a scatter of ~2-3e-3. Per-step signal and per-step noise are the SAME
    # ORDER, so a threshold at one sigma sits inside the distribution of genuine rises
    # and vetoes most of them.
    #
    # The window is what buys the margin, not the threshold: a median of
    # `RISING_WINDOW` readings has a standard error near `scatter / sqrt(window)`, which
    # at 5 is 0.45. That is the principled ceiling and this default sits under it.
    # Zero is not "no protection" — it is the window doing the work alone, which is the
    # arm `detour-3` never ran.
    rising_eps_scale: float = 0.45

    # provenance: fake — how far ahead the realizable detour puts the probe point it
    # routes to. The climb reads a *direction* from live energy; this turns it into a
    # place, so the navmesh follower can go there around whatever is in the way.
    #
    # Sized against `PlannerConfig.compass_dist_m` (1.5 m), a little longer so the probe
    # clears the obstacle that blocked the last forward rather than landing on it. Too
    # short and the snap puts it back at the agent's feet; too long and the cue that
    # chose it is stale before the follower arrives — which is what `decision_period`
    # bounds on the primary task and what the per-step re-query bounds here.
    investigate_probe_m: float = 2.0

    # provenance: fake — the heading offset a stall applies to the probe, in degrees.
    # The carried rule answers `turn_left` / `turn_right`; this is how far that turn is
    # worth in a *place* rather than in an action. It is deliberately larger than the
    # simulator's 30-degree turn increment: a probe one turn-step off the blocked
    # heading snaps back onto the same obstacle, which is the livelock ticket 26
    # measured.
    investigate_probe_turn_deg: float = 60.0

    # provenance: source — the ORACLE arm's arrival radius, and **only** the oracle
    # arm's (§4.2). In the realizable arm arrival is peak-or-plateau plus visual
    # confirm, and "reached the source" is a measured distance-at-STOP; ADR-0001's
    # asserted ~1 m ceiling was a grid-resolution artefact and is retired.
    investigate_arrive_radius_m: float = 1.5


@dataclass(frozen=True)
class DetectorConfig:
    """What makes a detection believable. Shared by both `GoalDetector` arms."""

    # provenance: source — the oracle arm's STOP radius, set to Find-SR's primary ring
    # (1.0 m, `CONTEXT.md`) so the agent stops where the metric counts it. **This is
    # the oracle STOP the smoke runs**, and the required disclosure rides with it:
    # `diagnose_spin` decomposed the 0.031 benchmark SPL as stop_miss ~50% +
    # explore_timeout ~45% + success ~5%, so an oracle STOP deletes the stop_miss half
    # outright and smoke find numbers are not capability numbers.
    oracle_radius_m: float = 1.0

    # provenance: source — the L3 snap gate, on the **floor plane (xz)**, never in 3D.
    # Furniture sits above the floor and `snap_point` drops a back-projected surface
    # point onto the navmesh, so a correct localization of a chair seat shows a large
    # vertical jump with a small horizontal offset. The 3D form rejected exactly those
    # (fixed in `3307f19` / `7fbf370`), and every consumer of a waypoint reads only
    # (x, z).
    max_snap_m: float = 0.5

    # provenance: source — carried from `DETECTOR_SNAP_FLOOR_EPS`. A point back-projected
    # this far BELOW the snapped navmesh height is depth overshoot — a marginal box whose
    # centre pierced past the object to a point underground — and is rejected before the
    # floor-plane gate can rescue it. Measured once on the box: a 0.058-score OWLv2 box
    # back-projected 0.76 m below the navmesh with only 0.21 m of horizontal offset.
    snap_floor_eps_m: float = 0.30

    # provenance: fake — how close the localized object must be for `detects()` to mean
    # "here". This is the caption arm's STOP radius and it has never been measured; the
    # arc that would have measured it is closed (detector OFF strictly dominates, Run
    # 11), so `CaptionDetector` ships live and untested until R2 by ticket 10's
    # disclosed choice. Set at Find-SR's primary ring, like the oracle's, so the two
    # arms answer the same question.
    here_radius_m: float = 1.0

    # provenance: source — the median window `robust_depth_at_pixel` takes around a
    # bbox centre, in pixels. Rejects NaN, 0.0 (habitat's "no return") and inf.
    depth_patch_px: int = 5
