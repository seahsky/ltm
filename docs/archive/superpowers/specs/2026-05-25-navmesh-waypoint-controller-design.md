# Navmesh point-goal waypoint controller — design

**Date:** 2026-05-25
**Branch:** phase2-readiness
**Status:** approved design → implementation

## Problem

The Phase-2 G4 ablation (valid, `--backbone remembr`, `runs/abl-s{1,2,3}-qwen`) FAILed
the gate on **C1 (backbone alive: n_success(setting 1) > 0)** — binary SPL is **0 on all
90 episodes**. Root cause (confirmed from the per-episode controller census, not guessed):

The chosen high-level waypoint (frontier / memory / remembr) is converted to a discrete
action by `FrontierPlanner.step_controller` (`frontier_planner.py:622`), which routes over a
**self-built occupancy grid from depth** using grid-A*. The grid disagrees with Habitat's
**navmesh** (which actually governs movement), so:

- A* finds a real path only ~8–24 of 249 steps (e.g. S1 ep0 `astar_path=18`,
  `astar_reachable_fallback=231`; ep1 `astar_path=12`, `astar_fallback=223`).
- 60–80 `collision_escape` per episode — the agent drives into geometry the grid missed.
- 76–98 forward actions/episode yet ends 3–28 m from goal: colliding and wandering, never
  routing. → SPL = 0 everywhere → C1 FAIL, and C2's soft-SPL is a degenerate proxy.

This is a **locomotion** failure, not a perception/memory one. The codebase already proves
the environment is navigable: the `oracle` backbone uses Habitat's `ShortestPathFollower`
(`episode_runner.py:678`) and reaches goals.

## Design

Replace the grid-A* locomotion with Habitat's navmesh **`ShortestPathFollower`, steering
toward the agent's SELF-CHOSEN waypoint** (`current_candidate.world_xy`) — *not* the
ground-truth goal. Every high-level decision is unchanged: which waypoint (frontier vs
memory vs remembr), the rerank, memory injection, and the keyword-STOP all stay exactly as
they are. Only *how the agent physically moves toward a waypoint it picked itself* changes.

This is **not** the oracle. The agent still does not know where the goal is; it navigates to
its own chosen waypoints and only succeeds if memory/frontier/remembr point it at the right
place and the keyword-STOP fires there. It is the standard ObjectNav decomposition (a
semantic policy picks the waypoint; a point-goal navigator executes locomotion) and is
faithful to how ReMEmbR runs on a real robot (it emits goal positions; a separate nav stack
does locomotion — ReMEmbR never does raw obstacle avoidance). **Eval claim becomes:** "given
a point-goal navigator, does hierarchical LTM improve object-goal search?"

### What stays (legitimate, perception-driven)

- The self-built occupancy grid (`planner.update`) and **frontier proposal** (`propose`):
  the agent still discovers where to explore from its own depth observations. Only
  locomotion uses the navmesh.
- The grid/controller census (`grid_stats`, `controller_stats`) stays for instrumentation.
- Keyword-STOP, memory injection, rerank, commit-to-candidate cadence — all unchanged.

### The controller

One `ShortestPathFollower` per episode, reused across waypoints (`get_next_action(goal)`
takes the goal each call, so a single instance serves any waypoint). Built lazily from
`self.source.get_sim()`.

`_waypoint_action(world_xy, agent_pos) -> int`:
1. If no follower yet, build it; if the source has no sim (cached mode), **fall back to the
   existing `planner.step_controller`** (preserves cached/sim-less behavior).
2. Build the 3D goal `[wx, agent_y, wz]` and snap to the navmesh
   (`sim.pathfinder.snap_point`); if the snap is non-finite, keep the raw point.
3. `raw = follower.get_next_action(goal)`; map to an action id with the same logic as
   `_oracle_action` (None → reached/unreachable; str → `_ACTION_NAMES.index`; int → pass).
4. **None handling — never STOP here** (only the keyword-STOP / explicit `stop_signal` may
   end an episode). On None, return `ACTION_TURN_LEFT` and signal the loop to re-propose next
   tick (set `current_candidate = None`), so an unreachable/reached waypoint yields a fresh
   pick rather than a premature STOP or an in-place spin.

Follower `goal_radius` = `self._waypoint_goal_radius` (default 0.5 m, ≈ `propose_reached_m`)
so "reached" aligns with the existing distance-based re-propose trigger.

### Wiring

In `_run_episode`, the non-oracle action derivation (`episode_runner.py:443-455`):
- `current_candidate is None` → `ACTION_FORWARD` (unchanged).
- `stop_signal` → `ACTION_STOP` (unchanged).
- else → `self._waypoint_action(current_candidate.world_xy, step.agent_state.position)`
  (was `self.planner.step_controller(...)`).

`planner.update(...)` still runs every step (grid + frontier upkeep). The follower is built
once at episode start for live non-oracle runs (or lazily on first `_waypoint_action`).

## Components touched

- `episode_runner.py`: add `_waypoint_goal_radius` attr; add `_init_waypoint_follower` +
  `_waypoint_action`; swap the `step_controller` call for `_waypoint_action`; handle the
  None→re-propose signal. `_oracle_action` / `_init_oracle_follower` unchanged.
- No change to `frontier_planner.py` (the grid controller stays as the cached-mode fallback),
  `memory_bridge.py`, `remembr_backbone.py`, or the dialogue path.

## Testing

Unit tests in `embodied_memory/scripts/test_propose_candidates.py`, stubbing a fake
`sim`/`pathfinder`/follower (no Habitat load — same pattern as the existing sanity cases):

1. Follower returns action name `"move_forward"` → `_waypoint_action` returns id 1; goal was
   snapped via `pathfinder.snap_point` and passed to `get_next_action`.
2. Follower returns int id (e.g. 2) → passed through.
3. Follower returns `None` → returns `ACTION_TURN_LEFT` **and** signals re-propose (a flag the
   loop reads to null `current_candidate`); never `ACTION_STOP`.
4. No sim (`get_sim()` is None) → falls back to `planner.step_controller` (called with the
   candidate + pose).
5. `snap_point` returns a non-finite point → the raw `[wx, agent_y, wz]` goal is used.

## Verification (on RACE)

1. Sanity suite green in the race-setup env.
2. Cheap smoke: `bash scripts/race-smoke.sh --backbone remembr --setting 3 --scenes
   wcojb4TFT35 --n-episodes 2 --target any --tag navmesh-ctrl`. Expect the controller census
   to change qualitatively — `collision_escape` collapses toward ~0, the agent reaches
   waypoints (distance-based re-propose fires), and path length / d2g improve. A **success
   (SPL>0)** on any episode is the goal but may need STOP-distance tuning (follow-up).
3. If locomotion is clean, re-run the G4 3×30 (`runs/abl-s{1,2,3}-qwen`) and read C1/C2.

## Out of scope (follow-ups, in order)

- **STOP-distance tuning** for success: the keyword-STOP fires at agent↔keyframe ≤ 1.5 m;
  Habitat success is to the goal viewpoints. If the agent routes but STOPs just outside the
  success radius, tune `REMEMBR_STOP_DIST` / the STOP trigger. Do this only after locomotion
  is verified clean.
- C2 (does memory help) — re-measured by the post-fix G4, not changed here.
- Perception embedding ceiling (ViT-B/32) — separate effort.
