# 23 — The `agent/` module

Type: task
Status: open
Blocked by: 20 (resolved 2026-08-04)

## Question

Build ADR-0008's candidate-pool explorer: proposers → reachability filter → scorer → waypoint, plus the detector seam and the anomaly controller.

`agent/` does **not** import `sim` — `snap_point`, `geodesic` and the oracle's distance function are injected (ADR-0013), so the whole module unit-tests on this Mac.

## What to build

- **`occupancy.py`** — depth → occupancy grid integration.
- **`proposers.py`** — the geometric `FrontierProposer`, **rewritten** to ~300 LOC against the old file's 1129. Four pieces survive: depth→occupancy integration, frontier-cell extraction and clustering, geometric candidate scoring, and the compass fallback for when no frontier exists.
- **`reachability.py`** — the navmesh filter and snap, with injected callables. This is ADR-0008's **invariant**, unconditional and no longer behind `REMEMBR_ANTITHRASH_SINGLEGOAL`: *the candidate pool is never empty, and every candidate in it is navmesh-reachable and snapped.* An ungated proposer emits waypoints the follower can never route to (`n_waypoint_unreachable` 60–99/ep, `min_d2g` stuck around 8 m).
- **`scorer.py`** — picks one waypoint from the pool.
- **`detector.py`** — the `GoalDetector` protocol with `detects(obj)` serving **both** the primary STOP and the anomaly CHECK, shipping `OracleDetector` (the smoke) and `CaptionDetector` (R2). Keep `parse_qwen_bbox`, `robust_depth_at_pixel`, `back_project_pinhole`, and the L3 floor-plane (xz) snap gate with `DETECTOR_SNAP_FLOOR_EPS`. **Drop** `_ensure_owlv2` / `_infer_owlv2`. Note the old class exposes `locate(...)`, so a straight port would not have fitted the seam anyway.
- **`controller.py`** — `NavMode`, `ControllerState`, `step_controller`, `realizable_investigate_step`, `is_diverting`. Ports near-verbatim **except** that `build_report` moves out to `earshot/report/` (ADR-0013's disclosed deviation from ticket 10) and the state mutation goes with it.
- **`config.py`** — `PlannerConfig`, `ControllerConfig`. `investigate_max_steps` defaults 40 (task spec §9, a number to set against measurement).

## Dropped, per ADR-0008

The 7B LLM planner, CLIP (both consumers), grid A\* and every steering fallback, the semantic value head, and the env-flag surface. Flags survive only as `RunConfig`'s two arm enums.

## Done when

Mac tests green for the controller (a pure decision function over `(energy_history, lateral_sign, visual_confirm)`), the reachability filter, and `OracleDetector`. `CaptionDetector` ships live but **untested until R2** — a disclosed cost from ticket 10, chosen over letting the seam ship with one side.

## Watch for

`realizable_investigate_step`'s greedy rule is unchanged — rising loudness means forward, peak-or-plateau plus visual confirm means STOP, a stall turns toward the louder half-plane — but §4.1 changed the frame underneath it. The controller consumes `lateral_sign` from ticket 22; do not re-apply the grid-era `heard == -right(world-bearing)` compensation.

`investigate_arrive_radius_m` (1.5 m) is **not** an arrival criterion in the realizable arm and survives only in the oracle arm (§4.2). Arrival is peak-or-plateau plus visual confirm; "reached the source" becomes a measured distance-at-STOP.
