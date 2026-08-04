# 23 — The `agent/` module

Type: task
Status: resolved
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

---

## Built, 2026-08-04 — Mac-green, and the frame was 180 degrees out

Eight modules (1,004 code lines against the three old files' 1,232), **187 new Mac tests
(235 → 422 green)**, ruff clean over 24 files, and one box file whose subject is the frame.
`proposers.py` is 238 code lines against `frontier_planner.py`'s 725, so the ticket's
"~300 LOC" target lands.

**The surviving arithmetic is bit-identical, and that was measured rather than asserted.**
A throwaway equivalence harness loaded the old `frontier_planner.py` and
`anomaly_controller.py` by path and compared them against the new modules: frontier-cell
extraction and clustering over 40 random grids each, the greedy climb over all 42
combinations of seven energy histories × three lateral signs × two confirms, and the two
score kernels plus `FrontierPhysicsScorer`'s planner branch over 100+ input combinations
(those three transcribed from source rather than executed, because
`memory_bridge.FrontierPhysicsScorer` drags faiss). **Two mismatches, both the same one:**
`wrap_pi(-pi)` returns `+pi` where the old returned `-pi`. Same angle, and the interval is
documented as `(-pi, pi]`; the only consumer of the sign is the turn rule, and a target
*exactly* behind is measure-zero either way. Everything else agrees to 1e-12.

### The finding: the old occupancy frame was 180 degrees out from habitat's

Read from source, not from a run. `habitat_env.py:620` extracts the yaw as
`atan2(2(wy+xz), 1-2(y²+z²))` — the rotation angle about `+y`, carried verbatim into
`sim/world.yaw_from_quaternion`. `episode_runner.py:1439` hands it to
`FrontierPlanner.update` unmodified. And `frontier_planner.py:541-557` marches along
`(sin theta, cos theta)`.

Habitat's agent forward is `-z` at zero yaw, so the world-frame forward is
`(-sin yaw, -cos yaw)`: the negative of what the splat used. **The whole depth cone landed
point-reflected through the agent, so the map was a 180-degree rotation of the room.** Two
knock-ons, both internally consistent and both wrong against the simulator:

- **The candidate bearing** (`episode_runner.py:2371`, `atan2(dx, dz) - yaw`, same
  convention) made a frontier straight ahead read as `|bearing| ~ pi`, so
  `FrontierPhysicsScorer`'s bearing-alignment term systematically preferred candidates
  **behind** the agent. Invisible in the scorer because only the magnitude is read.
- **The detector's back-projection** (`goal_detector.py:196`, "camera +Z is forward", with
  `episode_runner.py:2985` building the transform from the agent's *base* position) placed
  every detection behind the agent **and 0.88 m too low**.

The predicted symptoms are the recorded ones: no A\* path on ~92% of steps,
`n_waypoint_unreachable` 60-99/ep, `min_d2g` stuck around 8 m, `n_detector_localized` **0
across the whole c1 matrix**, and an L3 box back-projected 0.76 m below the navmesh.
**That is corroboration, not proof** — nothing here re-runs the old tree, and no closed
verdict is reopened (OWLv2's 0.031/0.058 box scores are a detector-quality finding that
survives regardless). What matters forward is that the clean room does not inherit it.

Two details worth keeping: the old **height gate was accidentally correct**, because a
camera 0.88 m too low was measured against a floor 0.88 m too deep and the two cancelled —
so the same test is written here with neither offset present. And the yaw *extraction*
was right all along; only the direction derived from it was wrong.

**The fix is structural, not a sign flip.** The frame algebra lives in exactly one module
(`agent/occupancy.py`: `forward_xz`, `right_xz`, `heading_to`, `bearing_rel`,
`camera_to_world`), and its three consumers — the splat, the candidate bearing, the
detector — all read it. That is what ticket 09's finding was missing: the convention was
written down in two places and checked in none.

### The check that would have caught it, at both layers

- **Mac** (`tests/mac/test_agent_frame.py`): `agent/` may not import `audio/` (ADR-0013),
  so nothing *inside* the package can compare the two frame consumers — but a test may
  import both. It asserts `occupancy.right_xz` and `audio.lateral.bearing_lateral_sign`
  name one axis, and that the two sign conventions **compose**: a source the audio layer
  calls `+1` (right) must be one the turn rule steers *right* toward. The controller's
  stall branch reads exactly that pair. One of these went red on first run and the *test*
  was wrong (it compared two signs where both modules abstain three ways on a source dead
  ahead) — which is itself the point: agreement had never been stated anywhere.
- **Box** (`tests/box/test_agent_frame_box.py`, new): whether they agree with the
  **simulator** is behaviour we did not write, so it is box-only (ADR-0014). It acts and
  measures: `move_forward` at four yaws must displace the agent within 20 degrees of
  `forward_xz`, `turn_left` must *increase* the yaw by ~30 degrees, a real depth frame's
  free cells must be overwhelmingly in the forward half-plane, and a back-projected centre
  pixel must land in front at its own depth. **The inverted frame fails all four.** It
  prints its measurements. `box_gate.sh` discovers, so nothing needed registering.

### Four more corrections, one of them found by a test going red

**(a) The investigate divert could lose the pick.** The old scorer gave it `score = 1.0`
and called that the max physics score — true only because the final rerank blended
`0.30 * S_sim + 0.70 * S_phys` and the memory term broke ties. With memory out of the
build the physics score *is* the final score and the geometric branch is clipped to 1.0,
so a maximal frontier ties the divert exactly and the emission-order tie-break hands it
the pick — i.e. the anomaly interrupt becomes advisory. Found by
`test_the_divert_beats_the_best_possible_frontier` going red. The divert is now an
**override by sort rank**; its 1.0 survives only as the number the audit record carries.

**(b) `DETECTOR_SNAP_FLOOR_EPS` cannot carry as an environment variable.** This ticket
names it; ADR-0008 removed the flag surface and `test_no_env_flags.py` enforces it. What
carries is the **gate**, as `DetectorConfig.snap_floor_eps_m`, with the old spelling in a
comment so the behaviour stays greppable to `3307f19`/`7fbf370`.

**(c) The detector's failure log cannot carry.** `_debug_log` appended a JSON line per
failure to a path the detector held, and ADR-0013 makes `report/artifacts.py` the only
module in the tree that writes anything. It is per-reason counters plus a `last_rejection`
string instead — the same information, in the artefact that already exists to hold it, and
the thing that made the c1 arc diagnosable at all.

**(d) `_random_walk_candidate` does not carry.** The live path never used it:
`propose_diverse` swapped it wholesale for the compass fan (`frontier_planner.py:1014`).
The no-frontier branch is the fan directly, and the two-step dance collapses.

### Four disclosed deviations

1. **`ControllerState` is frozen and `step_controller` returns `(state, decision)`.** The
   transition table is carried unchanged — every guard, ordering and one-shot directive,
   verified bit-identical above — but the in-place mutation is gone. This is the
   discipline `audio/onset.py` already applies to `OnsetState`, for its reason, and it is
   the same fix ADR-0013 already made when it moved `build_report` out. `reset()` does not
   exist because a new episode is a new value.
2. **`detects(obj)` needed a reach test that `locate()` did not have.** The old method
   returned a waypoint for the runner to approach; the seam returns a verdict, so
   "somewhere" had to become "here". Everything before it is carried; the radius is
   `DetectorConfig.here_radius_m`, set at Find-SR's 1.0 m ring so both arms answer the same
   question.
3. **A third dataclass where the ticket named two.** `DetectorConfig` holds the four
   numbers that decide whether a detection is believed. The alternative is constructor
   defaults, which no run record captures — and a number that gates a STOP and is invisible
   in `env_report.json` is the class of thing this map keeps finding after the fact.
4. **The arm reaches the controller as a `bool`, not as `Localization`.** ADR-0013 puts the
   enums in the root `config.py` and the layer graph forbids `agent/` importing it, so the
   bool is forced rather than chosen; ticket 25 maps the enum to it. The realizable arm is
   also the **default** here, because it is what the smoke runs (§8) — the old flag
   defaulted to the oracle path only so prior runs stayed byte-identical, and those runs
   are being deleted.

### One consequence for ticket 25's dataset builder

`SOURCE_PSEUDO_GOAL` carries, but in the realizable arm arrival is peak-or-plateau **plus
visual confirm**, and there is nothing to visually confirm about a sentinel. So an episode
whose source has no named object can only leave INVESTIGATE through the step-budget abort.
The builder should name the object; `investigate_aborted` in the funnel is what shows up if
it does not.

### An adversarial review pass found six defects, five of them in the tests

Run against the finished modules with the spec, the two ADRs and the old files in hand. It
independently re-derived the frame and cross-checked all three consumers against a
from-scratch reference over 5 yaws × 20 pixels × 3 depths — **max error 3.7e-15** — and
found no geometry defect and no structural violation. What it did find was mostly **tests
that would pass if the code were wrong**, which is the more useful result:

1. **`world_to_grid` floors where the old grid truncated, and that was an undisclosed
   divergence.** A real change and a fix — `int()` rounds toward zero, so a point in the
   half-open cell just *outside* the low edge aliased onto row/col 0, which is how a
   discarded ray endpoint gets written into the grid's edge cell. The equivalence harness
   above did not cover it (it compared extraction, clustering and the kernels). Now
   disclosed in the docstring, and **the test could not tell flooring from truncation**: it
   probed either side of the *world* origin, where a grid centred at 0 has its own origin at
   −2.0 and both quotients are positive. Re-pointed at the grid's low edge.
2. **`same_floor_m` had two homes.** My own late edit added the `PlannerConfig` field and
   left `reachable_pool`'s literal default in place — two values that can drift, with the
   effective one absent from `env_report.json`, which is exactly what `DetectorConfig`'s own
   justification argues against. `reachable_pool` now takes the config and has no local
   default, with a test that a non-default tolerance actually changes the verdict.
3. **`test_the_interpretation_can_be_forced` was vacuous.** `image_hw=(1000, 1000)` makes
   the normalization scale factor exactly 1.0, so it passed for `True`, `False` **and**
   `None` — it would have passed with the parameter deleted.
4. **The out-of-image bbox test passed for the wrong reason**, so `parse_qwen_bbox`'s bounds
   branch was never exercised: max coordinate 20 against a 15-pixel image triggers
   auto-detect, which scales the box to zero area and the *zero-area* guard drops it.
   Deleting the bounds check left the suite green. Now forced to pixel space, plus the
   positive case.
5. **The box test's back-projection assertions were two-thirds tautology**, and the half it
   existed for was only printed. The centre pixel sits exactly on the principal point, so
   `along == depth` and `lateral == 0` hold for any frame convention. It now also
   back-projects the **bottom-centre** pixel — which looks down and so hits the floor — and
   asserts its height sits between the floor and the sensor. That is the assertion the
   0.88 m camera-origin error fails, and it is measured against the simulator's own sensor
   placement rather than against our arithmetic. The two consistency checks are kept and
   **labelled as consistency checks**.
6. **`PlannerConfig.forward_fov_deg` / `eye_height_m` and `camera_sensor_specs`'s defaults
   are four independent literals with nothing asserting they agree** — and `config.py`
   itself says a disagreement "silently scales the map". `agent/` cannot import `sim`, but a
   *test* sits outside the layer graph and a **static** test sits outside the Mac/box split
   too, so the defaults are read out of `sim/world.py`'s `ast`, the same mechanism
   `test_layering.py` uses. Closes: someone widens the sensor to `hfov=90`, the splat keeps
   back-projecting at 79 degrees, the map skews, every test stays green.

Two more it raised that were **not** defects, checked rather than accepted. The
`n_candidates <= 0` truncation hole is real but reachable only from a degenerate frozen
config; fixed anyway with the `max(1, ...)` the compass fan already had. And its claim that
the old `_infer` extracted the assistant span before parsing is **wrong** — `_infer`
returned the full decode and the extraction only ever fed the debug log, so the new code
matches. But the *hazard* underneath it is real and was undocumented: the prompt inlines the
box tokens as a format hint, and the parse survives the echo only because the hint spells
its coordinates `(x1,y1),(x2,y2)` and the pattern requires digits. Now stated in
`Grounder`'s contract with a test pinning the placeholder case.

Mac suite 417 → **422 green**.

### The structural invariants are non-vacuous on `agent/`, verified by planting

Ticket 21's shape, applied to a tree that now exists. Five violations were planted in
`agent/controller.py` one at a time and the suite re-run: an attribute reach on the
line-of-sight probe, a `getattr` **string constant** for the same name, `agent/` importing
`sim`, `agent/` importing `audio`, and an `os.environ` read. **All five fired.** The file
was restored and compared byte-for-byte afterwards.

### Everything else the ticket asked for

`agent/` imports neither `sim` nor `audio`, and nothing in it names the line-of-sight probe
— `test_analyst_only.py` was armed and scanning nothing, and now scans a real `agent/`.
`test_walker_scope.py` gained `agent/controller.py` for the same reason ticket 21 added
`sim/world.py`: a walker that stopped reaching it would make that invariant pass by finding
nothing to check. `investigate_max_steps` defaults 40. The reachability invariant is
unconditional; an empty pool **raises** with a diagnosis naming which stage ate the
candidates, because the agent stands on the navmesh and a fan around it that yields nothing
is a broken episode rather than a degraded one. `CaptionDetector` ships live: every gate it
applies is pure and is pinned here, which is the difference between untested and unwritten
— but no run has exercised it against a real VLM, and that stays R2's, as ticket 10 chose.
