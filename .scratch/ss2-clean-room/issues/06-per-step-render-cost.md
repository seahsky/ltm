# 06 — What does live-every-step audio actually cost?

Type: task
Status: open
Blocked by: 04, 11

## Question

How many milliseconds does one in-sim audio render take on the RACE box, at default parameters and at a cheap preset, and does that make live-every-step affordable?

## Why it matters

"Live, every step" was chosen as a design decision before anyone measured it.
Rendering is Monte-Carlo path tracing at `indirectRayCount=5000`, `indirectRayDepth=200`, `irTime=4.0 s`, `threadCount=1`.
At a 500-step episode budget, 50 ms/step is 25 seconds of pure audio per episode and tolerable; 500 ms/step is four minutes per episode and forces the throttled variant instead.

The answer decides whether the destination as named is reachable, so it is worth measuring before any agent code is written.

## What would resolve it

On the box, in the ticket-04 env, with one HM3D scene loaded:

0. **Print the real defaults first.** Ticket 11 found they live inside the closed `.so` (`RLRA_ContextConfigurationDefault`) and that every number in ticket 01's table is unverified for our branch. Construct an `AudioSensorSpec` and dump every `acousticsConfig` field before timing anything, so the sweep has a real baseline.
1. Time `setAudioSourceTransform` + `get_sensor_observations()` at those defaults, over enough repeats to see variance (path tracing is stochastic).
2. Sweep the cost knobs one at a time and record the time/quality curve. **Names and defaults per ticket 11, not ticket 01** — `irTime` does not exist on our branch, and no default below is verified until printed on the box:
   - `maxIRLength` (was `irTime`) 4.0 → 1.0 → 0.5
   - `indirectRayCount` 5000 → 1000 → 500
   - `directRayCount` — new on our branch, ticket 01 never saw it
   - `threadCount` 1 → physical core count
   - `temporalCoherence` off → on
3. For each setting, also record whether the received energy still decreases monotonically as the listener moves away from the source along a navmesh path. Speed is worthless if the gradient stops being climbable.
4. Check whether cost depends on listener-source distance or scene size, since that determines whether the budget is per-scene stable.
5. **Sweep source count 1 → 2 → 3.** Ticket 02 found the engine is natively multi-source and that a wrapper patch exposes it at zero extra *renders* per step — but the `.so` is closed-source, so whether one `RLRA_Simulate` over 3 sources costs 1x or 3x cannot be read and must be measured. This decides whether the bed and distractor are real sources or post-render additions, so it is not optional.

Note two readouts ticket 02 found on the branch that this sweep should use:
- **`getRayEfficiency()`** gives a built-in quality number, so the ray-count sweep does not rest on eyeballing gradient monotonicity alone.
- **`sourceIsVisible()`** is a single-ray line-of-sight test, useful for labelling each timing sample as LOS or non-LOS, since diffraction-dominated paths are the expensive case.

Take the knob names and defaults from ticket 11, not ticket 01 — ticket 01 was researched against a different branch.

Deliverable: a table of ms/step against setting, a recommended preset, and an explicit verdict on whether live-every-step holds or whether the map's destination needs amending to the throttled variant.
