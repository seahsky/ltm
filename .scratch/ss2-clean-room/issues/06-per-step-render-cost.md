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

## Note added by ticket 03

**Good news for the budget: scene geometry is uploaded once, not per step.**
The whole mesh upload in `AudioSensor::runSimulation` is wrapped in `if (newInitialization_)`, so `loadMesh` / `loadSemanticMesh` run on the first simulation after `createAudioSimulator` and never again.
Only `RLRA_Simulate` re-runs per step.
Measure the two separately: first-call cost (geometry upload plus simulate) against steady-state cost (simulate only).
The ticket's affordability verdict hangs on the steady-state number, and the first-call number is a per-episode constant.

**The materials A/B this ticket was asked to run is now mostly answered, and its scope changes.**
Ticket 03 established that materials are off by default on this branch, that plain HM3D has no semantic scene at all, and that HM3D-Semantics v0.2 appears to break the semantic mesh path entirely (ticket 12).
So "render with `enableMaterials` off and on and compare" is not a meaningful A/B on HM3D.
What replaces it:

- **Add a gradient-contrast measurement to the sweep.** Fix a source, walk a navmesh path toward it, record broadband IR energy per step. Report the Spearman correlation between energy and negative geodesic distance, plus the far-to-near dynamic range in dB.
  Ticket 03 argues from the engine's physics that a uniform-absorption world still yields a climbable gradient, because every load-bearing term (direct-path spreading, occlusion, diffraction) is geometric and material-independent. What it cannot argue is **contrast**: a reflective built-in default flattens the within-room field and compresses the final-metres gradient.
  That number is the one thing section 7 of ticket 03 could not settle from source, and it belongs here because it is the same walk the timing sweep already does.
- Run it under HM3D materials-off, and under MP3D materials-on/off if ticket 08 keeps MP3D in play.
