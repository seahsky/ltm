# The clean-room agent is a candidate-pool frontier explorer with a detector seam, and no LLM in the loop

**Status:** accepted (2026-08-01, grilling session on ticket 07 of the `ss2-clean-room` map).

The SoundSpaces 2.0 clean-room rebuild reopens the agent architecture, which the old tree settled by accretion rather than by choice.
This ADR fixes what the rebuilt agent consists of: how it searches, how it knows it found the primary goal, when it STOPs, and which parts are deferred.

It is a **durable** decision, not a smoke-only one.
The smoke is built minimum-first and is the first thing this agent does, but the seams are decided now because they are cheap to decide and expensive to retrofit, and because ticket 10 deletes `embodied_memory/` once the smoke is green.

## The architecture

```
depth ─► exploration state (occupancy grid, agent-estimable)
          │
          ├─► proposers: [FrontierProposer]        ← memory plugs in here later
          │
          ├─► navmesh reachability filter + snap    (invariant, unconditional)
          ├─► scorer picks one waypoint
          └─► greedy follower ─► action

GoalDetector.detects(obj) ──┬─► primary-task STOP
                            └─► anomaly CHECK visual_confirm

AnomalyController (pure) overrides the pick during INVESTIGATE / CHECK / RESUME
```

### Kept

- **The candidate-pool seam.** A set of waypoint proposers feeds one scorer that picks a waypoint; a navmesh follower turns that into actions.
  With a single proposer this is a thin indirection, but it is the shape the follow-on memory effort plugs into, and it is the shape ADR-0006 protected when it ruled out an end-to-end policy.
- **A geometric frontier proposer, rewritten.** Four pieces survive: depth→occupancy integration, frontier-cell extraction and clustering, geometric candidate scoring, and the compass fallback for when no frontier exists.
  Target size is roughly 300 LOC against the old file's 1129.
- **The anomaly controller, ported near-verbatim.** It is already a pure decision function over `(energy_history, lateral_sign, visual_confirm)` with no simulator dependency, so it is the one module that unit-tests on a Mac, and it is the paper's single framing-independent positive.
  The only change is that `visual_confirm` is rerouted through the detector seam.
- **A `GoalDetector` seam with two implementations.** `detects(obj)` answers "is object X here", and serves both the primary-task STOP and the anomaly CHECK.
  The smoke runs the geodesic-oracle implementation; R2 runs the caption-grounded one.
- **The structured report.** `build_report` returns a dict, so dropping the LLM costs nothing here. Its content is ticket 09's question.

### Dropped

- **The Qwen2.5-7B LLM planner.** `n_remembr_chosen ≈ 0` across the whole arc, and every measured memory win came from `propose_memory_candidates` (SBERT cosine → waypoint injection), never from the LLM agent.
  Dropping it frees roughly 15 GB on a V100-32GB that now also holds live audio rendering and CLAP.
  The pool seam means an LLM proposer can be added later without restructuring anything.
- **CLIP, both consumers.** `visual_confirm` collapses into the detector seam. The ADR-0002 room classifier is ticket 09's call and is the only route by which CLIP returns.
- **Grid A\* and every steering fallback.** Already dead in the live path: `FrontierPlanner.step_controller` is a no-simulator fallback, and steering is the navmesh follower.
- **The semantic value head** (`observe_value` / `_semantic_value_at`). ADR-0006's documented negative, the fourth independent non-lift of a semantic frontier.
- **The env-flag surface.** The clean room carries the *behaviour* the flags gate, not the flags.
  Every default-OFF flag whose ON state is known-good becomes unconditional, and the invariant it protected is asserted instead.
  The concrete case: `_navmesh_reachable_frontiers` is default-OFF behind `REMEMBR_ANTITHRASH_SINGLEGOAL` purely so prior runs stayed byte-identical, and those runs are being deleted.
  Flags survive only where the choice is a genuine experimental arm: oracle vs realizable localization, oracle vs caption-grounded detector.

### The invariant that replaces the flags

The candidate pool is never empty, and every candidate in it is navmesh-reachable and snapped to a navmesh point.

This is the same discipline ticket 12 applies to the audio context, applied to the planner.
It matters because the depth-derived occupancy grid disagrees with the navmesh, so an ungated proposer emits waypoints the follower can never route to (`n_waypoint_unreachable` 60–99/ep, `min_d2g` stuck around 8 m).

## Why this is decided on evidence rather than on freed VRAM

Ticket 04's GREEN measured 31.73 GB free on the box with the audio build loaded, so capacity does not force any of this.
The note on ticket 07 was explicit that reopening ADR-0006 needs a better reason than "we have the VRAM now", because VRAM was never why the semantic frontier failed.

Each drop above rests on a measurement already in the record:

- LLM planner: `n_remembr_chosen ≈ 0`.
- Semantic value head: four independent non-lifts, the fourth being BLIP-2/S1+ at a GREEN vacuous-arm gate (13,405 scores, spread 0.45), so the lever is inert rather than unfired.
- A\* and steering fallbacks: dead code on the live path, verified by call-site inspection.
- Spin robustness: `diagnose_spin` on `r1spin2` returned **0% spin** in both arms, so there is nothing to fix and the anti-spin behaviour is simply carried.

## Considered and rejected

- **Port `frontier_planner.py` verbatim.** Maximum fidelity to ADR-0006's "deliberately-frozen backbone" and the cheapest path to green.
  Rejected: it moves roughly 800 LOC of known-dead and known-negative code into the new tree on day one, which costs the clean room its point.
  The freeze survives in the sense that matters: same algorithm, same geometric scoring, no capability added or removed.
- **Carry the caption-grounded detector into the smoke.** Tests the real thing end to end with no later integration.
  Rejected: roughly 5 GB and a much heavier first-green, coupling the audio spine's proof to a stack measured localization-bound.
- **Keep the LLM planner, or a small one (Phi-3.5-mini, already GREEN at the L3 gate).** Preserves the "extends ReMEmbR" framing at 4 GB instead of 15.
  Rejected: keeping a measured-inert component for a naming reason is the accretion this rebuild exists to undo.
- **No STOP, run to the step budget.** Honest about the localization bound, and the simplest thing that could work.
  Rejected: it removes Find-SR and benchmark SPL, so R2 would have no primary-task number at all.
- **A trained detector.** Rejected twice already on measurement: caption-grounding is net-neutral to negative with detector-OFF strictly dominating, and OWLv2 on GPU sits in the noise floor on HM3D sim renders (max box score 0.031 base / 0.058 large).

## Consequences

**The paper stops claiming a ReMEmbR extension.**
What is true after this: ReMEmbR-derived captioning, direct-retrieval memory injection, no LLM agent in the loop.
That is the more defensible claim, because the LLM agent is the component the arc measured as never chosen.

**Smoke numbers are not capability numbers.**
`diagnose_spin` decomposed the 0.031 benchmark SPL as stop_miss ~50% + explore_timeout ~45% + success ~5%.
An oracle STOP deletes the stop_miss half outright, so the smoke will look far better than 0.031 for a reason that must be disclosed rather than enjoyed.
The smoke does not exercise goal detection at all, and ticket 09's acceptance criteria must say so.

**No run in the new tree is byte-comparable to any run in the old tree.**
This follows from dropping the flag surface and is accepted deliberately.

**CLIP is dropped on parsimony plus a suggestive prior, not on a measurement in this regime.**
CLIP is measured flat on HM3D sim renders (separation 0.020 against a 0.05 bar, three independent measurements), but that was frontier value at distance, not close-range object confirmation.
If close-range confirmation later proves to need a visual channel the caption detector cannot supply, this is the decision to revisit, and the cheap test is the CapRL-gate pattern (render frames at known object viewpoints, measure within-vs-between separation) rather than a matrix.

**Carry list for ticket 10.** These move into the new tree before the deletion, whether or not the smoke calls them:

1. The caption-grounded detector stack: Qwen2-VL-2B captioner wrapper, caption store, SBERT, and the `_stop_check` near-agent match logic, behind the `GoalDetector` interface.
2. The STM/LTM calculation, as the follow-on memory effort's reference implementation.
3. The SPL / soft-SPL arithmetic, orphaned by dropping habitat-lab.
4. The ObjectNav `.json.gz` episode loader, also orphaned by dropping habitat-lab, carrying ticket 08's open question about which `scene_dataset_config` the dataset resolves against.

**Deferred to ticket 09, not decided here.** The ADR-0002 room classifier and whether it returns CLIP to the tree; any amendment of the controller's localization policy for live continuous audio; the report's content; and the smoke's acceptance criteria.
