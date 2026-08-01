# 07 — What is the rebuilt agent?

Type: grilling
Status: resolved
Assignee: Sky
Blocked by: 04 (resolved)
Resolved: 2026-08-01 — see the Answer section. ADR: `docs/adr/0008-clean-room-agent-architecture.md`

## Question

In the clean room, what does the agent actually consist of: how does it search, how does it know it found the primary goal, and when does it STOP?

## Why it matters

A clean-room rebuild reopens architecture decisions that were previously settled by accretion rather than by choice.
The prior stack was: geometric frontier planner, ReMEmbR backbone (Qwen2-VL-2B captioner + Qwen2.5-7B planner), SBERT-indexed LTM, anomaly controller on top.
With memory out of scope, most of that is no longer required, and the question is what minimum stack the audio spine needs.

The relevant history, so this is not re-derived from zero:
- ADR-0006 retreated to the **geometric** frontier as the spine. The BLIP-2 semantic frontier (S1+) was the fourth independent non-lift of a semantic frontier and is a documented negative.
- The searcher is weak in absolute terms: SPL 0.031 against VLFM's 0.304, roughly 10x under. That was accepted because the paper leads with the controller, not with absolute navigation numbers.
- Goal detection has been closed twice as a negative: caption-grounding (net-neutral to negative, detector OFF strictly dominates) and OWLv2 on GPU (noise floor on HM3D sim renders).
- Binary SPL at the 0.1 m ring is localization-bound. That finding survives the reset.

So the honest default is: geometric frontier, no detector, oracle-ish or geodesic STOP, and accept the absolute number.
But that default was reached under a *different* environment, and it is worth 20 minutes of grilling to ask whether the clean room should keep it or spend the freed VRAM differently.

The ticket-04 result constrains this hard. If the one env cannot hold torch plus a VLM, the answer is forced.

## What would resolve it

A grilling session covering:
- Search: geometric frontier rebuilt as-is, ported, or replaced by something else.
- Goal detection: none / geodesic oracle / a detector, given both detector arcs closed as negatives.
- STOP policy, and whether the 0.1 m localization bound is accepted again or attacked.
- Whether absolute find-performance matters at all for this map, given the destination is one green episode and the experiment matrix is out of scope.
- Which of these are decisions for *this* map versus decisions that belong to the follow-on memory effort.

Deliverable: a one-page architecture decision recorded as an ADR in the new tree, with the rejected options and why.

## Note added by ticket 04 (now resolved — this ticket is unblocked)

**The env does not constrain the model choice, so this is a design decision, not a capacity one.**

Ticket 04's GREEN measured the box: **Tesla V100-SXM3-32GB, 31.73 GB VRAM free, CUDA available, torch on GPU with an allocation smoke test passing**, in the same interpreter as the audio build.
The old worry — that the audio env could not hold the rest of the stack — is dead. With memory out of scope for this map there is no 7B planner and no VLM captioner competing for that VRAM either, so whatever this ticket picks, it fits.

**This absorbs a fog patch.** The map's *Not yet specified* carried "whether the geometric frontier searcher is rebuilt or replaced", explicitly waiting on ticket 04 for "what models can even run in the one env". The answer is *anything we would plausibly want*, so the question collapses back into this ticket's own scope — how the agent searches — and is no longer a separate unknown.

Two constraints worth carrying in, neither from capacity:

- **ADR-0006's retreat still stands on evidence, not on hardware.** The geometric spine was chosen after four independent non-lifts of a semantic frontier (the fourth being BLIP-2/S1+). A clean room reopens the question, but reopening it needs a reason better than "we have the VRAM now", because VRAM was never the reason it failed.
- **torch is pinned at 2.0.1 pending ticket 13**, which is likely to move it. Do not pick a model whose only supported runtime is a torch this env has not settled on yet.
  **Discharged by the answer:** the agent as decided runs no model at all in the smoke, and the one model it carries unused (Qwen2-VL-2B) already runs on this env's torch. Ticket 13's pin move cannot invalidate this ticket.

## Answer

**ADR:** `docs/adr/0008-clean-room-agent-architecture.md` — the one-page architecture decision this ticket asked for, with the rejected options and why.

The agent is a **candidate-pool frontier explorer with a detector seam and no LLM in the loop**.

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

### The six decisions

1. **Altitude: this is the agent R2 runs, built smoke-first.** Sequencing, not disposability. The seams are decided now because they are cheap now and expensive to retrofit; the parts behind them are deferred. Anything R2 might want is copied into the new tree before ticket 10's deletion, even if the smoke never calls it.
2. **Keep the candidate-pool seam.** Proposers → scorer → waypoint → follower. With one proposer it is a thin indirection, but it is what the follow-on memory effort plugs into and what ADR-0006 protected when it ruled out an end-to-end policy. It also gives the anomaly controller a clean place to sit: INVESTIGATE overrides the *pick*, it does not fight the planner.
3. **Rewrite the frontier proposer, keeping four pieces** — depth→occupancy, frontier extraction + clustering, geometric scoring, compass fallback. ~300 LOC against 1129. Drop A\* and the steering fallbacks (verified dead on the live path) and the semantic value head (ADR-0006's negative).
4. **One `GoalDetector` seam, oracle implementation first.** `detects(obj)` serves both the primary STOP and the anomaly CHECK. Smoke runs geodesic-oracle (zero VRAM, and the disclosed upper bound the arc has already used twice); R2 runs caption-grounded.
5. **Drop the Qwen2.5-7B planner, leave the seam open.** `n_remembr_chosen ≈ 0`; every measured memory win came from `propose_memory_candidates`, never from the LLM agent. Frees ~15 GB for live audio + CLAP.
6. **Carry the behaviour, not the flags.** Known-good default-OFF flags become unconditional and their invariant is asserted: *the candidate pool is never empty, and every candidate is navmesh-reachable and snapped*. Flags survive only for genuine experimental arms.

Plus one port: **the anomaly controller moves near-verbatim** (316 LOC, pure, Mac-testable, and the paper's only framing-independent positive). Only `visual_confirm` changes, rerouting to the detector.

### Facts established while resolving, that the ticket's framing had wrong

- **The 1129-LOC planner is mostly not what navigates.** `FrontierPlanner.step_controller` and its A\* are a no-simulator fallback; the live path is `_waypoint_action` → `ShortestPathFollower`. The live proposer is `propose_diverse` (+ compass fallback); `is_decision_step()` is ticked **for stats only** on the anomaly path, because the controller re-steers every tick.
- **The STOP is not a detector, it is a memory query.** `_stop_check` retrieves captions from ReMEmbR's flat caption store matching the goal (keyword or SBERT cosine), and fires STOP if a matching caption's recorded *position* is within `STOP_DIST_THRESHOLD`. So "keep caption-grounding STOP" means carrying a captioner **plus** a caption store **plus** SBERT, which is why it is a carry-list item rather than a smoke component.
- **`visual_confirm` and goal STOP are the same primitive.** Both ask "is object X here". This is what collapses CLIP out of the agent.
- **Spin is already closed and the ticket did not know it.** `diagnose_spin` on `r1spin2` returned **0% spin** in both arms; the 0.031 SPL decomposes as stop_miss ~50% + explore_timeout ~45% + success ~5%. Nothing to fix. The consequence for this ticket is sharper than the non-fix: **an oracle STOP deletes ~50% of the measured failure mass**, so the smoke will look far better than 0.031 for a reason that must be disclosed rather than enjoyed.
- **CLIP is still live in two places** the reset would otherwise have carried silently: `_anomaly_visual_confirm` (cos ≥ 0.26, never calibrated for that regime) and the ADR-0002 room classifier. The first is deleted here. The second is ticket 09's.

### What this costs, stated rather than buried

- **The paper stops claiming a ReMEmbR extension.** True after this: ReMEmbR-derived captioning, direct-retrieval memory injection, no LLM agent in the loop. More defensible than the current claim, since the LLM agent is the component measured never-chosen.
- **The smoke does not exercise goal detection at all.** Ticket 09's acceptance criteria must say so.
- **No run in the new tree is byte-comparable to any run in the old tree.** Accepted deliberately, as the price of dropping the flag surface.
- **CLIP is dropped on parsimony plus a suggestive prior**, not on a measurement in the close-range confirm regime. The cheap test if it is ever needed is the CapRL-gate pattern, not a matrix.

### Carry list handed to ticket 10

1. Caption-grounded detector stack (Qwen2-VL-2B wrapper, caption store, SBERT, `_stop_check` logic) behind the `GoalDetector` interface.
2. STM/LTM calculation, as the follow-on effort's reference implementation.
3. SPL / soft-SPL arithmetic, orphaned by dropping habitat-lab.
4. ObjectNav `.json.gz` episode loader, also orphaned, carrying ticket 08's open `scene_dataset_config` question.

### Explicitly not decided here

The ADR-0002 room classifier and whether CLIP returns for it; any amendment of the controller's localization policy for live continuous audio; the report's content; the smoke's acceptance criteria. All ticket 09.
