# 09 — Re-derive the anomaly-response task spec on live audio

Type: grilling
Status: resolved
Blocked by: none (02, 06, 08 all resolved)

## Note added by ticket 06 (resolved 2026-08-01)

Three inputs this ticket was waiting on:

1. **Cost is not a constraint on the task spec.** Live-every-step holds at **27.2 ms/step** on the `cheap_preset` (`indirectRayCount=500, indirectRayDepth=50, threadCount=4, temporalCoherence=1`) = 13.6 s per 500-step episode. Design the task for what it should test, not for a render budget.
2. **The multi-source patch is not budget-gated.** Three sequential renders is ~82 ms/step, still inside budget. So ADR-0002's room-normal distractor and ADR-0004's background bed can each be a *real positioned source* if the task wants them, at a cost that does not threaten anything. **This ticket now decides the ~40-line patch purely on onset provenance** — per-source IRs make "which source did this energy come from" structural rather than re-inferred from a summed signal, which is exactly the failure that invalidated the `anommxv` headline. That is the whole argument now; cost has dropped out of it.
3. **`transmission` and `diffraction` cost ~10% each**, so ADR-0003's floor constraint cannot be retired or kept on performance grounds either.

One caution on that last point. Ticket 06 measured that turning `diffraction` off still left a climbable gradient (rho_nlos −0.99), but its walks approach a source **on the same floor along a navmesh path**, where direct and transmitted paths dominate. That measurement is insensitive to what diffraction actually buys. **Do not read it as "diffraction is dispensable"** — a source around a corner or a floor away is untested, and that is precisely the regime ADR-0003 is about.

## Question

The task given to the robot is unchanged, but several of its constraints were workarounds for the precomputed grid.
Which of them survive live in-sim rendering, and what is the task spec now?

## Why it matters

Four ADRs encode constraints that exist *because* audio was a baked grid:

- **ADR-0003, floor-constrained source.** The source must sit on the goal's floor, because the grid renders one floor and `nearest` resolves by xz, so an off-floor source fabricates audio. Live rendering removes the grid entirely. With `transmission` enabled (ticket 01) a source a floor away produces real attenuated audio. This constraint is a candidate for retirement, and retiring it changes what the task tests.
- **ADR-0004, background bed as noise floor.** `bg_gain` was calibrated against the bed so it never triggers the interrupt. Under live rendering the level calibration point is `globalVolume`, and the old `onset_rms=0.065` is meaningless. This must be recalibrated from scratch, and if ticket 02 says single-source, the bed needs a mechanism at all.
- **ADR-0001, realizable localization.** The stated ceiling was "~1 grid cell (~1 m)" because the grid was the resolution limit. Continuous receiver positions remove that ceiling. The energy-gradient climb may now be genuinely climbable rather than quantised.
- **ADR-0002, scene-conditioned normality** and the room-normal distractor. Depends on ticket 02's source-count answer.

There is also unfinished business the reset should not silently inherit: the `anommxv` headline was invalidated by three structural breaks, one of which was the interrupt firing on the background bed at step 0 rather than on the alarm. Onset provenance has to be designed in, not bolted on.

## What would resolve it

A grilling session, after 02, 06 and 08 land, covering:
- Source placement rules, and whether the floor constraint is retired, kept as a simplification, or replaced by a transmission-aware rule.
- How the bed and the room-normal distractor are produced under the source-count reality.
  Ticket 02 turned this into a real choice rather than a forced workaround: habitat-sim exposes one source, but the engine is natively multi-source with **one IR per source**, and a ~40-line wrapper patch reaches it at zero extra renders per step. So the options are (a) patch and make the bed and distractor genuine positioned sources, (b) post-render diotic bed with no distractor mechanism, (c) sequential re-render at Nx cost. Decide this here; ticket 06's source-count sweep supplies the price of (a).
  Weigh in particular that per-source IRs make **onset provenance structural** — the `anommxv` break where the interrupt fired on the bed and was read as the alarm becomes impossible by construction, rather than something a heuristic has to disentangle from a summed signal. That is a strong argument for (a) independent of cost.
- Whether `sourceIsVisible()` (a single-ray LOS test the branch exposes for free) becomes a first-class per-step annotation, given the deferred "non-LOS but audible" seed idea depends on exactly this distinction.
- Level calibration: how `globalVolume`, the bed level and the onset threshold are set, and by what measurable gate rather than by hand.
- Onset provenance as a first-class output, not a log line.
- Investigate and resume criteria: what counts as reaching the source now that positions are continuous.
- The metric set, restated against `CONTEXT.md` (Find-SR, anomaly-response SR, benchmark SPL, cost).
- Which ADRs are superseded, and by what.

Deliverable: the task spec for the new tree, plus superseding ADRs for whichever of 0001 to 0004 no longer hold.

## Note added by ticket 08 (resolved 2026-08-01) — the dataset is fixed, and one option is closed

**HM3D, `minival` for the smoke, acoustic materials permanently off** (`docs/adr/0007-hm3d-stays-mp3d-out-of-scope.md`). MP3D is out of scope, so any task-spec option that leaned on material-dependent room character is closed before this ticket starts.

Two things this ticket can now take as given:

- **The acoustic world is uniform-absorption.** Room-scale RT60 variation survives via `V/S`; furnishing-dependent variation does not exist. A task spec must not depend on the agent distinguishing a carpeted room from a tiled one by ear.
- **The `val_mini` constraint is gone.** Ticket 05 measured HM3D `val` mesh coverage at 20/20, not the 2/20 that forced earlier work onto `val_mini`. The smoke stays on `minival` because it is small and ticket 04 already loaded a scene from it, but a task spec is free to assume full `val` is available downstream.

Ticket 06 remains the live blocker: whether audio renders live at every step, or the spec has to be written against a throttled variant.

## Note added by ticket 07 (resolved 2026-08-01) — three things land in this ticket's lap

The agent is decided (`docs/adr/0008-clean-room-agent-architecture.md`): a candidate-pool frontier explorer, one `GoalDetector.detects(obj)` seam serving both the primary STOP and the anomaly CHECK, the anomaly controller ported near-verbatim, no LLM and no CLIP in the agent.

07 deliberately did **not** decide the following, because they are task-spec questions and deciding them there would have decided this ticket by accident:

- **The ADR-0002 room classifier.** 07 removed CLIP from the agent, and the room classifier is now the *only* route by which CLIP returns to the tree. If scene-conditioned normality survives ticket 02's multi-source answer, this ticket owns whether CLIP comes back for it, and on what evidence — CLIP is measured flat on HM3D sim renders (separation 0.020 against a 0.05 bar, three times), though that was frontier value at distance, not room typing.
- **Whether the controller's localization policy is amended for live audio.** 07 ported it verbatim on purpose. Its `(energy_history, lateral_sign, visual_confirm)` inputs were shaped by the precomputed grid — quantised energy, ADR-0001's ~1 m ceiling — and continuous receiver positions may make the gradient climb genuinely climbable. Amending it is this ticket's call, not a port decision.
- **The report's content.** `build_report` returns a structured dict today (`primary_completed`, `investigated`, `investigate_aborted`, `resumed`, `anomaly_class`, `source_xyz`, `n_benign_ignored`). Dropping the LLM costs nothing here, since nothing in it was ever generated text. Whether the destination's "and reports" wants more than this is a task-spec question.

One thing this ticket must **state** rather than assume, for the smoke-green criteria: the smoke runs an **oracle STOP**, so it does not exercise goal detection at all. `diagnose_spin` decomposed the 0.031 benchmark SPL as stop_miss ~50% + explore_timeout ~45% + success ~5%, and an oracle STOP deletes the stop_miss half outright. Smoke find numbers are therefore not capability numbers and must not be quoted as such.

## Note added by ticket 10 (resolved 2026-08-01) — the audio module carry line is yours

Ticket 10 settled the reset spec: what is deleted, what carries, and the four-phase order with the deletion gated on this ticket's smoke-green criteria plus a hermeticity re-run.

It ruled on every carry question **except one**, and left it here on purpose.

**The audio module carry line is owned by this ticket.** Three files are unresolved:

- `embodied_memory/audio.py` (719 LOC)
- `embodied_memory/audio_task.py` (402 LOC)
- `embodied_memory/perception.py` (506 LOC — the CLIP image/text encoders)

The split inside them is real and this ticket has to draw it:

- The **precomputed-grid convolve path** (`render_rir_grid` lookup, `fftconvolve`, `cached_source`) is dead by chart-time decision — audio renders live in-sim every step, no grid.
- The **durable wins** are not: the CLAP open-set normal-vs-anomaly gate (`audio.is_anomaly`, whose calibration ran GO with perfect separation, EER 0.00), the onset detector and its calibration (`onset_rms` 0.065), and `resolve_anomaly_clip` / the ESC-50 fetch. These are mechanisms this ticket's spec will either keep, recalibrate, or replace — and it cannot say which until it has decided level calibration and onset provenance.
- `perception.py` is entirely conditional on **your** ADR-0002 call. Ticket 07 removed CLIP from the agent; the room classifier is the only route by which CLIP returns to the tree, and 07 routed that decision here.

Ticket 10 did not rule on these because doing so would have decided this ticket by accident — the same trap ticket 07 avoided when it declined to amend the controller's localization policy.

**The ordering holds and nothing is blocked.** You resolve before the smoke; the smoke gates the deletion. Whatever carry line you draw simply lands on ticket 10's phase-1 port list.

Two further inputs from ticket 10, for the record:

- **`metrics.py` (55 LOC, `compute_benchmark_spl`) is ported near-verbatim.** So when this ticket restates the metric set against `CONTEXT.md`, the arithmetic it is restating against already exists and is Mac-testable.
- **The smoke's oracle STOP is confirmed**, not just proposed: ticket 10's carry list ships `OracleDetector` (what the smoke runs) and `CaptionDetector` (what R2 runs) behind ADR-0008's `detects()` seam, with the OWLv2 backend dropped as a measured noise-floor negative. This ticket still owes the smoke-green criteria the statement 07 asked for — that smoke find numbers are not capability numbers.

---

## Note from ticket 15 (2026-08-03) — reviving CLIP now has a measured price

This ticket owns whether ADR-0002's scene-conditioned room classifier survives, and that is the only route by which CLIP returns to the tree (ADR-0008 dropped it). Ticket 15 tried to price CLIP and **could not load it at all**:

```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`,
we now require users to upgrade torch to at least v2.6 ... does not apply when loading files with safetensors.
```

transformers 4.57.6 refuses `torch.load` below torch 2.6 (CVE-2025-32434), the cached `openai/clip-vit-base-patch32` is a `.bin`, and **ticket 13 pinned torch at 2.2.2+cu118 deliberately** for this V100. Qwen2-VL-2B loaded without complaint because it ships safetensors.

So "keep the room classifier" is no longer a free choice. It costs one of:

- a safetensors re-fetch of CLIP (cheap, but it makes the env pin's model layer load-bearing — ticket 17's problem), or
- a torch bump past 2.6, which ticket 13's pin forbids on this card, or
- a different room-classification route that avoids CLIP entirely.

**VRAM is not the constraint** — ticket 15 measured 26 GiB of margin with the captioner resident, so there is room for CLIP many times over. The blocker is purely the pin.

---

## Answer

**The task spec is `docs/anomaly_response_task_spec.md`, and four ADRs supersede the four grid-era ones: 0009→0004, 0010→0003, 0011→0001, 0012→0002.**
The task the robot is given is unchanged.
Every constraint that existed because audio was a baked grid is re-derived, and the single decision that dissolved most of them is one this ticket was not expecting to make.

### The decision that carried the rest: one source, and the bed never renders

**One positioned source, no multi-source patch, no habitat-sim fork** (ADR-0009).
The bed is a fixed-level diotic signal generated directly and mixed after rendering, so it is position-invariant by construction.

Ticket 06 handed this ticket a free choice and named the one argument left standing: per-source IRs make onset provenance structural, so decide the ~40-line patch on provenance alone.
**That argument does not survive contact with the alternative, and this ticket corrects its own framing twice on the way.**

First, the patch was never the only route to per-source IRs.
Ticket 02's own option 2 (sequential re-render, move the single source, keep the IRs unsummed) yields exactly the same provenance at roughly 82 ms/step, which ticket 06 measured as affordable.
So the patch buys about 55 ms/step, not provenance.

Second, and decisively, **the task needs one source at a time**, which neither ticket noticed:

- **ADR-0002's room-normal distractor is not simultaneous.** Its design is same-sound / two-rooms *across episodes*, and the ADR states it rejected a simultaneous two-source distractor to preserve the O(1) invariant. The distractor needs a position, never at the same instant as the anomaly.
- **A bed is diotic by definition.** `CONTEXT.md` defines it as the continuous diotic noise floor. Rendering it from a position through the RIR is what it is not, and it is exactly what made ADR-0004's threshold unmeetable.

**So the fork question closes as "no fork"**, which also kills ticket 12's second patch candidate (binding the RLRA error channel), already downgraded by ticket 16 after it measured `RLRA_SetListenerHRTF` returning `Success` over a failed load.
That patch was only worth taking if the multi-source patch was taken anyway.

### What the unrendered bed retires, and what it makes structural

**ADR-0004's conclusion stands and its central argument is retired.**
ADR-0004 proved that no absolute RMS threshold could work: a 1.4x temporal step sitting inside an 8x spatial swing, with the temporal step detector named as the technically correct but unbuilt fix.
That argument depends entirely on the bed being *rendered*.
Unrendered, the bed is position-invariant and the anomaly contributes exactly zero before `t_anom`, so **the pre-onset signal is flat at the bed level, at every pose, in every scene**.
The 8x swing lives entirely in the post-onset term.
An absolute threshold is well-founded for the first time and the step detector stays unbuilt.
ADR-0004's conclusion (the bed is a noise floor, never the trigger) is unchanged and is now **structural rather than calibrated**; `bg_gain` is retired outright rather than recalibrated.

**`onset_step < t_anom` becomes structurally impossible.**
That is the `anommxv` break where the interrupt fired on the bed at step 0 and was read as the alarm.
It is closed by construction, and without the patch ticket 02 argued was the way to close it.

**Consequence for `CONTEXT.md`'s definition of onset provenance:** comparing `onset_step` to `t_anom` becomes a tautology, so it can no longer fail for the reason it was invented to catch.
It changes job rather than retiring.
It is now an **asserted invariant that raises** (pre-onset RMS equals the bed level within tolerance, `onset_step >= t_anom`), because a violation means the bed drifted or the source started early, and both are silent-fabrication bugs of the kind this map keeps finding.
Ticket 12's discipline, applied to the signal instead of the mesh.

### Levels: derived by a gate, not by hand

Bed level is a **chosen constant** (it is our signal, so there is nothing to calibrate against).
`onset_rms` is **derived**: render the anomaly across the intended audible band, set the threshold strictly between the bed level and the low percentile of the anomaly distribution, report the separation as the gate number in the pattern of ticket 13's EER and the CapRL gate.
Overlap fails the gate, and the correction is `globalVolume`, never a nudged threshold.
The old `onset_rms=0.065` is not carried.
Confirming ticket 11's warning: `globalVolume` is **1.0** on our branch (ticket 04, measured), not the 0.25 ticket 01 quoted from `main`.

**No build-time audibility screening.**
The grid-era `AUDIBLE` / `OUT_OF_COVERAGE` cell check has nothing to check, and pre-screening would reintroduce offline rendering by the back door.
The cost is deliberate and is carried in the metrics: `t_anom` is when the source starts *playing*, not when the agent hears it, so an agent that never gets close enough produces an episode with no onset.
That attrition is a visible funnel stage, not an absorbed one.
The smoke is the single exception and verifies audibility at its own start pose once.

### The floor constraint: right rule, dead reason

**Same-floor stays, as builder policy on controller-scope grounds** (ADR-0010).
ADR-0003 was a fabrication argument, and live rendering kills it: no grid, no `nearest`, no snapping, and with `transmission` measured ON an off-floor source now produces *real* attenuated audio.
Note precisely what that does to ADR-0003's second requirement: "`nearest` must refuse to snap across floors" is **not retired, it is unimplementable**, because the thing it guarded does not exist.

ADR-0003's *other* stated reason is untouched by live rendering: a cross-floor source turns the detour into a stair-climb the controller was not designed for.
That is a statement about a greedy energy climb, not about a renderer, and it fails ugly: energy rises toward a source through the ceiling while no navmesh path goes there.
Ticket 06's rho of −0.98/−0.99 (and −0.95/−0.98 NLOS) does not license relaxing it, because ticket 06 warned itself that those walks were same-floor navmesh walks where direct and transmitted paths dominate.
Relaxing the rule is now gated on **measuring the non-LOS regime**, not on a render budget.

### The silent correctness break: the lateral sign changed frame

**Found from source, before any run, and it is the most dangerous item in this resolution** (ADR-0011).

`lateral_sign` (`audio.py:595`) is a pure ILD sign: it reports which ear is louder and nothing else.
Its meaning comes from how the IR was rendered.
The grid rendered at **identity listener yaw**, so the cue was a **world-frame** bearing, and the fusion arc calibrated to `heard == -right(world-bearing)`.
Live rendering uses the agent's real listener transform, so the identical function now returns an **agent-frame** cue.

**The code does not change and the convention inverts underneath it.**
Ticket 07 ported the controller verbatim on purpose; verbatim plus the old compensation turns the wrong way on every stall, and it would look like a mediocre climb rather than a bug.
The convention is pinned by a test the tree owns, not by a comment and not by a calibration run.

Two smaller amendments: **ADR-0001's ~1 m ceiling was a grid-resolution artefact** and is replaced by a measured distance-at-STOP distribution (its stated spearman ≈ −0.45 is a grid number too; live is −0.98/−0.99).
And **rotation now moves the gradient** (`turn_left` changes RMS without changing distance, impossible on a one-yaw grid), which is **instrumented rather than fixed** by recording the action alongside RMS at every step.
Amending the rule on a hypothesis would be changing the one module ADR-0008 calls the paper's single framing-independent positive.

`sourceIsVisible()` becomes a **first-class per-step annotation and the controller must never read it** — it is computed from the ground-truth source position, so feeding it to the decision rule plants a hidden oracle inside the arm ADR-0001 exists to avoid one.

### ADR-0002 survives, and CLIP does not come back

**The claim survives** (ADR-0012).
Dropping it is not a small edit: ADR-0004 relocated the *entire* discrimination claim onto the room-normal distractor, so without scene-conditioned normality the CLAP gate is decorative, any onset interrupts, and there is no discrimination claim anywhere in the work.

**There is a fourth option neither ticket 15 nor this ticket's note listed, and it dominates all three.**
Ticket 15 priced CLIP and could not load it (transformers 4.57.6 refuses `torch.load` below torch 2.6 under CVE-2025-32434; the cached checkpoint is a `.bin`; ticket 13 pinned torch at 2.2.2+cu118 for this V100), leaving a safetensors re-fetch, a forbidden torch bump, or a non-CLIP route.
**The tree already carries a VLM.**
ADR-0008's carry list ships Qwen2-VL-2B, which ticket 15 noted loads cleanly precisely because it ships safetensors.
A room label from a caption costs no new dependency, no re-fetch, and nothing on the env pin's model layer — and it has the better prior, since CLIP is measured flat on HM3D sim renders three independent times while this project's Phase-3 fix was that VLM captions are discriminative where the cheap channel was degenerate.

So **`perception.py` (506 LOC) does not carry**, ADR-0008's drop of CLIP is confirmed rather than reopened, and ADR-0002's $0 room-classifier gate carries across the substitution unchanged, because the captioner has never been measured as a room classifier either.
Implementation defers to R2 behind a provider seam; the smoke does not exercise it.

### The carry line is not file-shaped

Ticket 10's open question, answered: **neither `audio.py` nor `audio_task.py` carries as a file.**
Porting either whole drags the grid into the clean room on day one.
Ticket 07's precedent applies (frontier proposer rewritten 1129 → ~300 LOC rather than ported with dead branches).

- **Re-homed:** CLAP gate + prompt banks + calibration (GO, EER 0.00), `resolve_anomaly_clip` + ESC-50 fetch, `normalize_clip`, `lateral_sign` (frame pinned), `ROOM_PRIOR`.
- **Dies with the grid:** `render_step_audio`, `fftconvolve` lookup, `cached_source`, `RIRGrid` resolution, `diotic_collapse`.
- **Dies on the record:** `should_audio_stop` (audio-energy STOP arc closed as GT-privileged, ~1 m capped), `estimate_doa` (`CONTEXT.md` retires DOA framing; ITD branch engine-weak), `audio_target_for_retrieval` + `gate_retrieval_target` (memory-facing, memory is out of this build).
- **Rewritten:** onset detector, its calibration, `process_audio_step`'s orchestration.
- **Does not carry:** `perception.py`.

CLAP is carried **despite being decorative in the smoke** (one sound, and it is the anomaly by construction), because the discrimination claim runs on it and ticket 13 spent real effort making it instantiate on this box.

### Metrics and outputs

**Anomaly-response SR is replaced by a staged funnel with no single headline fraction**: episodes run → `t_anom` reached → onset fired → investigate entered → source reached → primary resumed, denominator at stage 2.
A single fraction silently mixes "never heard it" with "heard it and failed to reach it", and section 2.5's unscreened audibility makes that mixture guaranteed rather than hypothetical.

Find-SR unchanged (1.0 m primary, 0.1 m diagnostic).
**soft-SPL computed but not headlined** — `CONTEXT.md` defines it as the memory-delta metric and memory is out of this build, so nothing here consumes it; it is computed anyway because `metrics.py` carries verbatim and the follow-on effort inherits it wired.
**Benchmark SPL computed and never cross-quoted from this map** (ADR-0005 makes it cross-quotable, ADR-0006 turned absolute numbers off as the lead, ADR-0008 established the oracle STOP deletes the stop_miss half).
**New: distance-at-STOP** (replacing ADR-0001's asserted ceiling) and **per-step audio render wall-clock**, so the claim ticket 06's 27.2 ms underwrites is auditable every run rather than trusted from one sweep.

**The report splits in two**, on whether the information is agent-estimable.
The **agent's testimony** carries only agent-estimable fields, with an **identical schema in both localization arms**; `source_xyz` leaves it, so the oracle arm's privilege shows in its trajectory and audit record, never in its testimony.
This makes the "the sound is just a stopwatch, the coordinate is handed to the agent" objection **checkable by reading the schema**: if a report cannot be built without ground truth, the arm is not realizable.
The **episode audit record** holds everything privileged, and lands with `AudioContextReport` and `env_report.json` — which the map's requirements 1(c) and 8 already say should be one place, not three.
Structured, no generated text (ADR-0008 dropped the LLM; nothing in the report was ever prose).

### Smoke-green

Nine checkable assertions, in the spec's section 8: render count equals step count; ticket 12's guard green through ticket 16's verified invariants; a real trimmed IR; provenance did not raise; the full SEARCH → onset → INVESTIGATE → CHECK → RESUME loop with a legitimate termination; a populated report; per-step audio wall-clock inside a **generously** stated ceiling (ticket 06 measured 2.3x pose variance against ticket 04 on the same scene, so a tight bound fails for a non-regression); `env_check.py` passed; then ticket 10's hermeticity re-run.

Two of those are decisions rather than bookkeeping:

- **The primary find-task is NOT required to succeed.** The destination says the agent *runs* its primary find-task. Requiring success would gate an irreversible deletion on a backbone measured at 0.031 benchmark SPL with ~45% explore-timeout, which is the capability ADR-0006 retreated from claiming.
- **The smoke runs the realizable arm, not the oracle one.** An oracle smoke leaves the entire live-audio path unexercised in the one episode that exists to prove it, and the sound really would be a stopwatch. The oracle arm is retained as a bisection tool: on failure it isolates audio from controller in one step. The risk is real and taken deliberately — a realizable climb that cannot reach an audible same-floor source is a finding better had *before* deleting the old tree.

**Required disclosure, as ticket 07 asked:** the smoke runs an oracle STOP and does not exercise goal detection at all.
`diagnose_spin` decomposed the 0.031 as stop_miss ~50% + explore_timeout ~45% + success ~5%, and an oracle STOP deletes the stop_miss half outright.
Smoke find numbers will look far better than 0.031 for a reason that must be disclosed rather than enjoyed, and they are not capability numbers.

### Left to the builder, deliberately

`investigate_max_steps` (currently 40), the bed level constant and the audible band the calibration sweeps, the wall-clock ceiling, and the tolerance on the pre-onset RMS assertion.
These are numbers to set against measurement, not decisions.

### What this unblocks

The map's fog entry **"How far the clean room is willing to fork habitat-sim" is settled without a ticket: no fork.**
**"Smoke-green acceptance criteria" is settled by this ticket** rather than graduating.
**"The new package's module layout and seams"** was waiting on this ticket and graduates now, with the test-strategy patch behind it.
ADR-0008's four deferrals to ticket 09 are all discharged, CLIP included, so nothing in it reopens.
