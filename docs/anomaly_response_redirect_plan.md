# Anomaly-Response Redirect — Plan & Gate 0a Reconciliation

**Date:** 2026-06-30 · **Branch:** `lifelong-revisit-eval` · **Status:** controller built+tested; rest staged.

This is the working plan for the new project direction. It is the durable companion to the
`anomaly-response-redirect` memory checkpoint and the 8-agent design workflow
(`/tmp/claude/wf_out/{design,review_soundness,review_confounds}.md`).

---

## 1. The new task

The robot is placed in a scene with a **primary find-task** (find a target object). While
searching, if an **abnormal sound** is heard it must: (1) **identify** the sound — anomaly vs
background noise; (2) if anomaly, **go to the source** to check what happened; (3) **resume** and
complete the primary find-task; (4) **report** back. Every scene mixes anomaly + background noise,
so anomaly-vs-background discrimination is load-bearing. Three setups: **(A)** viewed scene +
listened audio (fully warm), **(B)** viewed scene + new audio (warm visual / cold audio), **(C)**
new scene + new audio (fully cold).

**Key difference from all prior audio work:** previously the sound *was* the goal
(`anomaly_object_override` → sound==goal==target), which made audio a code-proven no-op for
retrieval. Now the sound is an **interrupt to a separate primary task** → a hierarchical
interrupt→investigate→resume→report controller that the flat per-step loop did not have.

---

## 2. Contribution framing (decided 2026-06-30: **anchor on the controller**)

The headline contribution is the **working interrupt-investigate-resume-report controller** on top
of the already-validated warm-recall memory (+0.17/+0.24, ~12 reproductions), with **audio causally
gating the interrupt** and **load-bearing anomaly-vs-background discrimination**. This is a
systems/integration result that is **positive and framing-independent** (it does not depend on
audio-memory beating vision).

### Gate 0a — reconciliation with the closed AudioGoal Step-2 negative ($0, on paper)

The adversarial confound review flagged that the *3-setup audio-memory claim* (A−B = "does
remembering the heard anomaly help?") is essentially **AudioGoal Step 2**, already run and closed:
write-ON vs write-OFF B−A = −0.170 → over-fire-fixed −0.012 (**redundant-with-vision**), because a
scene mapped visually already routes to the source, so an audio memory of the same spot is a
duplicate. "HELPS is unreachable by construction" on that harness (LOS seed + single instance +
static world + oracle-GT write each removes the only thing audio adds over vision). See
`PHASE2_ABLATION_REPORT.md` → "AudioGoal Step 2" and the `audio-visual-ltm-fusion-plan` memory.

**Why the controller is nonetheless a distinct, positive contribution (not a re-run of that
negative):**

1. **It measures a different thing.** Step 2 measured *audio-memory value for navigation* (a memory
   claim). The controller measures a *task-structure capability*: can the agent detect an anomaly
   mid-task, divert, investigate, and **resume + complete the primary task**? That is feasibility /
   behavior, independent of whether audio-memory beats vision.
2. **Audio is causally necessary here, by construction.** The interrupt only fires on
   onset+`is_anomaly`; turn the audio off and there is no divert. (The onset-gate already showed
   audio is causally necessary; here it gates a *behavior*, not just a retrieval target.)
3. **Discrimination is load-bearing for the first time.** Every scene has a benign bed; the agent
   must *not* chase benign onsets. False-investigate rate on a loud-benign distractor is a new
   measured capability absent from every prior arc.
4. **The two sub-goals make the eval non-degenerate.** There are now genuinely two destinations per
   episode (primary object AND source), and the agent must *return* — the single-goal-eval critique
   ("recall has nothing to do") does not apply to the controller behavior.

**Pre-registration (honest-negative).** The A−B *audio-event-memory-value* number is expected to
reproduce Step 2 (tie at the noise floor / redundant-with-vision) **unless** the regime changes
(non-static world between map & revisit + multi-same-category instances + a DOA-derived, not
oracle-GT, write). We are **not** doing that regime change now. We will **report A−B as a
pre-registered honest-negative replication**, not as the headline. A:S3−S1 in setup A is expected to
be positive but is **mostly the known +0.17 visual-recall effect re-measured** — we will say so and
attribute via the S2 decomposition, not claim it as new audio value.

**Net publishable claim:** "a working interrupt-resume anomaly-response controller on validated
warm-recall memory, audio causally gating the interrupt, with measured anomaly-vs-background
discrimination" — plus an honest map of *when audio-event memory does and doesn't add value over
vision* (the Step-2 negative, replicated and explained).

---

## 3. The controller (BUILT — `embodied_memory/anomaly_controller.py`, 18 TDD green)

Pure decision module (no sim import, no LTM), mirroring `audio_task.py`. States:

```
SEARCH ──(onset & is_anomaly & source cue)──► INVESTIGATE ──(arrived)──► CHECK
  ▲   ▲                                            │ (budget overflow)      │
  │   └────────── (benign onset: ignore+count) ────┘                        │
  │                                                                         ▼
  └──────────── SEARCH ◄── RESUME (restore primary + force re-query) ◄──────┘
        │
  (primary reached) ──► COMPLETE ──► REPORTED
```

- `step_controller(...)` emits a `ControllerDecision`: next `mode`, `active_goal`,
  `investigate_waypoint` (inject as `source="audio_investigate"`), and the one-shot directives
  `force_requery` / `save_primary_state` / `restore_primary_state`, plus the `investigation_event`
  (at CHECK) and end-of-episode `report` (`build_report`).
- One investigation per episode; benign onsets are ignored + counted (`n_benign_ignored`); the
  detour has its own sub-budget (`investigate_max_steps`, overflow → abort→RESUME).

---

## 4. Staged rollout & status

| Phase | Work | Status | Verifiable |
|---|---|---|---|
| **0a** | Reconcile with Step 2; pre-register A−B honest-negative | ✅ done (§2) | $0 / on paper |
| **Core** | `anomaly_controller.py` pure state machine + 18 TDD | ✅ done | local |
| **E4** | `active_goal` no-op rename — `_resolve_active_goal` helper + A1/A2 lifecycle + B1–B15 read-through routing; STAY sites (SPL/success/report) keep `ep.target_category` | ✅ done — **16/16 green, byte-identical** (15 existing regression + new `test_active_goal_noop.py`) | local |
| **E5/E6/E7** | Controller wired into `episode_runner` (spec §7): audio-guard widen, controller call site, force-requery, snapshot/restore, investigate-candidate injection, `memory_bridge` rerank branch, report hook, CLI. **+ review fixes (§8):** STOP-suppression while diverting, divert-tie filter, real extend-budget, `habitat_env` audio-render widen | ✅ done — **19/19 green**, byte-identical; adversarially reviewed (3 agents + 1 verifier) | local (live loop RACE-bound) |
| **0c** | Controller fires + changes behavior vs sound==goal on the existing M3 audiogoal dataset (`--task anomaly_response`); source==goal there so it's a wiring smoke, not the real eval | ⬜ next | RACE 1–2 cells |
| **E8/E11** | New metrics (discrimination_correct, investigated, investigate_source_error, resumed, primary_completed, report) + analyzer A−B/A/C contrasts | ⬜ | local (analyzer) + RACE |
| **0b** | CLAP separates the **mixed** anomaly+benign blend (prior GO was on clean clips) — reuse `diagnose_normal_anomaly_calib.py` | ⬜ | RACE/CLAP |
| **N3/E1/E2** | Dataset builder + multi-source render. **Reuse** `make_audiogoal_smoke.pick_non_los_seed` / `build_lifelong_dataset` / `fetch_anomaly_clips --include-benign`; mix-bus is gated on 0b GREEN (it may be the wrong primitive — see RISK below) | ⬜ (gated on 0b) | RACE |
| **N2/E10** | NVIDIA client — **offload + diagnostics only** (planner off-device, headline stays frozen-local). TDD-able with mocked OpenAI | ⬜ | local |
| **Matrix** | Full A/B/C × S1/S2/S3 on a **frozen local backbone**; mandatory A/Bs: realizable-DOA vs oracle-GT, budget-extend on/off | ⬜ (gated on 0a/0b/0c) | RACE multi-day |

---

## 5. Must-fix code corrections (from adversarial review)

- **C1** Setup B needs **per-episode clip re-loading** in `habitat_env` (clip is loaded once/run; re-load when `ep_info["anomaly_clip"]` changes, gated `task=="anomaly_response"`).
- **C2** The interrupt must **save/restore `mem_cands`** (and snapshot `consumed_memory_xys`/`unreachable_xys`/`consecutive_unreachable`); keep investigate candidates in a **separate** local so primary `mem_cands` is never clobbered (else STOP silently corrupts).
- **C3** The interrupt block is at the **wrong seam**: `process_audio_step` is gated `task=="audiogoal"` (must also run for `anomaly_response`), and the memory-injection seam is nested in `if cands:` (force an immediate re-propose on SEARCH→INVESTIGATE, `last_propose_step=-1e9`).
- **M4** `active_goal` touches ~15 read-through sites, not 7; some (`_observe_semantic_value`, audio-write/retrieval) must track `active_goal` while final SPL/report/summary stay `ep.target_category`. Land as a no-op rename first.
- **Reuse, don't rebuild:** non-LOS picker, benign fetcher, lifelong builder, consume flag (`REMEMBR_CONSUME_SINGLEGOAL`), lifelong analyzer all already exist.

## 6. Invariants & risks

- **Invariants:** candidate-proposer seam; byte-identical default paths (objectnav/audiogoal/revisit); two-env audio split; S1-vs-S3 differ only by `disable_ltm`.
- **GT-source = oracle teleport** (realizable DOA gives only a ±1 lateral sign): the realizable-investigate arm **must** be A/B'd against the oracle arm.
- **Discrimination can be trivially-100%** (loudness does the work) or destroyed by the mix-bus; needs Gate 0b **and** a loud-benign distractor condition (`is_anomaly_gt=False` onset that must be ignored).
- **NVIDIA:** offload planner only (per-replan, but it is wall-clock in the hot loop → ~25–60 calls/ep); hosted models version-drift → keep the **headline ablation on a frozen local backbone**; API for VRAM relief + $0 offline diagnostics only.
- **Power:** A/B/C splits the sample; generate A and B as **explicitly matched pairs** (same scene/category/start, differing only in `anomaly_clip`), pair on the renumbering-invariant `(scene, category, setup, visit_order)`, and quote the S1 run-to-run noise floor.

---

## 7. E5 wiring spec (NEXT — verified against the real file, anchors as of E4)

All `anomaly_response`-gated; every other task stays byte-identical (E4 already proved the `active_goal` seam is a no-op). The controller exists (`anomaly_controller.step_controller`). Order: S1 → S5 (state contract) → S2/S3 (call site + requery) → S4/S6 (inject + rerank) → S7 (report).

- **E5-S1 — C3 audio-guard widen** (`episode_runner.py` ~:1232): change `if self.task == "audiogoal" and self._audio_cfg.enabled:` → `if self.task in ("audiogoal", "anomaly_response") and self._audio_cfg.enabled:`. Body byte-identical; B9/B10 already route through `active_goal`. **DO NOT widen the audio-energy STOP guard** (`task=="audiogoal"`, ~:1435) — it must not hijack the anomaly_response STOP.
- **E5-S2 — controller call site** (insert ~:1283, after the audio block, before `is_decision_step`/`need_candidate` computation so `force_requery` can set the locals those reads consume). Call `step_controller(...)` with `onset_fired`/`is_anomaly` from `step.info`, `source_xyz` from `ep.metadata["audio_config"]["source_position"]`, `arrived_at_source` = floor-plane (xz) dist(agent, source) < `cfg.investigate_arrive_radius_m`, `primary_goal_reached` = the primary arrival/STOP test vs `ep.target_category`, `anomaly_class`/`anomaly_object` from `self._audio_state`, `keyframe_caption=keyframe.caption`. Then `active_goal = dec.active_goal`; apply `save_primary_state`/`restore_primary_state`/`force_requery`; stash `self._investigate_wp = dec.investigate_waypoint`.
- **E5-S3 — forced re-propose** (reuse MultiON handoff ~:1945-1947): on `force_requery` set `current_candidate=None`, `self._approach_waypoint=None`, `last_propose_step=-10**9` → the `if cands:` propose+inject seam fires THIS tick.
- **E5-S4 — investigate-candidate injection** (after `all_cands = cands + mem_cands` ~:1425): if `self._investigate_wp` is set, build a `source="audio_investigate"` candidate (new helper mirroring `_detector_candidate`), id `len(all_cands)+9000`, append. Survives consume filters (anomaly_response → `_consume_memory_applies` False by default).
- **E5-S5 — C2 snapshot/restore contract.** On SEARCH→INVESTIGATE snapshot, on RESUME restore. **Keep `mem_cands` + `current_candidate` in SEPARATE investigate-locals** (primary copies never clobbered → no STOP corruption at the detector-mem gate / audio-energy STOP / arrival-STOP). SNAPSHOT+RESTORE: `self._approach_waypoint`, `consumed_memory_xys`, `unreachable_xys`, `consecutive_unreachable`, `last_propose_step`, `last_reached_propose_step`, `last_follower_drop_step`. **MAJOR (from adversarial review — do NOT omit): also snapshot `self._semantic_goal` + `self._goal_text_emb`** (the goal-keyed CLIP re-encode cache in `_observe_semantic_value` ~:2266-2271) — else under `LTM_SEMANTIC_FRONTIER` the INVESTIGATE detour pollutes the planner's frontier value map with investigate-goal cosines at primary-search positions and biases the post-RESUME search. (Default-OFF, but the contract must include it.) Do NOT snapshot derived/per-tick/multion-only locals (`bearing_rad`/`distance_m`, `_waypoint_force_repropose`, `stm_captions`, `stop_cand`, `stop_event`, `no_progress_window`).
- **E5-S6 — source-aware rerank branch** (`memory_bridge.py` `FrontierPhysicsScorer.score`): add `elif source == "audio_investigate": return 1.0` so the divert deterministically wins during INVESTIGATE (else it falls into the frontier `else` ~0.5-0.7 and can lose to a strong memory match → controller emits the right waypoint but rerank silently discards it).
- **E5-S7 — report hook** (after success computation ~:2012): `anomaly_controller.build_report(self._anomaly_state, primary_completed=success)` into `ep_log`. **SPL stays scored against `ep.target_category`** — the success block is goal-expression-free, so the hook is additive and cannot perturb SPL.
- **Also needs:** `AnomalyControllerConfig`/`ControllerState` instantiated in `__init__` (mirror `_audio_cfg`/`_audio_state`); `--task anomaly_response` choice + `--investigate-max-steps` in `run_hm3d_pol.py`; `process_audio_step` must compute `is_anomaly` (anomaly_gate ON for anomaly_response) so `step.info["is_anomaly"]` is populated. TDD each seam (controller-call gating, separate-local STOP non-corruption, rerank branch, report additive).
- **Note:** the C1 STAY site is the dict entry `"target_category": ep.target_category,` (not the `ep_log[...] =` form the census quoted) — confirmed present/untouched.

---

## 8. E5 adversarial review — findings & fixes (2026-06-30)

3 review agents + 1 focused verifier on the wiring diff. Byte-identity APPROVED. Findings + resolutions:

- **CRITICAL — premature STOP during INVESTIGATE (FIXED).** The grounded backbone `stop_signal` (queried on `active_goal`=anomaly_object during the divert) fired AT the source and ended the episode as a failed primary STOP — RESUME unreachable. Fix: suppress `stop_cand` and the stop_signal action branch while `anomaly_controller.is_diverting(mode)` (∈ {INVESTIGATE,CHECK,RESUME}). **NB:** used `is_diverting`, NOT the reviewer's `mode \!= SEARCH`, because the latter also suppresses the legitimate STOP at COMPLETE (primary reached) → would break primary success. New pure helper `is_diverting` + 2 TDD cases.
- **MAJOR — divert-candidate tie (FIXED).** A saturated memory/frontier candidate could tie the `audio_investigate` S_phys=1.0 and win on S_sim. Fix: while `_investigate_wp` is set, filter `all_cands` to `{audio_investigate, frontier}`. Residual (documented, harmless): a near-perfect frontier (raw=1, dist≈2 m, bearing≈0) can still tie → 1-tick delay, re-injected next tick; no STOP, no stall. Hardening option if ever needed: force-select the investigate candidate before rerank like `stop_cand`.
- **HIGH — extend_budget was a dead flag (FIXED).** The runner extended only its own loop bound, but the Habitat env's `max_episode_steps` cap terminates first → extra steps unreachable. Fix: bump `args.max_steps` once at construction so it reaches BOTH the env cap and the runner loop; reverted the per-episode loop bump.
- **HIGH — INVESTIGATE goal/retrieval inconsistency (FIXED).** `anomaly_object` now reads `anomaly_object_override or target_override` (the dataset-mapped object, matching `audio_target_for_retrieval`) so `active_goal` and the memory query agree during the divert.
- **BLOCKER for the live loop — `habitat_env` audio gated to audiogoal only (FIXED).** `anomaly_response` got `Step.audio=None` → onset never fired → controller inert end-to-end. Widened the 3 render/config guards to include `anomaly_response`.
- **MEDIUM — oracle label (FIXED, documented).** `source_xyz` is the GT source (privileged); the realizable DOA-derived arm is pending (the plan's mandatory A/B). Any "investigates the source" number off this wiring is an oracle upper bound — labeled at the call site.
- **MEDIUM — both success rings (FIXED).** The report stamps `primary_completed` (0.1 m, strict) AND `primary_completed_1m` (1.0 m benchmark ring the controller's COMPLETE uses).

**Still NOT built (Phase-2 dependencies, correctly out of E5 scope):** the multi-source dataset with a co-present non-LOS source + a loud-benign distractor (the benign-ignore path / `n_benign_ignored` is dead until then — the headline discrimination capability has no data yet); the realizable DOA-investigate arm (only the oracle GT arm is wired). Gate 0b (CLAP on mixed blend) precedes the dataset.
