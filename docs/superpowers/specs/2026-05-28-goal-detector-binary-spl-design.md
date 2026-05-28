# Spec — Goal detector for precise final-approach (binary SPL milestone)

**Date:** 2026-05-28
**Status:** approved (brainstorming)
**Related:**
- `PHASE2_ABLATION_REPORT.md` Run 9 (Phase C — Gate-A GREEN on soft-SPL, binary
  SPL@0.1 m = 0.196 noted as perception-bound).
- `docs/superpowers/specs/2026-05-27-fold-revisit-into-harness.md` (the
  `analyze_ablation --revisit` front door this milestone reuses for eval).
- `docs/superpowers/specs/2026-05-25-navmesh-waypoint-controller-design.md`
  (the existing navmesh `_waypoint_action` this milestone drives at the end).

## Problem

Phase C closed with Gate-A **(a) GREEN** on warm soft-SPL (S3−S1 = +0.240,
p=0.008), and that result reproduces across runs. But **binary SPL@0.1 m on
warm S3 is stuck at 0.196** — and the diagnostic is precise:

- warm S3 `success@1m` = **67 %** — the LTM gets the agent to the goal region.
- warm S3 binary SPL@0.1 m = **0.196** — but the agent rarely lands within
  0.1 m of a goal viewpoint at STOP time.

The gap is in the *last metre*, not in exploration: the captioner-driven
keyword-STOP signal fires when the goal is "in view" (~0.5–2 m away), but a
text caption can't localize the agent to within the 0.1 m success ring.

CLAUDE.md's next-milestone section names "a real object detector / precise
goal-approach" as the headline lever for higher binary SPL. This spec realises
that lever as a **precise final-approach module** that intercepts the existing
keyword-STOP path: when the captioner says the goal is in view, instead of
stopping, locate the goal precisely via a detector and steer the last metre on
the navmesh.

## Decisions (locked in brainstorming, 2026-05-28)

1. **Detector role — precise final-approach only.** The captioner + LTM keep
   driving exploration unchanged. The detector fires only at the existing
   keyword-STOP point: when the caption already contains the goal keyword,
   *intercept* the STOP signal, localize the goal precisely, navigate the last
   metre, then STOP. Surgical seam; the proven LTM stack is untouched; cheapest
   GPU cost (a few calls per episode).
2. **Detector model — Qwen2-VL native grounding.** Reuse the
   `Qwen2-VL-2B-Instruct` model already loaded as the captioner; it natively
   emits bboxes via `<|box_start|>...<|box_end|>` tokens for queries like
   "locate the chair". **Zero new GPU memory, zero new dependencies, single
   extra forward pass on a model already validated on this scene set.** v2
   fallback: GroundingDINO-Tiny via the same `locate()` interface (~1-day swap
   if grounding quality on HM3D 0.5–2 m ranges is too noisy).
3. **Bbox → action — depth back-project + navmesh-snap.** bbox center pixel →
   read depth at that pixel → back-project to 3D world coords via camera
   intrinsics + agent pose → snap to nearest navigable point on the Habitat
   navmesh → install as approach waypoint → existing `_waypoint_action` drives
   `ShortestPathFollower` to it → STOP at success ring. Reuses proven infra; no
   second controller for the last metre.
4. **Gate — rigorous 2 × ablation.** 6 cells (S1/S2/S3 × detector ON/OFF),
   same Phase-C revisit dataset (2 scenes × {chair, bed} × n_warm 3 →
   16 eps/cell, 12 warm pairs/cell). Cleanly decomposes detector effect, LTM
   effect with detector, and full-system uplift. Worth the ~2× GPU time
   (~4 GPU-hours sequential).

## Architecture

The existing per-step loop runs unchanged. The only seam is at the
keyword-STOP branch, gated behind a new `--detector` flag:

```
[every step, unchanged]
  Qwen2-VL captioner       ──►  caption text
  LTM + planner            ──►  next waypoint
  navmesh waypoint ctrl    ──►  action (FORWARD / TURN / STOP)

[when caption contains goal keyword]
       │
       ▼
  if not args.detector:                            ◄── default: today's behaviour
      emit STOP
  else:
      wp = GoalDetector.locate(rgb, depth,
                               goal_category,
                               agent_pose, intrinsics)
      if wp is None:           # no bbox / parse fail / off-navmesh
          emit STOP             # fallback — no regression vs detector-OFF
      else:
          install wp as approach waypoint
          # subsequent steps: ShortestPathFollower drives toward wp,
          # emits STOP when it reaches the success ring.
```

The `None` fallback is the **monotonicity invariant**: detector-ON is
expectation-≥ detector-OFF because every detector-OFF outcome is reachable as
a detector-ON outcome via the None branch. This guarantees no soft-SPL
regression from the integration alone.

## Components & interfaces

### 1. `embodied_memory/goal_detector.py` (new)

```python
class GoalDetector:
    """Precise final-approach localizer.

    Reuses the already-loaded Qwen2-VL captioner — no new weights, no extra
    GPU memory. One additional forward pass per ``locate()`` call.
    """

    def __init__(self, qwen_vl_model, qwen_vl_processor, pathfinder,
                 max_snap_dist: float = 0.5):
        ...

    def locate(
        self,
        rgb: np.ndarray,            # H×W×3 uint8
        depth: np.ndarray,          # H×W float, metres (Habitat depth sensor)
        goal_category: str,         # "chair", "bed", …
        agent_pose: np.ndarray,     # 4×4 world transform
        intrinsics: dict,           # fx, fy, cx, cy, image_size
    ) -> Optional[np.ndarray]:
        """Return a navmesh-snapped 3D goal waypoint, or None.

        Pipeline:
          1. Prompt Qwen2-VL: "Locate the {goal_category}." Forward pass.
          2. Parse <|box_start|>x1,y1,x2,y2<|box_end|> tokens from output.
             Handles pixel-space and normalized [0, 1000] formats.
          3. If multiple bboxes: pick the one with smallest depth-at-center
             (closest physical surface — most likely the goal the agent has
             been navigating toward).
          4. Robust depth read: median of a 5×5 patch around bbox center.
          5. Back-project (u, v, d) → 3D world point via intrinsics +
             agent_pose.
          6. pathfinder.snap_point(world_point). If snap travels further
             than max_snap_dist (default 0.5 m), return None (implausible —
             back-projected point was off-floor far from any navigable
             surface, e.g. mid-air through a window).
          7. Return the snapped 3D point.

        Returns None if any of: no bbox in output, parse fails, depth at
        center is NaN/0/inf, snap_point exceeds max_snap_dist.
        """
```

Pure-Python geometry helpers (back-projection, robust depth) live alongside
the class as module-level functions so they're unit-testable without the
Qwen-VL model.

### 2. `embodied_memory/episode_runner.py` (modify, one block)

At the existing keyword-STOP branch (the whole-word goal-match check), gate
the action choice on `self.detector_enabled`:

- `detector_enabled is False` (default) → emit STOP exactly as today. The
  detector-OFF cells of the 2× ablation are byte-identical to a pre-milestone
  run, which is the **flag-gate regression check**.
- `detector_enabled is True` → call `self.goal_detector.locate(...)`. On
  `None`, emit STOP (fallback). On a 3D point, set
  `self._approach_waypoint = point` and continue stepping; subsequent
  iterations run the existing `_waypoint_action` toward `_approach_waypoint`
  until `ShortestPathFollower` returns its own STOP.

Five summary counters, all int, all default-0:

| counter | when it ticks |
|---|---|
| `n_detector_called` | each `locate()` invocation |
| `n_detector_localized` | `locate()` returned a non-None waypoint |
| `n_detector_offnavmesh` | bbox parsed but snap exceeded `max_snap_dist` |
| `n_detector_approach_success` | ShortestPathFollower emitted STOP at the snapped waypoint |
| `n_detector_approach_stop_distance` | per-episode: agent-to-waypoint distance at the STOP action (v1 is one-shot → at most one value per episode; float, NaN if detector never fired) |

These feed both per-episode JSON and the summary block; they are the
mechanism check in the gate (below).

### 3. `embodied_memory/run_hm3d_pol.py` — one new flag

```
--detector            enable precise final-approach localization (default off)
```

Embedded into `summary.json["ablation"]["detector"]` (bool) so the analyzer
can split the 6 cells cleanly.

### 4. `embodied_memory/scripts/analyze_revisit.py` — extend (small)

Current `print_report` prints paired **soft-SPL** deltas (`WARM S3 - S1`,
`S2 - S1`, `S3 - S2`, `COLD S3 - S1`). Add a parallel paired **binary-SPL**
block immediately after (`paired_warm_delta` already accepts a metric key,
mechanically the same call with `metric="spl"`). Stratified summary already
prints binary SPL per cell — only the paired delta + bootstrap CI needs to be
added. Touch one function (`print_report`); ~30 lines.

### 5. `scripts/race-revisit-detector.sh` (new, mirrors `race-revisit.sh`)

```
bash scripts/race-revisit-detector.sh --tag detector-c1
```

Pull → setup → pre-verify ([3/6] now includes the two new test suites) →
build the revisit dataset once (same as `race-revisit.sh`) → run **6 cells in
sequence** into `runs/<tag>-s{1,2,3}-{det,nodet}/` → finally run
`analyze_ablation.py --revisit` twice (the detector-OFF triple, then the
detector-ON triple) plus an explicit cross-condition contrast (gates A/B/C/D
below) and the mechanism-counter block (gate E).

`race-revisit.sh` is **not** modified — both drivers stay distinct, same
rationale as the two drivers we kept separate in the fold-revisit milestone
(they have different dataset/cell topologies).

## Ablation matrix + gate

### Matrix (96 episodes, ~4 GPU-hours sequential on the L4)

| Cell | Setting | Detector | Out-dir |
|---|---|---|---|
| s1-nodet | 1 (memory-off) | OFF | `runs/<tag>-s1-nodet/` |
| s2-nodet | 2 (STM-only)   | OFF | `runs/<tag>-s2-nodet/` |
| s3-nodet | 3 (full)       | OFF | `runs/<tag>-s3-nodet/` |
| s1-det   | 1 (memory-off) | ON  | `runs/<tag>-s1-det/`   |
| s2-det   | 2 (STM-only)   | ON  | `runs/<tag>-s2-det/`   |
| s3-det   | 3 (full)       | ON  | `runs/<tag>-s3-det/`   |

Same Phase-C revisit dataset (2 scenes × {chair, bed} × n_warm 3 = 16 eps/cell;
**12 warm pairs/cell** for the paired bootstrap — the same `n` that produced
Phase-C's p = 0.008, so signal is sufficient). Only the action-at-keyword-STOP
differs across the detector axis.

### Gate (paired-bootstrap on warm visits, 90 % CI, n = 12)

| Gate | Contrast | Metric | Bar | Question |
|---|---|---|---|---|
| **A** | s1-det − s1-nodet | binary SPL@0.1 m | mean > 0, CI excludes 0, p < 0.1 | Does the detector help by itself? *(mechanism check)* |
| **B** | s3-det − s1-det   | binary SPL@0.1 m | mean > 0, CI excludes 0, p < 0.1 | Does the LTM still help once perception is fixed? *(core scientific question)* |
| **C** | s3-det − s1-nodet | binary SPL@0.1 m | **mean ≥ +0.3**, CI excludes 0 | Full system vs nothing *(headline)* |
| **D** | s3-nodet − s1-nodet | warm soft-SPL | mean ≥ +0.15, p < 0.05 | No Phase-C regression *(controlled re-reproduction)* |
| **E** | mechanism counters | n_detector_* | localize rate > 0.4, off-navmesh rate < 0.3, detector-OFF cells all zero | Sanity, not a gate |

**Decision rules:**

- **A + B + C + D all pass** → milestone **GREEN**. Headline result is C.
- **A passes, B fails, C passes** → unexpected but still publishable: the
  LTM's contribution was *primarily* "reach the goal region" — now that the
  detector does precise approach, the LTM's marginal binary-SPL value is
  small. Reframe the writeup (LTM is necessary for the precondition the
  detector exploits, not directly for SPL).
- **A fails** → detector implementation bug (parse, back-project, or snap).
  Don't blame the LTM; fix the detector and re-run.
- **D fails** → detector regresses soft-SPL (false-positive keyword-STOPs
  installing bad waypoints). Integration bug — diagnose before claiming any
  binary-SPL win.
- **C alone fails** with A + B passing → main effect smaller than +0.3 but
  mechanism is real. Pick a defensible weaker bar in the writeup; **do not
  move the goalposts retroactively in this spec.**

## Testing strategy

Three layers; layers 1 + 2 gate the paid run via the `[3/6]` FATAL-on-fail
pre-test block (same pattern as `test_analyze_ablation.py` in the fold-revisit
milestone).

### Layer 1 — pure geometry & parsing (stdlib + numpy; no Habitat, no GPU)

`embodied_memory/scripts/test_goal_detector.py`:

- `case_parse_qwen_bbox_well_formed` — `...<|box_start|>100,200,300,400<|box_end|>...` → `(100, 200, 300, 400)`
- `case_parse_qwen_bbox_normalized` — Qwen2-VL also emits coords in the [0, 1000] normalized space; scaling to image pixels is correct
- `case_parse_no_bbox_returns_None` — caption with no box tokens
- `case_parse_malformed_returns_None` — partial / garbled tokens
- `case_parse_multi_bbox_picks_closest_depth` — multiple bboxes, lowest depth-at-center wins
- `case_back_project_pinhole_geometry` — synthetic depth + intrinsics → known 3D point (≤ 1e-6)
- `case_back_project_invalid_depth_returns_None` — NaN / 0 / inf at center pixel
- `case_robust_depth_5x5_median` — handles a few NaN pixels in the 5×5 window
- `case_snap_within_threshold_returns_point` — mock pathfinder returns nearby navigable point
- `case_snap_beyond_threshold_returns_None` — snap travelled > 0.5 m → None

### Layer 2 — `episode_runner` integration with a mock `GoalDetector`

`embodied_memory/scripts/test_episode_runner_detector.py`:

- `case_detector_off_unchanged` — flag off → keyword-STOP path byte-identical to today *(flag-gate regression)*
- `case_detector_on_locate_returns_None_fallback_STOP` — keyword fires, mock returns `None`, agent STOPs immediately *(no regression)*
- `case_detector_on_locate_installs_waypoint` — mock returns 3D point, waypoint installed, ShortestPathFollower drives until STOP
- `case_detector_counters_increment_correctly` — `n_detector_called` / `_localized` / `_offnavmesh` / `_approach_success` accounting

### Layer 3 — pre-flight smoke (during live run, before the 96-ep matrix)

**chair-warm × detector-ON × 1 episode**, eyeballed:

1. Qwen-VL emits a bbox on the goal chair (not a distractor).
2. Depth at bbox center is plausible (0.5–2.5 m).
3. Back-projected point is near the goal.
4. `snap_point` lands on adjacent navigable floor.
5. `ShortestPathFollower` drives there and STOPs at ≤ 0.1 m.

**GO/NO-GO** before spending the rest of the GPU-day. If NO-GO, fall back to
GroundingDINO-Tiny (same `locate()` interface; ~1-day swap).

## Edge cases (baked into the design)

1. **Multi-bbox** → pick lowest depth-at-center (closest physical surface).
2. **No bbox / parse-fail / off-navmesh** → `None` → fallback STOP. Monotonicity invariant.
3. **Noisy depth at bbox center** → 5×5 patch median (`_robust_depth_at_pixel`).
4. **Back-projected point inside a wall or off-floor** → `pathfinder.snap_point` still returns *something*; `max_snap_dist = 0.5 m` rejects implausible snaps → `None` → fallback STOP.
5. **False-positive captioner keyword** (e.g. "chair" mentioned but no chair visible) → detector returns `None` → fallback STOP. Identical to today.
6. **Approach overshoots horizon** → no new cap; existing step budget governs. ShortestPathFollower converges in < 10 steps on val_mini in practice.

## Risks → mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen2-VL grounding too noisy at 0.5–2 m HM3D ranges | Medium | Layer-3 pre-flight is the explicit GO/NO-GO. If NO-GO, swap to GroundingDINO-Tiny via the same `locate()` interface (~1 day). |
| GPU OOM from the extra Qwen forward pass | Low | Zero new weights; one extra inference per keyword-STOP event (~1–2 per episode). Verify in pre-flight. |
| `success_distance = 0.1 m` is **geodesic**; Euclidean back-project may land just outside the success ring | Medium | `pathfinder.snap_point` works in navmesh geodesic space; ShortestPathFollower's STOP heuristic targets < 0.1 m geodesic. Tested at Layer 2 via mock pathfinder. |
| Detector helps cold visits too, confounding the LTM contrast | None — already controlled | Gate B (`s3-det − s1-det`) subtracts out the detector's cold-visit benefit by construction. |

## Out of scope (explicitly)

- Periodic re-detection / multi-view triangulation during the approach phase
  (v2 lever, only if Layer-3 smoke or the 6-cell run shows residual
  mis-targeting from a single-shot locate).
- Replacing the captioner for LTM indexing (chose option 1 in brainstorming —
  the LTM-indexing path is untouched).
- Widening the revisit matrix to tv_monitor / plant / toilet or more scenes
  (separate, lower-effort milestone — Decision B from the brainstorming).
- Activating the wired learned seams (`train_predictor`, `train_scorer`,
  R-weighted consolidation; Decision C — separate milestone).
- Any change to the dialogue/MSC path (`dialogue_memory/`).
- Changing `success_distance` from 0.1 m — the whole point is to satisfy it.

## Effort estimate

- Implementation + tests: 1–2 days.
- Live ablation (96 eps × ~2.6 min/ep): ~4 GPU-hours sequential; possibly
  twice for a variance check → ~1 operator-day.
- Analysis + writeup: ~0.5 day.
- **Total: 2–3 days from `git checkout -b` to milestone GREEN/RED verdict.**
