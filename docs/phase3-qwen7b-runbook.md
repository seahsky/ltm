# Phase-3 RACE runbook — Qwen2.5-7B planner rerun

End-to-end runbook for the Phase-2 gate retry on RMIT RACE using a
larger planner. The Qwen lightweight pair from the 2026-05-22 run
(`PHASE2_ABLATION_REPORT.md` Run 2) failed C1 because Qwen2.5-3B
regurgitates the prompt's "Current position" as its ANSWER, producing
zero-displacement waypoints. The original ReMEmbR paper uses 7B-class
planners; Run 1's Mistral-7B succeeded at picking useful waypoints
(the agent navigated to within 0.59 m of a sofa). This session swaps
the 3B for **Qwen2.5-7B-Instruct** and retries the gate.

**Target gate:** `analyze_ablation.py` reports `phase 2 gate: PASS`
(C1 ∧ C2 — backbone alive + memory helps soft_spl with p<0.1).

**Total budget envelope:** ~$8–15 typical, $19 cap (remaining from the
$23 Phase-2 envelope; ~$4 already spent in Run 2).

**Pre-requisite reading:** `PHASE2_ABLATION_REPORT.md` Run 2 section
(three layered bugs, especially #2 regurgitation and #3 controller stall).

---

## Phase 0 — Pre-flight decisions (local, free, ~30 min)

Before provisioning anything on RACE, decide whether to land the
**step_controller collision-escape patch** locally first. This is the
Bug 3 fix from the Run 2 writeup — independent of which planner is
used, and almost certainly the difference between "gets close to
goals" and "actually succeeds". Without it, even a perfect planner
will stall on the starting wall like Run 2 did.

**Minimum patch sketch** (in `embodied_memory/episode_runner.py` or
`embodied_memory/frontier_planner.py`'s `step_controller`):

- Track the last K=3 agent positions.
- If all three are within 0.1 m of each other AND the last action was
  FORWARD, force a TURN_LEFT (or alternate L/R) on the next decision.
- Reset the counter on any successful translation.

If this lands cleanly with unit-level testing on cached scenes, push
before provisioning. If it's larger than expected, defer — the 7B
planner alone may produce non-zero successes if it picks waypoints
that route around the wall on its first turn.

**Gate:** decide land-now vs defer. Either way commit the decision in
a sentence before Phase 1.

---

## Phase 1 — RACE bring-up (G15, ~$0.40, ~20 min)

Provision **G15 (g6.2xlarge: 1×NVIDIA L4 24 GB VRAM, 4 CPU, 32 GB RAM,
$1.27/hr)**. **Root volume: 150 GB** (Run 2 settled on this — 50 GB
is too tight once Qwen-7B weights land).

```bash
cd ~ && git clone https://github.com/seahsky/ltm.git && cd ltm
git checkout phase2-readiness
git pull
conda env create -f embodied_memory/environment.yml
conda activate ltm-embodied

cat >> ~/.bashrc <<'EOF'
export REMEMBR_CAPTIONER_MODEL=Qwen/Qwen2-VL-2B-Instruct
export REMEMBR_PLANNER_MODEL=Qwen/Qwen2.5-7B-Instruct
EOF
source ~/.bashrc

python models/download_remembr_models.py
```

The captioner stays at Qwen2-VL-2B (it worked fine in Run 2 — the failure
was on the planner side, not the visual side). The planner is the only
swap.

**VRAM budget on L4 (24 GB):**

| Component | fp16 |
|---|---|
| Qwen2-VL-2B captioner | ~4.5 GB |
| Qwen2.5-7B-Instruct planner | ~14 GB |
| CLIP + open_clip | ~1 GB |
| Habitat-sim GPU buffers | ~1 GB |
| KV cache + activations headroom | ~3 GB |
| **Total** | **~23.5 GB** |

Tight but should fit. **If OOM at smoke time**, fallback options in
order of preference:

1. Quantize the planner to 8-bit via `bitsandbytes` — set
   `REMEMBR_PLANNER_LOAD_IN_8BIT=1` (will need a tiny env-var hook in
   `_lazy_load_llm`, ~5 lines).
2. Drop captioner to bf16 + planner to bf16 (`REMEMBR_*_DTYPE=bfloat16`
   if the loader honors it).
3. Swap captioner for `Qwen/Qwen2-VL-2B-Instruct-AWQ` (4-bit, ~1.5 GB).
4. Last resort: drop the planner to **Qwen2.5-7B-Instruct-AWQ** (4-bit,
   ~5.5 GB) — accuracy trade-off but a known-good fit.

Don't fall back to 3B again — that's the failure we're escaping from.

**HM3D data:** rsync from laptop (same as Run 2) — gitignored:

```bash
# From laptop, NOT RACE:
rsync -avz --progress \
  -e "ssh -i /Users/kyseah/Documents/Keys/Sky-race.pem" \
  /Users/kyseah/Documents/GitHub/ltm/data/hm3d/ \
  ec2-user@<RACE_DNS>:~/ltm/data/hm3d/
```

After rsync, **rebuild the symlink on RACE** (the local one is
absolute and dangles on RACE):

```bash
cd ~/ltm/data/hm3d/scene_datasets
rm hm3d
ln -s ../versioned_data/hm3d-0.2/hm3d hm3d
ls -L hm3d/ | head    # should list 00800-... etc.
```

**Gate:** conda env activates, both snapshots cached (`ls ~/.cache/huggingface/hub/ | grep -i qwen` shows both), HM3D scenes link resolves, no HF auth errors.

---

## Phase 2 — Smoke gate (~$0.40, ~10 min)

```bash
cd ~/ltm
nvidia-smi    # confirm ~22 GB free before launching

mkdir -p runs
python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting 3 --scene all --n-episodes 1 --target any \
    --out-dir runs/remembr-smoke-qwen7b 2>&1 | tee runs/remembr-smoke-qwen7b.log

python - <<'PY'
import json, math
ep = json.load(open('runs/remembr-smoke-qwen7b/episode_000.json'))
steps = ep.get('steps', [])
pos = [(s['agent_pos'][0], s['agent_pos'][2]) for s in steps]
path = sum(math.hypot(pos[i+1][0]-pos[i][0], pos[i+1][1]-pos[i][1]) for i in range(len(pos)-1))
print(f"n_steps={ep['n_steps']} stops={ep['n_stop_signals']} "
      f"success={ep['success']} soft_spl={ep['soft_spl']:.3f}")
print(f"dist_to_goal={ep['distance_to_goal']:.2f}m  path_traveled={path:.2f}m")
PY
```

**Pass conditions** (any failure → consult the matching branch below):

| Field | Pass | If fail → |
|---|---|---|
| Crash-free | run finishes, JSON parses | check VRAM (`nvidia-smi`), see OOM fallbacks above |
| `n_steps` | > 30 | controller-stall still active — Phase 0 patch needed |
| `path_traveled` | ≥ 2 m | agent is actually walking, not just spinning |
| `dist_to_goal` | < starting dist | navigated *toward* goal, not away |

**Soft signals (informative, not gating):**
- `n_stop_signals` could be 0 or 1. 0 means STOP never triggered (LLM
  may need ANSWER instead). 1 means grounded STOP fired — that's fine
  as long as `dist_to_goal` is small.
- `success=False` is acceptable on a single smoke — minival episodes
  are hard.

If smoke passes the four conditions: kick off Phase 3. If not, paste
the failure output back and we triage before the meter eats budget.

---

## Phase 3 — Full ablation (~$6–10, ~5–8 h in tmux)

```bash
tmux new -s phase3
unset REMEMBR_STOP_MIN_STEP REMEMBR_STOP_COS REMEMBR_MIN_WAYPOINT_DIST   # defaults
for s in 1 2 3; do
  python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting $s --scene all --n-episodes 30 --target any \
    --out-dir runs/abl-s${s}-qwen7b 2>&1 | tee runs/abl-s${s}-qwen7b.log
done
# Ctrl+B then D to detach
```

Monitor from laptop with the SSH alias (or one-off `ssh` command with
`-i`):

```bash
ssh ec2-user@<RACE_DNS> -i /Users/kyseah/Documents/Keys/Sky-race.pem \
  'tail -f ~/ltm/runs/abl-s3-qwen7b.log | grep -E "episode|spl|success|n_stop_signals"'
```

**Mid-run sanity** (every ~1 hr):
- `ls runs/abl-s*-qwen7b/episode_*.json | wc -l` grows ~3–6 per hour
  per setting. Run 2 was faster (~10 episodes/min) because every
  episode hit the 9-step floor — that was the bug, expect 7B to take
  more steps per episode.
- `tail -1 runs/abl-s3-qwen7b.log` shows a recent episode number, not
  stuck.

**Cost gate:** if a setting hasn't completed 5 episodes after 90 min,
something is wrong (VRAM thrash, prompt drift, sim hang). Kill,
diagnose, restart — don't let it burn 8 h to fail.

---

## Phase 4 — Analysis (free, local)

```bash
# Locally:
rsync -avz -e "ssh -i /Users/kyseah/Documents/Keys/Sky-race.pem" \
  ec2-user@<RACE_DNS>:~/ltm/runs/abl-s1-qwen7b \
  ec2-user@<RACE_DNS>:~/ltm/runs/abl-s2-qwen7b \
  ec2-user@<RACE_DNS>:~/ltm/runs/abl-s3-qwen7b \
  /Users/kyseah/Documents/GitHub/ltm/runs/

python embodied_memory/scripts/analyze_ablation.py \
    runs/abl-s1-qwen7b runs/abl-s2-qwen7b runs/abl-s3-qwen7b \
    | tee runs/phase3-qwen7b-gate.txt
```

---

## Branch on gate result

### PASS (C1 ∧ C2)

1. Append a "Run 3" section to `PHASE2_ABLATION_REPORT.md` mirroring
   the Run 2 structure — keep Run 1 and Run 2 intact as parallel
   baselines.
2. Force-add `runs/abl-s{1,2,3}-qwen7b/` + `runs/phase3-qwen7b-gate.txt`
   past the `runs/` gitignore (see `2a6f6ac` for the pattern).
3. Phase-2 milestone done — Phase-3-facing deliverables (G3 trainers,
   G5 affordance refresh, val scale-up) become eligible to schedule.
   Don't bundle them into this session.

### FAIL C1 (still 0 successes in S1)

Run the action histogram diagnostic:

```bash
python embodied_memory/scripts/diagnose_stop.py runs/abl-s3-qwen7b
```

Then check (in priority order):

1. **`stop=0` in the action histogram** — grounded STOP not firing.
   Test cosine knob: `REMEMBR_STOP_COS=0.20` on a single smoke.
2. **`fwd≈100%` and `dist_to_goal` unchanged** — controller stall
   (same as Run 2 Bug 3). If Phase 0 patch wasn't landed, this is the
   moment to write it before re-running.
3. **`fwd` distributed but never reaches goals** — planner is
   exploring but not goal-directed. Inspect the LLM tool_calls in
   episode JSON; if all unparseable, the chat template needs work.

Write up as Run 3 negative result with whichever diagnosis above
applies. Budget allows for ~$2–3 of re-runs at this stage; beyond that,
defer to a future session.

### FAIL C2 (soft_SPL Δ not significant)

If C1 passes but the soft_SPL delta straddles zero, one disambiguator
run (perturb seed on S3 only, ~$2–3): is the gap real but underpowered,
or genuinely null? If still flat across two seeds, write up as
"memory neutral at this backbone capability" — different conclusion
from Run 1/Run 2.

---

## Cost ceiling

| Phase | Best | Worst |
|---|---|---|
| Pre-flight (Phase 0 patch) | $0 | $0 |
| Phase 1 bring-up | $0.40 | $1 |
| Phase 2 smoke (1–3×) | $0.40 | $2 |
| Phase 3 full ablation | $6 | $10 |
| Phase 4 analysis | $0 | $0 |
| Rerun buffer (one FAIL branch) | $0 | $5 |
| **Total** | **$7** | **$18** |

**Hard cap: $19** (remaining envelope from the Phase-2 budget). If
costs trend past that without a gate read, stop and escalate.

---

## What this runbook deliberately does NOT do

- **G3 trainers, G5 affordance refresh, val scale-up.** All deferred
  to a separate session after the gate read. Same logic as the Phase-2
  playbook.
- **Mistral-7B fallback.** If Qwen2.5-7B fails on C1 with controller
  stall fixed, the next debug step is the bridge-CLIP-image-LTM
  refactor for `_maybe_stop`, not a swap to a third planner. Different
  axis of variation.
- **Multi-seed averaging.** Single-seed paired bootstrap is the gate
  protocol; only add seeds in the C2-FAIL disambiguator branch.

---

## Critical files (unchanged from Phase-2)

- `embodied_memory/run_hm3d_pol.py` — runner CLI
- `embodied_memory/remembr_backbone.py` — `_maybe_stop` (line 454),
  `_llm_propose` (line 601), regurgitation guard (`bd60288`)
- `embodied_memory/episode_runner.py` — STOP force-select (line 244),
  `current_step` threading (`2f2d141`)
- `embodied_memory/scripts/analyze_ablation.py` — gate logic
- `embodied_memory/scripts/diagnose_stop.py` — action histogram
- `models/download_remembr_models.py` — honors `REMEMBR_*_MODEL` env vars

---

## Run-4 amendment (2026-05-23)

After Run 3 (this runbook's named session) failed the smoke gate at the
movement layer (0.04 m total across 21 steps), the immediate next session
**did not** take the runbook-recommended Option 1 (bridge-CLIP-image STOP
refactor for `_maybe_stop`). Instead it picked the deferred **Option 2a**
from §"What's next" §2: inject 2–3 frontier-planner candidates into the
LLM proposal pool when `backbone=remembr`.

### Why Option 2a over Option 1

Run 3 showed C1 was gated by **movement**, not STOP precision. With
`STOP_COS=0.40` + `STOP_MIN_STEP=20` already in place, the smoke logged
zero false STOPs. The bridge-CLIP STOP refactor only helps once the agent
is near a goal — Run 3 never got there. Obstacle-aware proposals are the
direct lever for movement.

### Where it lands in the code

Single-seam change in
`embodied_memory/episode_runner.py::_propose_candidates`. The `remembr`
branch now concatenates up to `REMEMBR_FRONTIER_INJECT=3` frontier-planner
candidates onto the LLM output, de-duped against existing LLM picks by
`REMEMBR_MIN_WAYPOINT_DIST` (default 0.5 m). STOP short-circuit is
preserved: if the LLM emitted a `stop_signal` candidate, it returns alone.

The 3-setting (memory off / STM / full) protocol is unchanged. Frontier
injection is applied uniformly across all settings — the same way
`509dbc8` (Run 1 → Run 2 STOP fix) and `117028d` / `6265870` (Run 2 → Run
3 controller patches) were applied uniformly.

### Status when this amendment landed

Code change + module-level sanity test passing locally. **No RACE
execution yet.** The full operator runbook for the paid run lives in
`PHASE2_ABLATION_REPORT.md`'s Run-4 section. The original Phase-1 through
Phase-4 structure of *this* runbook still applies verbatim for the next
session — Run 4 only changes what code is checked in before Phase 1.

### Bridge-CLIP-image STOP refactor — still deferred

If Run 4 succeeds at the movement layer (smoke passes `path_traveled ≥
4 m` and `n_frontier_chosen ≥ 1`) but fails C1 because the agent doesn't
STOP at goals, the bridge-CLIP STOP refactor becomes the next session's
lever. That's the original Option 1 from this runbook's §"What's next",
now scoped one layer up.

## Run-5 amendment (2026-05-24)

Run 4's prep landed the obstacle-aware proposal pool but the agent still
hadn't been shown to move (it was never RACE-executed). Run 5 lands two
complementary levers before the next paid bring-up, plus a decisive
diagnostic, all verified faiss/habitat-free locally. Commits `a26b1b6`
(densified splat + `grid_stats`) and `f713119` (oracle + grid logging + tests).

### Two levers + a diagnostic

1. **Densified depth splat** (`frontier_planner.py::update`). The old single
   middle-row scanline (64 cols) sat at eye height, hit walls, and carved too
   few FREE cells, so frontiers clustered against walls and the agent had no
   navigable subgoal. Replaced with a multi-row (~28×28) per-pixel
   back-projection from `hfov=79°` pinhole intrinsics + a **height gate**
   (`camera_height_m=0.88`, `obstacle_min_h=0.3`): march FREE along each ray's
   ground range, mark the endpoint OCCUPIED only if it rises >0.3 m above the
   floor (set in `reset(agent_pos)`), else FREE — so floor/doorways become
   walkable. Local sanity: 926 FREE cells vs 26 for the old scanline (35×).
2. **`grid_stats()` instrumentation** logged per episode
   (`grid_cells_{free,occupied,unknown}`, `grid_frontier_cells`) so the next
   smoke is interpretable — if `cells_free` is still tiny, densification didn't
   take and we iterate locally, not on RACE.
3. **`--backbone oracle`** (decisive diagnostic). A `ShortestPathFollower`
   steers straight to the goal, bypassing all candidate/scorer/memory/model
   loads (`bridge=None`, no CLIP/captioner) but logging the same metrics. It
   answers the question Runs 1–4 never did: *is the env/episode navigable at
   all with a perfect planner?*

### The smoke this enables (Phase 1 bring-up, then:)

```bash
# A) Oracle env check — no model loads, both scenes
for sc in TEEsavR23oF wcojb4TFT35; do
  python -m embodied_memory.run_hm3d_pol --mode live --backbone oracle \
    --setting 1 --scene $sc --n-episodes 2 --target any --no-strict-pass \
    --out-dir runs/oracle-smoke-$sc
done
# B) Densified-grid escape check — full stack
python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting 3 --scene wcojb4TFT35 --n-episodes 2 --target any \
    --out-dir runs/remembr-dense-smoke
```

Read the result with the decision tree in `PHASE2_ABLATION_REPORT.md` Run-5
("Decision tree on the read"): oracle-reaches + densified-passes → full 3×30
ablation (Phase 3); oracle-reaches + densified-stalls → iterate the splat
locally; oracle-also-stalls → env debugging, no planner/perception fix matters.
Phases 1–4 of this runbook still apply verbatim; Run 5 only changes the code
checked in before Phase 1 and runs the two cheap smokes above before Phase 3.

### Status when this amendment landed

Code change + 13-case module-level sanity suite passing locally. **No RACE
execution yet** (CUDA-host operator step). Run-3 stopgap
(`REMEMBR_STOP_COS=0.40 REMEMBR_STOP_MIN_STEP=20`) stays in place for smoke B.
