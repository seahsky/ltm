# Phase-2 RACE runbook — lightweight pair

End-to-end runbook for the Phase-2 ablation rerun on RMIT RACE using the
Qwen2-VL-2B + Qwen2.5-3B-Instruct pair (~11 GB VRAM fp16, fits a single
L4). Replaces the JarvisLabs A100 path for university-billable compute.

**Target gate:** `analyze_ablation.py` reports `phase 2 gate: PASS`
(C1 ∧ C2 — backbone alive + memory helps soft_spl with p<0.1).

**Total budget envelope:** $10–23 typical, $50 cap.

---

## Phase 0 — Pre-flight (local, free, ~10 min)

```bash
# 1. Commit backbone fix + README updates
git add embodied_memory/remembr_backbone.py models/README.md \
        docs/phase2-race-runbook.md
git commit -m "Backbone: model-agnostic chat template + Qwen pair docs"

# 2. Push so RACE can pull
git push origin phase2-readiness

# 3. Verify RACE SSH
ssh <your-race-username>@race.rmit.edu.au true && echo "SSH OK"
```

**Gate:** clean push, SSH works.

---

## Phase 1 — RACE smoke (G15, ~$0.40, ~20 min)

Provision **G15 (g6.2xlarge: 1×L4, 4 CPU, 32 GB RAM, $1.27/hr)** via the
RACE portal. Once the workspace boots:

```bash
# Setup
cd ~ && git clone <your-repo-url> ltm && cd ltm
git checkout phase2-readiness
conda env create -f environment.yml  # or however ltm-embodied is built
conda activate ltm-embodied

# Persist the lightweight pair env vars
cat >> ~/.bashrc <<'EOF'
export REMEMBR_CAPTIONER_MODEL=Qwen/Qwen2-VL-2B-Instruct
export REMEMBR_PLANNER_MODEL=Qwen/Qwen2.5-3B-Instruct
EOF
source ~/.bashrc

# L1: pull weights (~10 GB, ~5–10 min on RACE)
python models/download_remembr_models.py

# L2: 1-episode smoke
python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting 3 --n-episodes 1 --target any \
    --out-dir runs/remembr-smoke-qwen
```

### Gate (manual inspection of `runs/remembr-smoke-qwen/episode_000.json`)

- `stub_mode: false` everywhere → real weights loaded
- ≥1 `kind: "tool"` or `kind: "answer"` in `trace.tool_calls` → planner is
  emitting parseable replies
- `n_steps` reaches 250 without crash → end-to-end pipeline works

### Failure mode: `kind: "unparseable"` every turn

Qwen2.5-3B is ignoring the line-format constraint. Cheapest fix:

1. Edit `sys_prompt` in `remembr_backbone._llm_propose` (around line 616)
   to include a one-shot exemplar:
   ```
   Example output: TOOL: retrieve_from_text(sofa)
   ```
2. Re-run L2 (~$0.40 each iteration).
3. Budget 2–3 retries before considering `meta-llama/Llama-3.2-3B-Instruct`
   as an alternative planner (stronger BFCL, but a second chat template
   in play).

**Cost gate:** if smoke iterations exceed $2 with no parseable reply,
stop and reconsider. Don't burn budget on a failing config.

---

## Phase 2 — Full ablation (G15, ~$6–10, 5–8 hr wall-time)

Once L2 passes, kick off the full 3-setting run inside `tmux` so
disconnect doesn't kill the job:

```bash
tmux new -s phase2
# Inside tmux:
for s in 1 2 3; do
  python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting $s --scene all --n-episodes 30 --target any \
    --out-dir runs/abl-s${s}-qwen 2>&1 | tee runs/abl-s${s}-qwen.log
done
# Detach: Ctrl+B then D
```

### Monitor without staying logged in

```bash
# From your laptop:
ssh ... 'tail -f ~/ltm/runs/abl-s3-qwen.log | grep -E "episode|spl|success"'
```

### Mid-run sanity checks (every ~1 hr)

- `ls runs/abl-s*-qwen/episode_*.json | wc -l` grows ~3–6 per hour per setting
- `tail -1 runs/abl-s3-qwen.log` shows a recent episode number, not stuck

**Cost gate:** if a setting hasn't completed 5 episodes after 90 min,
something is wrong (VRAM thrash, template drift, sim hang). Kill,
diagnose, restart — don't let it burn 8 hr to fail.

---

## Phase 3 — Analysis (local or G14, free–$1)

Pull run dirs locally (free) or analyze on a cheap G14 ($1.05/hr):

```bash
# Locally:
rsync -avz race:~/ltm/runs/abl-s{1,2,3}-qwen/ runs/

python embodied_memory/scripts/analyze_ablation.py \
    runs/abl-s1-qwen runs/abl-s2-qwen runs/abl-s3-qwen \
    | tee runs/phase2-qwen-gate.txt
```

### Gate criteria (from `analyze_ablation.py:250-322`)

| Criterion | Pass condition |
|---|---|
| C1 | `n_success(S1) ≥ 1` — backbone alive |
| C2 | S3−S1 soft_spl Δ > 0, one-sided p < 0.1 |
| C3 (stretch) | S3−S1 SPL Δ > 0 |
| **Phase-2 gate** | C1 ∧ C2 |

---

## Phase 4 — Branch on result

### If gate PASSES → write up + optional scale-up

```bash
# Update PHASE2_ABLATION_REPORT.md with a new section for the Qwen-pair
# baseline (parallel to the existing JarvisLabs fp16 section — both
# stand as independent baselines).

# Optional: full HM3D val scale-up (~$19–32, ~15–25 hr on G15)
# Worth it if credits permit and you want a stronger writeup.
```

### If gate FAILS C1 (no successes)

```bash
python embodied_memory/scripts/diagnose_stop.py runs/abl-s3-qwen
```

Check the action histogram. If `stop=0`, the STOP fix (commit `509dbc8`)
isn't firing for Qwen captions. Likely cause: smaller-VLM captions
don't cosine-match goal text at `REMEMBR_STOP_COS=0.25`. Workaround:

```bash
REMEMBR_STOP_COS=0.20 \
  python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
  --setting 3 --scene all --n-episodes 30 --target any \
  --out-dir runs/abl-s3-qwen-stop020
```

### If gate FAILS C2 (no soft_spl signal)

Two possibilities:

1. Memory genuinely isn't helping — a real negative result, publishable
   as such.
2. Smaller backbone isn't capable enough to benefit from memory.

Before concluding: run setting 3 once more with perturbed episode order
to confirm reproducibility (~$6–10). If still flat, write up as
negative result with clear backbone-capability framing.

---

## Total cost envelope

| Phase | Best case | Worst case |
|---|---|---|
| Pre-flight | $0 | $0 |
| Smoke (1×–3×) | $0.40 | $2 |
| Full ablation (1 pass) | $6 | $10 |
| Analysis | $0 | $1 |
| Rerun buffer | $0 | $10 |
| **Phase-2 milestone total** | **$6.40** | **$23** |
| Stretch: HM3D val scale-up | — | +$32 |

**Median expected: $10–12** out of $50 RACE credits.

---

## Pitfalls to watch

1. **Conda env on RACE.** If `environment.yml` / `requirements.txt`
   isn't in the repo, document the install steps on first boot. Budget
   +30 min before tearing the workspace down.
2. **`models/download_remembr_models.py` respecting env vars.** Confirm
   it reads `REMEMBR_CAPTIONER_MODEL` / `REMEMBR_PLANNER_MODEL` rather
   than hardcoding the 7B+7B defaults. Avoid accidentally pulling 30 GB
   of LLaVA+Mistral.
3. **Qwen2.5-3B `TOOL:` adherence at temperature 0** is unproven for
   this prompt. Phase-1 smoke gate catches it.
4. **Suspend G15 between sessions.** RACE meters wall-clock; idle
   instances burn credit.
5. **Don't compare Qwen-pair results directly to the JarvisLabs fp16
   numbers** as "same experiment". They're parallel baselines — report
   both, frame appropriately.
