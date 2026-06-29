#!/bin/bash
# scripts/race-blip2-frontier.sh — the BETTER-SEARCHER lever: a VLFM-style
# semantic frontier value map whose SIGNAL is a BLIP-2 ITM image-text MATCH
# probability (not the flat CLIP cosine). Measures the ABSOLUTE soft-SPL lift on
# the L4 by swapping the 7B planner for Phi-3.5-mini to make room for BLIP-2.
#
# WHY: the absolute soft-SPL ceiling (~0.39) is path-length-bound (L_b) — the
# frontier scorer is PURELY GEOMETRIC (frontier_planner raw_score = 0.6*size +
# 0.4*dist, zero goal-semantics), so the memory-OFF agent WANDERS. VLFM (ICRA-2024)
# fixes exactly this by scoring each frontier with how much the view toward it
# MATCHES the goal, reaching HM3D ObjectNav SPL 0.304. The value-map plumbing,
# blend, ceiling renorm and memory-injection seam already exist (commit 4ff9584);
# the ONLY change is the value SIGNAL: CLIP cosine (flat on HM3D sim renders, $0
# gate measured 0.020 < 0.05 sep, 3rd flatness measurement) -> BLIP-2 ITM cross-
# attention head (the model VLFM used). The lever stays MODULAR so the LTM still
# injects waypoint candidates into the same pool and the +0.2505 memory delta is
# still measurable; the ceiling renorm (LTM_SEMANTIC_FRONTIER_CEILING=0.45) keeps a
# semantic frontier BELOW a true memory match (>=0.8) so a recall is never crowded.
#
# L4 VRAM FIT: BLIP-2-ITM (~3.5-5 GB resident) does not co-fit under the Qwen2.5-7B
# planner (the 7B backbone alone sits ~20.6 GB on 22.5 GB). So this driver SWAPS
# THE PLANNER to Phi-3.5-mini-instruct (~7.6 GB, frees ~5.5 GB resident) for THIS
# process ONLY — exported BEFORE `source scripts/race-setup.sh` (its line uses :-
# so a pre-export wins), identical to race-owlv2-detector.sh. The 2B captioner is
# KEPT (the LTM indexes its rich captions). Budget with Phi:
#   Phi 7.6 + captioner 4.4 + BLIP-2-ITM ~3.5-5 + CLIP 0.6 + KV ~2 = ~18.5-19.6 GB
#   < 22.5 GB usable  => FITS (headroom ~3 GB; the VRAM PREFLIGHT self-verifies).
#
# CAVEAT (document in any write-up): a Phi planner makes the ABSOLUTE soft-SPL
# NON-cross-quotable to the +0.171/+0.24/+0.2505 7B arc — ACCEPTED, this is a
# DIFFERENT contribution (a stronger searcher / higher absolute). The value-OFF vs
# value-ON and S1-vs-S3 A/Bs stay internally valid (both arms run the SAME Phi
# planner; memory/value injection is planner-independent, n_remembr_chosen~=0).
#
# Pipeline: [1] pull [2] Phi-swap + setup [3] pre-verify (unit tests) [4] planner-
# fit GATE (check_planner_fit; GREEN required) [5] VRAM PREFLIGHT (1-ep S3 value-ON
# + nvidia-smi peak + OOM guard + --blip2-cpu hatch) [6] S1/S3 A/B (value OFF
# baseline vs value ON) via race-revisit.sh [7] analyze + decision rule.
#
# EXECUTE it (do NOT source) — conda activates in its own process:
#
#   nrun bash scripts/race-blip2-frontier.sh --tag blip2v1
#   nrun bash scripts/race-blip2-frontier.sh --tag blip2v1 --weight 0.4
#   nrun bash scripts/race-blip2-frontier.sh --tag blip2cpu --blip2-cpu   # CPU value model
#   nrun bash scripts/race-blip2-frontier.sh --tag blip2v1 --planner Qwen/Qwen2.5-3B-Instruct
#
# git pull FIRST (this driver + perception.Blip2ITMScorer are new; the driver
# self-pulls at step 1 but that only takes effect on the 2nd invocation).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults ---
WEIGHT="0.5"                                   # LTM_SEMANTIC_FRONTIER blend weight
CEILING="0.45"                                 # LTM_SEMANTIC_FRONTIER_CEILING renorm
VALUE_MODEL="Salesforce/blip2-itm-vit-g"       # BLIP-2 ITM checkpoint
PROMPT="Seems like there is a {goal} ahead."   # VLFM value prompt
AB_SCENES="wcojb4TFT35 TEEsavR23oF"
AB_CATS="toilet chair"                         # category gradient extremes
NWARM="3"
TAG=""
PLANNER="microsoft/Phi-3.5-mini-instruct"      # VRAM-freeing swap (validated in L3)
BLIP2_CPU=0                                     # --blip2-cpu => value model on CPU
VRAM_GUARD_MIB="21000"                          # FATAL if preflight peak exceeds this
RUN_AB=1

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)         TAG="$2"; shift 2 ;;
    --weight)      WEIGHT="$2"; shift 2 ;;
    --ceiling)     CEILING="$2"; shift 2 ;;
    --value-model) VALUE_MODEL="$2"; shift 2 ;;
    --prompt)      PROMPT="$2"; shift 2 ;;
    --scenes)      AB_SCENES="$2"; shift 2 ;;
    --categories)  AB_CATS="$2"; shift 2 ;;
    --n-warm)      NWARM="$2"; shift 2 ;;
    --planner)     PLANNER="$2"; shift 2 ;;
    --blip2-cpu)   BLIP2_CPU=1; shift ;;
    --vram-guard)  VRAM_GUARD_MIB="$2"; shift 2 ;;
    --skip-ab)     RUN_AB=0; shift ;;
    -h|--help)     sed -n '1,55p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
AB_CATS="${AB_CATS//,/ }"
AB_SCENES="${AB_SCENES//,/ }"
[ -z "$TAG" ] && { echo "FATAL: --tag <name> required"; exit 1; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }

CPU_FLAG=""
VALUE_CPU_ENV=""   # exported into the A/B arms so the OOM hatch reaches the paid matrix
if [ "$BLIP2_CPU" -eq 1 ]; then
  CPU_FLAG="--blip2-cpu"
  VALUE_CPU_ENV="LTM_VALUE_CPU=1"
  echo "  [blip2-cpu] value model forced to CPU for BOTH the preflight AND the A/B matrix."
  echo "  WARNING: a CPU preflight cannot certify the GPU matrix fits — but arm B now also"
  echo "  runs the value model on CPU (LTM_VALUE_CPU=1), so preflight and matrix are consistent"
  echo "  (no false-GREEN). This is the slow OOM-safe mode, not a GPU fit certification."
fi

OUT_DIR="runs/${TAG}"
mkdir -p "$OUT_DIR"
PREFLIGHT_DIR="${OUT_DIR}-preflight"
VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
PF_DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/blip2pf_${TAG}"
PF_DS="${PF_DS_DIR}/blip2pf_${TAG}.json.gz"
PF_SCENE="$(echo "$AB_SCENES" | awk '{print $1}')"
PF_CAT="$(echo "$AB_CATS" | awk '{print $1}')"

banner() { printf '\n########## %s ##########\n' "$1"; }

# --- 1. git pull ---
banner "[1/7] git pull --ff-only"
if [ -n "${RACE_SKIP_PULL:-}" ]; then echo "  RACE_SKIP_PULL set — skipping"; else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

# --- 2. Phi-swap + conda setup ---
banner "[2/7] Phi-swap + conda setup (source scripts/race-setup.sh)"
# Swap the planner BEFORE sourcing race-setup.sh — its line uses :- so a pre-export
# wins (identical to race-owlv2-detector.sh). Frees the VRAM BLIP-2 needs; local to
# this process so L1/L2 and the published 7B arc are untouched.
export REMEMBR_PLANNER_MODEL="$PLANNER"
echo "  [planner-swap] REMEMBR_PLANNER_MODEL=$REMEMBR_PLANNER_MODEL (small planner => VRAM for BLIP-2)"
set +u
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }
set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
# expandable_segments reduces fragmentation between the heterogeneous model
# allocations (Phi + Qwen2-VL + BLIP-2 + CLIP co-resident).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU capacity banner (mirror race-owlv2-detector.sh:135-141).
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader \
    | sed 's/^/  GPU: /'
  echo "  budget(Phi): Phi(~7.6) + Qwen2-VL-2B(~4.4) + BLIP-2-ITM(~3.5-5) + CLIP(~0.6) + KV(~2) = ~18.5-19.6 < 22.5 GB"
else
  echo "  (nvidia-smi not on PATH — cannot verify GPU capacity)"
fi

# --- 3. pre-test code verify (free unit tests) ---
banner "[3/7] pre-test code verify (value-scorer + semantic-frontier + planner-fit + builder)"
python embodied_memory/test_value_scorer.py \
  || { echo "FATAL: test_value_scorer failed."; exit 1; }
python embodied_memory/test_semantic_frontier.py \
  || { echo "FATAL: test_semantic_frontier failed."; exit 1; }
python embodied_memory/scripts/test_check_planner_fit.py \
  || { echo "FATAL: test_check_planner_fit failed."; exit 1; }
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: test_make_revisit_smoke failed."; exit 1; }

# --- 4. planner-fit GATE (certify the Phi swap fits + navigates + LTM fires) ---
banner "[4/7] planner-fit GATE — Phi swap viable? (scripts/race-planner-fit-smoke.sh)"
# The planner-fit smoke pulls + sets up in its OWN process; pass the same planner.
# GREEN ⇒ FIT(no OOM) + NAVIGATE + LTM-FIRES + PARSEABLE. RED ⇒ abort before spend.
# (it self-pulls at its step 1; a 2nd ff-only pull here is a harmless no-op.)
bash scripts/race-planner-fit-smoke.sh \
    --planner "$PLANNER" --scene "$PF_SCENE" --category "$PF_CAT" \
    --tag "fit-${TAG}" 2>&1 | tee "${OUT_DIR}/planner_fit.log"
fit_rc=${PIPESTATUS[0]}
if [ "$fit_rc" -ne 0 ]; then
  echo "FATAL: planner-fit gate RED (rc=$fit_rc) — Phi swap not viable on this host."
  echo "  See ${OUT_DIR}/planner_fit.log. Try --planner Qwen/Qwen2.5-3B-Instruct or fix OOM."
  exit 1
fi
echo "  planner-fit GATE GREEN — Phi swap fits + navigates + LTM fires."

# --- 5. VRAM PREFLIGHT: 1-ep S3 with the BLIP-2 value model ON ---
banner "[5/7] VRAM PREFLIGHT: 1-ep S3 value-ON (backend=blip2 weight=$WEIGHT) + nvidia-smi peak"
rm -rf "$PF_DS_DIR"
SRC="${VALMINI}/${PF_SCENE}.json.gz"
[ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
python embodied_memory/scripts/make_revisit_smoke.py \
    --src "$SRC" --scene "$PF_SCENE" --categories "$PF_CAT" --n-warm 1 \
    --out-dir "$PF_DS_DIR" \
  || { echo "FATAL: preflight dataset build failed (does $PF_CAT exist in $PF_SCENE?)"; exit 1; }
[ -f "$PF_DS" ] || { echo "FATAL: preflight dataset not written: $PF_DS"; exit 1; }

# Background nvidia-smi peak sampler (best-effort) while the preflight episode runs.
PEAK_FILE="${PREFLIGHT_DIR}.peak"
mkdir -p "$PREFLIGHT_DIR"
: > "$PEAK_FILE"
if command -v nvidia-smi >/dev/null 2>&1; then
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -dc '0-9' >> "$PEAK_FILE"; echo >> "$PEAK_FILE"; sleep 2;
    done ) &
  SAMPLER_PID=$!
else
  SAMPLER_PID=""
fi

# Run the 1-episode S3 smoke with the value model ON (cold+warm, 2 eps, one
# process so the cold sighting seeds the LTM the warm visit retrieves).
LTM_SEMANTIC_FRONTIER_CEILING="$CEILING" \
LTM_SEMANTIC_FRONTIER_PROMPT="$PROMPT" \
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --setting 3 --episodes-path "$PF_DS" \
    --scene "$PF_SCENE" --target any --n-episodes 2 \
    --semantic-frontier-weight "$WEIGHT" --semantic-frontier-backend blip2 \
    --value-model "$VALUE_MODEL" $CPU_FLAG \
    --out-dir "$PREFLIGHT_DIR" 2>&1 | tee "${PREFLIGHT_DIR}.log"
pf_rc=${PIPESTATUS[0]}
[ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null

# OOM guard: a CUDA OOM anywhere in the log is FATAL (the matrix would crash).
if grep -qiE "out of memory|CUDA error|torch.cuda.OutOfMemoryError" "${PREFLIGHT_DIR}.log"; then
  echo "FATAL: VRAM PREFLIGHT hit a CUDA OOM. The full stack does not fit on this host."
  echo "  Re-run with --blip2-cpu (value model on CPU, slow but cannot OOM), or use a"
  echo "  smaller --value-model / --planner. Log: ${PREFLIGHT_DIR}.log"
  exit 1
fi
if [ ! -f "${PREFLIGHT_DIR}/summary.json" ]; then
  echo "FATAL: VRAM PREFLIGHT wrote no summary.json (the run crashed before writing). rc=$pf_rc"
  exit 1
fi

# nvidia-smi PEAK guard: leave a ~1.4 GB margin under the 22479 MiB L4.
PEAK_MIB="$(sort -nr "$PEAK_FILE" 2>/dev/null | head -1 | tr -dc '0-9')"
if [ -n "$PEAK_MIB" ]; then
  echo "  VRAM PREFLIGHT peak memory.used = ${PEAK_MIB} MiB (guard ${VRAM_GUARD_MIB} MiB)"
  if [ "$PEAK_MIB" -gt "$VRAM_GUARD_MIB" ] 2>/dev/null; then
    echo "FATAL: preflight peak ${PEAK_MIB} MiB > guard ${VRAM_GUARD_MIB} MiB — too close to OOM."
    echo "  Re-run with --blip2-cpu, or raise --vram-guard only if you accept the OOM risk."
    exit 1
  fi
else
  echo "  (no nvidia-smi peak samples captured — relying on the OOM-grep guard above)"
fi

# Verify the value backend actually fired (a flat/erroring scorer would null the
# A/B by construction). n_memory_chosen should also stay >0 (ceiling held).
N_MEM="$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(s.get('n_memory_chosen', 0))" "${PREFLIGHT_DIR}/summary.json" 2>/dev/null || echo 0)"
echo "  preflight n_memory_chosen=$N_MEM (must stay >0 — the ceiling renorm guards the memory delta)"
# Enforce it (was a dead print). If the BLIP-2 value frontier saturates and crowds
# out memory (the LTM_SEMANTIC_FRONTIER_CEILING=$CEILING renorm risk) or the value
# scorer never fired, n_memory_chosen collapses to 0 — the A/B would then measure a
# value frontier that has ALREADY destroyed the +0.2505 memory delta. FATAL so the
# paid matrix never runs under a crowded-out memory.
if ! [ "$N_MEM" -gt 0 ] 2>/dev/null; then
  echo "FATAL: preflight n_memory_chosen=$N_MEM (==0) — the value frontier crowded out memory"
  echo "  (or the value scorer never fired). The ceiling renorm (LTM_SEMANTIC_FRONTIER_CEILING="
  echo "  $CEILING) failed to keep a semantic frontier below a memory match. Lower --weight or"
  echo "  --ceiling, or inspect the value distribution, before spending on the matrix."
  echo "  Log: ${PREFLIGHT_DIR}.log"
  exit 1
fi
echo "  VRAM PREFLIGHT GREEN — value-ON stack fits, ran, and memory survived (n_memory_chosen=$N_MEM)."

# --- 6. S1/S3 A/B: value OFF baseline vs value ON ---
if [ "$RUN_AB" -eq 0 ]; then
  banner "DONE (--skip-ab): gates GREEN, A/B skipped by request."
  echo "  preflight: ${PREFLIGHT_DIR}.log"
  exit 0
fi
export LTM_SEMANTIC_FRONTIER_CEILING="$CEILING"

banner "[6/7] A/B arm A: value OFF (geometric baseline) — S1/S3 on [$AB_SCENES]x[$AB_CATS]"
# Arm A: geometric baseline (LTM_SEMANTIC_FRONTIER=0 -> weight 0.0, byte-identical).
# shellcheck disable=SC2086
LTM_SEMANTIC_FRONTIER=0 LTM_SEMANTIC_FRONTIER_BACKEND=clip \
  bash scripts/race-revisit.sh \
    --settings "1 3" --scenes "$AB_SCENES" --categories "$AB_CATS" \
    --n-warm "$NWARM" --tag "${TAG}-base" 2>&1 | tee "${OUT_DIR}/ab_base.log"
arc=${PIPESTATUS[0]}

banner "[6/7] A/B arm B: value ON (BLIP-2 ITM, weight=$WEIGHT) — S1/S3"
# Arm B: BLIP-2 semantic frontier ON. The value scorer is built inside
# run_hm3d_pol only because weight>0 AND backend=blip2. VALUE_CPU_ENV
# (LTM_VALUE_CPU=1 when --blip2-cpu) carries the OOM hatch into the paid matrix —
# race-revisit.sh cannot pass --blip2-cpu, so run_hm3d_pol reads it from the env.
# shellcheck disable=SC2086
env $VALUE_CPU_ENV \
  LTM_SEMANTIC_FRONTIER="$WEIGHT" LTM_SEMANTIC_FRONTIER_BACKEND=blip2 \
  LTM_VALUE_MODEL="$VALUE_MODEL" LTM_SEMANTIC_FRONTIER_PROMPT="$PROMPT" \
  bash scripts/race-revisit.sh \
    --settings "1 3" --scenes "$AB_SCENES" --categories "$AB_CATS" \
    --n-warm "$NWARM" --tag "${TAG}-on" 2>&1 | tee "${OUT_DIR}/ab_on.log"
brc=${PIPESTATUS[0]}

# --- 6b. A/B OOM + exit-code guards (the matrix is the spend; the preflight
# OOM-grep covered ONLY the 1-ep preflight on the first scene/category. The
# matrix also runs TEEsavR23oF — a scene the preflight never exercised, which can
# peak higher — and fragmentation can accumulate across episodes. episode_runner
# swallows a per-episode torch OOM (except Exception -> log 'crashed' -> continue)
# so a mid-matrix OOM would otherwise complete with degraded arms and exit 0. Grep
# both arm logs and FATAL on any OOM or non-zero arm rc. ---
OOM_RE="out of memory|CUDA error|torch.cuda.OutOfMemoryError"
if grep -qiE "$OOM_RE" "${OUT_DIR}/ab_base.log" 2>/dev/null; then
  echo "FATAL: A/B arm A (value OFF) hit a CUDA OOM — arms are degraded, results invalid."
  echo "  Log: ${OUT_DIR}/ab_base.log. Re-run with --blip2-cpu (forces value model to CPU"
  echo "  in BOTH preflight AND matrix) or a smaller --value-model / --planner."
  exit 1
fi
if grep -qiE "$OOM_RE" "${OUT_DIR}/ab_on.log" 2>/dev/null; then
  echo "FATAL: A/B arm B (value ON) hit a CUDA OOM — the value-ON arm is degraded, results invalid."
  echo "  Log: ${OUT_DIR}/ab_on.log. Re-run with --blip2-cpu (forces value model to CPU"
  echo "  in BOTH preflight AND matrix) or a smaller --value-model / --planner."
  exit 1
fi
if [ "$arc" -ne 0 ] || [ "$brc" -ne 0 ]; then
  echo "FATAL: an A/B arm exited non-zero (base rc=$arc, on rc=$brc) — results invalid."
  echo "  Logs: ${OUT_DIR}/ab_base.log ${OUT_DIR}/ab_on.log"
  exit 1
fi

# --- 7. decision rule ---
banner "[7/7] VERDICT (read both arms): A/B complete (base rc=$arc, on rc=$brc)"
cat <<EOF
  DECISION RULE (pre-registered) — absolute lift over the Phi value-OFF baseline:
    (i)  S1(on) soft-SPL > S1(base) -> the BLIP-2 value map cuts L_b (the memory-OFF
         agent explores SMARTER toward goal-affording views). This is the headline
         "better searcher / higher absolute" contribution. If flat -> the value
         signal still doesn't help here (run the $0 CLIP/BLIP-2 separation gate).
    (ii) warm S3-S1(on) still >> 0 AND n_memory_chosen(on) >> 0 -> the memory delta
         SURVIVES a stronger explorer (the ceiling renorm held; not crowded out).
         If S3-S1 collapses -> reframe as "memory adds on top of semantic
         exploration (delta +X)".
  Per-arm summaries:
    base S1/S3:  runs/${TAG}-base-s{1,3}/   (value OFF, geometric frontiers)
    on   S1/S3:  runs/${TAG}-on-s{1,3}/     (value ON, BLIP-2 ITM, weight=$WEIGHT, ceiling=$CEILING)
  Check n_memory_chosen in the S3 ON summaries stays >> 0 (renorm guardrail held).
  CAVEAT: ABSOLUTE soft-SPL is on the Phi planner — NON-cross-quotable to the
  +0.171/+0.24/+0.2505 7B arc. The value-OFF vs value-ON A/B is internally valid.
  preflight: ${PREFLIGHT_DIR}.log  (peak=${PEAK_MIB:-?} MiB)
EOF
exit 0
