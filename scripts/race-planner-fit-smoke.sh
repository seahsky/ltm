#!/bin/bash
# scripts/race-planner-fit-smoke.sh — CHEAP pre-flight that certifies a
# planner/GPU config is VIABLE before spending on a multi-hour matrix.
#
# WHY: a RACE run accidentally landed on a Tesla T4 (15 GB) instead of the usual
# L4 (22 GB); the ReMEmbR backbone (Qwen2-VL-2B captioner + Qwen2.5-7B planner)
# does NOT fit in 15 GB, so every episode of all 3 levers OOM'd and ~8 min were
# burned. This smoke runs ONE cold+warm pair (setting 3, real backbone) and the
# pure-python checker (check_planner_fit.py) certifies four things, GREEN/RED:
#   (a) FIT      — no CUDA OOM / crash; full completion
#   (b) NAVIGATE — warm episode cleared the ~9-step stall floor
#   (c) LTM FIRES— warm visit retrieved AND chose a memory candidate (SBERT path,
#                  planner-independent => the control that proves the LTM seam)
#   (d) PARSEABLE— planner emitted a parseable ANSWER (goto/explore > 0)
# It ALSO does an up-front GPU-capacity check that would have caught the T4
# placement in 2 s. Reusable: --planner overrides REMEMBR_PLANNER_MODEL so the
# SAME smoke validates a 3B swap / future 4-bit config before trusting it.
#
# git pull FIRST (new files: this driver + check_planner_fit.py; the driver
# self-pulls at step 1 but that only takes effect on the 2nd invocation).
#
#   nrun bash scripts/race-planner-fit-smoke.sh                              # 7B, default scene/cat
#   nrun bash scripts/race-planner-fit-smoke.sh --planner Qwen/Qwen2.5-3B-Instruct   # validate a swap
#   nrun bash scripts/race-planner-fit-smoke.sh --scene TEEsavR23oF --category bed
#
# EXECUTE it (do NOT source) — conda activates in its own process.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

SCENE="wcojb4TFT35"
CATEGORY="chair"
TAG="fit-smoke"
PLANNER=""              # optional override of REMEMBR_PLANNER_MODEL (else race-setup default 7B)
MIN_STEPS="20"         # warm episode must exceed this (the ~9-step 3B stall floor)
MIN_VRAM_MIB="20000"   # warn if total GPU VRAM is below this (T4=15 GB => 7B backbone OOMs)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --planner) PLANNER="$2"; shift 2 ;;
    --min-steps) MIN_STEPS="$2"; shift 2 ;;
    -h|--help) sed -n '1,33p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/fit_${TAG}"
NAME="fit_${TAG}"
DS="${DS_DIR}/${NAME}.json.gz"
OUT_DIR="runs/${TAG}"

banner() { printf '\n========== %s ==========\n' "$1"; }

banner "[1/6] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# Override the planner BEFORE sourcing race-setup.sh (its line uses :- so a
# pre-export wins). Lets the SAME smoke validate a swap without editing the setup.
[ -n "$PLANNER" ] && { export REMEMBR_PLANNER_MODEL="$PLANNER"; echo "  planner override: REMEMBR_PLANNER_MODEL=$PLANNER"; }

banner "[2/6] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/6] GPU capacity pre-check (the T4-placement guard)"
total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  GPU: ${gpu_name:-unknown}  total=${total_mib:-?} MiB  (planner=${REMEMBR_PLANNER_MODEL:-?})"
if [ -n "$total_mib" ] && [ "$total_mib" -lt "$MIN_VRAM_MIB" ] && [ -z "$PLANNER" ]; then
  echo "  ⚠️  WARNING: ${total_mib} MiB < ${MIN_VRAM_MIB} MiB. The fp16 Qwen2-VL-2B + Qwen2.5-7B"
  echo "      backbone needs ~20 GB and will likely OOM on this host (this is the T4 trap)."
  echo "      Re-queue on an L4 (>=24 GB), or pass --planner Qwen/Qwen2.5-3B-Instruct to fit."
  echo "      Continuing the cheap smoke so the OOM (if any) is surfaced concretely."
fi

banner "[4/6] pre-test code verify (fit checker + builder + propose)"
python embodied_memory/scripts/test_check_planner_fit.py \
  || { echo "FATAL: check_planner_fit sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_propose_candidates.py \
  || { echo "FATAL: propose_candidates sanity suite failed"; exit 1; }

banner "[5/6] build revisit smoke (1 cold + 1 warm) scene=$SCENE cat=$CATEGORY -> $DS_DIR"
rm -rf "$DS_DIR"
SRC="${VALMINI}/${SCENE}.json.gz"
[ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
python embodied_memory/scripts/make_revisit_smoke.py \
    --src "$SRC" --scene "$SCENE" --categories "$CATEGORY" --n-warm 1 \
    --out-dir "$DS_DIR" \
  || { echo "FATAL: dataset build failed (does $CATEGORY exist in $SCENE?)"; exit 1; }
[ -f "$DS" ] || { echo "FATAL: dataset not written: $DS"; exit 1; }

banner "[6/6] run cold+warm S3 (real backbone, strict) -> $OUT_DIR, then certify"
# One process so the cold sighting seeds the LTM that the warm visit retrieves.
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --setting 3 --episodes-path "$DS" \
    --scene "$SCENE" --target any --n-episodes 2 \
    --out-dir "$OUT_DIR" 2>&1 | tee "${OUT_DIR}.log"

[ -f "$OUT_DIR/summary.json" ] || { echo "FATAL: no summary.json written (the run crashed before writing)."; exit 1; }
echo
python embodied_memory/scripts/check_planner_fit.py "$OUT_DIR/summary.json" --min-steps "$MIN_STEPS"
rc=$?
echo
if [ "$rc" -eq 0 ]; then
  echo "DONE — GREEN: planner=${REMEMBR_PLANNER_MODEL:-?} on ${gpu_name:-?} is viable; safe to run the matrix."
else
  echo "DONE — RED: do NOT run the matrix yet. Fix the FAIL(s) above (OOM => bigger GPU or smaller/4-bit planner)."
fi
exit "$rc"
