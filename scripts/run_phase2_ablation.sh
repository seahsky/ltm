#!/bin/bash
# Full Phase-2 ablation: 3 settings x 30 episodes x 250-step cap, --backbone remembr.
# Designed for any CUDA host. On compute-only containers (no NVIDIA EGL),
# set HABITAT_SIM_GPU_DEVICE_ID=-1 + EGL_PLATFORM=surfaceless and ensure
# Mesa software EGL is installed.

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltm-embodied
cd "$(dirname "$0")/.."

LOG="${PHASE2_LOG:-/tmp/phase2.log}"
OUT_PREFIX="${PHASE2_OUT_PREFIX:-runs/abl-s}"
SUFFIX="${PHASE2_OUT_SUFFIX:--remembr}"
N_EPISODES="${PHASE2_N_EPISODES:-30}"
MAX_STEPS="${PHASE2_MAX_STEPS:-250}"

: > "$LOG"
rm -rf "${OUT_PREFIX}1${SUFFIX}" "${OUT_PREFIX}2${SUFFIX}" "${OUT_PREFIX}3${SUFFIX}"

for s in 1 2 3; do
  OUT="${OUT_PREFIX}${s}${SUFFIX}"
  echo "=== SETTING ${s} starting at $(date) -> ${OUT} ===" | tee -a "$LOG"
  python -u -m embodied_memory.run_hm3d_pol \
      --mode live --backbone remembr --scene all \
      --setting "$s" --n-episodes "$N_EPISODES" \
      --target any --max-steps "$MAX_STEPS" \
      --out-dir "$OUT" 2>&1 | tee -a "$LOG"
  echo "=== SETTING ${s} done at $(date), exit=$? ===" | tee -a "$LOG"
done

echo "=== ALL SETTINGS DONE ===" | tee -a "$LOG"
python -u embodied_memory/scripts/analyze_ablation.py \
    "${OUT_PREFIX}1${SUFFIX}" "${OUT_PREFIX}2${SUFFIX}" "${OUT_PREFIX}3${SUFFIX}" 2>&1 | tee -a "$LOG"
echo "=== ANALYZER DONE ===" | tee -a "$LOG"
