#!/bin/bash
# scripts/race-wide-s2.sh — S2 (STM-only) on the Run-14 wide revisit matrix.
#
# Phase C showed S2-S1 = exactly 0.000 on the 2x{chair,bed} matrix (STM alone
# does nothing; the whole effect is S3-S2). Run 14 widened the matrix to
# 6 categories x 2 scenes but only ran S1/S3 — this fills in S2 so the
# decomposition holds at the wider n (expect S2-S1 ~= 0 at n=26 warm).
#
# REUSES the scorer-d3 dataset already on RACE — deliberately NO rebuild
# (race-revisit.sh would rm -rf the dataset dir and resample it; the paired
# analysis needs the EXACT episodes S1/S3 ran). EXECUTE it (do NOT source):
#
#   bash scripts/race-wide-s2.sh
#   bash scripts/race-wide-s2.sh --dataset data/.../revisit_scorer-d3/revisit_scorer-d3.json.gz \
#       --s1 runs/scorer-d3-s1 --s3 runs/scorer-d3-s3-heur --out runs/wide-s2

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults (the Run-14 wide-matrix artifacts on RACE) ---
DS="data/hm3d/datasets/objectnav/hm3d/v1/revisit_scorer-d3/revisit_scorer-d3.json.gz"
S1_DIR="runs/scorer-d3-s1"
S3_DIR="runs/scorer-d3-s3-heur"
OUT_DIR="runs/wide-s2"
N_EPISODES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)    DS="$2"; shift 2 ;;
    --s1)         S1_DIR="$2"; shift 2 ;;
    --s3)         S3_DIR="$2"; shift 2 ;;
    --out)        OUT_DIR="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/5] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup ---
banner "[2/5] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-flight: dataset + S1/S3 runs must already exist (NO rebuild) ---
banner "[3/5] pre-flight (existing dataset + S1/S3 runs; analyzer sanity)"
[ -f "$DS" ] || { echo "FATAL: dataset not found: $DS (this script never rebuilds it)"; exit 1; }
[ -d "$S1_DIR" ] || { echo "FATAL: S1 run missing: $S1_DIR"; exit 1; }
[ -d "$S3_DIR" ] || { echo "FATAL: S3 run missing: $S3_DIR"; exit 1; }
python embodied_memory/scripts/test_analyze_revisit.py \
  || { echo "FATAL: analyze_revisit sanity suite failed — not spending."; exit 1; }
if [ -z "$N_EPISODES" ]; then
  CONTENT_GLOB="$(dirname "$DS")/content/*.json.gz"
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "$CONTENT_GLOB")" \
    || { echo "FATAL: could not count dataset episodes."; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass, matching S1/S3)"
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: episode count '$N_EPISODES' <=0."; exit 1; }

# --- 4. run S2 (STM-only) on the SAME dataset ---
banner "[4/5] run: setting=2 backbone=remembr -> $OUT_DIR"
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --setting 2 --episodes-path "$DS" \
    --scene all --target any --n-episodes "$N_EPISODES" \
    --out-dir "$OUT_DIR" 2>&1 | tee "${OUT_DIR}.log"
rc=${PIPESTATUS[0]}
completed="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${OUT_DIR}/summary.json" 2>/dev/null || echo 0)"
if [ "$completed" != "$N_EPISODES" ]; then
  echo "WARN: S2 completed ${completed}/${N_EPISODES} episodes (exit $rc) — decomposition may be partial."
fi

# --- 5. 3-setting revisit analysis (S2 decomposition on the wide matrix) ---
banner "[5/5] analysis: analyze_ablation.py --revisit $S1_DIR $OUT_DIR $S3_DIR"
python embodied_memory/scripts/analyze_ablation.py --revisit "$S1_DIR" "$OUT_DIR" "$S3_DIR"

banner "DONE — paste everything above (esp. WARM S2 - S1 and S3 - S2)"
