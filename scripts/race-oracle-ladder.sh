#!/bin/bash
# scripts/race-oracle-ladder.sh — RACE driver for the bottleneck-isolation
# oracle ladder on the Phase-C revisit dataset. Five cells, all backbone=remembr:
#
#   1. nomem        --setting 1                              (No memory baseline)
#   2. ours         --setting 3                              (Our memory)
#   3. oracle-loc   --setting 3 --oracle-location            (perfect target; own STOP)
#   4. oracle-stop  --setting 3 --oracle-stop                (own nav; perfect STOP)
#   5. oracle-both  --setting 3 --oracle-location --oracle-stop  (locomotion ceiling)
#
# Reading (success@0.1m, warm): if oracle-stop >> ours, the bottleneck is
# TERMINATION; if oracle-loc is also needed, exploration/retrieval is implicated;
# oracle-both is the locomotion upper bound. Diagnosis (diagnose_pipeline.py over
# the c9 logs) already showed several warm eps reach min_d2g=0.00 but fail to
# STOP there -> oracle-stop is the rung expected to move.
#
# EXECUTE it (do NOT source) — conda is activated in its own process:
#
#   bash scripts/race-oracle-ladder.sh --tag oracle-1
#
# Mirrors race-revisit-detector.sh (pull -> setup -> sanity -> build -> preflight
# -> 5 cells -> diagnose). Each cell is a SEPARATE process/out-dir (LTM persists
# within a process; mixing settings would corrupt it).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair bed"
NWARM="3"
TAG=""
N_EPISODES=""
TARGET="any"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --categories) CATS="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"
SCENES="${SCENES//,/ }"
[ -z "$TAG" ] && { echo "FATAL: --tag <name> required"; exit 1; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_${TAG}"
NAME="revisit_${TAG}"
DS="${DS_DIR}/${NAME}.json.gz"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/7] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup ---
banner "[2/7] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify ---
banner "[3/7] pre-test code verify (runner + oracle helpers + diagnose_pipeline)"
python embodied_memory/scripts/test_episode_runner_detector.py \
  || { echo "FATAL: episode_runner_detector sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_diagnose_pipeline.py \
  || { echo "FATAL: diagnose_pipeline sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed"; exit 1; }

# --- 4. build revisit dataset (same as race-revisit.sh) ---
banner "[4/7] build revisit dataset: scenes=[$SCENES] cats=[$CATS] n-warm=$NWARM -> $DS_DIR"
rm -rf "$DS_DIR"
for SCENE in $SCENES; do
  SRC="${VALMINI}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_revisit_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories $CATS --n-warm "$NWARM" \
      --out-dir "$DS_DIR" \
    || { echo "FATAL: dataset build failed for $SCENE"; exit 1; }
done
[ -f "$DS" ] || { echo "FATAL: top-level dataset not written: $DS"; exit 1; }
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes"; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass over all built scenes)"
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: n-episodes <=0"; exit 1; }

# --- 5. pre-flight: oracle-both on 1 ep (GO/NO-GO: flags don't crash; STOP fires) ---
banner "[5/7] pre-flight: setting=3 --oracle-location --oracle-stop  scene=wcojb4TFT35  n=1"
PREFLIGHT_DIR="runs/${TAG}-preflight"
# --no-strict-pass: oracle-location beelines to the GT goal so memory is never
# CHOSEN -> the 'memory_influences' pass-condition is FALSE by design and would
# exit non-zero. The ladder readout comes from diagnose_pipeline, not the gate,
# so disable strict pass and treat a real exception (non-zero) as the only abort.
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --setting 3 --oracle-location --oracle-stop --no-strict-pass \
    --episodes-path "$DS" --scene wcojb4TFT35 --target chair --n-episodes 1 \
    --out-dir "$PREFLIGHT_DIR" 2>&1 | tee "${PREFLIGHT_DIR}.log"
rc=${PIPESTATUS[0]}
[ "$rc" = "0" ] || { echo "FATAL: pre-flight crashed (exit $rc) — fix before the matrix."; exit 1; }
n_succ="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('n_successful_episodes', 0))" "${PREFLIGHT_DIR}/summary.json" 2>/dev/null || echo 0)"
echo "preflight: n_successful_episodes=$n_succ (oracle-both should usually reach the goal)"

# --- 6. run 5 ladder cells in SEPARATE processes ---
declare -a CELLS=(
  "nomem|--setting 1"
  "ours|--setting 3"
  "oracle-loc|--setting 3 --oracle-location"
  "oracle-stop|--setting 3 --oracle-stop"
  "oracle-both|--setting 3 --oracle-location --oracle-stop"
)
OUT_DIRS=""
for CELL in "${CELLS[@]}"; do
  NAME_C="${CELL%%|*}"
  ARGS_C="${CELL#*|}"
  out_dir="runs/${TAG}-${NAME_C}"
  banner "[6/7] run: ${NAME_C} (${ARGS_C}) -> $out_dir"
  # shellcheck disable=SC2086
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr $ARGS_C --no-strict-pass --episodes-path "$DS" \
      --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
      --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  rc=${PIPESTATUS[0]}
  completed="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${out_dir}/summary.json" 2>/dev/null || echo 0)"
  [ "$completed" = "$N_EPISODES" ] || echo "WARN: ${NAME_C} completed ${completed}/${N_EPISODES} (exit $rc)."
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- 7. ladder readout: warm success@1m / success@0.1m per cell ---
banner "[7/7] Oracle ladder — per-cell observation/retrieval + warm success rates"
# shellcheck disable=SC2086
python embodied_memory/scripts/diagnose_pipeline.py $OUT_DIRS

banner "DONE — read the warm 'succ@0.1m' line per cell:
  nomem -> ours       : memory's contribution
  ours  -> oracle-stop : how much TERMINATION (STOP-timing) is costing
  ours  -> oracle-loc  : how much EXPLORATION+RETRIEVAL is costing
  oracle-both          : locomotion upper bound
Paste the five per-cell blocks above."
