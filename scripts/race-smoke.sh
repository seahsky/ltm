#!/bin/bash
# scripts/race-smoke.sh — one-shot RACE smoke driver.
#
# Runs the whole RACE test cycle as a SINGLE command and prints clearly-banner'd
# sections so the entire output can be pasted back in one block:
#
#   [1/5] git pull --ff-only
#   [2/5] conda setup            (source scripts/race-setup.sh)
#   [3/5] pre-test code verify   (faiss/habitat-free sanity suite)
#   [4/5] run                    (embodied_memory.run_hm3d_pol, per scene)
#   [5/5] post-test verify       (embodied_memory.scripts.verify_smoke_gate)
#
# EXECUTE it (do NOT source) — it activates conda in its own process and runs
# everything there:
#
#   bash scripts/race-smoke.sh --backbone oracle --setting 1 \
#       --scenes "TEEsavR23oF wcojb4TFT35" --n-episodes 2 --target any \
#       --no-strict-pass --tag oracle-smoke
#
#   bash scripts/race-smoke.sh --backbone remembr --setting 3 \
#       --scenes wcojb4TFT35 --n-episodes 2 --target any --tag remembr-dense
#
# Env vars pass straight through to the run, e.g. disable the false STOP:
#   REMEMBR_STOP_MIN_STEP=9999 bash scripts/race-smoke.sh --backbone remembr \
#       --setting 3 --scenes wcojb4TFT35 --n-episodes 2 --target any --tag dense-nostop
#
# Aborts early (before the paid run) if git pull, conda setup, or the pre-test
# sanity suite fail. A failing post-test gate is NOT an abort — it's a valid
# result that still gets printed and continues to the next scene.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults ---
BACKBONE=""
SETTING=""
SCENES=""
N_EPISODES="2"
TARGET="any"
TAG=""
MAX_STEPS=""
NO_STRICT_PASS=""

# --- arg parse ---
while [ $# -gt 0 ]; do
  case "$1" in
    --backbone)        BACKBONE="$2"; shift 2 ;;
    --setting)         SETTING="$2"; shift 2 ;;
    --scenes|--scene)  SCENES="$2"; shift 2 ;;
    --n-episodes)      N_EPISODES="$2"; shift 2 ;;
    --target)          TARGET="$2"; shift 2 ;;
    --tag)             TAG="$2"; shift 2 ;;
    --max-steps)       MAX_STEPS="$2"; shift 2 ;;
    --no-strict-pass)  NO_STRICT_PASS="--no-strict-pass"; shift ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done

if [ -z "$BACKBONE" ] || [ -z "$SETTING" ] || [ -z "$SCENES" ]; then
  echo "FATAL: --backbone, --setting and --scenes are required."
  echo "  e.g. bash scripts/race-smoke.sh --backbone oracle --setting 1 \\"
  echo "         --scenes \"TEEsavR23oF wcojb4TFT35\" --n-episodes 2 --target any \\"
  echo "         --no-strict-pass --tag oracle-smoke"
  exit 1
fi
[ -z "$TAG" ] && TAG="$BACKBONE"
SCENES="${SCENES//,/ }"   # accept comma- or space-separated

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/5] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup (sourced so the env persists in THIS process) ---
banner "[2/5] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify (free; aborts before any paid run if broken) ---
banner "[3/5] pre-test code verify (sanity suite)"
python embodied_memory/scripts/test_propose_candidates.py \
  || { echo "FATAL: sanity suite failed — not spending on the live run."; exit 1; }

# --- 4. run + 5. verify, per scene ---
MAXSTEPS_ARG=""
[ -n "$MAX_STEPS" ] && MAXSTEPS_ARG="--max-steps $MAX_STEPS"
OUT_DIRS=""
for sc in $SCENES; do
  out_dir="runs/${TAG}-${sc}"
  banner "[4/5] run: backbone=$BACKBONE setting=$SETTING scene=$sc -> $out_dir"
  # shellcheck disable=SC2086
  python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone "$BACKBONE" --setting "$SETTING" --scene "$sc" \
    --n-episodes "$N_EPISODES" --target "$TARGET" $NO_STRICT_PASS $MAXSTEPS_ARG \
    --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

for out_dir in $OUT_DIRS; do
  banner "[5/5] post-test verify: $out_dir"
  python embodied_memory/scripts/verify_smoke_gate.py "$out_dir"
done

banner "DONE — paste everything above"
