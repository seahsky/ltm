#!/bin/bash
# scripts/race-benchmark-success.sh — recompute the TRUE HM3D benchmark
# success rate (STOP within 1.0 m) from existing episode logs. No GPU, no
# Habitat — pure log mining via diagnose_pipeline.py --benchmark.
#
# Context (Run 15): neither previously-reported number is the benchmark
# metric — "8%" used the 0.1 m radius (10x too strict) and "67%"
# (success_1m) is STOP-independent reach. Since STOP terminates the episode,
# final distance_to_goal IS distance-at-STOP, so the true number is
# recoverable from logs already on RACE.
#
#   [1/3] git pull --ff-only
#   [2/3] sanity tests (test_diagnose_benchmark.py; abort on fail)
#   [3/3] diagnose_pipeline.py --benchmark over the run dirs
#
# EXECUTE it (do NOT source):
#
#   bash scripts/race-benchmark-success.sh                 # default run dirs
#   bash scripts/race-benchmark-success.sh runs/foo runs/bar   # override

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# Default = the Run-13/14 wide-matrix runs + any revisit-tagged runs found
# on disk (these dirs exist on RACE only). Override by passing run dirs as
# args.
RUN_DIRS=("$@")
if [ ${#RUN_DIRS[@]} -eq 0 ]; then
  RUN_DIRS=(
    runs/scorer-d3-s1
    runs/scorer-d3-s3-heur
    runs/scorer-d3-s3-trained
  )
  for d in runs/*revisit*-s[123]; do
    [ -d "$d" ] && RUN_DIRS+=("$d")
  done
fi

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/3] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. sanity tests (habitat/torch-free; abort on fail) ---
banner "[2/3] sanity: test_diagnose_benchmark.py"
python embodied_memory/scripts/test_diagnose_benchmark.py \
  || { echo "FATAL: benchmark-success sanity suite failed."; exit 1; }

# --- 3. recompute benchmark success over existing logs ---
EXISTING=()
for rd in "${RUN_DIRS[@]}"; do
  if [ -d "$rd" ]; then EXISTING+=("$rd"); else echo "SKIP (missing): $rd"; fi
done
if [ ${#EXISTING[@]} -eq 0 ]; then
  echo "FATAL: none of the run dirs exist here. Pass run dirs as args."
  exit 1
fi
banner "[3/3] benchmark success recompute (${#EXISTING[@]} run dirs)"
python embodied_memory/scripts/diagnose_pipeline.py --benchmark "${EXISTING[@]}"

# Spot-check the at-STOP assumption on one stopped episode: STOP terminates
# the episode, so final distance_to_goal should be distance-at-STOP.
banner "spot-check: one action_stop>0 episode (d2g is at-STOP)"
python - "${EXISTING[@]}" <<'EOF'
import glob, json, os, sys
for rd in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(rd, "episode_*.json"))):
        if p.endswith("_error.json"):
            continue
        ep = json.load(open(p))
        if ep.get("action_stop"):
            print(f"{p}: action_stop={ep['action_stop']} n_steps={ep['n_steps']} "
                  f"distance_to_goal={ep.get('distance_to_goal'):.3f} "
                  f"min_d2g={ep.get('min_distance_to_goal')}")
            sys.exit(0)
print("no action_stop>0 episode found in given dirs")
EOF

banner "DONE — paste everything above"
