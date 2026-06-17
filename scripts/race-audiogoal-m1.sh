#!/bin/bash
# scripts/race-audiogoal-m1.sh — M1 WIRING SMOKE for the AudioGoal task.
#
# Proves the audio observation flows end to end through the LIVE loop: a cached
# RIR grid + an anomaly clip render into Step.audio, audio_task detects onset →
# CLAP-classifies → overrides the retrieval target, and the run completes with
# the objectnav path unaffected. This is the M1 gate BEFORE M2 (the dataset
# builder that seeds warm episodes); it is a WIRING check, not a metrics claim
# (the warm S1-vs-S3 soft-SPL measurement comes after M2 seeds warm sightings).
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull \
#       && bash scripts/race-audiogoal-m1.sh 2>&1 | tee runs/audiogoal-m1/m1.log
#
# Prereqs: soundspaces-spike env (RIR grid render) + ltm-embodied env. Reuses the
# M0 grid runs/audiogoal/<scene>_rir_grid.npz if present, else renders it.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
MINICONDA="${HOME}/miniconda3"
SS_ENV="soundspaces-spike"
SCENE="${AUDIOGOAL_M1_SCENE:-TEEsavR23oF}"
GRID="runs/audiogoal/${SCENE}_rir_grid.npz"
OUT_DIR="runs/audiogoal-m1"
T_ANOM="${AUDIOGOAL_T_ANOM:-30}"
ANOM_CLASS="${AUDIOGOAL_CLASS:-baby_cry}"
CLIP_ARG=""
[ -n "${AUDIOGOAL_CLIP:-}" ] && CLIP_ARG="--anomaly-clip ${AUDIOGOAL_CLIP}"
mkdir -p "$OUT_DIR"
banner() { printf '\n========== %s ==========\n' "$1"; }
fail=0
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }

# --- 1/4. ensure the RIR grid (soundspaces-spike env) ---------------------
if [ ! -f "$GRID" ]; then
  banner "[1/4] render RIR grid for $SCENE ($SS_ENV)"
  eval "$("$MINICONDA/bin/conda" shell.bash hook)"
  set +u; conda activate "$SS_ENV" || { echo "FATAL: activate $SS_ENV failed"; exit 1; }; set -u
  GLB="$(find data/hm3d -name "${SCENE}.basis.glb" 2>/dev/null | head -1)"
  [ -n "$GLB" ] || GLB="$(find data/hm3d -name "*${SCENE}*.glb" 2>/dev/null | grep -v semantic | head -1)"
  [ -n "$GLB" ] || { echo "RED: no .glb for $SCENE"; exit 1; }
  python embodied_memory/scripts/render_rir_grid.py --scene "$GLB" --out "$GRID" --n-cells 24 \
      2>&1 | tee "$OUT_DIR/${SCENE}_render.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "RED: render failed"; exit 1; }
  set +u; conda deactivate 2>/dev/null || true; set -u
else
  banner "[1/4] reusing existing grid $GRID"
fi

# --- 2/4. activate ltm-embodied ------------------------------------------
banner "[2/4] activate ltm-embodied"
set +u; source scripts/race-setup.sh; rc=$?; set -u
[ "$rc" -eq 0 ] || { echo "FATAL: race-setup.sh failed (rc=$rc)"; exit 1; }
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# --- 3/4. audiogoal wiring episode (frontier backbone = fast, no LLM) ------
# CLAP still loads (classify is the point); frontier keeps the loop cheap so the
# wiring proof is minutes, not a full 7B run. The remembr gate run is step 4.
banner "[3/4] audiogoal wiring episode (frontier)"
python -m embodied_memory.run_hm3d_pol \
    --mode live --scene "$SCENE" --backbone frontier \
    --task audiogoal --rir-grid "$GRID" --anomaly-class "$ANOM_CLASS" \
    --t-anom "$T_ANOM" $CLIP_ARG \
    --n-episodes 1 --max-steps 120 --target any --no-strict-pass \
    --out-dir "$OUT_DIR/frontier" 2>&1 | tee "$OUT_DIR/frontier.log"
rc=${PIPESTATUS[0]}

banner "wiring assertions"
python embodied_memory/scripts/verify_audiogoal_wiring.py \
    --log "$OUT_DIR/frontier.log" --run "$OUT_DIR/frontier" \
    --t-anom "$T_ANOM" --expect-object crib 2>&1 | tee "$OUT_DIR/verify.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "RED: wiring assertions failed"; fail=1; }

# --- 4/4. objectnav regression (same scene, no audio) ---------------------
banner "[4/4] objectnav regression (audio path must be inert)"
python -m embodied_memory.run_hm3d_pol \
    --mode live --scene "$SCENE" --backbone frontier \
    --n-episodes 1 --max-steps 60 --target any --no-strict-pass \
    --out-dir "$OUT_DIR/objectnav" 2>&1 | tee "$OUT_DIR/objectnav.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "RED: objectnav regression run failed"; fail=1; }
if grep -q "\[audio\]" "$OUT_DIR/objectnav.log"; then
  echo "RED: objectnav run emitted [audio] output — audio path not gated off"; fail=1
fi

banner "M1 WIRING GATE"
if [ "$fail" -eq 0 ]; then
  echo "GREEN: M1 wiring PASS — audio renders into the live loop, onset+classify"
  echo "       override the retrieval target, objectnav path inert. Next: real"
  echo "       --backbone remembr run, then M2 (warm-episode dataset builder)."
  echo
  echo "  Real backbone gate (after this is green):"
  echo "    python -m embodied_memory.run_hm3d_pol --mode live --scene $SCENE \\"
  echo "      --backbone remembr --task audiogoal --rir-grid $GRID \\"
  echo "      --anomaly-class $ANOM_CLASS --t-anom $T_ANOM --n-episodes 1 \\"
  echo "      --target any --out-dir $OUT_DIR/remembr"
else
  echo "RED: M1 wiring has failures above."
fi
exit "$fail"
