#!/bin/bash
# scripts/race-audiogoal-m0.sh — M0 GATE for the AudioGoal/FSD50K task.
#
# Renders the binaural RIR grids for the two SoundSpaces-GREEN HM3D scenes in
# the dedicated `soundspaces-spike` env, then verifies — in the working
# `ltm-embodied` env — that the rendered audio is (a) HEARD (RMS rises toward
# the source) + LOCALIZABLE (ILD/ITD DOA within ~30°) and (b) that the
# anomaly→LTM→waypoint retrieval discriminates the right instance.
#
# Three GREEN gates here unblock M1 (single end-to-end audio episode). This
# driver only RENDERS + VERIFIES; it assumes both envs already exist:
#   * soundspaces-spike  — built by scripts/race-soundspaces-spike.sh (GREEN)
#   * ltm-embodied       — scripts/race-setup.sh
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull \
#       && nrun bash scripts/race-audiogoal-m0.sh
#
# (RACE checks out lifelong-revisit-eval; pull first so this brand-new driver
# and the audio modules are present — see the RACE branch/self-update gotchas.)
# Idempotent: re-runs re-render the grids and re-verify.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MINICONDA="${HOME}/miniconda3"
SS_ENV="soundspaces-spike"
OUT_DIR="runs/audiogoal"
SCENES="${AUDIOGOAL_SCENES:-wcojb4TFT35 TEEsavR23oF}"
N_CELLS="${AUDIOGOAL_N_CELLS:-24}"
MIN_CELLS="${AUDIOGOAL_MIN_CELLS:-8}"
mkdir -p "$OUT_DIR"

banner() { printf '\n========== %s ==========\n' "$1"; }
fail=0
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }

# --- 1/3. render RIR grids in the soundspaces-spike env -------------------
banner "[1/3] render RIR grids ($SS_ENV)"
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
set +u
conda activate "$SS_ENV" || {
  echo "FATAL: conda activate $SS_ENV failed — build it first: bash scripts/race-soundspaces-spike.sh"
  exit 1
}
set -u
for S in $SCENES; do
  GLB="$(find data/hm3d -name "${S}.basis.glb" 2>/dev/null | head -1)"
  [ -n "$GLB" ] || GLB="$(find data/hm3d -name "*${S}*.glb" 2>/dev/null | grep -v semantic | head -1)"
  if [ -z "$GLB" ]; then
    echo "RED: no .glb for scene $S under data/hm3d"; fail=1; continue
  fi
  echo "  scene $S -> $GLB"
  python embodied_memory/scripts/render_rir_grid.py \
      --scene "$GLB" --out "$OUT_DIR/${S}_rir_grid.npz" \
      --n-cells "$N_CELLS" --min-cells "$MIN_CELLS" \
      2>&1 | tee "$OUT_DIR/${S}_render.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "RED: render_rir_grid.py failed for $S (rc=$rc)"; fail=1; }
done
set +u; conda deactivate 2>/dev/null || true; set -u

# --- 2/3. heard + localizable verification (ltm-embodied env) -------------
banner "[2/3] activate ltm-embodied"
set +u
source scripts/race-setup.sh
rc=$?
set -u
[ "$rc" -eq 0 ] || { echo "FATAL: race-setup.sh failed (rc=$rc)"; exit 1; }
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

for S in $SCENES; do
  GRID="$OUT_DIR/${S}_rir_grid.npz"
  if [ ! -f "$GRID" ]; then
    echo "RED: missing grid $GRID (render step failed) — skipping loop smoke"; fail=1; continue
  fi
  banner "audio_loop_smoke: $S"
  python embodied_memory/scripts/audio_loop_smoke.py --rir-grid "$GRID" \
      2>&1 | tee "$OUT_DIR/${S}_loop.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "RED: audio_loop_smoke failed for $S"; fail=1; }
done

# --- 3/3. retrieval discrimination (ltm-embodied env) ---------------------
banner "[3/3] audiogoal_retrieval_smoke"
python embodied_memory/scripts/audiogoal_retrieval_smoke.py \
    2>&1 | tee "$OUT_DIR/retrieval.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "RED: audiogoal_retrieval_smoke failed"; fail=1; }

banner "M0 GATE"
if [ "$fail" -eq 0 ]; then
  echo "GREEN: M0 PASS — RIR grids render; audio is heard + localizable; "
  echo "       retrieval discriminates the right instance. M1 is unblocked."
else
  echo "RED: M0 has failures above — fix before starting M1."
fi
exit "$fail"
