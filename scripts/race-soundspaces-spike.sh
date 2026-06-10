#!/bin/bash
# scripts/race-soundspaces-spike.sh — SoundSpaces 2.0 feasibility spike (GO/NO-GO).
#
# Decision this de-risks (2026-06-10, ICRA-2027 framing): audio-goal navigation
# on OUR HM3D stack via SoundSpaces 2.0 on-the-fly RIR rendering. The spike's
# only goal is to render ONE audible room impulse response in an HM3D val_mini
# scene we already own. GREEN => the audio layer is buildable on RACE and task
# design can start. RED => the printed blocker/version-delta list IS the
# deliverable (reassess Friday; fallback = descope audio, paper stands on
# LTM + baselines + demo).
#
# EVERYTHING lives in a dedicated conda env (soundspaces-spike) + build dir —
# the working ltm-embodied env is NEVER touched (SoundSpaces is archived,
# Nov 2024, and its habitat-sim branch predates our pins).
#
#   nrun bash scripts/race-soundspaces-spike.sh          # full spike (~1h build)
#
# Idempotent: re-runs reuse the env/clone and skip the build when an
# audio-capable habitat_sim already imports.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="soundspaces-spike"
PY_VER="3.9"                       # SoundSpaces INSTALLATION.md pin
CMAKE_VER="3.14.0"                 # ditto
BUILD_ROOT="${HOME}/soundspaces-build"
SIM_DIR="${BUILD_ROOT}/habitat-sim"
LAB_DIR="${BUILD_ROOT}/habitat-lab"
SIM_BRANCH="RLRAudioPropagationUpdate"
LAB_TAG="v0.2.2"
OUT_DIR="runs/soundspaces-spike"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull (repo scripts current) ---
banner "[1/7] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda env (FRESH, isolated; never ltm-embodied) ---
banner "[2/7] conda env: $ENV_NAME (python=$PY_VER cmake=$CMAKE_VER)"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER" "cmake=$CMAKE_VER" \
    || { echo "FATAL: conda create failed"; exit 1; }
fi
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
[ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ] || { echo "FATAL: wrong env active: $CONDA_DEFAULT_ENV"; exit 1; }
python -c "import numpy" 2>/dev/null || pip install numpy

# --- 3. habitat-sim from the audio branch (skip if already audio-capable) ---
banner "[3/7] habitat-sim @ $SIM_BRANCH with --audio"
if python -c "import habitat_sim; assert hasattr(habitat_sim, 'AudioSensorSpec')" 2>/dev/null; then
  echo "  audio-capable habitat_sim already importable — skipping build"
else
  mkdir -p "$BUILD_ROOT"
  if [ ! -d "$SIM_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/habitat-sim.git "$SIM_DIR" \
      || { echo "FATAL: habitat-sim clone failed"; exit 1; }
  fi
  cd "$SIM_DIR" || exit 1
  git fetch origin "$SIM_BRANCH" \
    || { echo "FATAL: branch $SIM_BRANCH not fetchable (repo archived — mirror needed?)"; exit 1; }
  git checkout "$SIM_BRANCH" || { echo "FATAL: checkout $SIM_BRANCH failed"; exit 1; }
  git rev-parse --short HEAD
  pip install -r requirements.txt || { echo "FATAL: habitat-sim requirements failed"; exit 1; }
  export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
  # --headless: RACE has no display. --audio: builds RLRAudioPropagation.
  python setup.py install --headless --audio 2>&1 | tail -40
  rc=${PIPESTATUS[0]}
  cd "$REPO_ROOT" || exit 1
  [ "$rc" = "0" ] || { echo "FATAL: habitat-sim audio build failed (exit $rc) — full log above; this version delta is the spike deliverable"; exit 1; }
fi

# --- 4. import + audio-API verify ---
banner "[4/7] verify import + AudioSensorSpec"
python - <<'EOF' || { echo "FATAL: audio API verify failed"; exit 1; }
import habitat_sim
assert hasattr(habitat_sim, "AudioSensorSpec"), "no AudioSensorSpec — built without --audio?"
print(f"  habitat_sim {getattr(habitat_sim, '__version__', '?')} OK (AudioSensorSpec present)")
EOF

# --- 5. habitat-lab v0.2.2 (best-effort; NOT needed for the RIR smoke) ---
# Installs the task-layer pin so the spike also reports whether it goes in
# cleanly. A failure here is a WARN, not a blocker — task design needs it,
# the RIR render does not.
banner "[5/7] habitat-lab $LAB_TAG (best-effort)"
if python -c "import habitat" 2>/dev/null; then
  echo "  habitat-lab already importable — skipping"
else
  if [ ! -d "$LAB_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/habitat-lab.git "$LAB_DIR" || true
  fi
  if [ -d "$LAB_DIR/.git" ]; then
    ( cd "$LAB_DIR" && git fetch origin tag "$LAB_TAG" 2>/dev/null; git checkout "$LAB_TAG" \
        && pip install -e . 2>&1 | tail -5 ) \
      || echo "WARN: habitat-lab $LAB_TAG install failed — record as version delta (RIR smoke unaffected)"
  else
    echo "WARN: habitat-lab clone failed — record as version delta (RIR smoke unaffected)"
  fi
fi

# --- 6. find an HM3D val_mini scene asset we own ---
banner "[6/7] locate HM3D scene .glb"
SCENE_GLB="$(find data/hm3d -name '*.basis.glb' 2>/dev/null | grep -E 'wcojb4TFT35|TEEsavR23oF' | head -1)"
[ -n "$SCENE_GLB" ] || SCENE_GLB="$(find data/hm3d -name '*.basis.glb' 2>/dev/null | head -1)"
[ -n "$SCENE_GLB" ] || SCENE_GLB="$(find data/hm3d -name '*.glb' 2>/dev/null | grep -v semantic | head -1)"
[ -n "$SCENE_GLB" ] || { echo "FATAL: no HM3D .glb found under data/hm3d"; exit 1; }
echo "  scene: $SCENE_GLB"

# --- 7. render ONE RIR (materials OFF — HM3D semantics are absent/broken) ---
banner "[7/7] RIR smoke"
mkdir -p "$OUT_DIR"
python embodied_memory/scripts/soundspaces_rir_smoke.py \
    --scene "$SCENE_GLB" --out "$OUT_DIR/rir.npy" 2>&1 | tee "$OUT_DIR/smoke.log"
rc=${PIPESTATUS[0]}

if [ "$rc" = "0" ]; then
  banner "SPIKE GREEN — SoundSpaces 2.0 renders RIRs on our HM3D stack"
  echo "  IR saved: $OUT_DIR/rir.npy — audio-goal task design can start."
else
  banner "SPIKE RED (exit $rc) — blockers above are the deliverable"
  echo "  Paste $OUT_DIR/smoke.log + the build tail; reassess Friday"
  echo "  (fallback: descope audio — paper stands on LTM + baselines + demo)."
fi
exit "$rc"
