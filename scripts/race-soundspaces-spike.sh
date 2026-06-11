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
#
# Setup-audit notes (2026-06-10, verified against SoundSpaces INSTALLATION.md +
# habitat-sim@RLRAudioPropagationUpdate sources):
#   * setup.py auto-runs `git submodule update --init --recursive` (we also run
#     it explicitly for clean failure attribution — the prebuilt closed-source
#     libRLRAudioPropagation.so ships as the rlr-audio-propagation submodule,
#     Linux x64 only, needs GLIBC >= 2.29).
#   * numpy MUST stay <1.24 (2022-era tree; numpy 2.x breaks it).
#   * `import quaternion` before habitat_sim (issue #1813 official workaround).
#   * Audio-capability probes use RLRAudioPropagationChannelLayoutType.Binaural
#     — AudioSensorSpec is bound even in non-audio builds (issue #2340).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="soundspaces-spike"
PY_VER="3.9"                       # SoundSpaces INSTALLATION.md pin
CMAKE_VER="3.14.0"                 # ditto (also dodges CMake-4.x refusing the 2022 dep tree)
BUILD_ROOT="${HOME}/soundspaces-build"
SIM_DIR="${BUILD_ROOT}/habitat-sim"
LAB_DIR="${BUILD_ROOT}/habitat-lab"
SIM_BRANCH="RLRAudioPropagationUpdate"
LAB_TAG="v0.2.2"
OUT_DIR="runs/soundspaces-spike"
APT_LINE="sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev libglvnd-dev"

banner() { printf '\n========== %s ==========\n' "$1"; }

# Audio-capable import probe (shared by the build-skip check and the verify).
audio_probe() {
  python -c "
import quaternion  # noqa: F401  must precede habitat_sim (issue #1813)
import habitat_sim, habitat_sim.sensor
t = getattr(habitat_sim.sensor, 'RLRAudioPropagationChannelLayoutType', None)
assert t is not None and hasattr(t, 'Binaural'), 'habitat_sim built WITHOUT --audio'
print(f'  habitat_sim {getattr(habitat_sim, \"__version__\", \"?\")} audio-capable OK')
" 2>&1
}

# --- 1. git pull (repo scripts current) ---
banner "[1/8] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
mkdir -p "$OUT_DIR"

# --- 2. conda env (FRESH, isolated; never ltm-embodied) ---
banner "[2/8] conda env: $ENV_NAME (python=$PY_VER cmake=$CMAKE_VER, numpy<1.24, gcc-10 toolchain)"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
# conda's compiler-package hooks dereference CONDA_BACKUP_* vars that are
# unset on first install (observed on RACE 2026-06-11: `conda install
# gcc_linux-64` triggers a reactivate that sources the brand-new
# deactivate-gxx_linux-64.sh -> "CONDA_BACKUP_CXX: unbound variable", which
# under `set -u` kills the whole script). Run every conda state change with
# nounset OFF.
set +u
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
# Existence check via the env dir, NOT `conda env list | grep -q`: under
# pipefail, grep -q exiting early can SIGPIPE the conda writer and turn a
# found-it result into a pipeline failure (same class as the GLIBC bug below).
if [ ! -d "$MINICONDA/envs/$ENV_NAME" ]; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER" "cmake=$CMAKE_VER" \
    || { echo "FATAL: conda create failed"; exit 1; }
fi
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
[ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] || { echo "FATAL: wrong env active: ${CONDA_DEFAULT_ENV:-<none>}"; exit 1; }
set -u
# numpy 2.x breaks the 2022-era tree (and np.float removal breaks habitat-lab
# v0.2.2) — pin BEFORE anything else can resolve numpy. quaternion is the
# import-order workaround dep; it must exist even on build-skip re-runs.
pip install "numpy>=1.16.1,<1.24" numpy-quaternion \
  || { echo "FATAL: numpy/numpy-quaternion pin install failed"; exit 1; }
# The only attested toolchain for this 2022-era tree is gcc 7-10 (docs build
# with 7.4.0; modern gcc 12/13 predates-era magnum/corrade headers). Prefer a
# conda gcc-10; fall back to system gcc with a loud version print.
if [ ! -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
  set +u   # the install reactivates the env -> sources compiler hooks (see above)
  conda install -y -c conda-forge 'gcc_linux-64=10.*' 'gxx_linux-64=10.*' sysroot_linux-64 \
    || echo "WARN: conda gcc-10 install failed — will build with system gcc $(gcc -dumpversion 2>/dev/null || echo '?') (era-tested is 7-10)"
  set -u
fi
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
  echo "  toolchain: conda gcc $("$CC" -dumpversion)"
else
  echo "  toolchain: system gcc $(gcc -dumpversion 2>/dev/null || echo '?')"
fi

# --- 3. system preflight (fail in seconds, not 40 min into the build) ---
banner "[3/8] system preflight: GLIBC >= 2.29 + GL/EGL dev libs"
# Pipe-safe extraction (cost the 2nd spike run, 2026-06-11): `ldd | head -1 |
# grep` lets head/grep exit early -> SIGPIPEs ldd -> under pipefail the
# pipeline "fails" despite a good match, so an `|| echo 0.0` fallback APPENDS
# a bogus 0.0 line and sort -V flunks a healthy GLIBC 2.35. getconf + awk
# consume their whole input, so nothing SIGPIPEs.
GLIBC_VER="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
[ -n "$GLIBC_VER" ] || GLIBC_VER="$(ldd --version 2>/dev/null | awk 'NR==1{print $NF}')"
[ -n "$GLIBC_VER" ] || GLIBC_VER="0.0"
if [ "$(printf '2.29\n%s\n' "$GLIBC_VER" | sort -V | head -1)" != "2.29" ]; then
  echo "FATAL: GLIBC $GLIBC_VER < 2.29 — the prebuilt libRLRAudioPropagation.so"
  echo "  will not link (undefined reference to pow@GLIBC_2.29, habitat-sim #1810)."
  exit 1
fi
echo "  GLIBC $GLIBC_VER OK"
missing=""
ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'    || missing="$missing libegl1-mesa-dev"
ldconfig -p 2>/dev/null | grep -q 'libOpenGL\.so\.0' || missing="$missing libglvnd(libOpenGL)"
{ [ -f /usr/include/glm/glm.hpp ]; }                 || missing="$missing libglm-dev"
{ [ -f /usr/include/jpeglib.h ] || [ -f /usr/include/x86_64-linux-gnu/jpeglib.h ]; } \
                                                     || missing="$missing libjpeg-dev"
if [ -n "$missing" ]; then
  echo "  missing system deps:$missing"
  if sudo -n true 2>/dev/null; then
    echo "  passwordless sudo available — installing them now"
    sudo apt-get update -qq || true
    sudo apt-get install -y --no-install-recommends \
        libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils \
        xorg-dev freeglut3-dev libglvnd-dev \
      || { echo "FATAL: apt install failed — run manually: $APT_LINE"; exit 1; }
  else
    echo "FATAL: missing system deps:$missing"
    echo "  fix: $APT_LINE"
    exit 1
  fi
fi
ls /usr/lib/x86_64-linux-gnu/libGLX.so >/dev/null 2>&1 \
  || echo "WARN: no libGLX.so dev symlink — if the build dies with 'No rule to make target .../libGLX.so' (issue #215), run: $APT_LINE"
echo "  GL/EGL dev libs OK"

# --- 4. habitat-sim from the audio branch (skip if already audio-capable) ---
banner "[4/8] habitat-sim @ $SIM_BRANCH with --audio"
if audio_probe; then
  echo "  audio-capable habitat_sim already importable — skipping build"
else
  mkdir -p "$BUILD_ROOT"
  if [ ! -d "$SIM_DIR/.git" ]; then
    # MUST be a git clone (not a tarball): setup.py only auto-inits submodules
    # inside a git repo, and the audio engine itself is a submodule.
    git clone https://github.com/facebookresearch/habitat-sim.git "$SIM_DIR" \
      || { echo "FATAL: habitat-sim clone failed"; exit 1; }
  fi
  cd "$SIM_DIR" || exit 1
  git fetch origin "$SIM_BRANCH" \
    || { echo "FATAL: branch $SIM_BRANCH not fetchable (repo archived — mirror needed?)"; exit 1; }
  git checkout "$SIM_BRANCH" || { echo "FATAL: checkout $SIM_BRANCH failed"; exit 1; }
  git rev-parse --short HEAD
  # Explicit submodule init (setup.py would also do it) so a transient network
  # failure surfaces HERE, not as 'Could NOT find Corrade' mid-configure.
  git submodule update --init --recursive \
    || { echo "FATAL: submodule init failed (network needed once; the RLR audio engine is a submodule)"; exit 1; }
  [ -f src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so ] \
    || { echo "FATAL: prebuilt RLR audio engine missing after submodule init (src/deps/rlr-audio-propagation)"; exit 1; }
  # Constraint file keeps requirements.txt's unpinned numpy from resolving 2.x.
  echo "numpy<1.24" > "$BUILD_ROOT/np-constraint.txt"
  pip install -r requirements.txt -c "$BUILD_ROOT/np-constraint.txt" \
    || { echo "FATAL: habitat-sim requirements failed"; exit 1; }
  export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
  # --headless: RACE has no display. --audio: builds RLRAudioPropagation.
  # --with-cuda deliberately omitted (audio engine is CPU; EGL needs no CUDA).
  python setup.py install --headless --audio 2>&1 \
    | tee "$REPO_ROOT/$OUT_DIR/sim-build.log" | tail -40
  rc=${PIPESTATUS[0]}
  cd "$REPO_ROOT" || exit 1
  if [ "$rc" != "0" ]; then
    echo "FATAL: habitat-sim audio build failed (exit $rc) — error excerpt:"
    # Surface the real failure (the tail -40 window is usually eaten by the
    # python traceback); the cmake/compiler error lines live mid-log.
    grep -n -iE "cmake error|error:|Could NOT|No such file|undefined reference" \
        -B3 -A20 "$OUT_DIR/sim-build.log" | head -120 || true
    [ -f "$SIM_DIR/build/CMakeFiles/CMakeError.log" ] \
      && { echo "--- CMakeError.log tail ---"; tail -40 "$SIM_DIR/build/CMakeFiles/CMakeError.log"; }
    echo "  full log: $OUT_DIR/sim-build.log; this version delta is the spike deliverable"
    exit 1
  fi
fi

# --- 5. import + audio-API verify ---
banner "[5/8] verify import + audio bindings (Binaural enum, not just AudioSensorSpec)"
audio_probe || { echo "FATAL: audio API verify failed (see probe output above)"; exit 1; }

# --- 6. habitat-lab v0.2.2 (best-effort; NOT needed for the RIR smoke) ---
# Installs the task-layer pin so the spike also reports whether it goes in
# cleanly. A failure here is a WARN, not a blocker — task design needs it,
# the RIR render does not.
banner "[6/8] habitat-lab $LAB_TAG (best-effort)"
if python -c "import habitat" 2>/dev/null; then
  echo "  habitat-lab already importable — skipping"
else
  if [ ! -d "$LAB_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/habitat-lab.git "$LAB_DIR" || true
  fi
  if [ -d "$LAB_DIR/.git" ]; then
    ( cd "$LAB_DIR" \
        && { git fetch origin tag "$LAB_TAG" 2>/dev/null || true; } \
        && git checkout "$LAB_TAG" \
        && sed -i '/from habitat\.robots\.fetch_robot import FetchRobot/d' \
             habitat/tasks/rearrange/rearrange_sim.py \
        && pip install -e . -c "$BUILD_ROOT/np-constraint.txt" 2>&1 | tail -5 ) \
      || echo "WARN: habitat-lab $LAB_TAG install failed — record as version delta (RIR smoke unaffected)"
    # The sed is the OFFICIAL post-install edit (INSTALLATION.md: remove the
    # FetchRobot import, line 36) — without it `import habitat` breaks.
    python -c "import habitat; print('  habitat-lab', habitat.__version__, 'imports OK')" 2>/dev/null \
      || echo "WARN: habitat-lab installed but import fails — record as version delta (RIR smoke unaffected)"
  else
    echo "WARN: habitat-lab clone failed — record as version delta (RIR smoke unaffected)"
  fi
fi

# --- 7. find an HM3D val_mini scene asset we own ---
banner "[7/8] locate HM3D scene .glb"
SCENE_GLB="$(find data/hm3d -name '*.basis.glb' 2>/dev/null | grep -E 'wcojb4TFT35|TEEsavR23oF' | head -1)"
[ -n "$SCENE_GLB" ] || SCENE_GLB="$(find data/hm3d -name '*.basis.glb' 2>/dev/null | head -1)"
[ -n "$SCENE_GLB" ] || SCENE_GLB="$(find data/hm3d -name '*.glb' 2>/dev/null | grep -v semantic | head -1)"
[ -n "$SCENE_GLB" ] || { echo "FATAL: no HM3D .glb found under data/hm3d"; exit 1; }
echo "  scene: $SCENE_GLB"

# --- 8. render ONE RIR (materials OFF — HM3D semantics are absent/broken) ---
banner "[8/8] RIR smoke"
python embodied_memory/scripts/soundspaces_rir_smoke.py \
    --scene "$SCENE_GLB" --out "$OUT_DIR/rir.npy" 2>&1 | tee "$OUT_DIR/smoke.log"
rc=${PIPESTATUS[0]}

if [ "$rc" = "0" ]; then
  banner "SPIKE GREEN — SoundSpaces 2.0 renders RIRs on our HM3D stack"
  echo "  IR saved: $OUT_DIR/rir.npy — audio-goal task design can start."
else
  banner "SPIKE RED (exit $rc) — blockers above are the deliverable"
  echo "  Paste $OUT_DIR/smoke.log + $OUT_DIR/sim-build.log tail; reassess Friday"
  echo "  (fallback: descope audio — paper stands on LTM + baselines + demo)."
fi
exit "$rc"
