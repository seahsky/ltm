#!/bin/bash
# .scratch/ss2-clean-room/probes/oneenv_gate.sh — ticket 04's box gate.
#
# Question: can ONE conda env on RACE hold an audio-capable habitat-sim build
# AND everything else the rebuilt agent imports, or does "purely SoundSpaces 2.0"
# force something out of process?
#
#   nrun bash .scratch/ss2-clean-room/probes/oneenv_gate.sh
#
# GREEN = one process, all imports, GPU visible, audio sensor renders in a scene.
# RED   = the printed blocker list IS the deliverable; ticket 07 gets re-scoped
#         around whatever cannot coexist.
#
# Deliberately does NOT reuse the `soundspaces-spike` env — ticket 04 calls it a
# spike artifact with unknown drift. Builds clean so the result is trustworthy.
# Never touches `ltm-embodied` either.
#
# --- what this adds over race-soundspaces-spike.sh -------------------------
#  1. ONE env: torch + transformers(CLAP) + scipy layer onto the audio build,
#     and the audio probe re-runs after each layer. Layering is the actual
#     question; the spike never tested it.
#  2. Patch-capable from the start (ticket 02's ~40-line multi-source patch is
#     likely), with applied patches recorded so box state is reproducible.
#  3. Records habitat-sim HEAD + rlr-audio-propagation submodule SHAs, which is
#     what makes tickets 01/11's parameter sheet falsifiable.
#  4. Dumps every AudioSensorSpec / acousticsConfig field, so the defaults column
#     stops being hearsay (ticket 06 blocks on this).
#  5. habitat-lab is installed LAST, after the core verdict is already written to
#     disk, because it is the layer most likely to rot the numpy pin. Two reports
#     get produced and compared.
#
# Idempotent: re-runs reuse the env and clone, and skip the build when an
# audio-capable habitat_sim already imports.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
PY_VER="3.9"                       # SoundSpaces INSTALLATION.md pin
CMAKE_VER="3.14.0"                 # ditto (also dodges CMake 4.x refusing the 2022 dep tree)
BUILD_ROOT="${SS2_BUILD_ROOT:-${HOME}/ss2-build}"
SIM_DIR="${BUILD_ROOT}/habitat-sim"
LAB_DIR="${BUILD_ROOT}/habitat-lab"
SIM_BRANCH="RLRAudioPropagationUpdate"
LAB_TAG="v0.2.2"
PROBE_DIR="$REPO_ROOT/.scratch/ss2-clean-room/probes"
PATCH_DIR="$PROBE_DIR/patches"
OUT_DIR="${SS2_OUT_DIR:-runs/ss2-oneenv-gate}"
NP_CONSTRAINT="$BUILD_ROOT/np-constraint.txt"

# V100 is compute 7.0 / CUDA 11.x era. Overridable because ticket 05's inventory
# may find a driver that wants a different wheel.
TORCH_SPEC="${SS2_TORCH_SPEC:-torch==2.0.1}"
TORCH_INDEX="${SS2_TORCH_INDEX:-https://download.pytorch.org/whl/cu117}"
# ClapModel/ClapProcessor landed in transformers 4.27; cap below 5 so a major
# bump cannot silently change the API under us.
TRANSFORMERS_SPEC="${SS2_TRANSFORMERS_SPEC:-transformers>=4.30,<5}"

APT_LINE="sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1 libglx-mesa0 libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev libglvnd-dev"

banner() { printf '\n========== %s ==========\n' "$1"; }

# Quick shell-level audio probe. Re-run after every install layer, because the
# whole question is whether layering breaks the audio build.
quick_audio_probe() {
  python - <<'PY' 2>&1
import sys
try:
    import quaternion  # noqa: F401  must precede habitat_sim (issue #1813)
    import habitat_sim, habitat_sim.sensor, numpy
    t = getattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType", None)
    assert t is not None and hasattr(t, "Binaural"), "habitat_sim built WITHOUT --audio"
    print("  OK  habitat_sim {} audio-capable, numpy {}".format(
        getattr(habitat_sim, "__version__", "?"), numpy.__version__))
except Exception as exc:
    print("  BROKEN  {!r}".format(exc))
    sys.exit(1)
PY
}

layer_check() {   # layer_check <label>
  local label="$1"
  echo "  audio probe after $label:"
  if quick_audio_probe; then
    return 0
  fi
  echo "  *** the '$label' layer BROKE the audio build — that is a ticket-04 finding,"
  echo "      not a script bug. Record it and stop."
  return 1
}

mkdir -p "$OUT_DIR" "$BUILD_ROOT"

# --- 1. self-update -------------------------------------------------------
# Gotcha: this script git-pulls itself, so a change to THIS file only takes
# effect on the second invocation.
banner "[1/9] git pull --ff-only"
git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
echo "  running commit: $(git rev-parse --short HEAD)"

# --- 2. conda env (FRESH; never ltm-embodied, never soundspaces-spike) -----
banner "[2/9] conda env: $ENV_NAME (python=$PY_VER cmake=$CMAKE_VER, numpy<1.24, gcc-10)"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
case "$ENV_NAME" in
  ltm-embodied|soundspaces-spike)
    echo "FATAL: refusing to touch '$ENV_NAME' — ticket 04 requires a clean env"; exit 1;;
esac
# conda's compiler-package hooks dereference CONDA_BACKUP_* vars that are unset on
# first install; under `set -u` that kills the script. Run every conda state
# change with nounset OFF.
set +u
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
# Existence via the env dir, NOT `conda env list | grep -q`: under pipefail a
# grep -q exiting early can SIGPIPE the conda writer and turn found-it into a
# pipeline failure.
if [ ! -d "$MINICONDA/envs/$ENV_NAME" ]; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER" "cmake=$CMAKE_VER" \
    || { echo "FATAL: conda create failed"; exit 1; }
fi
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
[ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] || { echo "FATAL: wrong env: ${CONDA_DEFAULT_ENV:-<none>}"; exit 1; }
set -u

# numpy 2.x breaks the 2022-era tree. Pin BEFORE anything else resolves numpy,
# and apply the constraint file to EVERY later pip install.
echo "numpy<1.24" > "$NP_CONSTRAINT"
pip install -q "numpy>=1.16.1,<1.24" numpy-quaternion -c "$NP_CONSTRAINT" \
  || { echo "FATAL: numpy/numpy-quaternion pin install failed"; exit 1; }

# The only attested toolchain for this 2022-era tree is gcc 7-10.
if [ ! -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
  set +u   # the install reactivates the env -> sources compiler hooks
  conda install -y -q -c conda-forge 'gcc_linux-64=10.*' 'gxx_linux-64=10.*' sysroot_linux-64 \
    || echo "WARN: conda gcc-10 install failed — building with system gcc $(gcc -dumpversion 2>/dev/null || echo '?')"
  set -u
fi
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
  echo "  toolchain: conda gcc $("$CC" -dumpversion)"
else
  echo "  toolchain: system gcc $(gcc -dumpversion 2>/dev/null || echo '?')"
fi

# --- 3. system preflight (fail in seconds, not 40 min into the build) ------
banner "[3/9] system preflight: GLIBC >= 2.29 + GL/EGL dev libs"
# Pipe-safe: `ldd | head -1 | grep` lets head/grep exit early -> SIGPIPEs ldd ->
# under pipefail the pipeline "fails" despite a good match. getconf + awk consume
# their whole input, so nothing SIGPIPEs.
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
    echo "  passwordless sudo — installing"
    sudo apt-get update -qq || true
    # libgl1-mesa-glx was DROPPED in Ubuntu 24.04 (split into libgl1 +
    # libglx-mesa0). Those exist on 20.04/22.04 too, so this line resolves
    # across releases; the build is EGL-headless and never needs old GLX.
    sudo apt-get install -y --no-install-recommends \
        libjpeg-dev libglm-dev libgl1 libglx-mesa0 libegl1-mesa-dev mesa-utils \
        xorg-dev freeglut3-dev libglvnd-dev \
      || { echo "FATAL: apt install failed — run manually: $APT_LINE"; exit 1; }
  else
    echo "FATAL: missing system deps:$missing"
    echo "  fix: $APT_LINE"
    exit 1
  fi
fi
echo "  GL/EGL dev libs OK"
echo "  nproc: $(nproc)  (threadCount is a free speed knob — ticket 06)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/  gpu: /' || echo "  gpu: nvidia-smi unavailable"

# --- 4. habitat-sim from the audio branch, patch-capable -------------------
banner "[4/9] habitat-sim @ $SIM_BRANCH with --audio"
APPLIED_PATCHES=""
if quick_audio_probe >/dev/null 2>&1; then
  echo "  audio-capable habitat_sim already importable — skipping build"
  if [ -d "$SIM_DIR/.git" ] && [ -f "$BUILD_ROOT/applied-patches.txt" ]; then
    APPLIED_PATCHES="$(cat "$BUILD_ROOT/applied-patches.txt")"
    echo "  previously applied patches: ${APPLIED_PATCHES:-none}"
  fi
else
  if [ ! -d "$SIM_DIR/.git" ]; then
    # MUST be a git clone, not a tarball: setup.py only auto-inits submodules
    # inside a git repo, and the closed-source audio engine IS a submodule.
    git clone https://github.com/facebookresearch/habitat-sim.git "$SIM_DIR" \
      || { echo "FATAL: habitat-sim clone failed"; exit 1; }
  fi
  cd "$SIM_DIR" || exit 1
  git fetch origin "$SIM_BRANCH" \
    || { echo "FATAL: branch $SIM_BRANCH not fetchable (repo archived — mirror needed?)"; exit 1; }
  # Reset hard so a re-run after a failed patch starts from a known tree.
  git checkout "$SIM_BRANCH" || { echo "FATAL: checkout $SIM_BRANCH failed"; exit 1; }
  git reset --hard "origin/$SIM_BRANCH" >/dev/null 2>&1 || true
  echo "  habitat-sim HEAD: $(git rev-parse --short HEAD)"
  # Explicit submodule init so a transient network failure surfaces HERE, not as
  # 'Could NOT find Corrade' mid-configure.
  git submodule update --init --recursive \
    || { echo "FATAL: submodule init failed (the RLR audio engine is a submodule)"; exit 1; }
  [ -f src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so ] \
    || { echo "FATAL: prebuilt RLR audio engine missing after submodule init"; exit 1; }

  # --- local patches (ticket 02's multi-source patch is the likely first) ---
  # Applied BEFORE the build and recorded, so box state is reproducible. Whether
  # we actually take the multi-source patch is ticket 09's call, gated on ticket
  # 06's cost sweep; what ticket 04 owes is a build that CAN take one.
  if [ -d "$PATCH_DIR" ] && ls "$PATCH_DIR"/*.patch >/dev/null 2>&1; then
    for p in "$PATCH_DIR"/*.patch; do
      name="$(basename "$p")"
      if git apply --check "$p" 2>/dev/null; then
        git apply "$p" && APPLIED_PATCHES="$APPLIED_PATCHES $name"
        echo "  applied patch: $name"
      else
        echo "FATAL: patch does not apply cleanly: $name"
        git apply --check "$p" || true
        exit 1
      fi
    done
  else
    echo "  no patches in $PATCH_DIR — building stock branch"
  fi
  echo "${APPLIED_PATCHES# }" > "$BUILD_ROOT/applied-patches.txt"

  pip install -q -r requirements.txt -c "$NP_CONSTRAINT" \
    || { echo "FATAL: habitat-sim requirements failed"; exit 1; }

  # A failed configure leaves CMakeCache.txt WITHOUT compile_commands.json;
  # setup.py's arg-cache then SKIPS re-running cmake and dies on the missing
  # file. A build dir without compile_commands.json is a poisoned half-configure.
  if [ -d "$SIM_DIR/build" ] && [ ! -f "$SIM_DIR/build/compile_commands.json" ]; then
    echo "  stale half-configured build dir — wiping $SIM_DIR/build"
    rm -rf "$SIM_DIR/build"
  fi
  export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
  # The conda cross-toolchain's triplet stops cmake searching Ubuntu's multiarch
  # dir, so magnum's find_package(OpenGL) misses the GLVND libs even with
  # libglvnd-dev installed. Point find_library/find_path at the system GL stack.
  export CMAKE_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
  export CMAKE_INCLUDE_PATH="/usr/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
  # COMPILE-time counterpart: magnum's EGL object library includes <EGL/egl.h>
  # via the compiler's DEFAULT include path, which for the conda gcc is its
  # sysroot (glibc + kernel headers only). Expose ONLY the GL header trees via a
  # shim on CPATH — deliberately NOT all of /usr/include, which would shadow the
  # sysroot's glibc headers with the host's newer ones.
  SHIM="$BUILD_ROOT/include-shim"
  mkdir -p "$SHIM"
  for d in EGL KHR GL X11; do
    [ -d "/usr/include/$d" ] && ln -sfn "/usr/include/$d" "$SHIM/$d"
  done
  export CPATH="$SHIM${CPATH:+:$CPATH}"
  # --headless: RACE has no display. --audio: builds RLRAudioPropagation.
  python setup.py install --headless --audio 2>&1 \
    | tee "$REPO_ROOT/$OUT_DIR/sim-build.log" | tail -30
  rc=${PIPESTATUS[0]}
  cd "$REPO_ROOT" || exit 1
  if [ "$rc" != "0" ]; then
    echo "FATAL: habitat-sim audio build failed (exit $rc) — error excerpt:"
    grep -n -iE "cmake error|error:|Could NOT|No such file|undefined reference" \
        -B3 -A20 "$OUT_DIR/sim-build.log" | head -120 || true
    [ -f "$SIM_DIR/build/CMakeFiles/CMakeError.log" ] \
      && { echo "--- CMakeError.log tail ---"; tail -40 "$SIM_DIR/build/CMakeFiles/CMakeError.log"; }
    echo "  full log: $OUT_DIR/sim-build.log — this version delta IS the deliverable"
    exit 1
  fi
fi
layer_check "habitat-sim" || exit 1

# --- 5. torch on top, in the SAME env --------------------------------------
banner "[5/9] torch ($TORCH_SPEC from $TORCH_INDEX)"
if python -c "import torch" 2>/dev/null; then
  echo "  torch already installed — skipping"
else
  pip install -q "$TORCH_SPEC" --index-url "$TORCH_INDEX" -c "$NP_CONSTRAINT" \
    || { echo "FATAL: torch install failed (wrong CUDA wheel for this driver? override SS2_TORCH_SPEC / SS2_TORCH_INDEX)"; exit 1; }
fi
layer_check "torch" || exit 1

# --- 6. CLAP stack (transformers + scipy) ----------------------------------
# The rebuilt agent's ONLY model dependency: memory is out of scope, so the 7B
# planner and the VLM captioner are no longer required imports.
banner "[6/9] CLAP stack ($TRANSFORMERS_SPEC + scipy)"
pip install -q "$TRANSFORMERS_SPEC" scipy soundfile -c "$NP_CONSTRAINT" \
  || { echo "FATAL: transformers/scipy install failed"; exit 1; }
layer_check "clap-stack" || exit 1

# --- 7. THE CORE VERDICT (written to disk before habitat-lab can rot it) ----
banner "[7/9] core probe — this is the ticket-04 gate"
python "$PROBE_DIR/oneenv_probe.py" \
    --out "$OUT_DIR/report-core.json" \
    --sim-dir "$SIM_DIR" \
    --label core \
    ${SS2_SCENE:+--scene "$SS2_SCENE"} \
    ${SS2_LOAD_CLAP:+--load-clap} 2>&1 | tee "$OUT_DIR/probe-core.log"
CORE_RC=${PIPESTATUS[0]}

# --- 8. habitat-lab, deliberately last and clearly optional -----------------
# Ticket 04's source read found the clean-room runner can drive habitat_sim
# directly (the audio branch's own tutorial does exactly that), so habitat-lab is
# measured, not required. It goes last because it is the layer most likely to
# break the numpy pin — and if it does, THAT is the finding.
banner "[8/9] habitat-lab $LAB_TAG (measured, not required)"
if [ "${SS2_SKIP_HABITAT_LAB:-0}" = "1" ]; then
  echo "  skipped (SS2_SKIP_HABITAT_LAB=1)"
elif python -c "import habitat" 2>/dev/null; then
  echo "  habitat-lab already importable — skipping install"
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
        && pip install -q -e . -c "$NP_CONSTRAINT" 2>&1 | tail -5 ) \
      || echo "WARN: habitat-lab $LAB_TAG install failed — record as version delta"
    # The sed is the OFFICIAL post-install edit (INSTALLATION.md: remove the
    # FetchRobot import) — without it `import habitat` breaks.
  else
    echo "WARN: habitat-lab clone failed — record as version delta"
  fi
fi
if python -c "import habitat" 2>/dev/null; then
  echo "  habitat-lab importable — re-running the full probe to see what it cost"
  if quick_audio_probe; then :; else
    echo "  *** DECISIVE: habitat-lab BROKE the audio build. The core verdict above"
    echo "      still stands; habitat-lab is not free and the clean room should"
    echo "      drive habitat_sim directly."
  fi
  python "$PROBE_DIR/oneenv_probe.py" \
      --out "$OUT_DIR/report-with-lab.json" \
      --sim-dir "$SIM_DIR" \
      --label with-habitat-lab \
      ${SS2_SCENE:+--scene "$SS2_SCENE"} 2>&1 | tee "$OUT_DIR/probe-with-lab.log"
fi

# --- 9. verdict -------------------------------------------------------------
banner "[9/9] TICKET 04 VERDICT"
python - "$OUT_DIR" <<'PY'
import json, os, sys
out = sys.argv[1]
def load(name):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)

core = load("report-core.json")
lab = load("report-with-lab.json")
if core is None:
    print("  NO CORE REPORT — the probe did not run; see probe-core.log")
    sys.exit(1)

v = core["_verdict"]
print("  core verdict: {}".format("GREEN" if v["green"] else "RED"))
if not v["green"]:
    print("  failed stages: {}".format(", ".join(v["failed_stages"]) or "(render did not succeed)"))
print("  numpy pin held: {}".format(v["numpy_pin_held"]))
print("  habitat-lab importable at core time: {}".format(v["habitat_lab_importable"]))

d = core.get("03_defaults_dump", {})
print("\n  measured defaults that other tickets block on:")
print("    transmission        = {}".format(d.get("transmission_default")))
print("    enableMaterials     = {} (on {})".format(
    d.get("enableMaterials_default"), d.get("enableMaterials_location")))
print("    maxIRLength present = {} / irTime present = {}".format(
    d.get("has_maxIRLength"), d.get("has_irTime")))
print("    directRayCount      = {}".format(d.get("has_directRayCount")))
trap = d.get("dynamic_attr_trap", {})
print("    spec swallows unknown keys = {} (validator must live on the SPEC)".format(
    trap.get("spec_swallows_unknown_key")))

r = core.get("07_live_render", {})
print("\n  first render: {} s, IR shape {}, non-silent {}".format(
    r.get("first_render_s"), r.get("ir_shape"), r.get("ir_nonzero")))
print("  (single timing at defaults — the cost sweep is ticket 06)")

p = core.get("08_provenance", {})
print("\n  provenance:")
print("    habitat-sim {} @ {}".format(p.get("habitat_sim_branch"), p.get("habitat_sim_sha")))
print("    rlr-audio-propagation submodule @ {}".format(p.get("rlr_audio_propagation_sha")))

if lab is not None:
    lv = lab["_verdict"]
    print("\n  with habitat-lab: {} (numpy pin held: {})".format(
        "GREEN" if lv["green"] else "RED", lv["numpy_pin_held"]))
    if v["green"] and not lv["green"]:
        print("  => habitat-lab COSTS the audio build. Drive habitat_sim directly.")
    elif lv["green"]:
        print("  => habitat-lab coexists. Keeping it is a choice, not a constraint.")
PY

echo
echo "  reports: $OUT_DIR/report-core.json"
[ -f "$OUT_DIR/report-with-lab.json" ] && echo "           $OUT_DIR/report-with-lab.json"
echo "  paste report-core.json back into ticket 04 to resolve it."
exit "$CORE_RC"
