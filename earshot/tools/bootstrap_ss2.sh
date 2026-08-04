#!/bin/bash
# earshot/tools/bootstrap_ss2.sh — the clean room's rebuild path for the `ss2` env.
#
# MOVED from .scratch/ss2-clean-room/probes/oneenv_gate.sh (ticket 17). Two copies of a
# build recipe is a drift trap, and a probe inside a wayfinder ticket directory is not
# where an operator looks — `.scratch` reads as disposable even though it is tracked.
# The footgun hardening below is CARRIED, not rewritten: the SIGPIPE-safe conda
# directory check, the poisoned half-configure wipe, the cmake/CPATH shim, and the
# explicit submodule init all cost a box trip each to learn.
#
#   nrun bash earshot/tools/bootstrap_ss2.sh
#
# GREEN = one process, all imports, GPU visible, audio sensor renders in a scene.
# RED   = the printed blocker list IS the deliverable.
#
# Deliberately does NOT reuse the `soundspaces-spike` env — a spike artifact with
# unknown drift. Builds clean so the result is trustworthy. Never touches
# `ltm-embodied` either; ticket 27 retires it.
#
# --- what ticket 17 changed on the way in ----------------------------------
#  1. THE PIN IS A FILE, not a one-line numpy constraint. `earshot/tools/
#     ss2-constraints.txt` carries nine exact versions and is passed as `-c` to every
#     pip install here — the numpy layer, habitat-sim's own requirements.txt, torch,
#     and the CLAP stack. That plumbing already existed; only the content widened.
#  2. habitat-sim is pinned to a SHA, and THE BUILD-SKIP CHECKS IT. The old skip fired
#     whenever an audio-capable habitat_sim merely imported, never asking which tree it
#     was built from — the exact version-blind bug ticket 13 killed in the torch layer,
#     still alive here, and it made a SHA pin inert on every re-run against an existing
#     env.
#  3. A BOOTSTRAP-TIME PROVENANCE CHECK. A constraint on a package that is never
#     installed is a silent no-op, so a misspelled name is an inert pin that reports
#     success. Only a resolved-versus-constraints comparison can see that; a capability
#     probe cannot.
#  4. The habitat-lab arm and the two-report comparison are GONE. habitat-lab is
#     deliberately not installed: the runner drives habitat_sim directly, so that arm
#     was feasibility-experiment shape, not production shape.
#
# --- what ticket 24 changed ------------------------------------------------
#  5. THE VERDICT IS NOW THE ASSERTION THE RUNTIME SHARES. Stage 8 runs
#     `python -m earshot.env_check --strict` — one implementation of "can this env run
#     an episode", called from here and from `task/`'s entry point. An assertion that
#     lives only in the gate cannot run at episode time, which is exactly when a drifted
#     env produces results instead of an error.
#  6. STAGE 7'S COMPARISON MOVED OUT OF BASH. It was an inline heredoc; ticket 19 assigns
#     that same rule to `env_check.py`'s Mac-testable half, so the two were one rule in
#     two languages. One implementation now, unit-tested against this very constraints
#     file.
#  7. THE TICKET-04 PROBE IS KEPT, DEMOTED TO STAGE 9. `env_check` never opens a scene,
#     and "audio sensor renders in a scene" is this script's headline — so the probe is
#     the scene render rather than the verdict.
#
# Idempotent: re-runs reuse the env and clone, and skip the build when an audio-capable
# habitat_sim already imports AND is built from the pinned SHA.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
PY_VER="3.9"                       # SoundSpaces INSTALLATION.md pin
CMAKE_VER="3.14.0"                 # ditto (also dodges CMake 4.x refusing the 2022 dep tree)
BUILD_ROOT="${SS2_BUILD_ROOT:-${HOME}/ss2-build}"
SIM_DIR="${BUILD_ROOT}/habitat-sim"
# The audio work never left this branch: its HEAD IS the SHA below, last committed
# 2022-11-04 and dormant since. So the source half of the pin costs one line, and the
# genuinely unpinned surface was always the PyPI half in ss2-constraints.txt.
SIM_BRANCH="RLRAudioPropagationUpdate"
SIM_SHA="${SS2_SIM_SHA:-4f61e321}"
PROBE_DIR="$REPO_ROOT/.scratch/ss2-clean-room/probes"
PATCH_DIR="$PROBE_DIR/patches"
OUT_DIR="${SS2_OUT_DIR:-runs/ss2-bootstrap}"
CONSTRAINTS="$REPO_ROOT/earshot/tools/ss2-constraints.txt"
[ -f "$CONSTRAINTS" ] || { echo "FATAL: $CONSTRAINTS missing — the pin IS the recipe"; exit 1; }

# V100 is compute 7.0 / CUDA 11.x era. Overridable because ticket 05's inventory
# may find a driver that wants a different wheel.
#
# Ticket 13 moved this pin UP from torch==2.0.1+cu117, which was an unverified
# guess that transformers then silently refused to run against. 2.2.2+cu118,
# because:
#   - transformers gates `is_torch_available()` on `>= 2.1.0` (source-verified),
#     so 2.2.x clears it with headroom instead of sitting on the boundary;
#   - torch declares NO numpy dependency, so this install physically cannot move
#     the `numpy<1.24` pin the 2022-era habitat-sim tree depends on;
#   - 2.2.x predates numpy 2.0, so numpy 1.23.x is its NATIVE ABI, not a
#     tolerated downgrade — which is why we take the OLD end of the 2.1-2.6 band
#     that has cp39+cu118 wheels, not the new end. Nothing here needs torch 2.3+;
#   - cu118 is the last CUDA line where the V100's sm_70 is a first-class target.
# Ticket 05 measured torch 2.8.0+cu128 running on this exact V100, so the driver
# was never the constraint. The numpy floor is, and it argues downward.
TORCH_SPEC="${SS2_TORCH_SPEC:-torch==2.2.2}"
TORCH_INDEX="${SS2_TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
# transformers' hard floor for enabling its torch backend. Kept as a variable
# because it is the number the whole pin choice above is derived from.
TORCH_MIN="${SS2_TORCH_MIN:-2.1}"
# ClapModel/ClapProcessor landed in transformers 4.27.
# NOTE (ticket 13): the `<5` cap is INERT on this env — transformers 5.x declares
# `requires_python >= 3.10` and the SoundSpaces pin is Python 3.9, so the
# interpreter already caps resolution at the 4.x line. The floor is what matters,
# and it is raised here so a resolver cannot drift DOWN to a pre-CLAP release
# either. The real protection is not this range at all: it is stage 05 of the
# probe, which now fails loudly when the resolved pair does not actually work.
TRANSFORMERS_SPEC="${SS2_TRANSFORMERS_SPEC:-transformers>=4.40,<5}"

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
#
# The constraint file is now git-tracked and READ, never written. It used to be
# generated here as a single `numpy<1.24` line into $BUILD_ROOT — which meant the pin
# lived on the box and nowhere in the repo.
pip install -q "numpy>=1.16.1,<1.24" numpy-quaternion -c "$CONSTRAINTS" \
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

# --- 4. habitat-sim at the pinned SHA, patch-capable -----------------------
banner "[4/9] habitat-sim @ $SIM_BRANCH:$SIM_SHA with --audio"
APPLIED_PATCHES=""
# TICKET 17'S FIX. This skip used to be `quick_audio_probe` alone: it fired whenever an
# audio-capable habitat_sim merely imported, and never asked which tree it was built
# from. Against an existing env that made the SHA pin INERT on every re-run — the same
# version-blind class ticket 13 killed in the torch layer below. Both halves must hold.
_sim_head="$( [ -d "$SIM_DIR/.git" ] && git -C "$SIM_DIR" rev-parse HEAD 2>/dev/null || echo "" )"
if quick_audio_probe >/dev/null 2>&1 && [ "${_sim_head:0:${#SIM_SHA}}" = "$SIM_SHA" ]; then
  echo "  audio-capable habitat_sim already built from $SIM_SHA — skipping build"
  if [ -d "$SIM_DIR/.git" ] && [ -f "$BUILD_ROOT/applied-patches.txt" ]; then
    APPLIED_PATCHES="$(cat "$BUILD_ROOT/applied-patches.txt")"
    echo "  previously applied patches: ${APPLIED_PATCHES:-none}"
  fi
else
  if [ -n "$_sim_head" ] && [ "${_sim_head:0:${#SIM_SHA}}" != "$SIM_SHA" ]; then
    echo "  habitat-sim at ${_sim_head:0:8}, pinned to $SIM_SHA — rebuilding"
  fi
  if [ ! -d "$SIM_DIR/.git" ]; then
    # MUST be a git clone, not a tarball: setup.py only auto-inits submodules
    # inside a git repo, and the closed-source audio engine IS a submodule.
    git clone https://github.com/facebookresearch/habitat-sim.git "$SIM_DIR" \
      || { echo "FATAL: habitat-sim clone failed"; exit 1; }
  fi
  cd "$SIM_DIR" || exit 1
  # FETCH THE BRANCH, RESET TO THE SHA. Not `git fetch origin <sha>`, which needs the
  # server to allow SHA-in-want. This form also fails LOUDLY if a force-push ever makes
  # the SHA unreachable from the branch — which is exactly the signal the pin exists to
  # produce, and the reason the reset is no longer `|| true`.
  git fetch origin "$SIM_BRANCH" \
    || { echo "FATAL: branch $SIM_BRANCH not fetchable (repo archived — mirror needed?)"; exit 1; }
  git checkout "$SIM_BRANCH" || { echo "FATAL: checkout $SIM_BRANCH failed"; exit 1; }
  git reset --hard "$SIM_SHA" >/dev/null 2>&1 \
    || { echo "FATAL: $SIM_SHA unreachable from origin/$SIM_BRANCH — the branch moved or"
         echo "  was force-pushed. The pin is doing its job; decide on the new tree before"
         echo "  overriding with SS2_SIM_SHA."; exit 1; }
  echo "  habitat-sim HEAD: $(git rev-parse --short HEAD) (pinned $SIM_SHA)"
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

  pip install -q -r requirements.txt -c "$CONSTRAINTS" \
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
# Ticket 13: this skip used to be `python -c "import torch"`, i.e. version-blind.
# The box's ss2 env ALREADY has torch 2.0.1 installed, so a bare importability
# check would have skipped the layer and made the pin bump a silent no-op — the
# gate would re-run, skip, and reproduce the exact failure it was fixing.
# Gate on the version that transformers actually requires.
if python - "$TORCH_MIN" <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
have = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
need = tuple(int(p) for p in sys.argv[1].split("."))
print("  found torch {} (need >= {})".format(torch.__version__, sys.argv[1]))
sys.exit(0 if have >= need else 2)
PY
then
  echo "  torch already satisfies the transformers backend gate — skipping"
else
  echo "  installing $TORCH_SPEC (replacing any older torch in this env)"
  pip install -q "$TORCH_SPEC" --index-url "$TORCH_INDEX" -c "$CONSTRAINTS" \
    || { echo "FATAL: torch install failed (wrong CUDA wheel for this driver? override SS2_TORCH_SPEC / SS2_TORCH_INDEX)"; exit 1; }
fi
layer_check "torch" || exit 1

# --- 6. CLAP stack (transformers + scipy) ----------------------------------
# The rebuilt agent's ONLY model dependency: memory is out of scope, so the 7B
# planner and the VLM captioner are no longer required imports.
banner "[6/9] CLAP stack ($TRANSFORMERS_SPEC + scipy)"
pip install -q "$TRANSFORMERS_SPEC" scipy soundfile -c "$CONSTRAINTS" \
  || { echo "FATAL: transformers/scipy install failed"; exit 1; }
layer_check "clap-stack" || exit 1

# Ticket 13: assert the numpy pin, do not assume it. The constraint file applies
# to every install above, but the whole 2022-era habitat-sim tree dies on numpy
# 2.x, and a later pip layer quietly resolving numpy up is the single most likely
# way this env breaks. `ltm-embodied` on this same box carries numpy 1.26.4, so
# the failure is one careless `pip install` away, not hypothetical.
python - <<'PY' || exit 1
import sys
import numpy
parts = tuple(int(p) for p in numpy.__version__.split(".")[:2])
if parts >= (1, 24):
    print("  FATAL: numpy {} >= 1.24 — a pip layer defeated the constraint file."
          "\n  The 2022-era habitat-sim tree cannot run on this. Find the layer that"
          "\n  pulled it and constrain that install.".format(numpy.__version__))
    sys.exit(1)
print("  numpy pin held: {}".format(numpy.__version__))
PY

# --- 7. provenance: did the pin actually take? ------------------------------
# Ticket 17 section 4. A constraint on a package that is never installed is a SILENT
# NO-OP, so a misspelled or dropped line in ss2-constraints.txt is an inert pin that
# reports success — the same class as the version-blind build-skips fixed above, and
# invisible to any capability probe. Only comparing the RESOLVED set against the file
# can see it. This is the bootstrap-time half; the runtime capability half is stage 8.
#
# TICKET 24 MOVED THE COMPARISON INTO PYTHON. It used to be an inline heredoc here, and
# ticket 19 assigns exactly this rule to `env_check.py`'s Mac-testable half — so two
# implementations of one rule were living in two languages, which is the drift trap
# ticket 17 named when it refused to leave the build recipe in two places. There is now
# one implementation, and it is unit-tested against this very constraints file.
#
# WARN, not FATAL (no `--strict`): the recipe already succeeded by this point, and
# killing a 40-minute build over a version skew nobody has judged yet would destroy the
# evidence. The printed diff IS the deliverable.
banner "[7/9] provenance — resolved versus ss2-constraints.txt"
pip freeze > "$OUT_DIR/freeze.txt" 2>/dev/null \
  || echo "  WARN: pip freeze failed — provenance unverified, which is not the same as verified"
PYTHONPATH="$REPO_ROOT" python -m earshot.env_check --provenance \
    --constraints "$CONSTRAINTS" \
    --freeze "$OUT_DIR/freeze.txt"

# --- 8. the runtime assertion ------------------------------------------------
# TICKET 24, and it is the reason this file and the runtime cannot disagree about what a
# working env is: `assert_env()` is ONE implementation with TWO callers — this line, and
# `task/`'s entry point before an episode. An assertion stranded in bash cannot run at
# episode time, which is exactly when a drifted env produces results instead of an error.
#
# CAPABILITY-SHAPED, NEVER PROVENANCE-SHAPED. Every probe does the thing: allocates on
# the GPU and reads the result back, resolves the audio enum MEMBER (AudioSensorSpec is
# bound even in non-audio builds), instantiates CLAP and reads a finite logit. Ticket 13
# is the whole argument — `transformers` reported 4.57.6 both before and after the fix
# and `ClapModel` imported cleanly the entire time it was a DummyObject, so a version
# comparison would have printed green through the whole failure.
#
# A PROBE THAT DID NOT RUN IS NOT A PASS. `--strict` exits non-zero on a FAIL, on a
# NOT_RUN, and on an expected probe that was never emitted at all.
#
# SS2_LOAD_CLAP adds the CLAP instantiation (153.5M params, ~0.7 GB VRAM), which is
# requested rather than required for the same reason ticket 17 gave: it is paid only by
# runs that use it.
banner "[8/9] env_check --strict — the runtime assertion"
PYTHONPATH="$REPO_ROOT" python -m earshot.env_check --strict \
    ${SS2_LOAD_CLAP:+--clap} \
    | tee "$OUT_DIR/env_check.log"
ENV_CHECK_RC=${PIPESTATUS[0]}
if [ "$ENV_CHECK_RC" -ne 0 ]; then
  echo
  echo "  RED: the env cannot run an episode. The probe list above IS the blocker list."
  exit "$ENV_CHECK_RC"
fi

# --- 9. the scene render ------------------------------------------------------
# The verdict `env_check` cannot give, and the reason the ticket-04 probe is KEPT rather
# than replaced: the assertion above proves the stack imports, allocates and resolves the
# audio enum, but this script's headline is "audio sensor renders IN A SCENE" and no
# probe in `env_check` opens a scene. That belongs to `audio/guard.arm_audio_context`,
# whose own box coverage is `tests/box/test_audio_guard_box.py`.
#
# The payload stays at .scratch/ss2-clean-room/probes/oneenv_probe.py (ADR-0013): it is
# the ticket-04 feasibility probe, and its defaults dump is what makes the 23-knob
# parameter sheet measured rather than quoted.
banner "[9/9] core probe — the scene render"
python "$PROBE_DIR/oneenv_probe.py" \
    --out "$OUT_DIR/report-core.json" \
    --sim-dir "$SIM_DIR" \
    --label core \
    ${SS2_SCENE:+--scene "$SS2_SCENE"} \
    ${SS2_LOAD_CLAP:+--load-clap} 2>&1 | tee "$OUT_DIR/probe-core.log"
CORE_RC=${PIPESTATUS[0]}

python - "$OUT_DIR" <<'PY'
import json, os, sys
out = sys.argv[1]
path = os.path.join(out, "report-core.json")
if not os.path.exists(path):
    print("  NO CORE REPORT — the probe did not run; see probe-core.log")
    sys.exit(1)
with open(path) as fh:
    core = json.load(fh)

v = core["_verdict"]
print("  core verdict: {}".format("GREEN" if v["green"] else "RED"))
if not v["green"]:
    print("  failed stages: {}".format(", ".join(v["failed_stages"]) or "(render did not succeed)"))
print("  numpy pin held: {}".format(v["numpy_pin_held"]))

d = core.get("03_defaults_dump", {})
print("\n  measured defaults the parameter sheet rests on:")
print("    transmission        = {}".format(d.get("transmission_default")))
print("    enableMaterials     = {} (on {})".format(
    d.get("enableMaterials_default"), d.get("enableMaterials_location")))
print("    maxIRLength present = {} / irTime present = {}".format(
    d.get("has_maxIRLength"), d.get("has_irTime")))
print("    directRayCount      = {}".format(d.get("has_directRayCount")))
trap = d.get("dynamic_attr_trap", {})
print("    spec swallows unknown keys = {} (validator must live on the SPEC)".format(
    trap.get("spec_swallows_unknown_key")))

# Ticket 13: a GREEN must never again be ambiguous about what it proved of CLAP.
c = core.get("05_clap", {})
print("\n  CLAP:")
print("    transformers {} / torch backend enabled = {}".format(
    c.get("transformers"), c.get("torch_backend_available")))
print("    ClapModel is a dummy object = {}".format(c.get("clap_is_dummy")))
if c.get("clap_weights_loaded"):
    print("    forward pass: logits {} finite={} on {} => PROVEN".format(
        c.get("clap_logits_shape"), c.get("clap_logits_finite"), c.get("clap_device")))
else:
    print("    forward pass: NOT RUN — backend checked only.")
    print("    a GREEN needs the logit: re-run with SS2_LOAD_CLAP=1")

r = core.get("07_live_render", {})
print("\n  first render: {} s, IR shape {}, non-silent {}".format(
    r.get("first_render_s"), r.get("ir_shape"), r.get("ir_nonzero")))
print("  (single timing at defaults — the tuned preset is ticket 06's cheap_preset)")

p = core.get("08_provenance", {})
print("\n  provenance:")
print("    habitat-sim {} @ {}".format(p.get("habitat_sim_branch"), p.get("habitat_sim_sha")))
print("    rlr-audio-propagation submodule @ {}".format(p.get("rlr_audio_propagation_sha")))
PY

echo
echo "  report:    $OUT_DIR/report-core.json"
echo "  env_check: $OUT_DIR/env_check.log   (the assertion the runtime shares)"
echo "  freeze:    $OUT_DIR/freeze.txt      (forensic evidence — never installed from)"
exit "$CORE_RC"
