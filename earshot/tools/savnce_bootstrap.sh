#!/bin/bash
# earshot/tools/savnce_bootstrap.sh — build the `savnce` env for the reproduced
# reference (ADR-0015).
#
#   bash earshot/tools/savnce_bootstrap.sh
#
# GREEN = one env, all imports, GPU visible, SAVN-CE's own simulator renders audio in a
# scene. RED = the printed blocker list IS the deliverable.
#
# BOX ONLY. A Mac cannot load habitat-sim at all: `libRLRAudioPropagation.so` is a
# prebuilt Linux-x64 binary needing GLIBC >= 2.29. That is a structural exclusion, not a
# preference.
#
# --- what this deliberately does NOT do -----------------------------------------
#
#  1. IT DOES NOT BUILD habitat-sim FROM SCRATCH. SAVN-CE's INSTALLATION.md asks for
#     branch `RLRAudioPropagationUpdate`, which is the branch `bootstrap_ss2.sh` already
#     builds at 4f61e321. We install from that existing checkout into a second env
#     instead of spending a second multi-hour compile on a 4-core box.
#
#     THE COST OF THAT CHOICE, STATED: habitat-sim's setup.py builds into `build/`
#     inside the source tree, and reconfiguring it for a second interpreter invalidates
#     that cache. A later `bootstrap_ss2.sh` run will recompile. That is recoverable;
#     hours of compile are not free.
#
#  2. IT DOES NOT PATCH EARSHOT'S SOURCE TREE. SAVN-CE needs a one-line change to
#     `simulator.py` for multiple audio sensors. Their INSTALLATION.md sanctions applying
#     it to the INSTALLED copy, so it lands in this env's site-packages and nowhere else.
#     Stage 4 asserts BOTH halves: that the installed copy carries it, and that the
#     shared source tree does not. `ss2` must not silently inherit it.
#
#  3. IT DOES NOT RECONCILE THE TWO PINS. See savnce-constraints.txt: `ss2` says
#     numpy 1.23.5 and this says 1.26.0, against the same habitat-sim SHA. Theirs wins on
#     their side of the fence, and stage 8 is where a wrong guess dies loudly.
#
# Idempotent: re-runs reuse the env, skip an install that is already correct, and
# re-assert every check.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SAVNCE_ENV_NAME:-savnce}"
PY_VER="3.9"
CMAKE_VER="3.14.0"
SS2_BUILD_ROOT="${SS2_BUILD_ROOT:-${HOME}/ss2-build}"
SIM_DIR="${SS2_BUILD_ROOT}/habitat-sim"
SIM_SHA="${SS2_SIM_SHA:-4f61e321}"
SAVNCE_DIR="$REPO_ROOT/earshot/reference/savnce"
SAVNCE_DATA_ROOT="${SAVNCE_DATA_ROOT:-${HOME}/savnce-data}"
CONSTRAINTS="$REPO_ROOT/earshot/tools/savnce-constraints.txt"
TORCH_INDEX="${SAVNCE_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

# The multi-audio-sensor change, verbatim from their INSTALLATION.md.
PATCH_FROM='audio_sensor = self._agent._sensors["audio_sensor"]'
PATCH_TO='audio_sensor = self._agent._sensors[self._spec.uuid]'

BLOCKERS=()
blocker() { BLOCKERS+=("$1"); echo "  RED   $1"; }
ok()      { echo "  OK    $1"; }
banner()  { printf '\n========== %s ==========\n' "$1"; }

# ----------------------------------------------------------------------
banner "0  preflight"
# ----------------------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || blocker "not Linux — habitat-sim's audio library is a prebuilt Linux-x64 binary"
command -v nvidia-smi >/dev/null 2>&1 || blocker "nvidia-smi absent — this needs the GPU box"

if ! command -v conda >/dev/null 2>&1; then
  # A RACE pod restart drops the conda CLI from PATH but leaves ~/miniconda3 (runbook §2).
  if [ -x "${HOME}/miniconda3/bin/conda" ]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
    ok "conda hook restored from ~/miniconda3"
  else
    blocker "conda not on PATH and ~/miniconda3/bin/conda absent"
  fi
fi

[ -f "$CONSTRAINTS" ] || blocker "$CONSTRAINTS missing — the pin IS the recipe"
[ -f "$SAVNCE_DIR/setup.py" ] || blocker "submodule empty — run: git submodule update --init earshot/reference/savnce"

if [ -d "$SIM_DIR/.git" ]; then
  have_sha="$(git -C "$SIM_DIR" rev-parse --short HEAD 2>/dev/null)"
  case "$SIM_SHA" in
    "$have_sha"*) ok "habitat-sim source at $have_sha (pinned $SIM_SHA)" ;;
    *) blocker "habitat-sim source at $have_sha, pinned $SIM_SHA — this env must not diverge from ss2's tree" ;;
  esac
else
  blocker "no habitat-sim checkout at $SIM_DIR — run 'nrun bash earshot/tools/bootstrap_ss2.sh' first"
fi

if [ ${#BLOCKERS[@]} -gt 0 ]; then
  banner "STOPPED IN PREFLIGHT"
  printf '  - %s\n' "${BLOCKERS[@]}"
  exit 1
fi

# ----------------------------------------------------------------------
banner "1  the env"
# ----------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  ok "env '$ENV_NAME' exists, reusing"
else
  conda create -n "$ENV_NAME" "python=$PY_VER" "cmake=$CMAKE_VER" -y || blocker "conda create failed"
fi
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)" && conda activate "$ENV_NAME" || { echo "FATAL: cannot activate $ENV_NAME"; exit 1; }
PY="$(command -v python)"
ok "python: $PY"
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# ----------------------------------------------------------------------
banner "2  system deps"
# ----------------------------------------------------------------------
# THE UNION OF TWO LISTS, AND THE UNION IS THE POINT.
#
# The first half is SAVN-CE's INSTALLATION.md verbatim. The second half is
# `bootstrap_ss2.sh`'s line, which is MEASURED on this box against this exact
# habitat-sim SHA. Shipping only SAVN-CE's list cost a full failed compile on
# 2026-08-27: their list has no EGL development headers, and `--headless` builds
# magnum's `platform/egl.cpp`, which dies on `EGL/egl.h: No such file or directory`.
#
# Their list is not wrong; it is written for a machine that already had them. Ours is
# the one with evidence attached, so both go in.
APT_SAVNCE="libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxft-dev libxext-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev libxcb-xinerama0 libx11-xcb-dev"
APT_EARSHOT="libjpeg-dev libglm-dev libgl1 libglx-mesa0 libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev libglvnd-dev"
APT_LINE="sudo apt-get install -y --no-install-recommends $APT_SAVNCE $APT_EARSHOT"
if sudo -n true 2>/dev/null; then
  eval "$APT_LINE" >/dev/null 2>&1 && ok "apt deps installed" || echo "  WARN  apt install returned nonzero; the header check below is the real verdict"
else
  echo "  NOTE  passwordless sudo unavailable. Run this yourself:"
  echo "        $APT_LINE"
fi

# CHECK THE HEADERS, DO NOT TRUST THE INSTALL. apt reported success on the run that
# then failed to compile, because the package that mattered was not in the list at all.
# A header check costs milliseconds; discovering the same fact through cmake costs the
# whole build. Fail here, before stage 3.
MISSING_HEADERS=()
for header in EGL/egl.h GL/gl.h X11/Xlib.h; do
  compgen -G "/usr/include/$header" >/dev/null 2>&1 || compgen -G "/usr/include/*/$header" >/dev/null 2>&1 || MISSING_HEADERS+=("$header")
done
if [ ${#MISSING_HEADERS[@]} -gt 0 ]; then
  blocker "missing development headers: ${MISSING_HEADERS[*]} — a --headless habitat-sim build needs them. Run: $APT_LINE"
  banner "STOPPED BEFORE THE BUILD"
  echo "  Not starting a compile that is already known to fail."
  printf '    - %s\n' "${BLOCKERS[@]}"
  exit 1
fi
ok "development headers present (EGL, GL, X11)"

# CMAKE CACHES NOTFOUND. A build that failed for a missing header records
# `EGL_INCLUDE_DIR-NOTFOUND` and will keep failing after the header is installed,
# because the cache is consulted before the filesystem. So a poisoned cache is wiped
# rather than reused. Object files survive; only the configure step repeats.
CMAKE_CACHE="$SIM_DIR/build/CMakeCache.txt"
if [ -f "$CMAKE_CACHE" ] && grep -q "NOTFOUND" "$CMAKE_CACHE"; then
  rm -f "$CMAKE_CACHE"
  ok "wiped a poisoned CMake cache (it held NOTFOUND entries from a failed configure)"
fi

# ----------------------------------------------------------------------
banner "3  habitat-sim into this env (from ss2's checkout, no second clone)"
# ----------------------------------------------------------------------
audio_probe() {
  python - <<'PY' 2>&1
import sys
try:
    import quaternion  # noqa: F401  must precede habitat_sim (issue #1813)
    import habitat_sim, habitat_sim.sensor
    t = getattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType", None)
    assert t is not None and hasattr(t, "Binaural"), "built WITHOUT --audio"
    print("AUDIO_OK")
except Exception as exc:
    print("AUDIO_FAIL {}: {}".format(type(exc).__name__, exc))
PY
}
if audio_probe | grep -q AUDIO_OK; then
  ok "audio-capable habitat_sim already importable in '$ENV_NAME'"
else
  echo "  building/installing habitat-sim from $SIM_DIR (reuses its cmake cache)"
  ( cd "$SIM_DIR" && python setup.py install --headless --audio ) || blocker "habitat-sim install failed"
  audio_probe | grep -q AUDIO_OK || blocker "habitat_sim still not audio-capable after install"
fi

# ----------------------------------------------------------------------
banner "4  the multi-audio-sensor patch — installed copy only"
# ----------------------------------------------------------------------
INSTALLED_SIM="$SITE_PACKAGES/habitat_sim/simulator.py"
SOURCE_SIM="$SIM_DIR/src_python/habitat_sim/simulator.py"
if [ -f "$INSTALLED_SIM" ]; then
  if grep -qF "$PATCH_TO" "$INSTALLED_SIM"; then
    ok "patch already present in $INSTALLED_SIM"
  elif grep -qF "$PATCH_FROM" "$INSTALLED_SIM"; then
    python - "$INSTALLED_SIM" "$PATCH_FROM" "$PATCH_TO" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path, encoding="utf-8").read()
open(path, "w", encoding="utf-8").write(text.replace(old, new))
PY
    grep -qF "$PATCH_TO" "$INSTALLED_SIM" && ok "patch applied" || blocker "patch did not take in $INSTALLED_SIM"
  else
    blocker "neither the original nor the patched line found in $INSTALLED_SIM — habitat-sim moved, re-read INSTALLATION.md"
  fi
else
  blocker "$INSTALLED_SIM not found — habitat-sim did not install into this env"
fi
# The other half of the assertion. ss2 must not inherit this.
if [ -f "$SOURCE_SIM" ]; then
  if grep -qF "$PATCH_TO" "$SOURCE_SIM"; then
    blocker "SHARED SOURCE TREE IS PATCHED at $SOURCE_SIM — ss2 would silently inherit it; revert with: git -C $SIM_DIR checkout -- src_python/habitat_sim/simulator.py"
  else
    ok "shared source tree is unpatched (ss2 unaffected)"
  fi
fi

# ----------------------------------------------------------------------
banner "5  python deps, in THEIR documented order"
# ----------------------------------------------------------------------
# Order is load-bearing: habitat-sim first (stage 3), then torch, then their
# requirements.txt, which is where numpy lands at 1.26.0. See savnce-constraints.txt.
python -m pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url "$TORCH_INDEX" -c "$CONSTRAINTS" || blocker "torch install failed"
python -m pip install -q -r "$SAVNCE_DIR/requirements.txt" -c "$CONSTRAINTS" || blocker "requirements.txt install failed"
python -m pip install -q -e "$SAVNCE_DIR" -c "$CONSTRAINTS" || blocker "pip install -e savnce failed"

# ----------------------------------------------------------------------
banner "6  the data root"
# ----------------------------------------------------------------------
# Their configs read `data/...` relative to cwd, and cwd is the submodule. The submodule's
# own .gitignore ignores `data/`, so a symlink here is invisible to git on both sides.
mkdir -p "$SAVNCE_DATA_ROOT"
if [ -L "$SAVNCE_DIR/data" ] || [ -d "$SAVNCE_DIR/data" ]; then
  ok "data root already linked: $(readlink -f "$SAVNCE_DIR/data" 2>/dev/null || echo "$SAVNCE_DIR/data")"
else
  ln -s "$SAVNCE_DATA_ROOT" "$SAVNCE_DIR/data" && ok "linked $SAVNCE_DIR/data -> $SAVNCE_DATA_ROOT"
fi
# FOOTGUN, found before the first box trip: INSTALLATION.md draws `datasets/savnce-dataset`
# with a hyphen; savnce_clean.yaml:72 reads `savnce_dataset` with an underscore. The
# config wins, so reconcile rather than let a path silently miss.
HYPHEN="$SAVNCE_DATA_ROOT/datasets/savnce-dataset"
UNDER="$SAVNCE_DATA_ROOT/datasets/savnce_dataset"
if [ -d "$HYPHEN" ] && [ ! -e "$UNDER" ]; then
  ln -s "$HYPHEN" "$UNDER" && ok "reconciled savnce-dataset -> savnce_dataset (their docs and their config disagree)"
fi

# ----------------------------------------------------------------------
banner "7  a scene that needs no licence"
# ----------------------------------------------------------------------
# MP3D proper needs a signed Matterport Terms of Use (savnce_licence_wizard.sh). Habitat
# ships one example MP3D scene freely, which is enough to prove this env renders audio.
if [ -d "$SAVNCE_DATA_ROOT/scene_datasets/mp3d_example" ] || [ -d "$SAVNCE_DATA_ROOT/scene_datasets/mp3d/17DRP5sb8fy" ]; then
  ok "an MP3D scene is already present"
else
  python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene \
    --data-path "$SAVNCE_DATA_ROOT" >/dev/null 2>&1 \
    && ok "downloaded mp3d_example_scene" \
    || echo "  WARN  example-scene download failed (it runs through habitat_sim.utils, so an earlier RED explains it); stage 8 will say so"
fi

# ----------------------------------------------------------------------
banner "8  the verdict — imports, GPU, and audio that is actually rendered"
# ----------------------------------------------------------------------
VERIFY_OUT="$(python "$REPO_ROOT/earshot/tools/savnce_verify.py" --data-root "$SAVNCE_DATA_ROOT" 2>&1)"
echo "$VERIFY_OUT"
echo "$VERIFY_OUT" | grep -q "VERIFY_OK" || blocker "stage 8 verification failed — see its lines above"

# ----------------------------------------------------------------------
banner "verdict"
# ----------------------------------------------------------------------
if [ ${#BLOCKERS[@]} -eq 0 ]; then
  echo "  GREEN — '$ENV_NAME' is ready."
  echo "  Next:  bash earshot/tools/savnce_eval.sh --tag smoke1 --episodes 20"
  exit 0
fi
echo "  RED — ${#BLOCKERS[@]} blocker(s):"
printf '    - %s\n' "${BLOCKERS[@]}"
exit 1
