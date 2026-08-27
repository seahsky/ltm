#!/bin/bash
# earshot/tools/habitat_sim_toolchain.sh — the environment a 2022-era habitat-sim needs
# in order to compile at all. Sourced, not executed.
#
#   source earshot/tools/habitat_sim_toolchain.sh
#   habitat_sim_toolchain_setup "$BUILD_ROOT" "$REPO_ROOT"
#
# EVERY LINE HERE COST A BOX TRIP, and all four were learned by `bootstrap_ss2.sh`
# before this file existed. It is extracted so the SAVN-CE env stops reinventing them:
# the first savnce build failed on a missing EGL header, and the second failed on
# `'std::uint32_t' has not been declared` — the GCC 13 signature, because Ubuntu 24.04's
# default compiler is a decade newer than this tree.
#
# ON THE DUPLICATION, DELIBERATELY: `bootstrap_ss2.sh` still carries its own inline copy
# and is NOT changed to source this file yet. Two copies of a build recipe is a drift
# trap and that is a real cost, accepted here for a specific reason — `ss2` is the
# working environment every earshot result depends on, this file has never run, and
# editing an untestable-from-here build script to depend on new untested code is the
# wrong order. Fold `bootstrap_ss2.sh` into this once the savnce build is green, which
# is a build we can afford to have fail.

# shellcheck shell=bash

habitat_sim_toolchain_setup() {
  local build_root="$1"
  local repo_root="$2"

  # --- 1. the compiler -------------------------------------------------------
  # The only attested toolchain for this tree is gcc 7-10. GCC 13 dropped the
  # transitive <cstdint> includes that corrade relies on, so a modern system compiler
  # fails with 'std::uint32_t has not been declared' deep in Corrade/Utility.
  if [ ! -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
    set +u   # the install reactivates the env -> sources compiler hooks
    conda install -y -q -c conda-forge 'gcc_linux-64=10.*' 'gxx_linux-64=10.*' sysroot_linux-64 \
      || echo "  WARN  conda gcc-10 install failed — the build will use system gcc $(gcc -dumpversion 2>/dev/null || echo '?') and is expected to fail"
    set -u
  fi
  if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    echo "  OK    toolchain: conda gcc $("$CC" -dumpversion)"
  else
    echo "  WARN  toolchain: system gcc $(gcc -dumpversion 2>/dev/null || echo '?') — this tree wants 7-10"
  fi

  # --- 2. where cmake looks for the GL stack ---------------------------------
  # The conda cross-toolchain's triplet stops cmake searching Ubuntu's multiarch dir,
  # so magnum's find_package(OpenGL) misses the GLVND libs even with libglvnd-dev
  # installed.
  export CMAKE_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
  export CMAKE_INCLUDE_PATH="/usr/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"

  # --- 3. the include shim ---------------------------------------------------
  # COMPILE-time counterpart to the above: magnum's EGL object library includes
  # <EGL/egl.h> via the compiler's DEFAULT include path, which for the conda gcc is its
  # sysroot (glibc + kernel headers only). Expose ONLY the GL header trees, deliberately
  # NOT all of /usr/include, which would shadow the sysroot's glibc headers with the
  # host's newer ones and reintroduce the GCC-13 class of failure by another route.
  local shim="$build_root/include-shim"
  mkdir -p "$shim"
  local d
  for d in EGL KHR GL X11; do
    [ -d "/usr/include/$d" ] && ln -sfn "/usr/include/$d" "$shim/$d"
  done
  export CPATH="$shim${CPATH:+:$CPATH}"

  # --- 4. how many compilers to start ----------------------------------------
  # NOT unbounded `-j`, which is habitat-sim's own default and WHICH CRASHED THE BOX.
  # `nproc` ignores a cgroup CPU quota, so a four-CPU pod on a large node reports the
  # node's count; and core count is the wrong budget anyway, because magnum's and
  # bullet's heavy translation units peak in the low gigabytes each and memory runs out
  # first. A build that OOMs the host does not fail, it takes the host down, and `nrun`
  # then sends no mail — the symptom is silence, not a red run.
  local jobs
  jobs="$(python "$repo_root/earshot/tools/build_jobs.py" 2>/dev/null)"
  case "${jobs:-}" in
    ''|*[!0-9]*)
      echo "  WARN  could not size the build — falling back to 1 job"
      jobs=1 ;;
  esac
  export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
  echo "  OK    build parallelism: $CMAKE_BUILD_PARALLEL_LEVEL job(s)"
}
