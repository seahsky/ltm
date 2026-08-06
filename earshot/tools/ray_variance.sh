#!/bin/bash
# earshot/tools/ray_variance.sh — is the run-to-run variance a knob or a fact?
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/ray_variance.sh --tag rays-1
#
# `yield-1` and `detour-1` ran the same scene under the same configuration and 4 of 20
# episodes came out with different outcomes. Both funnels said 8/20 source-reached; they
# were not the same eight. The onset step was identical in all twenty, so the trigger is
# deterministic — the render is not: calibration thresholds moved up to 13%, separation
# 2.5 dB, and the live RMS at the trigger pose 24%. The navmesh `seed` is fixed and does
# not touch this, because it seeds pose DRAWS and the same poses rendered differently.
#
# `spec.ACOUSTICS_PRESET` sets `indirectRayCount` to 500, cut from habitat's 5000 by
# ticket 06 for a 63x speedup. That is a Monte Carlo estimate with a tenth of the samples.
# THE VARIANCE IS THE PRICE OF THAT CUT, and it was paid without being measured. This runs
# the same scene N times at each of several ray counts and reports what buying some of it
# back returns.
#
# WHAT IT DECIDES. If a higher count removes most of the flips, every future number gets
# cheaper and cleaner and the preset should move. If it does not, repeats are mandatory
# and a matrix has to budget for them — which is a different paper's worth of GPU either
# way. There is no threshold here and no verdict: the arms are the arms, and the wall
# clock is the price.
#
# COST. Roughly linear in ray count. `detour-1` measured mean 0.059 s/step at 500 against
# criterion 7's 0.5 s ceiling, so 2500 lands near 0.3 s/step — inside the ceiling, and
# about 20 min per repeat instead of 7.
#
# Flags: --tag T (required-ish; a fresh one, the sweep refuses a reused directory),
#        --scene S (default ziup5kvtCCR — the scene both prior runs used),
#        --rays "500 2500" (the arms), --repeats N (default 3),
#        --n-episodes N (default 20), --max-steps M (default 250), --no-pull.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
TAG="rays-$(date +%Y%m%d-%H%M%S)"
SCENE="ziup5kvtCCR"
RAYS="500 2500"
REPEATS=3
N_EPISODES=20
MAX_STEPS=250
NO_PULL=0

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)         need_value $# "$1"; TAG="$2";        shift 2 ;;
    --scene)       need_value $# "$1"; SCENE="$2";      shift 2 ;;
    --rays)        need_value $# "$1"; RAYS="$2";       shift 2 ;;
    --repeats)     need_value $# "$1"; REPEATS="$2";    shift 2 ;;
    --n-episodes)  need_value $# "$1"; N_EPISODES="$2"; shift 2 ;;
    --max-steps)   need_value $# "$1"; MAX_STEPS="$2";  shift 2 ;;
    --no-pull)     NO_PULL=1;                           shift ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }

# Same rule as the sweep, same reason: one directory is one run. Checked before the pull
# and the env because it depends on nothing else, and because the cost of getting it
# wrong is paid hours later.
for rays in $RAYS; do
  for i in $(seq 1 "$REPEATS"); do
    d="runs/${TAG}-r${rays}-${i}"
    if [ -e "$d" ]; then
      echo "FATAL: $d already exists. One directory is one run — pass a fresh --tag."
      exit 1
    fi
  done
done

if [ "$NO_PULL" = 0 ]; then
  banner "[1/3] git pull --ff-only"
  git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
fi
echo "  commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

banner "[2/3] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
[ -d "$MINICONDA/envs/$ENV_NAME" ] || { echo "FATAL: env '$ENV_NAME' missing"; exit 1; }
set +u
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "  python: $(python -V 2>&1)"

# --- the arms -------------------------------------------------------------
FAILED=""
DIRS=""
for rays in $RAYS; do
  for i in $(seq 1 "$REPEATS"); do
    d="runs/${TAG}-r${rays}-${i}"
    banner "$SCENE at ${rays} rays, repeat ${i}/${REPEATS}"
    # `--seed` is left at its default ON PURPOSE and identically across every arm: it
    # seeds the navmesh pose draws, so holding it fixed is what makes the calibration
    # sweep visit the same poses and leaves the RENDER as the only thing that differs.
    # Varying it here would confound the two.
    python -m earshot --run-dir "$d" --scene "$SCENE" \
        --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
        --indirect-ray-count "$rays"
    ec=$?
    if [ "$ec" -ne 0 ]; then
      echo "  WARN: ${rays} rays repeat ${i} exited $ec — continuing"
      FAILED="$FAILED ${rays}/${i}"
    else
      DIRS="$DIRS $d"
    fi
  done
done

# --- the number -----------------------------------------------------------
banner "[3/3] flips"
if [ -z "$DIRS" ]; then
  echo "  no run completed — nothing to compare"
  exit 1
fi
# shellcheck disable=SC2086
python -m earshot.tools.flip_report $DIRS
echo
echo "  per-run detail: python -m earshot.tools.detour_report runs/${TAG}-r<rays>-<i>"
if [ -n "$FAILED" ]; then
  echo "  FAILED (their arms are short by that many repeats, so their flip rates are"
  echo "  over fewer comparisons than the others):$FAILED"
  exit 1
fi
exit 0
