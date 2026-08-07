#!/usr/bin/env bash
# The climb's threshold, swept — and repeated, because one cell cannot be read.
#
#   nrun bash earshot/tools/eps_sweep.sh --tag eps-1
#
# `detour-3` set the rise threshold to ONE SIGMA of the renderer's measured scatter and
# returned 3/20 source-reached against 7/20, with median walked distance collapsing from
# 9.75 m to 4.00 m. The same run says why: the reached arm's gradient is 2.18e-2 per
# metre, so ~5.5e-3 per 0.25 m step, against a scatter of ~2-3e-3. Per-step signal and
# per-step noise are the SAME ORDER. A threshold at one sigma sits inside the distribution
# of genuine rises and vetoes most of them.
#
# So the question is not "windowed rule or not" — it is where on that axis the knee sits,
# and the axis is `ControllerConfig.rising_eps_scale`, in multiples of the measured
# scatter. 0.0 is the median window working alone (the arm `detour-3` never ran); 1.0 is
# `detour-3` itself and is included as the CONTROL, because a sweep whose worst cell is
# not the known-bad one cannot tell a real curve from a broken harness.
#
# **REPEAT-MAJOR, deliberately.** The outer loop is the repeat and the inner loop is the
# scale, so an interrupted sweep leaves complete passes rather than a finished 0.0 and
# nothing else. Per-episode outcome instability on this scene is ~22% (`detour-1` vs
# `detour-2` vs `detour-3` disagree on which episodes reach), so a single 20-episode cell
# has an error bar of roughly +/- 4 episodes and NO single cell here is a result. Three
# repeats is the minimum that shows a spread; it is not enough for a published number.
#
# CONTINUE-ON-FAILURE, like `yield_sweep.sh`: one dead cell must not cost the other
# fourteen overnight. Continuing is not passing — the exit code is nonzero if any cell
# failed, and the failures are named at the end.
#
# Flags: --tag T (required, fresh), --scenes "a b c" (default ziup5kvtCCR),
#        --scales "0.0 0.45 1.0" (default below), --repeats N (default 3),
#        --n-episodes N (default 20), --max-steps M (default 250).
set -u
set -o pipefail

TAG=""
SCENES="ziup5kvtCCR"
SCALES="0.0 0.25 0.45 0.7 1.0"
REPEATS=3
N_EPISODES=20
MAX_STEPS=250

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)        need_value $# "$1"; TAG="$2";        shift 2 ;;
    --scenes)     need_value $# "$1"; SCENES="$2";     shift 2 ;;
    --scales)     need_value $# "$1"; SCALES="$2";     shift 2 ;;
    --repeats)    need_value $# "$1"; REPEATS="$2";    shift 2 ;;
    --n-episodes) need_value $# "$1"; N_EPISODES="$2"; shift 2 ;;
    --max-steps)  need_value $# "$1"; MAX_STEPS="$2";  shift 2 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done
[ -n "$TAG" ] || { echo "FATAL: --tag is required"; exit 2; }

banner() { printf '\n========== %s ==========\n' "$1"; }

# ONE DIRECTORY IS ONE RUN, checked before any work — the same rule `yield_sweep.sh`
# enforces, and for the same reason: `yield-1` reused its tag and produced a funnel that
# pooled three invocations under one number.
[ -e "runs/$TAG" ] && { echo "FATAL: runs/$TAG exists. Pass a fresh --tag."; exit 2; }
mkdir -p "runs/$TAG"

# Pull ONCE, here, rather than letting each of the fifteen cells re-pull and possibly
# land on a different commit mid-sweep. Every cell then runs `--no-pull`, so the whole
# sweep is one code version and the report can say which.
banner "[1/2] git pull --ff-only"
git pull --ff-only || { echo "FATAL: pull failed"; exit 1; }
COMMIT="$(git rev-parse --short HEAD)"
echo "  commit: $COMMIT"

N_CELLS=0
FAILED=""
banner "[2/2] $(echo "$SCALES" | wc -w) scale(s) x $REPEATS repeat(s)"
echo "  scales:  $SCALES"
echo "  scenes:  $SCENES"
echo "  commit:  $COMMIT"

for repeat in $(seq 1 "$REPEATS"); do
  for scale in $SCALES; do
    # The scale goes in the tag with its dot stripped, so `runs/<tag>/s045-r2` is a
    # directory name that survives every shell and reads back as (scale 0.45, repeat 2).
    cell="s$(echo "$scale" | tr -d '.')-r${repeat}"
    banner "cell $cell — rising_eps_scale=$scale repeat $repeat/$REPEATS"
    N_CELLS=$((N_CELLS + 1))
    if ! bash earshot/tools/yield_sweep.sh \
        --tag "${TAG}-${cell}" --out-dir "runs/$TAG/$cell" \
        --scenes "$SCENES" --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
        --rising-eps-scale "$scale" --no-pull; then
      echo "CELL FAILED: $cell (continuing — the other cells are still worth having)"
      FAILED="$FAILED $cell"
    fi
  done
done

banner "done"
echo "  $N_CELLS cell(s) attempted, commit $COMMIT"
echo "  read each with: python -m earshot.tools.detour_report runs/$TAG/<cell>/<scene>"
echo "  the headline per cell is SOURCE_REACHED in its own summary; compare ACROSS"
echo "  repeats before comparing across scales — one cell is +/- ~4 episodes."
if [ -n "$FAILED" ]; then
  echo "  FAILED cell(s):$FAILED"
  echo "  A sweep missing cells is not a complete sweep; that is why this exits nonzero."
  exit 1
fi
echo "  all cells completed."
