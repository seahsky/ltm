#!/bin/bash
# earshot/tools/leg_probe.sh — does the level fall along a walking leg? TC on and off.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/leg_probe.sh --tag walk-1
#   python -m earshot.tools.leg_probe read runs/walk-1     # re-read later: off-box, seconds
#
# WHY THIS EXISTS. The leg replay found the cue falling along about three quarters of
# sounding cast legs whichever way they walked, by a median 9.0% per loop (PR #152).
# That is the number the controller reads: `is_rising` compares five readings against
# five along a leg. The hold probe (`hold-2`, PR #159) rendered the STANDING first scans
# again and read them flat in both arms, which ruled the preset, the renderer at a fixed
# pose, the pipeline and the heading out for a standing agent and not for a walking one.
# Its only moving sequence was also its only leaning row: -2.6% per loop with the preset
# on against -0.9% with it off, p = 0.05. This walks. `leg_probe.py`'s docstring is the
# design.
#
# WHAT IT RUNS, IN ORDER.
#   1. preflight: one directory is one run, checked first, then `git pull --ff-only`.
#   2. select: the legs off the runs named by --from. Read-only, seconds, and it refuses
#      before any render if a run is not `full`'s rule.
#   3. render tc1: the shipped preset, first, so a crash later still leaves the arm every
#      run used.
#   4. render tc0: `temporalCoherence` off, in its own process.
#   5. read: the table and the branch.
#
# THE BRANCH, PRE-REGISTERED IN leg_probe.py BEFORE THIS FIRST RAN. Read `BRANCH:` at the
# end. RENDERER: the level falls along the leg whatever the preset. PRESET: it falls with
# TC 1 alone, and the next run is a sweep. MIXED: read the table. SELECTION: the legs'
# fall does not reproduce either, and `is_rising`'s own cut is what both populations
# share. NOT_RUN is red and exits nonzero.
#
# THE DEFAULT RUNS. `oracle-2/full` then `oracle-1/full`: `full` at identical behaviour.
# One leg per episode, the episode's first completed sounding one, and the first run
# named wins it. The clip, the split and the audio configuration come from each scene's
# `env_report.json`, and the runs are refused unless they all agree.
#
# COST, AN ESTIMATE AND NOT A MEASUREMENT. Roughly 80 to 100 legs, walk-in 10 and a leg
# of 9 readings, is about 19 renders a pose an arm: about 2k renders an arm at ticket
# 06's 27 ms. One World per scene, so about 36 scene loads over both arms. Well under
# half an hour for both arms, and cheaper than the hold probe.
#
# Flags: --tag T (required), --from "A B" (default "runs/oracle-2/full runs/oracle-1/full"),
#        --walk-in N (default 10), --scenes "A,B" (default all),
#        --data-root D (default .), --no-pull.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -u

TAG=""
FROM="runs/oracle-2/full runs/oracle-1/full"
WALK_IN=10
SCENES=""
DATA_ROOT="."
PULL=1

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value" >&2; exit 2; }; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)            need_value $# "$1"; TAG="$2";           shift 2 ;;
    --from)           need_value $# "$1"; FROM="$2";          shift 2 ;;
    --walk-in)        need_value $# "$1"; WALK_IN="$2";       shift 2 ;;
    --scenes)         need_value $# "$1"; SCENES="$2";        shift 2 ;;
    --data-root)      need_value $# "$1"; DATA_ROOT="$2";     shift 2 ;;
    --no-pull)        PULL=0; shift ;;
    *) echo "FATAL: unknown flag $1" >&2; exit 2 ;;
  esac
done

[ -n "$TAG" ] || { echo "FATAL: --tag is required" >&2; exit 2; }

OUT_DIR="runs/$TAG"
banner() { echo; echo "=== $* ==="; }

# --- 1. one directory is one run, checked FIRST ---------------------------
banner "[1/5] preflight"
if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  echo "FATAL: $OUT_DIR already exists and is not empty. One directory is one run:"
  echo "  pick a fresh --tag. A finished run is re-read with"
  echo "  python -m earshot.tools.leg_probe read $OUT_DIR"
  exit 2
fi
mkdir -p "$OUT_DIR" || { echo "FATAL: cannot create $OUT_DIR"; exit 2; }

if [ "$PULL" -eq 1 ]; then
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 2; }
fi

# shellcheck disable=SC2206  # FROM is a space-separated list of paths with no spaces in them
RUNS=($FROM)
SELECT_ARGS=("${RUNS[@]}" --walk-in "$WALK_IN")
[ -n "$SCENES" ] && SELECT_ARGS+=(--scenes "$SCENES")
RENDER_ARGS=("${SELECT_ARGS[@]}" --out "$OUT_DIR" --data-root "$DATA_ROOT")

{
  echo "tag=$TAG"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "from=$FROM"
  echo "walk_in=$WALK_IN scenes=${SCENES:-all} data_root=$DATA_ROOT"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/provenance.txt"
cat "$OUT_DIR/provenance.txt"

# --- 2. the legs, refused here rather than after a scene load -------------
banner "[2/5] select"
python -m earshot.tools.leg_probe select "${SELECT_ARGS[@]}" 2>&1 | tee "$OUT_DIR/select.txt"
STATUS=${PIPESTATUS[0]}
[ "$STATUS" -eq 0 ] || { echo "FATAL: select exited $STATUS — see $OUT_DIR/select.txt"; exit "$STATUS"; }

# --- 3 and 4. the two arms, each in its own process -----------------------
STEP=3
for ARM in tc1 tc0; do
  banner "[$STEP/5] render $ARM"
  python -m earshot.tools.leg_probe render "${RENDER_ARGS[@]}" --arm "$ARM" \
    2>&1 | tee "$OUT_DIR/render-$ARM.log"
  STATUS=${PIPESTATUS[0]}
  [ "$STATUS" -eq 0 ] || { echo "FATAL: render $ARM exited $STATUS — see $OUT_DIR/render-$ARM.log"; exit "$STATUS"; }
  STEP=$((STEP + 1))
done

# --- 5. the readout -------------------------------------------------------
banner "[5/5] read"
python -m earshot.tools.leg_probe read "$OUT_DIR" 2>&1 | tee "$OUT_DIR/readout.txt"
STATUS=${PIPESTATUS[0]}

echo
echo "  run:     $OUT_DIR"
echo "  readout: $OUT_DIR/readout.txt"
echo
echo "  READ THE BRANCH LINE. The walk rows say whether the recorded fall survives a"
echo "  re-render of the same leg, and the two arms whether temporalCoherence makes it."
echo "  The CHECK row must read FALLS in both arms, or the branch is NOT_RUN."

exit "$STATUS"
