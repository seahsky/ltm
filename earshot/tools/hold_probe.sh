#!/bin/bash
# earshot/tools/hold_probe.sh — does the cue fall while the agent stands still? TC on and off.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/hold_probe.sh --tag hold-1
#   python -m earshot.tools.hold_probe read runs/hold-1     # re-read later: off-box, seconds
#
# WHY THIS EXISTS. The leg replay (PRs #149 to #156) found the cue falling along about
# three quarters of sounding cast legs whichever way they walked, and falling in the
# detour's first scan too (83.7% of 184, median -11.1% per loop). PR #156 read that as a
# trend in time. Selection (a scan the level rose through is cut by a surge), heading (a
# scan's same-phase pairs are five turns apart) and the render preset
# (`temporalCoherence: 1`, never A/B'd against a leg) all stand between that reading and a
# cause. This renders the recorded first scans again, three ways, with the preset on and
# off, and nothing else changes. `hold_probe.py`'s docstring is the design.
#
# WHAT IT RUNS, IN ORDER.
#   1. preflight: one directory is one run, checked first, then `git pull --ff-only`.
#   2. select: the poses off the runs named by --from. Read-only, seconds, and it refuses
#      before any render if a run is not `full`'s rule.
#   3. render tc1: the shipped preset. First, so a crash later still leaves the arm every
#      run used.
#   4. render tc0: `temporalCoherence` off, in its own process.
#   5. read: the table and the branch.
#
# THE BRANCH, PRE-REGISTERED IN hold_probe.py BEFORE THIS FIRST RAN. Read `BRANCH:` at the
# end. The hold, at a fixed pose and heading, is read before the turning replay.
# PRESET: run `full` with TC off beside `full`. RENDERER: the renderer drifts at a fixed
# pose whatever the preset. PIPELINE: a defect after the renderer, find it first.
# HEADING: the scan's turns are the fall. SELECTION: the fall does not reproduce with
# nothing cut. MIXED: read the table. NOT_RUN is red and exits nonzero.
#
# THE DEFAULT RUNS. `oracle-2/full` then `oracle-1/full`: `full` at identical behaviour.
# One pose per episode, and the first run named wins it, so the second only adds the
# episodes whose first scan the first did not read. The clip, the split and the audio
# configuration come from each scene's `env_report.json`, and the runs are refused unless
# they all agree.
#
# COST, AN ESTIMATE AND NOT A MEASUREMENT. Roughly 60 to 90 poses, walk-in 10 and hold 60,
# is about 148 renders a pose an arm: about 13k renders an arm at ticket 06's 27 ms. Each
# of the three sequence kinds opens its own World, so three scene loads a scene an arm,
# about 114 over both arms. Under an hour for both arms.
#
# Flags: --tag T (required), --from "A B" (default "runs/oracle-2/full runs/oracle-1/full"),
#        --walk-in N (default 10), --hold N (default 60), --scenes "A,B" (default all),
#        --data-root D (default .), --no-pull.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -u

TAG=""
FROM="runs/oracle-2/full runs/oracle-1/full"
WALK_IN=10
HOLD=60
SCENES=""
DATA_ROOT="."
PULL=1

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value" >&2; exit 2; }; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)            need_value $# "$1"; TAG="$2";           shift 2 ;;
    --from)           need_value $# "$1"; FROM="$2";          shift 2 ;;
    --walk-in)        need_value $# "$1"; WALK_IN="$2";       shift 2 ;;
    --hold)           need_value $# "$1"; HOLD="$2";          shift 2 ;;
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
  echo "  python -m earshot.tools.hold_probe read $OUT_DIR"
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
RENDER_ARGS=("${SELECT_ARGS[@]}" --out "$OUT_DIR" --hold "$HOLD" --data-root "$DATA_ROOT")

{
  echo "tag=$TAG"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "from=$FROM"
  echo "walk_in=$WALK_IN hold=$HOLD scenes=${SCENES:-all} data_root=$DATA_ROOT"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/provenance.txt"
cat "$OUT_DIR/provenance.txt"

# --- 2. the poses, refused here rather than after a scene load ------------
banner "[2/5] select"
python -m earshot.tools.hold_probe select "${SELECT_ARGS[@]}" 2>&1 | tee "$OUT_DIR/select.txt"
STATUS=${PIPESTATUS[0]}
[ "$STATUS" -eq 0 ] || { echo "FATAL: select exited $STATUS — see $OUT_DIR/select.txt"; exit "$STATUS"; }

# --- 3 and 4. the two arms, each in its own process -----------------------
STEP=3
for ARM in tc1 tc0; do
  banner "[$STEP/5] render $ARM"
  python -m earshot.tools.hold_probe render "${RENDER_ARGS[@]}" --arm "$ARM" \
    2>&1 | tee "$OUT_DIR/render-$ARM.log"
  STATUS=${PIPESTATUS[0]}
  [ "$STATUS" -eq 0 ] || { echo "FATAL: render $ARM exited $STATUS — see $OUT_DIR/render-$ARM.log"; exit "$STATUS"; }
  STEP=$((STEP + 1))
done

# --- 5. the readout -------------------------------------------------------
banner "[5/5] read"
python -m earshot.tools.hold_probe read "$OUT_DIR" 2>&1 | tee "$OUT_DIR/readout.txt"
STATUS=${PIPESTATUS[0]}

echo
echo "  run:     $OUT_DIR"
echo "  readout: $OUT_DIR/readout.txt"
echo
echo "  READ THE BRANCH LINE. The rescan rows say whether the fall survives with nothing"
echo "  cut, the hold rows whether it survives at a fixed heading, and the two arms whether"
echo "  temporalCoherence makes it. The CHECK rows must read as the docstring says, or the"
echo "  branch is NOT_RUN."

exit "$STATUS"
