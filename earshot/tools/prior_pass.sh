#!/bin/bash
# earshot/tools/prior_pass.sh — walk the scripted tour and dump the store `run()` reads.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/prior_pass.sh --tag prior-1 \
#     --scenes "sceneA sceneB sceneC" --classes "toilet_flush snoring keyboard_typing"
#
# WHAT THIS FILLS IN. `window_pilot.sh`'s own header named the gap: "no caller of
# `task/prior_pass.walk_tour` exists outside its own tests, and `task/runner.py` names
# neither `plan_episodes` nor the bank of record." `task/prior_driver.py` is that caller;
# this is its box entrypoint. The output is one file, `<tag>/store.json`, built by
# `memory_build.dump_stores` and read back by `run()`'s new `--memory-store` flag.
#
# --SCENES AND --CLASSES HAVE NO DEFAULT, ON PURPOSE. Which scenes to tour and which
# classes to sound at them is `earshot.tools.anchor_yield`'s decision (the room-balanced
# class-per-scene assignment), not this driver's — a default here would let a sweep run
# against the wrong assignment silently. Pass the assignment `anchor_yield` printed.
#
# ONE TOUR PER SCENE SERVES THE WHOLE CLASS BANK. `class_at_category` reads `--classes`
# per stop, so a scene toured once yields a row for whichever class in the bank anchors
# at each room it has -- pass every class the matrix's assignment uses, in one
# invocation, not one invocation per class.
#
# THE REAL SENSOR, THE REAL ENCODER, EVERY REACHED STOP. `render_embedding_at_stop`
# mirrors `clap_gate.py`'s own render -> bed -> classify chain exactly, because it is the
# same question asked at a different pose: what would the agent's own sensor hand CLAP if
# this class sounded from this stop right now. `prior_pass.py`'s own docstring calls the
# alternative — writing associations straight into the store — the one thing CLAUDE.md's
# "a capability is exercised, never proxied" rule forbids.
#
# CONTINUE-ON-FAILURE AT THE SCENE GRAIN, same rule `clap_gate.run_gate` and
# `yield_sweep.sh` already hold. A scene that cannot load, or whose tour comes back
# incomplete, is recorded and excluded from the merged store -- `TourRecord.complete`'s
# own docstring is explicit that a partial tour must never read as a seen scene. Nonzero
# exit only if EVERY scene failed; the per-scene provenance in `store.json` names which
# ones did.
#
# ONE DIRECTORY IS ONE RUN, enforced before any work starts, for yield-1's reason.
#
# Flags: --tag T (required), --scenes "a b c" (required), --classes "a b c" (required),
#        --split S (default val), --data-root D (default .), --seed N (default 20260821),
#        --leg-budget N (default 200), --goal-radius M (default 1.0), --no-pull, --overwrite.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -u

TAG=""
SCENES=""
CLASSES=""
SPLIT="val"
DATA_ROOT="."
SEED=20260821
LEG_BUDGET=200
GOAL_RADIUS=1.0
PULL=1
OVERWRITE=0

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value" >&2; exit 2; }; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)         need_value $# "$1"; TAG="$2";         shift 2 ;;
    --scenes)      need_value $# "$1"; SCENES="$2";      shift 2 ;;
    --classes)     need_value $# "$1"; CLASSES="$2";     shift 2 ;;
    --split)       need_value $# "$1"; SPLIT="$2";       shift 2 ;;
    --data-root)   need_value $# "$1"; DATA_ROOT="$2";   shift 2 ;;
    --seed)        need_value $# "$1"; SEED="$2";        shift 2 ;;
    --leg-budget)  need_value $# "$1"; LEG_BUDGET="$2";  shift 2 ;;
    --goal-radius) need_value $# "$1"; GOAL_RADIUS="$2"; shift 2 ;;
    --no-pull)     PULL=0; shift ;;
    --overwrite)   OVERWRITE=1; shift ;;
    *) echo "FATAL: unknown flag $1" >&2; exit 2 ;;
  esac
done

[ -n "$TAG" ] || { echo "FATAL: --tag is required" >&2; exit 2; }
[ -n "$SCENES" ] || { echo "FATAL: --scenes is required — pass anchor_yield's own assignment" >&2; exit 2; }
[ -n "$CLASSES" ] || { echo "FATAL: --classes is required — the bank anchor_yield assigned" >&2; exit 2; }

OUT_DIR="runs/$TAG"
banner() { echo; echo "=== $* ==="; }

# --- 1. one directory is one run, checked FIRST ---------------------------
banner "[1/3] preflight"
if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ] && [ "$OVERWRITE" -eq 0 ]; then
  echo "FATAL: $OUT_DIR already exists and is not empty."
  echo "  One directory is one run. Pick a fresh --tag, or pass --overwrite if replacing"
  echo "  it is the intent."
  exit 2
fi
mkdir -p "$OUT_DIR" || { echo "FATAL: cannot create $OUT_DIR"; exit 2; }

if [ "$PULL" -eq 1 ]; then
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 2; }
fi

# Provenance, written before the work rather than after it: a finished pass that cannot
# say what code and what assignment produced it is the failure 6561434 fixed.
{
  echo "tag=$TAG"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "split=$SPLIT data_root=$DATA_ROOT"
  echo "scenes=$SCENES"
  echo "classes=$CLASSES"
  echo "seed=$SEED leg_budget=$LEG_BUDGET goal_radius=$GOAL_RADIUS"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/provenance.txt"
cat "$OUT_DIR/provenance.txt"

# --- 2. the CLAP checkpoint ------------------------------------------------
# Idempotent — a second run prints "already staged" and returns — so it is safe to leave
# in the path rather than trusted to have run before this script was called.
banner "[2/3] CLAP checkpoint"
python -m earshot.task.models \
  || { echo "FATAL: could not stage the CLAP checkpoint — the tour cannot render"; exit 2; }

# --- 3. the tour -------------------------------------------------------
banner "[3/3] touring $SCENES for $CLASSES"
python -m earshot.task.prior_driver \
  --run-dir "$OUT_DIR" \
  --split "$SPLIT" \
  --data-root "$DATA_ROOT" \
  --scenes "$SCENES" \
  --classes "$CLASSES" \
  --seed "$SEED" \
  --leg-budget "$LEG_BUDGET" \
  --goal-radius "$GOAL_RADIUS" \
  $([ "$OVERWRITE" -eq 1 ] && echo --overwrite) \
  2>&1 | tee "$OUT_DIR/prior_pass.log"
PASS_STATUS=${PIPESTATUS[0]}

banner "done"
echo "  artefacts: $OUT_DIR/{provenance.txt,store.json,prior_pass.log}"
echo "  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$PASS_STATUS" -ne 0 ]; then
  echo "  EXIT NONZERO: every scene's tour was incomplete or failed to load."
fi
exit "$PASS_STATUS"
