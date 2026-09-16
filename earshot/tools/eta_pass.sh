#!/bin/bash
# earshot/tools/eta_pass.sh — ADR-0024 step 1: price `eta` on one scene, before booking a night.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/eta_pass.sh --tag eta-1
#   python -m earshot.tools.dream_report runs/eta-1
#
# WHY THIS EXISTS. `dream-2` set `eta = 0.5` against the box's EMPTY-memory `I_j` range of
# 1.0000-1.2901, which is a regime that occurs exactly once per chain: `novelty` returns
# 1.0 for every segment of an empty store. With `M^E` non-empty its keys sat at 0.909 to
# 0.989 cosine, `N_j` fell into [0.011, 0.091], and `max I_j` was about 0.38 against a
# threshold of 0.5. The gate never opened again: 275 of 282 episodes retained NOTHING and
# 38 of the arm's 45 rows came from one walk in one house. ADR-0024 fixed the cause --
# `C_j` and `U_j` were shares carrying a hidden `1/J` that `N_j` does not -- so
# `mean(C_j + U_j)` is now exactly 2.0 for ANY `J` and any memory state.
#
# THAT MADE `eta` MEANINGFUL AND LEFT IT UNPRICED. The shipped 2.0 reads as "this segment
# carried more than an average segment's worth", which is stationary in a way 0.5 never
# was, and it is still a guess about a distribution nobody has measured. A synthetic probe
# over 250-step trajectories differing only in whether progress arrives evenly or in
# bursts had 9 to 14 of 21 segments clearing 2.0, and 0 to 3 clearing 3.0. That spread
# over fixtures that similar is the finding: THE DISTRIBUTION IS NOT PREDICTABLE OFF-BOX.
#
# WHAT TO READ, AND THE BRANCH IT DECIDES. `dream_report` section C prints
# `dream_segments_over_eta`, which `runner.py` writes on every episode: how many segments
# cleared `eta` BEFORE the cap. Set `eta` so its median sits BELOW `--max-retained`.
#
#   * median BELOW the cap  -> `eta` is the retention rule and the cap is the bound it was
#     added to be. Proceed to ADR-0024 step 2.
#   * median AT OR ABOVE the cap on most episodes -> THE CAP IS THE RETENTION RULE and
#     `eta` is decoration. That is the top-k deviation ADR-0024 declined to ship, and it
#     needs its own ADR rather than a quietly-raised cap. STOP AND SAY SO.
#   * `rows added` still 0 nearly everywhere -> `eta` is too high; the rescale did not
#     open the gate and the diagnosis is incomplete.
#
# ONE SCENE, NOT A SWEEP, AND THAT IS THE POINT. A reportable DREAM comparison costs
# 14h15m (three repeats an arm at n=282, `power.py`'s 80%-power figure, not ADR-0016's
# 2-sigma one). This is minutes, and it can rule the whole line of work out before any of
# that is spent. `--n-episodes 15` matches one scene's share of the ablation sweep, so the
# `I_j` distribution it measures is the one a real arm would see.
#
# THE KNOBS ARE `ablation_sweep.sh`'s `DREAM_KNOBS`, VERBATIM EXCEPT THE TWO BEING PRICED.
# They are duplicated rather than sourced because that driver defines them inside a script
# that runs a sweep on sight; a `--dream-eta`/`--dream-max-retained` passed here overrides
# only those two, so every other number this measures against is the one the sweep uses.
# `tests/mac/test_eta_pass.py` holds the two lists against each other, so a knob added to
# the sweep and not to this file is a test failure rather than a silent divergence.
#
# ONE DIRECTORY IS ONE RUN, enforced before any work starts, for `yield-1`'s reason.
#
# Flags: --tag T (required), --scene S (default 4ok3usBNeis, the sweep's first scene and
#        the one `dream-2` built its 38-row flood in), --eta E (default 2.0),
#        --max-retained N (default 12), --n-episodes N (default 15),
#        --max-steps N (default 250), --split S (default val), --data-root D (default .),
#        --seed N (default 20260821), --no-pull, --overwrite.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -u

TAG=""
SCENE="4ok3usBNeis"
ETA="2.0"
MAX_RETAINED="12"
N_EPISODES=15
MAX_STEPS=250
SPLIT="val"
DATA_ROOT="."
SEED=20260821
PULL=1
OVERWRITE=0

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value" >&2; exit 2; }; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)           need_value $# "$1"; TAG="$2";          shift 2 ;;
    --scene)         need_value $# "$1"; SCENE="$2";        shift 2 ;;
    --eta)           need_value $# "$1"; ETA="$2";          shift 2 ;;
    --max-retained)  need_value $# "$1"; MAX_RETAINED="$2"; shift 2 ;;
    --n-episodes)    need_value $# "$1"; N_EPISODES="$2";   shift 2 ;;
    --max-steps)     need_value $# "$1"; MAX_STEPS="$2";    shift 2 ;;
    --split)         need_value $# "$1"; SPLIT="$2";        shift 2 ;;
    --data-root)     need_value $# "$1"; DATA_ROOT="$2";    shift 2 ;;
    --seed)          need_value $# "$1"; SEED="$2";         shift 2 ;;
    --no-pull)       PULL=0; shift ;;
    --overwrite)     OVERWRITE=1; shift ;;
    *) echo "FATAL: unknown flag $1" >&2; exit 2 ;;
  esac
done

[ -n "$TAG" ] || { echo "FATAL: --tag is required" >&2; exit 2; }

OUT_DIR="runs/$TAG"
banner() { echo; echo "=== $* ==="; }

# The sweep's knobs. `eta` and `max-retained` are the two this run prices, so they come
# from the flags above; every other number is `ablation_sweep.sh`'s, held identical by
# `tests/mac/test_eta_pass.py` so the measurement is against the arm's own configuration.
DREAM_KNOBS="--dream \
  --dream-stm-horizon 8 --dream-stm-decay 0.8 --dream-present-weight 0.7 \
  --dream-coherence 0.99 --dream-min-segment 3 --dream-max-segment 12 \
  --dream-alpha 1.0 --dream-beta 1.0 --dream-gamma 1.0 \
  --dream-eta $ETA --dream-max-retained $MAX_RETAINED \
  --dream-min-support 2 \
  --dream-k-experience 3 --dream-k-pattern 2 --dream-k-knowledge 1 \
  --dream-temperature 0.5 \
  --dream-lambda-plan 1.0 --dream-lambda-memory 0.5 --dream-lambda-feasibility 0.5"

# --- 1. one directory is one run, checked FIRST ---------------------------
banner "[1/4] preflight"
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

{
  echo "tag=$TAG"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "scene=$SCENE split=$SPLIT data_root=$DATA_ROOT"
  echo "eta=$ETA max_retained=$MAX_RETAINED"
  echo "n_episodes=$N_EPISODES max_steps=$MAX_STEPS seed=$SEED"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/provenance.txt"
cat "$OUT_DIR/provenance.txt"

# --- 2. the checkpoints ---------------------------------------------------
# Idempotent — a second run prints "already staged" and returns — so it is safe to leave
# in the path rather than trusted to have run before this script was called. `--dream`
# needs CLAP for `--clap` and CLIP for `f_v`, and a missing one fails 40 steps in rather
# than here, which is the shape `prior_pass.sh` already removed from its own path.
banner "[2/4] CLAP and CLIP checkpoints"
python -m earshot.task.models \
  || { echo "FATAL: could not stage the checkpoints — the DREAM arm cannot run"; exit 2; }

# --- 3. the episodes ------------------------------------------------------
banner "[3/4] $N_EPISODES episode(s) in $SCENE at eta=$ETA max_retained=$MAX_RETAINED"
# shellcheck disable=SC2086  # DREAM_KNOBS is a literal flag list with no spaces in any value
python -m earshot \
  --run-dir "$OUT_DIR" \
  --scene "$SCENE" \
  --split "$SPLIT" \
  --data-root "$DATA_ROOT" \
  --n-episodes "$N_EPISODES" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED" \
  --clap \
  $DREAM_KNOBS \
  2>&1 | tee "$OUT_DIR/eta_pass.log"
RUN_STATUS=${PIPESTATUS[0]}
[ "$RUN_STATUS" -eq 0 ] || { echo "FATAL: the run exited $RUN_STATUS — see $OUT_DIR/eta_pass.log"; exit "$RUN_STATUS"; }

# --- 4. the readout -------------------------------------------------------
# The whole reason this driver exists is that the NUMBERS decide the next spend, so it
# prints them rather than leaving a second command to remember. Read-only, seconds.
banner "[4/4] what eta did"
python -m earshot.tools.dream_report "$OUT_DIR" 2>&1 | tee "$OUT_DIR/dream_report.txt"
REPORT_STATUS=${PIPESTATUS[0]}

echo
echo "  run:    $OUT_DIR"
echo "  log:    $OUT_DIR/eta_pass.log"
echo "  report: $OUT_DIR/dream_report.txt"
echo
echo "  READ SECTION C. It prints dream_segments_over_eta beside the cap ($MAX_RETAINED) and"
echo "  NAMES which of the two retained. \"THE CAP IS THE RETENTION RULE\" means eta is"
echo "  decoration — ADR-0024's top-k deviation, which needs its own ADR rather than a"
echo "  quietly-raised cap. Raise eta until the cap binds on a minority."
echo
echo "  This prices a knob. It is NOT a measurement of whether DREAM helps: one scene,"
echo "  one run, and repeat-1 measured a 16.2% outcome flip rate on byte-identical reruns."

exit "$REPORT_STATUS"
