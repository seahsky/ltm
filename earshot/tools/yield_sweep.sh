#!/bin/bash
# earshot/tools/yield_sweep.sh — how much of HM3D can pose the task at all.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/yield_sweep.sh --tag yield-1
#
# Runs the S1 arm across every scene whose mesh is on this box and totals the builder's
# attrition. The number it produces is a DENOMINATOR: §2.1 refuses an episode whose scene
# cannot place an anomaly source far enough from every primary goal, on the same floor as
# both the anchor and the agent's start, at a real view point. Every refusal is correct,
# and the refusal RATE bounds every `n` an experiment matrix can quote. The smoke ran one
# scene and skipped 1 of 2 — a sample of one, consistent with anything.
#
# THIS IS NOT AN ABLATION. There is no memory in the tree (ADR-0008 deferred it), so every
# arm here is S1 and there is no S3 to difference against. It also runs the ORACLE STOP by
# §8's default, so its find numbers are an upper bound with roughly half the failure mass
# deleted — the run says so in its own notes. What it measures is yield, cost per step and
# the funnel at n >> 1, none of which need memory or a detector.
#
# CONTINUE-ON-FAILURE, deliberately: one scene that cannot load must not cost the other
# nineteen. But continuing is not passing — the exit code is NONZERO if any scene failed.
# yield-1 lost 12 of 20 scenes and emailed a green tick, because this script ended in an
# unconditional `exit 0`. CLAUDE.md's rule is that a criterion which could not be
# evaluated is never green, and a sweep missing 60% of its scenes is that rule's case.
#
# ONE DIRECTORY IS ONE RUN, enforced before any work starts. yield-1 reused its tag: the
# per-scene ArtifactExistsError fired correctly on the scenes that already had records,
# the sweep swallowed it, and `yield_report` then globbed the leftovers from the earlier
# invocation in with the fresh ones. The 41% it printed was a pool of two runs with
# nothing on disk saying so, under a line claiming failures were excluded. The pre-flight
# below makes that arithmetic correct by construction rather than by inspection.
#
# Flags: --tag T, --n-episodes N (default 20), --max-steps M (default 250),
#        --scenes "a b c" (default: every scene with a mesh), --category C, --limit N,
#        --out-dir DIR (default runs/<tag>), --no-pull, --force (reuse a non-empty
#        out-dir; mixes runs, and says so in the report).

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
TAG="yield-$(date +%Y%m%d-%H%M%S)"
N_EPISODES=20
MAX_STEPS=250
SCENES=""
CATEGORY=""
LIMIT=0
OUT_DIR=""
NO_PULL=0
FORCE=0

# Captured before the parse loop shifts them away, for `provenance.txt` below. After a
# self-update re-exec these are the RECONSTRUCTED arguments, which is the honest thing to
# record: they are what the body that actually ran was given.
ORIGINAL_ARGS="$*"

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)         need_value $# "$1"; TAG="$2";        shift 2 ;;
    --n-episodes)  need_value $# "$1"; N_EPISODES="$2"; shift 2 ;;
    --max-steps)   need_value $# "$1"; MAX_STEPS="$2";  shift 2 ;;
    --scenes)      need_value $# "$1"; SCENES="$2";     shift 2 ;;
    --category)    need_value $# "$1"; CATEGORY="$2";   shift 2 ;;
    --limit)       need_value $# "$1"; LIMIT="$2";      shift 2 ;;
    --out-dir)     need_value $# "$1"; OUT_DIR="$2";    shift 2 ;;
    --no-pull)     NO_PULL=1;                           shift ;;
    --force)       FORCE=1;                             shift ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done
OUT_DIR="${OUT_DIR:-runs/$TAG}"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- ONE DIRECTORY IS ONE RUN --------------------------------------------
# FIRST, before the pull, the env and the scene discovery, because it depends on nothing
# but OUT_DIR and because the whole cost of getting it wrong is paid at the END: yield-1
# ran for 1h17m and then pooled two invocations into a single 41% under a line claiming
# it had not. Everything under OUT_DIR now came from this run, which is what makes
# `yield_report`'s glob correct by construction rather than by inspection. The per-scene
# ArtifactExistsError stays as the second line of defence; it cannot be the first,
# because by the time it fires the scenes before it have already run.
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  if [ "$FORCE" = 0 ]; then
    echo "FATAL: $OUT_DIR already exists and is not empty."
    echo "       One directory is one run. Re-using it mixes two sweeps into one"
    echo "       aggregate with nothing on disk saying so — which is exactly what"
    echo "       yield-1 did, and its 41% was a pool of two runs."
    echo "       Pass a fresh --tag, or --force if replacing it is the intent."
    exit 1
  fi
  echo "WARN: --force — reusing a non-empty $OUT_DIR. Records from an earlier run will"
  echo "      be pooled into the aggregate below and cannot be told apart."
fi

# --- 1. self-update by re-exec (bash runs the body it loaded, not the file) -
if [ "$NO_PULL" = 0 ]; then
  banner "[1/4] git pull --ff-only"
  _self_before="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
  _self_after="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  if [ -n "$_self_before" ] && [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
    echo "  this script changed in the pull — re-execing the new body"
    export _REEXEC=1
    # `--force` has to survive the re-exec or the new body refuses the out-dir the old
    # body was told to reuse, and a deliberate overwrite turns into a FATAL half a second
    # in. Held as a variable rather than a `$(...)`: a command substitution here runs
    # under `pipefail` and its exit status is the re-exec's.
    _force_flag=""
    [ "$FORCE" = 1 ] && _force_flag="--force"
    exec bash "$0" --tag "$TAG" --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
         ${SCENES:+--scenes "$SCENES"} ${CATEGORY:+--category "$CATEGORY"} \
         --limit "$LIMIT" --out-dir "$OUT_DIR" ${_force_flag:+--force}
  fi
else
  banner "[1/4] git pull SKIPPED (--no-pull)"
fi
echo "  commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# --- 2. the env -----------------------------------------------------------
banner "[2/4] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
# A DIRECTORY check, never `conda env list | grep -q`: under pipefail a matching grep
# exits early, SIGPIPEs conda, and turns found-it into a pipeline failure.
[ -d "$MINICONDA/envs/$ENV_NAME" ] || { echo "FATAL: env '$ENV_NAME' missing — run bootstrap_ss2.sh"; exit 1; }
set +u   # conda's compiler hooks dereference unset CONDA_BACKUP_* vars
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "  python: $(python -V 2>&1)"

[ -d "data/anomaly_audio" ] || {
  echo "FATAL: no data/anomaly_audio — stage the ESC-50 recordings once:"
  echo "       python -m earshot.audio.clips --out-dir data/anomaly_audio"
  exit 1
}

# --- 3. discover the scenes, then run each one ----------------------------
banner "[3/4] scenes"
if [ -z "$SCENES" ]; then
  # Only scenes whose MESH is on this box: `available_scenes` lists content files, and a
  # content file without its .glb is a scene that will fail at load, not a scene with a
  # zero yield. Counting those as refusals would understate the denominator.
  SCENES="$(python - <<'PY'
# The same check task/runner._pick_scene makes, over every label instead of stopping at
# the first hit: load the content file, then test that its resolved mesh is on this box.
# NOTE: no apostrophes or backticks in this heredoc. It sits inside a $( ) command
# substitution, where bash scans for quote pairs even in a quoted heredoc, and a single
# apostrophe in a comment is enough to break the whole script with an EOF error.
# Reusing the loader rather than re-deriving the path layout: the last time this repo
# had two copies of a path rule, one of them quietly stopped being true.
import os
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene

split_dir = find_split_dir("val", ".")
scenes_dir = find_scenes_dir(".")
have = []
for label in available_scenes(split_dir):
    try:
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
    except Exception:
        continue          # unreadable or empty content: not a scene with a zero yield
    if os.path.exists(dataset.scene_path):
        have.append(label)
print(" ".join(have))
PY
)" || { echo "FATAL: scene discovery failed"; exit 1; }
fi
set -- $SCENES
[ "$#" -gt 0 ] || { echo "FATAL: no scenes with a mesh on this box"; exit 1; }
if [ "$LIMIT" -gt 0 ] && [ "$#" -gt "$LIMIT" ]; then
  SCENES="$(echo "$SCENES" | tr ' ' '\n' | head -n "$LIMIT" | tr '\n' ' ')"
  set -- $SCENES
  echo "  --limit $LIMIT applied"
fi
echo "  $# scene(s): $*"
mkdir -p "$OUT_DIR"

# --- what code produced this sweep, recorded INSIDE the sweep -------------
# `env_report.json` carries the resolved config, the interpreter and the conda prefix,
# and NOT the commit. The commit went only to `notify-run.sh`'s log, which lives beside
# `runs/` rather than inside the run and can be deleted on its own — so a finished sweep
# could not say what built it. That cost a real reproduction: repeating `yield-2` meant
# reconstructing its commit from mtimes and the shape of the controller.
#
# Written here rather than from Python ON PURPOSE. The null arm (`repeat-1`) rests on
# `git diff` over `agent/ audio/ sim/ task/ config.py report/` being EMPTY between two
# runs; a provenance write inside `task/runner.py` would put a diff there and turn a
# demonstration into an argument that the diff is inert. A shell line cannot.
#
# It never fails the sweep. A tarball with no `.git` is a legitimate way to run this, and
# an unrecorded commit must read as unknown rather than as absent — CLAUDE.md's rule that
# a criterion which could not be evaluated is never green.
{
  echo "command: $0 $ORIGINAL_ARGS"
  echo "started: $(date -Is)"
  echo "host:    $(hostname)"
  if COMMIT="$(git rev-parse HEAD 2>/dev/null)"; then
    echo "commit:  $COMMIT"
    if git diff --quiet HEAD 2>/dev/null && git diff --cached --quiet HEAD 2>/dev/null; then
      echo "tree:    clean"
    else
      # A dirty tree means the commit does NOT identify the code that ran, so say it
      # here rather than let the sha imply a reproducibility it does not have.
      echo "tree:    DIRTY — the commit above does NOT identify what ran"
      git status --porcelain 2>/dev/null | sed 's/^/         /'
    fi
  else
    echo "commit:  UNKNOWN — no git repository answered here"
    echo "tree:    UNKNOWN"
  fi
} > "$OUT_DIR/provenance.txt"
echo "  provenance: $OUT_DIR/provenance.txt"

# --- criterion 9's evidence, armed once around the whole sweep ------------
# `reset_manifest --verify-absent` is a filesystem existence check over the delete set,
# so it costs milliseconds and there is no reason a sweep should leave criterion 9
# NOT_RUN. NOT_RUN is red (CLAUDE.md), and a criterion that is structurally red on every
# ordinary run is one the reader learns to skip — which is how a never-armed canary read
# as a pass in the first place. Green here means measured, not excused.
HERM_BEFORE="$OUT_DIR/.hermeticity-before.json"
if ! python -m earshot.tools.reset_manifest --verify-absent --when before > "$HERM_BEFORE"; then
  echo "WARN: could not record the pre-run hermeticity check — criterion 9 will be NOT_RUN"
  rm -f "$HERM_BEFORE"
fi

FAILED=""
N_OK=0
GATE_RED=""
ZERO_YIELD=""
for scene in "$@"; do
  banner "[3/4] $scene"
  python -m earshot --run-dir "$OUT_DIR/$scene" --scene "$scene" \
      --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
      ${CATEGORY:+--category "$CATEGORY"}
  ec=$?
  if [ "$ec" -ne 0 ]; then
    # A scene that could place NO episode still writes its summary.json now
    # (EmptyDatasetError carries the build), so it is a measured 0% rather than a
    # scene that broke — yield-1's mL8ThkuaVTM, 99 candidates and none placed, which
    # the old code lost entirely. It still exits nonzero, because a run asked for
    # episodes and produced none; it is listed apart so the FAILED line means "could
    # not be measured" and nothing else.
    if [ -f "$OUT_DIR/$scene/summary.json" ]; then
      echo "  $scene: 0 episode(s) buildable — a measured 0% yield, counted below"
      ZERO_YIELD="$ZERO_YIELD $scene"
    else
      echo "  WARN: $scene exited $ec — continuing"
      FAILED="$FAILED $scene"
    fi
    continue
  fi
  N_OK=$((N_OK + 1))

  # §8's nine over EVERY episode of this scene, not over episode 0. The gate reads the
  # run directory the scene just wrote, so it costs no simulator time — and until now a
  # sweep evaluated none of the nine at all: 20 scenes, 1h17m, zero criteria.
  if [ -f "$HERM_BEFORE" ]; then
    # stderr is NOT suppressed: `--verify-absent` prints "STILL PRESENT: <paths>" there,
    # and that list is the entire diagnostic. A criterion 9 that went red with the reason
    # discarded is the shape of failure this repo keeps paying for.
    python -m earshot.tools.reset_manifest --verify-absent --when after \
        > "$OUT_DIR/$scene/.hermeticity-after.json" \
      && python -m earshot.tools.reset_manifest --write-record \
           --run-dir "$OUT_DIR/$scene" --before "$HERM_BEFORE" \
           --after "$OUT_DIR/$scene/.hermeticity-after.json" \
           --commit "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
           >/dev/null \
      || echo "  WARN: hermeticity incomplete for $scene — criterion 9 will not be green"
  fi
  if ! python -m earshot.task.smoke --run-dir "$OUT_DIR/$scene"; then
    GATE_RED="$GATE_RED $scene"
  fi
done

# --- 4. the number --------------------------------------------------------
banner "[4/4] yield"
python -m earshot.tools.yield_report "$OUT_DIR"
echo
echo "  $N_OK of $# scene(s) completed."
if [ -n "$ZERO_YIELD" ]; then
  echo "  ZERO YIELD (measured, counted in the totals above — the scene cannot pose the"
  echo "  task at all, which is the denominator's most informative point):$ZERO_YIELD"
fi
if [ -n "$FAILED" ]; then
  # NOT "excluded from the totals above" — that line was false, and it was false in the
  # direction that flatters. `yield_report` aggregates every summary.json under OUT_DIR;
  # a scene that failed AFTER writing one is in the table. What the pre-flight guarantees
  # is narrower and true: every record here came from THIS run.
  echo "  FAILED:$FAILED"
  echo "  A scene that failed after writing its summary.json is still counted above."
  echo "  The totals are over the scenes with a record, not over the $# attempted."
fi
if [ -n "$GATE_RED" ]; then
  echo "  SMOKE RED (§8 criteria, tallied over every episode):$GATE_RED"
fi
echo "  records: $OUT_DIR/<scene>/summary.json"
echo "  why a detour ended: python -m earshot.tools.detour_report $OUT_DIR/<scene>"

# CONTINUE-ON-FAILURE is about not abandoning the sweep, not about calling it a success.
# yield-1 lost 12 of 20 scenes, pooled a second run's records into its headline, and
# arrived as a green tick because this line used to read `exit 0`.
#
# ZERO_YIELD is deliberately NOT here. A scene that can pose no episode was *measured*,
# and its record is in the totals — that is the sweep doing its job, not failing at it.
# The rule being kept is "a criterion that could not be evaluated is never green", and a
# yield of zero was evaluated. `yield_report.aggregate` draws the identical line in code:
# `None`, not 0.0, only when nothing was offered.
if [ -n "$FAILED" ] || [ -n "$GATE_RED" ]; then
  exit 1
fi
exit 0
