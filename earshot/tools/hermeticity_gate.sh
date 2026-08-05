#!/bin/bash
# earshot/tools/hermeticity_gate.sh — ticket 10 phase 2, the gate the irreversible
# deletion commit hangs off. Run on the RACE V100, in the `ss2` env.
#
#   git fetch && git checkout <branch>
#   source earshot/tools/notify/notify-run.sh && nrun bash earshot/tools/hermeticity_gate.sh
#
# WHAT IT DOES: moves everything phase 3 deletes OUT of the repo, re-runs the smoke with
# them gone, moves them back, and writes the evidence into the run directory so criterion
# 9 is answered by an artefact rather than by the operator's memory.
#
# WHY THIS AND NOT A GREP: a static scan misses `importlib`, a `sys.path` append, a
# hardcoded "embodied_memory/…" in a config, and a data path only the old tree knew
# about. Ticket 10 priced the alternative at one smoke run and took it.
#
# WHY IT MOVES MORE THAN TICKET 10 SAYS: phase 2 as written moves the two old trees;
# phase 3 deletes three more groups. `earshot/tools/reset_manifest.py` is the one list,
# and it covers all five — see its docstring. The widening is not hypothetical: ticket 27
# found the carried `earshot/tools/notify/` trio still executing `$REPO_ROOT/scripts/…`,
# inside a group the narrow gate would not have moved, so a green gate would have been
# followed by a deletion that broke the box's own launcher.
#
# RESTORE IS THE SAFETY PROPERTY, and it has three layers:
#   1. an EXIT trap, so a failure, a Ctrl-C or a SIGTERM restores;
#   2. `git status` is verified clean afterwards, and a mismatch is shouted, not logged;
#   3. every moved path is TRACKED, so even a SIGKILL leaves a one-line recovery:
#        git checkout -- embodied_memory dialogue_memory scripts README_LTM_MSC_EVAL.md …
#      (printed below before anything moves, so it is in the log the operator has).
#
# It refuses to run on a dirty tree. Not fussiness: an uncommitted edit inside a moved
# path is indistinguishable from a restore failure afterwards, and the recovery above
# would silently discard it.
#
# THE MOVE AND THE RESTORE NEED NO BOX, so they are not box-only: `--dry-run` performs
# them and stops, and `tests/mac/test_hermeticity_gate.py` drives it against a scratch
# git repo — including `--self-test-abort`, which kills the run at the one moment the
# repo is taken apart. ADR-0014: a detector ships both arms, and the arm that matters
# here is the failing one. Only the env, the box suite and the smoke are box-only.
#
# Flags: --run-dir DIR (default runs/hermetic-<ts>), --branch B, --skip-box-tests,
#        --max-steps N, --tag T (so `nrun … --tag t` names the log), --no-pull,
#        --dry-run, --self-test-abort.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
RUN_DIR=""
BRANCH="${SS2_BRANCH:-}"
MAX_STEPS=250
SKIP_BOX_TESTS=0
DRY_RUN=0
NO_PULL=0
SELF_TEST_ABORT=0

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir)         need_value $# "$1"; RUN_DIR="$2";   shift 2 ;;
    --branch)          need_value $# "$1"; BRANCH="$2";    shift 2 ;;
    --max-steps)       need_value $# "$1"; MAX_STEPS="$2"; shift 2 ;;
    --tag)             need_value $# "$1";                 shift 2 ;;
    --skip-box-tests)  SKIP_BOX_TESTS=1;                   shift ;;
    --dry-run)         DRY_RUN=1;                          shift ;;
    --no-pull)         NO_PULL=1;                          shift ;;
    --self-test-abort) SELF_TEST_ABORT=1;                  shift ;;
    -h|--help) sed -n '2,41p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done
RUN_DIR="${RUN_DIR:-runs/hermetic-$(date +%Y%m%d-%H%M%S)}"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. self-update by re-exec, BEFORE anything moves ---------------------
# Bash executes the body it loaded at launch, so a script that git-pulls itself updates
# the file on disk and not the running body. That has already cost this project a 10-hour
# run. It happens here and nowhere later: a pull with tracked files moved aside is a way
# to lose them.
if [ "$NO_PULL" = 0 ]; then
  banner "[1/8] git pull --ff-only"
  if [ -n "$BRANCH" ]; then
    git fetch --quiet origin "$BRANCH" || echo "WARN: fetch of $BRANCH failed"
    git checkout "$BRANCH" || { echo "FATAL: cannot checkout $BRANCH"; exit 1; }
  fi
  _self_before="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
  _self_after="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  if [ -n "$_self_before" ] && [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
    echo "  this script changed in the pull — re-execing the new body"
    export _REEXEC=1
    exec bash "$0" ${BRANCH:+--branch "$BRANCH"} --run-dir "$RUN_DIR" \
         --max-steps "$MAX_STEPS" \
         $([ "$SKIP_BOX_TESTS" = 1 ] && echo --skip-box-tests) \
         $([ "$DRY_RUN" = 1 ] && echo --dry-run)
  fi
else
  banner "[1/8] git pull SKIPPED (--no-pull)"
fi
echo "  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)   commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# --- 2. refuse a dirty tree ------------------------------------------------
banner "[2/8] working tree must be clean"
DIRT="$(git status --porcelain)"
if [ -n "$DIRT" ]; then
  echo "FATAL: the working tree is dirty. This gate moves TRACKED files out of the repo"
  echo "       and back; an uncommitted edit inside one of them cannot be told apart"
  echo "       from a restore failure, and the documented recovery would discard it."
  echo "       Commit or stash first. Offending paths:"
  echo "$DIRT" | sed 's/^/         /'
  exit 1
fi
echo "  clean"

# --- 3. the env (box-only; --dry-run stops before it) ---------------------
if [ "$DRY_RUN" = 0 ]; then
  banner "[3/8] conda env: $ENV_NAME"
  MINICONDA="${HOME}/miniconda3"
  [ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
  # A DIRECTORY check, never `conda env list | grep -q`: under `pipefail` a matching grep
  # exits early, SIGPIPEs conda, and turns found-it into a pipeline failure.
  if [ ! -d "$MINICONDA/envs/$ENV_NAME" ]; then
    echo "FATAL: env '$ENV_NAME' does not exist — build it first:"
    echo "       nrun bash earshot/tools/bootstrap_ss2.sh"
    exit 1
  fi
  set +u   # conda's compiler hooks dereference unset CONDA_BACKUP_* vars
  eval "$("$MINICONDA/bin/conda" shell.bash hook)"
  conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
  set -u
  [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] || { echo "FATAL: wrong env: ${CONDA_DEFAULT_ENV:-<none>}"; exit 1; }
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
  echo "  python: $(python -V 2>&1)  at $(command -v python)"

  # The one prerequisite the tree cannot supply itself. There is deliberately no synthetic
  # fallback (ticket 22: a silent synthetic clip is how a run calibrates CLAP on real audio
  # and classifies a noise burst), so a missing staging must fail here, not mid-run.
  if [ ! -d "data/anomaly_audio" ]; then
    echo "FATAL: no data/anomaly_audio — stage the ESC-50 recordings once:"
    echo "       python -m earshot.audio.clips --out-dir data/anomaly_audio"
    exit 1
  fi
else
  banner "[3/8] env SKIPPED (--dry-run: move and restore only)"
fi

PY="${PYTHON:-python}"

# --- 4. move the delete set out of the repo -------------------------------
banner "[4/8] moving the phase-3 delete set out"
HOLD="${HERMETICITY_HOLD_DIR:-${HOME}/.earshot-hermeticity-$(date +%Y%m%d-%H%M%S)}"
PATHS=()
while IFS= read -r line; do
  [ -n "$line" ] && PATHS+=("$line")
done < <("$PY" -m earshot.tools.reset_manifest --print-paths)
[ "${#PATHS[@]}" -gt 0 ] || { echo "FATAL: the manifest is empty or unreadable"; exit 1; }

echo "  holding directory: $HOLD"
echo "  IF THIS PROCESS IS KILLED, restore with:"
echo "      cd $REPO_ROOT && git checkout -- ${PATHS[*]}"
echo

MOVED=()
restore() {
  local status=$?
  banner "restore"
  local p i
  for ((i=${#MOVED[@]}-1; i>=0; i--)); do
    p="${MOVED[$i]}"
    mkdir -p "$(dirname "$REPO_ROOT/$p")"
    if [ -e "$HOLD/$p" ]; then
      mv "$HOLD/$p" "$REPO_ROOT/$p" && echo "  restored $p"
    fi
  done
  local dirt
  dirt="$(git -C "$REPO_ROOT" status --porcelain)"
  if [ -n "$dirt" ]; then
    echo
    echo "  !!! THE TREE IS NOT BACK THE WAY IT WAS. Recover with:"
    echo "      cd $REPO_ROOT && git checkout -- ${PATHS[*]}"
    echo "  still differing:"
    echo "$dirt" | sed 's/^/    /'
    status=1
  else
    echo "  working tree clean again"
  fi
  exit "$status"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for p in "${PATHS[@]}"; do
  if [ ! -e "$p" ]; then
    echo "  WARN: $p is already absent — the manifest may be stale"
    continue
  fi
  mkdir -p "$HOLD/$(dirname "$p")"
  mv "$p" "$HOLD/$p" || { echo "FATAL: cannot move $p"; exit 1; }
  MOVED+=("$p")
  echo "  moved $p"
done

# The forced-failure arm. Its whole job is to abort while the repo is taken apart, so a
# test can assert the trap put it back. Never reached in a real run.
if [ "$SELF_TEST_ABORT" = 1 ]; then
  echo "  --self-test-abort: aborting with the delete set moved out"
  exit 42
fi

# --- 5. verify absence, before the run ------------------------------------
banner "[5/8] verifying absence (before)"
BEFORE="$HOLD/verify-before.json"
"$PY" -m earshot.tools.reset_manifest --verify-absent --when before > "$BEFORE" \
  || { echo "FATAL: something in the delete set is still present"; cat "$BEFORE"; exit 1; }
echo "  all ${#PATHS[@]} paths absent"

if [ "$DRY_RUN" = 1 ]; then
  banner "[6-8/8] SKIPPED (--dry-run)"
  echo "  the move and the absence check are what --dry-run exercises; the restore"
  echo "  below is the rest of it."
  exit 0
fi

# --- 6. the box suite (supporting evidence, not the criterion) ------------
if [ "$SKIP_BOX_TESTS" = 0 ]; then
  banner "[6/8] box suite, with the delete set gone"
  "$PY" -m unittest discover earshot/tests/box
  BOX_EC=$?
  [ "$BOX_EC" -eq 0 ] || { echo "FATAL: the box suite is red without the old trees — that is a leak"; exit 1; }
else
  banner "[6/8] box suite SKIPPED (--skip-box-tests)"
fi

# --- 7. the smoke ---------------------------------------------------------
banner "[7/8] the smoke, with the delete set gone"
echo "  run dir: $RUN_DIR"
"$PY" -m earshot --run-dir "$RUN_DIR" --n-episodes 1 --max-steps "$MAX_STEPS"
RUN_EC=$?
if [ "$RUN_EC" -ne 0 ]; then
  echo "FATAL: the smoke did not complete without the old trees (exit $RUN_EC)."
  echo "       That is the gate working. Find the leak, restore, repeat."
  exit "$RUN_EC"
fi

# --- 8. verify absence again, then record ---------------------------------
banner "[8/8] verifying absence (after) and writing the record"
AFTER="$HOLD/verify-after.json"
"$PY" -m earshot.tools.reset_manifest --verify-absent --when after > "$AFTER" \
  || { echo "FATAL: the delete set reappeared DURING the run"; cat "$AFTER"; exit 1; }

"$PY" -m earshot.tools.reset_manifest --write-record \
    --run-dir "$RUN_DIR" --before "$BEFORE" --after "$AFTER" \
    --commit "$COMMIT" --holding-dir "$HOLD" \
  || { echo "FATAL: could not write the hermeticity record"; exit 1; }

echo
echo "  smoke complete and recorded. The restore runs next; judge AFTER it, so the last"
echo "  thing in this log is a clean tree rather than a green with the repo taken apart:"
echo "      python -m earshot.task.smoke --run-dir $RUN_DIR"
echo "  criterion 9 reads $RUN_DIR/hermeticity.json and is now answerable."
exit 0
