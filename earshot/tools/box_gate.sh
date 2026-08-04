#!/bin/bash
# earshot/tools/box_gate.sh — run the box suite on the RACE V100, in the `ss2` env.
#
#   git fetch && git checkout <branch>     # once, to get this file
#   bash earshot/tools/box_gate.sh
#
# A few minutes, read-only. Installs nothing, builds nothing, applies no patches, and
# reuses the `ss2` env `bootstrap_ss2.sh` builds. If `ss2` is missing, run that first.
#
# Five suites now: the audio guard (tickets 12/16), `sim.World` + the ObjectNav loader
# (ticket 21), the audio layer (ticket 22 — the lateral FRAME CONVENTION, the real spec,
# the received signal, the per-step bill), the agent's frame against the simulator
# (ticket 23), and `env_check`'s capability half (ticket 24 — the real allocation, the
# audio enum MEMBER, CLAP instantiated, plus the two forced-failure arms). The
# scene-loading ones scan the ObjectNav content files until they find a scene whose mesh
# is on this box, so their setup is the slow part; `SS2_SCENE_LABEL` pins one and skips
# the scan. `env_check`'s costs seconds and opens no scene, which is why the bootstrap
# can afford to run the same assertion as its stage-8 verdict.
#
# The single most important line in the log is ticket 22's frame verdict. A red there
# does not mean "fix the test": it means live rendering behaves as the grid did, and
# ticket 23's controller needs the compensation term back.
#
# CARRIED, NOT REWRITTEN, from .scratch/ss2-clean-room/probes/audioguard_gate.sh. Only
# the driver concerns live here — ADR-0014 draws the split there, and everything below
# is a footgun that cost a box trip to learn:
#
#   * self-update by RE-EXEC, not by warning. Bash executes the body it loaded at
#     launch, so a script that git-pulls itself updates the file on DISK and not the
#     running body. That gotcha has already cost this project a 10-hour run.
#   * conda activation via a DIRECTORY check, never `conda env list | grep -q`: under
#     `pipefail` a matching grep exits early, SIGPIPEs conda, and turns found-it into a
#     pipeline failure (runbook section 7).
#   * the audio ENUM MEMBER preflight, so a non-audio build fails in seconds rather
#     than 90 (AudioSensorSpec is bound even in non-audio builds — habitat-sim #2340).
#   * `pip freeze` FIRST, so a suite failure still leaves ticket 17's evidence behind.
#   * PYTHONPATH set to the repo root, because `import earshot` has to resolve.
#   * a two-way source/execute guard: this script's `exit` calls would kill an
#     interactive shell that sourced it.
#
# WHAT THE SUITE IS: `python -m unittest discover earshot/tests/box`. The four negative
# controls of the old probe are test methods now.
#
# WHERE THE EVIDENCE LIVES: the captured driver log, not a JSON report. Box tests PRINT
# their measurements (ADR-0014) — ticket 16's box trip left numbers, not just green, and
# those numbers are what made tickets 09, 15 and 17 decidable. This retires the old
# gate's `--tails` mode, whose subject was a report.json the probe no longer writes.
#
# Wrap in `nrun` for the emailed report; it is short enough not to need it:
#   source earshot/tools/notify/notify-run.sh && nrun bash earshot/tools/box_gate.sh

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
OUT_DIR="${SS2_OUT_DIR:-runs/ss2-box-gate}"
BRANCH="${SS2_BRANCH:-}"

# `shift 2` on a flag with no value would leave $# unchanged and spin forever, so every
# value-taking flag checks first.
need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --branch)  need_value $# "$1"; BRANCH="$2";  shift 2 ;;
    --out-dir) need_value $# "$1"; OUT_DIR="$2"; shift 2 ;;
    --tag)     need_value $# "$1";               shift 2 ;;  # so `nrun ... --tag t` names the log
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. self-update, by re-exec -------------------------------------------
banner "[1/4] git pull --ff-only"
if [ -n "$BRANCH" ]; then
  git fetch --quiet origin "$BRANCH" || echo "WARN: fetch of $BRANCH failed"
  git checkout "$BRANCH" || { echo "FATAL: cannot checkout $BRANCH"; exit 1; }
fi
_self_before="$(md5sum "$0" | awk '{print $1}')"
git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
_self_after="$(md5sum "$0" | awk '{print $1}')"
if [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
  echo "  this script changed in the pull — re-execing the new body"
  export _REEXEC=1
  exec bash "$0" ${BRANCH:+--branch "$BRANCH"} --out-dir "$OUT_DIR"
fi
echo "  branch: $(git rev-parse --abbrev-ref HEAD)   commit: $(git rev-parse --short HEAD)"

mkdir -p "$OUT_DIR"

# --- 2. activate the env bootstrap_ss2.sh built ---------------------------
banner "[2/4] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
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

# Probe the enum MEMBER, not the class: AudioSensorSpec is bound even in non-audio
# builds. Capability-shaped, not provenance-shaped — ticket 13's failure would have
# passed every version check.
python - <<'PY' || exit 1
import sys
try:
    import quaternion  # noqa: F401  (must precede habitat_sim — issue #1813)
    import habitat_sim, habitat_sim.sensor
    t = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType
    assert t is not None and hasattr(t, "Binaural"), "habitat_sim built WITHOUT --audio"
except Exception as exc:
    print("FATAL: audio-capable habitat_sim not importable: {!r}".format(exc))
    sys.exit(1)
print("  habitat_sim: audio-capable (Binaural enum member present)")
PY

# --- 3. pip freeze, first --------------------------------------------------
# Forensic evidence, never installed from. Runs BEFORE the suite so a red run still
# leaves it behind — the constraints file pins the inputs and this records the outputs.
banner "[3/4] pip freeze — the resolved set"
if pip freeze > "$OUT_DIR/freeze.txt" 2>"$OUT_DIR/freeze.err"; then
  echo "  $(wc -l < "$OUT_DIR/freeze.txt" | tr -d ' ') packages -> $OUT_DIR/freeze.txt"
  # `-` and `_` both, because pip normalises distribution names inconsistently across
  # versions, and `[=@ ]` because an editable or VCS install renders as `name @ url`.
  echo "  the nine pinned in earshot/tools/ss2-constraints.txt:"
  grep -iE '^(numpy|numpy[-_]quaternion|torch|transformers|scipy|soundfile|huggingface[-_]hub|tokenizers|safetensors)[=@ ]' \
    "$OUT_DIR/freeze.txt" | sed 's/^/    /' || echo "    *** none matched — check the freeze by hand"
else
  echo "  WARN: pip freeze failed; see $OUT_DIR/freeze.err"
fi

# --- 4. the suite ----------------------------------------------------------
# The suite pins HABITAT_SIM_LOG before it imports habitat_sim (via `import earshot`),
# so it must own the process — do not pre-import anything here.
#
# `-b` is deliberately NOT passed: buffering stdout would swallow the measurements the
# box tests print, and those are the whole reason a box trip is worth taking.
banner "[4/4] python -m unittest discover earshot/tests/box"
python -m unittest discover earshot/tests/box -v 2>&1 | tee "$OUT_DIR/box-suite.log"
SUITE_RC="${PIPESTATUS[0]}"

echo
echo "  suite log: $OUT_DIR/box-suite.log   <- THE EVIDENCE RECORD, measurements included"
echo "  freeze:    $OUT_DIR/freeze.txt"
if [ "$SUITE_RC" = "0" ]; then
  echo "  GREEN — this gate protects the smoke run, and licenses nothing beyond the guard."
else
  echo "  RED — the printed blocker list IS the deliverable. Paste box-suite.log back."
fi
exit "$SUITE_RC"
