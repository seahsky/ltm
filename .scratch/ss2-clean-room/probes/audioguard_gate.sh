#!/bin/bash
# .scratch/ss2-clean-room/probes/audioguard_gate.sh — ticket 16's box verification.
#
# Question: does ticket 12's audio context guard hold on the real `ss2` build — does it
# pass the healthy path, fire under forced failure, and is its calibration right?
#
#   git fetch && git checkout wayfinder/ss2-clean-room-16     # once, to get this file
#   bash .scratch/ss2-clean-room/probes/audioguard_gate.sh
#
# ~2 minutes, read-only. Installs nothing, builds nothing, applies no patches, and
# reuses the `ss2` env ticket 04 built. If `ss2` is missing, run ticket 04's gate first.
#
# Four stages. Stage 0 is not part of the guard at all — it is ticket 17's `pip freeze`,
# riding along because this is a read-only box trip that was already going to happen, and
# five of its nine pinned versions are recorded nowhere in this repo. It runs FIRST so
# that a guard failure still leaves the freeze behind.
#
#   0  pip freeze                 -> ticket 17's constraints evidence
#   1  key validator              vars(spec) against the real AudioSensorSpec
#   2  healthy path               arm_audio_context must return, ~392,356 verts
#   3  negative controls          the guard must FIRE, and on the right file descriptor
#
# Wrap in `nrun` if you want the emailed report; it is short enough not to need it:
#   source scripts/notify-run.sh && nrun bash .scratch/ss2-clean-room/probes/audioguard_gate.sh

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
OUT_DIR="${SS2_OUT_DIR:-runs/ss2-audioguard}"
PROBE="$REPO_ROOT/.scratch/ss2-clean-room/probes/audioguard_probe.py"
BRANCH="${SS2_BRANCH:-}"

# `shift 2` on a flag with no value would leave $# unchanged and spin forever, so every
# value-taking flag checks first.
need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --branch)  need_value $# "$1"; BRANCH="$2";  shift 2 ;;
    --out-dir) need_value $# "$1"; OUT_DIR="$2"; shift 2 ;;
    --tag)     need_value $# "$1";              shift 2 ;;  # so `nrun ... --tag t` names the log
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. self-update -------------------------------------------------------
# Bash executes the body it loaded at launch, so a script that git-pulls itself updates
# the file on DISK, not the running body — an edit would otherwise only take effect on
# the SECOND invocation. That gotcha has already cost this project a 10-hour run, so this
# re-execs itself instead of warning about it (runbook section 7).
banner "[1/5] git pull --ff-only"
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

# --- 2. activate the env ticket 04 built ----------------------------------
banner "[2/5] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
# Checked as a directory, not `conda env list | grep -q`: under pipefail a matching grep
# exits early, SIGPIPEs conda, and turns found-it into a failure (runbook section 7).
if [ ! -d "$MINICONDA/envs/$ENV_NAME" ]; then
  echo "FATAL: env '$ENV_NAME' does not exist — run ticket 04's gate first:"
  echo "       nrun bash .scratch/ss2-clean-room/probes/oneenv_gate.sh"
  exit 1
fi
set +u   # conda's compiler hooks dereference unset CONDA_BACKUP_* vars
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
set -u
[ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] || { echo "FATAL: wrong env: ${CONDA_DEFAULT_ENV:-<none>}"; exit 1; }
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "  python: $(python -V 2>&1)  at $(command -v python)"

# The documented trap: AudioSensorSpec is bound even in non-audio builds (habitat-sim
# issue #2340), so probe the enum MEMBER. Fail here rather than 90 seconds into stage 2.
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

# --- 3. stage 0: the pip freeze, for ticket 17 ----------------------------
# First, so a guard failure still leaves the deliverable behind. Ticket 17 pinned the env
# behind a nine-line constraints file and five of those versions are recorded nowhere.
banner "[3/5] stage 0 — pip freeze (ticket 17)"
if pip freeze > "$OUT_DIR/freeze.txt" 2>"$OUT_DIR/freeze.err"; then
  echo "  $(wc -l < "$OUT_DIR/freeze.txt" | tr -d ' ') packages -> $OUT_DIR/freeze.txt"
  echo "  the five ticket 17 could not find anywhere in the repo:"
  # `-` and `_` both, because pip normalises distribution names inconsistently across
  # versions, and `[=@ ]` because an editable or VCS install renders as `name @ url`.
  grep -iE '^(soundfile|numpy[-_]quaternion|huggingface[-_]hub|tokenizers|safetensors)[=@ ]' \
    "$OUT_DIR/freeze.txt" | sed 's/^/    /' || echo "    *** none matched — check the freeze by hand"
else
  echo "  WARN: pip freeze failed; see $OUT_DIR/freeze.err"
fi

# --- 4. stages 1-3: the guard ---------------------------------------------
# The probe pins HABITAT_SIM_LOG before it imports habitat_sim, so it must own the
# process — do not pre-import anything here.
banner "[4/5] stages 1-3 — the guard against the real binary"
python "$PROBE" \
  --out "$OUT_DIR/report.json" \
  ${SS2_SCENE:+--scene "$SS2_SCENE"} \
  ${SS2_EXTRA_ARGS:-} 2>&1 | tee "$OUT_DIR/probe.log"
PROBE_RC="${PIPESTATUS[0]}"

# --- 5. verdict -----------------------------------------------------------
# Pulls out exactly what ticket 16 asks, so pasting this block back resolves it.
banner "[5/5] verdict"
python - "$OUT_DIR/report.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as fh:
        rep = json.load(fh)
except Exception as exc:
    print("  could not read report: {!r}".format(exc)); sys.exit(0)

print("  VERDICT: {}".format(rep.get("verdict", "?")))
for name in sorted(k for k in rep if k[:1].isdigit()):
    st = rep[name]
    print("    {:<22} {:<5} {:>7.2f}s {}".format(
        name, "ok" if st.get("ok") else "FAIL", st.get("elapsed_s", 0.0),
        "" if st.get("ok") else st.get("error", "")))

s1 = rep.get("01_key_validator", {})
if s1:
    print("\n  stage 1 — the spec")
    print("    vars(spec) after configure : {}".format(s1.get("vars_after_configure") or "empty"))
    print("    irTime rejected            : {}".format(s1.get("rejects_renamed_key")))
    print("    swallowed key detected     : {}".format(s1.get("detects_swallowed_key")))

s2 = rep.get("02_healthy_path", {})
r = s2.get("report") or {}
# Gated on the report, not the stage: a stage that raised has no numbers, and a column
# of Nones under it reads like measurements that came back empty.
if r:
    print("\n  stage 2 — the healthy path")
    print("    n_vertices  engine / logged: {} / {}  (delta {})".format(
        r.get("n_vertices"), r.get("submitted_n_vertices"),
        s2.get("submitted_vs_engine_delta")))
    print("    ir_shape / peak            : {} / {}".format(r.get("ir_shape"), r.get("ir_peak_abs")))
    print("    log split  stdout / stderr : {} / {} chars".format(
        r.get("stdout_chars"), r.get("stderr_chars")))
    print("    canary seen                : {}".format(r.get("log_canary_seen")))
    print("    fatal log lines            : {}".format(r.get("fatal_log_lines") or "none"))
    print("    guard total / OBJ write    : {}s / {}s ({} bytes)".format(
        s2.get("guard_total_s"), s2.get("obj_write_s"), s2.get("obj_bytes")))
    print("    placement                  : {}".format(s2.get("placement")))

s3 = rep.get("03_negative_controls", {})
if s3:
    print("\n  stage 3 — does it fire, and on which descriptor")
    print("    impossible floor fires     : {}".format(s3.get("floor_fires")))
    print("    prefix RE matched on fd 2  : {}".format(s3.get("prefix_re_validated_on_stderr")))
    print("    canary on stdout / stderr  : {} / {}".format(
        s3.get("canary_on_stdout"), s3.get("canary_on_stderr")))
    print("    canary on 2nd render       : {}  (expected False — mesh uploads once)".format(
        s3.get("canary_seen_on_second_render")))
    for p in s3.get("provocations", []):
        print("      {:<22} out {:>5}c / err {:>5}c  prefix-on-stderr {}".format(
            p.get("provocation"), p.get("stdout_chars"), p.get("stderr_chars"),
            p.get("prefix_re_matched_on_stderr")))

# The two claims ticket 16 pre-flighted from source. If either comes back False the
# constant is wrong on the binary and the raw tails in report.json say how.
if s2 and s3:
    canary_ok = bool(r.get("log_canary_seen")) and bool(s3.get("canary_on_stdout"))
    print("\n  pre-flight claims, now measured:")
    print("    ESP_DEBUG canary is on fd 1        : {}".format(canary_ok))
    print("    ESP_ERROR reaches fd 2 + matches   : {}".format(
        s3.get("prefix_re_validated_on_stderr")))
PY

echo
echo "  report: $OUT_DIR/report.json"
echo "  freeze: $OUT_DIR/freeze.txt"
echo "  log:    $OUT_DIR/probe.log"
echo "  Resolve ticket 16 by pasting report.json and freeze.txt back."
exit "$PROBE_RC"
