#!/bin/bash
# .scratch/ss2-clean-room/probes/rendercost_sweep.sh — ticket 06's box measurement.
#
# Question: how many ms does one in-sim audio render take on the RACE box, and
# does that make live-every-step affordable?
#
#   nrun bash .scratch/ss2-clean-room/probes/rendercost_sweep.sh
#
# ~10-20 min. Reuses the `ss2` env ticket 04 built — installs nothing, builds
# nothing, and applies no patches. If `ss2` is missing, run ticket 04's gate first.
#
# Verdict is computed by the probe against thresholds pre-registered in ticket 06,
# so pasting the report back resolves the ticket rather than opening a new
# judgement call:
#   LIVE_EVERY_STEP_HOLDS      <= 50 ms/step
#   LIVE_EVERY_STEP_TOLERABLE  <= 150 ms/step
#   THROTTLE_REQUIRED          otherwise -> the map's destination gets amended
# ...and every one of those is gated on the config still producing a CLIMBABLE
# energy gradient. A fast preset with a flat field does not count as a win.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
OUT_DIR="${SS2_OUT_DIR:-runs/ss2-render-cost}"
PROBE="$REPO_ROOT/.scratch/ss2-clean-room/probes/rendercost_probe.py"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. self-update -------------------------------------------------------
# The gotcha ticket 04 hit: this script git-pulls itself, so an edit to it lands
# on the SECOND invocation. The probe it calls is pulled in the same step, so a
# probe-only change is live immediately.
banner "[1/4] git pull --ff-only"
git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
echo "  running commit: $(git rev-parse --short HEAD)"

mkdir -p "$OUT_DIR"

# --- 2. activate the env ticket 04 built ----------------------------------
banner "[2/4] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
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
echo "  python: $(python -V 2>&1)  at $(command -v python)"
echo "  cores:  $(nproc)   (threadCount ceiling is this, not an order of magnitude)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/  gpu: /' || echo "  gpu: nvidia-smi unavailable"

# --- 3. the sweep ---------------------------------------------------------
# Timed through get_sensor_observations() with the agent carrying NO camera, so
# the number is audio alone. --with-camera-delta re-measures one config with the
# RGB sensor attached, which is what ticket 04's 0.6013 s actually included.
banner "[3/4] render-cost sweep"
python "$PROBE" \
  --out "$OUT_DIR/report.json" \
  --walk-steps "${SS2_WALK_STEPS:-20}" \
  --max-scenes "${SS2_MAX_SCENES:-2}" \
  --with-camera-delta \
  ${SS2_SCENE:+--scene "$SS2_SCENE"} \
  ${SS2_EXTRA_ARGS:-} 2>&1 | tee "$OUT_DIR/probe.log"
PROBE_RC="${PIPESTATUS[0]}"

# --- 4. verdict -----------------------------------------------------------
banner "[4/4] verdict"
python - "$OUT_DIR/report.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as fh:
        rep = json.load(fh)
except Exception as exc:
    print("  could not read report: {!r}".format(exc)); sys.exit(0)
v = rep.get("_verdict", {})
print("  {}".format(v.get("verdict", "?")))
print("  {}".format(v.get("reason", "")))
best = v.get("best_admissible")
if best:
    print("\n  cheapest gradient-admissible preset:")
    for k in ("label", "steady_ms_worst", "episode_s_worst", "scenes",
              "admissible_everywhere", "rho_worst"):
        print("    {:<22} {}".format(k, best.get(k)))
    cheap = rep.get("02_sweep", {}).get("cheap_preset")
    if cheap:
        print("    {:<22} {}".format("cheap_preset config", cheap))
sc = rep.get("03_source_count", {}).get("per_scene", [{}])
if sc and sc[0].get("scaling_vs_1"):
    print("\n  sequential source scaling (upper bound, stock build): {}".format(
        sc[0]["scaling_vs_1"]))
drift = rep.get("01_defaults_recheck", {}).get("drift_from_ticket_04")
if drift:
    print("\n  *** acousticsConfig DRIFTED from ticket 04's measured defaults: {}".format(
        sorted(drift)))
PY

echo
echo "  report: $OUT_DIR/report.json"
echo "  log:    $OUT_DIR/probe.log"
echo "  Resolve ticket 06 by pasting report.json (or the verdict block) back."
exit "$PROBE_RC"
