#!/bin/bash
# scripts/race-nonlos-seed-gate.sh — the $0 FEASIBILITY LADDER for a non-LOS
# lifelong AudioGoal build. NO paid GPU matrix: it only decides whether a
# redundancy-removing (non-line-of-sight-but-audible) seed is even constructible
# on a given scene+source, BEFORE any A/B run is worth queuing.
#
# Why this exists: Step 2 (audio→LTM write) is closed as REDUNDANT-WITH-VISION
# because the default seed is line-of-sight to the source (vision maps it → the
# oracle write is a duplicate). A measured HELPS needs a seed that HEARS but can't
# SEE the source. This gate tests, for $0, whether such a seed exists here — and a
# design workflow judged that gate itself likely-RED (the energy-STOP gate tends to
# force the agent to approach and re-acquire LOS). Run this to find out cheaply
# instead of spending the matrix to discover it.
#
# THREE TIERS (all $0). Stops at the first RED:
#   TIER-1 (local): the pure TDD suites for the new code are green.
#   TIER-2 (builder): make_audiogoal_smoke --lifelong --non-los-seed picks an
#       audible occluded seed and the construction check prints NO FAIL
#       (the non-LOS branch promotes any residual LOS to a hard FAIL).
#   TIER-3 (captioner, DECISIVE): caption the seed pose ONCE with the real
#       captioner and assert vision does NOT name the goal object
#       (check_seed_not_los.py). Operator-fed: pass --caption "<seed caption>"
#       (read it off a 1-step run / the video HUD); without it, Tier-3 is left
#       PENDING with the exact command printed.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull --ff-only \
#     && bash scripts/race-nonlos-seed-gate.sh \
#          --scene wcojb4TFT35 \
#          --src data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/wcojb4TFT35.json.gz \
#          --category chair --class glass_break \
#          --rir-grid runs/audiogoal/wcojb4TFT35_rir_grid.npz \
#          [--caption "a hallway with a closed door"]
#
# EXECUTE (do NOT source). No nrun needed — it spends nothing.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
LTM_ENV="ltm-embodied"

SCENE=""; SRC=""; CATEGORY="chair"; CLASS="glass_break"; RIR_GRID=""
CAPTION=""; DETOUR="1.3"; MIN_GEO="2.0"; ENERGY_FLOOR="0.0"; COS_BAR="0.23"
OUT_DIR="runs/nonlos-gate"
while [ $# -gt 0 ]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --rir-grid) RIR_GRID="$2"; shift 2 ;;
    --caption) CAPTION="$2"; shift 2 ;;
    --detour-ratio) DETOUR="$2"; shift 2 ;;
    --min-geo-m) MIN_GEO="$2"; shift 2 ;;
    --energy-floor) ENERGY_FLOOR="$2"; shift 2 ;;
    --cos-bar) COS_BAR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
[ -n "$SCENE" ] && [ -n "$SRC" ] && [ -n "$RIR_GRID" ] || {
  echo "FATAL: --scene, --src and --rir-grid are required"; exit 2; }

banner() { printf '\n========== %s ==========\n' "$1"; }

if [ -z "${RACE_SKIP_PULL:-}" ]; then
  banner "[0/4] git pull --ff-only"
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[1/4] conda setup ($LTM_ENV)"
set +u; source scripts/race-setup.sh 2>/dev/null || conda activate "$LTM_ENV" || {
  echo "FATAL: could not activate $LTM_ENV"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[2/4] TIER-1 — pure TDD suites (local, \$0)"
for t in test_audio test_make_audiogoal_smoke test_analyze_lifelong_ab \
         test_check_seed_not_los; do
  python embodied_memory/scripts/$t.py \
    || { echo "RED (Tier-1): $t failed — fix before anything else."; exit 1; }
done
echo "  Tier-1 GREEN"

[ -f "$RIR_GRID" ] || { echo "RED: --rir-grid $RIR_GRID not found (render it first "; \
  echo "  with render_rir_grid.py at the manifest source)"; exit 1; }
[ -f "$SRC" ] || { echo "RED: --src $SRC not found"; exit 1; }

banner "[3/4] TIER-2 — non-LOS construction (builder, \$0)"
echo "  scene=$SCENE class=$CLASS category=$CATEGORY"
echo "  detour>=$DETOUR  geo>=${MIN_GEO}m  energy>=$ENERGY_FLOOR"
T2_LOG="$OUT_DIR/tier2.log"; mkdir -p "$OUT_DIR"
python embodied_memory/scripts/make_audiogoal_smoke.py \
    --src "$SRC" --scene "$SCENE" --categories "$CATEGORY" --n-warm 6 \
    --anomaly-class "$CLASS" --out-dir "$OUT_DIR" \
    --lifelong --non-los-seed --rir-grid "$RIR_GRID" \
    --detour-ratio "$DETOUR" --min-geo-m "$MIN_GEO" --energy-floor "$ENERGY_FLOOR" \
    2>&1 | tee "$T2_LOG"
T2=${PIPESTATUS[0]}
if [ "$T2" -ne 0 ]; then
  echo
  echo "RED (Tier-2): no constructible non-LOS seed (FAIL above, or pick_non_los_seed"
  echo "  found no audible occluded cell). This is the cheap answer that the seed is"
  echo "  hard to place — re-render the grid with --source tucked behind a doorway/"
  echo "  alcove so around-a-corner audible cells exist, or relax --detour-ratio."
  exit 1
fi
SEED_LINE="$(grep -m1 '^NONLOS_SEED ' "$T2_LOG" || true)"
echo "  Tier-2 GREEN — $SEED_LINE"

banner "[4/4] TIER-3 — seed-not-LOS captioner gate (DECISIVE, \$0)"
if [ -z "$CAPTION" ]; then
  echo "  PENDING: caption the seed pose ONCE with the real captioner (1-step run at"
  echo "  the NONLOS_SEED start_xyz above, or read the video HUD), then run:"
  echo
  echo "    python embodied_memory/scripts/check_seed_not_los.py \\"
  echo "        --goal $CATEGORY --cos-bar $COS_BAR --caption \"<seed caption>\""
  echo
  echo "  GREEN (exit 0) iff vision does NOT name '$CATEGORY' at the seed → the non-LOS"
  echo "  build is feasible and a write-ON vs write-OFF A/B is worth queuing. RED ⇒ the"
  echo "  seed still sees the source ⇒ redundant-with-vision again ⇒ do NOT spend."
  echo
  echo "RESULT: Tier-1 GREEN, Tier-2 GREEN, Tier-3 PENDING (supply --caption to decide)."
  exit 0
fi
python embodied_memory/scripts/check_seed_not_los.py \
    --goal "$CATEGORY" --cos-bar "$COS_BAR" --caption "$CAPTION"
T3=$?
echo
if [ "$T3" -eq 0 ]; then
  echo "RESULT: ALL THREE TIERS GREEN — a non-LOS-but-audible seed is constructible AND"
  echo "  vision does not map the source. NOW a write-ON vs write-OFF A/B (the oracle"
  echo "  upper bound) is worth queuing; a caveat-free HELPS still needs a range-capable"
  echo "  DOA write (deferred — the sim's audio gives sign, not range)."
else
  echo "RESULT: Tier-3 RED — the seed still visually maps the source → redundant again."
  echo "  Do NOT spend the matrix. Move the source further off-LOS and re-render, or"
  echo "  accept the documented boundary (paper §7)."
fi
exit "$T3"
