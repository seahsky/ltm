#!/bin/bash
# scripts/race-nonlos-tier3.sh — automate TIER-3 of the non-LOS audio-write gate:
# caption the non-LOS SEED pose with the real ReMEmbR captioner (1 step), then
# adjudicate with check_seed_not_los.py. This is the DECISIVE tier the $0 gate
# (race-nonlos-seed-gate.sh) leaves PENDING for an operator caption — here we
# produce the caption ourselves from a 1-step run, so the whole gate becomes one
# command. Tiny GPU spend (model load + 1 step, ~1-2 min) — NOT the paid A/B.
#
# Assumes Tier-1+Tier-2 already GREEN (the gate built the non-LOS dataset at
# --episodes-path; the SEED is episode 0, started at its away-facing yaw). GREEN
# here ⇒ vision does NOT map the source at the seed ⇒ a write-ON/OFF A/B is worth
# queuing. RED ⇒ the seed still sees the source ⇒ redundant-with-vision ⇒ stop.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull --ff-only \
#     && nrun bash scripts/race-nonlos-tier3.sh \
#          --scene wcojb4TFT35 --category chair --class glass_break \
#          --rir-grid runs/audiogoal/wcojb4TFT35_glass_break_rir_grid.npz \
#          --episodes-path runs/nonlos-gate/audiogoal.json.gz
#
# Bare invocation uses exactly those defaults (the current wcojb chair/glass_break
# cell). EXECUTE (do NOT source).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
LTM_ENV="ltm-embodied"

SCENE="wcojb4TFT35"; CATEGORY="chair"; CLASS="glass_break"
RIR_GRID="runs/audiogoal/wcojb4TFT35_glass_break_rir_grid.npz"
EPISODES="runs/nonlos-gate/audiogoal.json.gz"
COS_BAR="0.23"; OUT_DIR="runs/nonlos-tier3"
while [ $# -gt 0 ]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --rir-grid) RIR_GRID="$2"; shift 2 ;;
    --episodes-path) EPISODES="$2"; shift 2 ;;
    --cos-bar) COS_BAR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }

if [ -z "${RACE_SKIP_PULL:-}" ]; then
  banner "[0/3] git pull --ff-only"
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[1/3] conda setup ($LTM_ENV)"
set +u; source scripts/race-setup.sh 2>/dev/null || conda activate "$LTM_ENV" || {
  echo "FATAL: could not activate $LTM_ENV"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Preflight (fail fast BEFORE loading 7B+2B): the gate artifacts must exist.
[ -f "$EPISODES" ] || { echo "RED: --episodes-path $EPISODES not found — run the \$0 gate first (race-nonlos-seed-gate.sh) to build the non-LOS dataset, or rebuild it."; exit 1; }
[ -f "$RIR_GRID" ] || { echo "RED: --rir-grid $RIR_GRID not found — re-render it (race-rerender-grid.sh)."; exit 1; }

# Preflight (cheap, no GPU): the content file resolved from --episodes-path must
# exist AND contain a *-cold-* SEED with a start_position. If the gate built the
# dataset somewhere Habitat can't resolve (or the non-LOS build regressed), find
# out for $0 before spending the 7B+2B load. Prints the seed start_position so the
# operator can eyeball the expected away-facing seed (≈0.7821,-0.0051,-5.1784).
banner "[1.5/3] preflight — seed exists in the content for $SCENE"
mkdir -p "$OUT_DIR"
python embodied_memory/scripts/check_seed_pose.py \
    --episodes-path "$EPISODES" --scene "$SCENE" --anomaly-class "$CLASS" \
    --check-seed-exists 2>"$OUT_DIR/.preflight.err" || {
      echo "RED: no resolvable non-LOS SEED for scene=$SCENE class=$CLASS from"
      echo "  --episodes-path $EPISODES. Either the content file isn't where Habitat"
      echo "  resolves it (<dir-of-episodes-path>/content/$SCENE.json.gz) or the"
      echo "  non-LOS build regressed (no *-cold-* episode / no start_position)."
      cat "$OUT_DIR/.preflight.err" 2>/dev/null
      echo "  Rebuild under the standard root, e.g.:"
      echo "    python embodied_memory/scripts/make_audiogoal_smoke.py \\"
      echo "      --src data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/$SCENE.json.gz \\"
      echo "      --scene $SCENE --categories $CATEGORY --anomaly-class $CLASS \\"
      echo "      --lifelong --non-los-seed --rir-grid $RIR_GRID \\"
      echo "      --out-dir data/hm3d/datasets/objectnav/hm3d/v1/nonlos-$SCENE"
      echo "    then re-run with --episodes-path data/.../nonlos-$SCENE/audiogoal.json.gz"
      exit 1
    }
echo "  preflight OK — content resolves and a non-LOS seed is present."

banner "[2/3] caption the non-LOS SEED pose (1 step, real captioner)"
mkdir -p "$OUT_DIR"
CAP_LOG="$OUT_DIR/caption_run.log"
# --setting 1 + 1 step ⇒ pass_conditions FAIL ⇒ exit 1 EXPECTED; --no-strict-pass
# keeps it a soft fail. REMEMBR_STRICT=1 hard-crashes on a captioner load failure
# (so we never trust a stub). We gate on summary.json, NOT the exit code.
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol \
    --mode live --backbone remembr --task audiogoal --setting 1 \
    --rir-grid "$RIR_GRID" --anomaly-class "$CLASS" \
    --episodes-path "$EPISODES" --scene "$SCENE" --target any \
    --n-episodes 1 --max-steps 1 --no-strict-pass --out-dir "$OUT_DIR" \
    2>&1 | tee "$CAP_LOG"
RUN_RC=${PIPESTATUS[0]}
echo "  (run exit=$RUN_RC — exit 1 is EXPECTED for a setting-1 1-step run)"

if [ ! -f "$OUT_DIR/summary.json" ]; then
  echo "RED: no $OUT_DIR/summary.json — the caption run genuinely failed (not the"
  echo "  expected soft pass-fail). Tail of the log:"; tail -n 25 "$CAP_LOG"; exit 1
fi

# ── HARD seed-pose gate (renumbering-invariant) ───────────────────────────────
# habitat overwrites episode_id with str(load_index) and its iterator default is
# shuffle=True, so a 1-step run can caption a RANDOM (warm) episode, not the cold
# seed at index 0. We do NOT trust episode_id ordering: instead we compare the
# captioned pose's start_position (now in summary.json) against the SEED's authored
# start_position read straight from the gate-built content file. Mismatch ⇒ a
# different episode was captioned ⇒ HARD-ABORT RED-INVALID (the verdict is garbage).
banner "[2.5/3] HARD seed-pose check (captioned pose == non-LOS seed?)"
SEED_EPS="0.05"
python embodied_memory/scripts/check_seed_pose.py \
    --episodes-path "$EPISODES" --scene "$SCENE" \
    --anomaly-class "$CLASS" --summary "$OUT_DIR/summary.json" --eps "$SEED_EPS"
POSE_RC=$?
if [ "$POSE_RC" -ne 0 ]; then
  echo
  echo "RED-INVALID: the captioned pose is NOT the non-LOS seed (check_seed_pose rc=$POSE_RC)."
  echo "  The Tier-3 caption ran against the WRONG episode (habitat shuffle/renumber),"
  echo "  so any GREEN/RED verdict below would be meaningless. ABORTING before the gate."
  echo "  Fix: ensure the episode iterator is pinned (shuffle=False — episode_order"
  echo "  must target config.habitat.environment.iterator_options) so index 0 = the"
  echo "  cold seed is captioned, then re-run this driver. If summary.json lacks"
  echo "  episodes[0].start_position, the runner is stale — git pull on RACE."
  exit 1
fi
echo "  seed-pose check PASSED — the captioned pose IS the non-LOS seed."

# Extract the SEED's caption (raw, to stdout); diagnostics to stderr; non-zero on
# stub / missing so we never feed a garbage caption to the decisive gate.
CAP="$(python - "$OUT_DIR/summary.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as ex:
    sys.stderr.write("could not read summary.json (%s)\n" % ex); sys.exit(2)
eps = d.get("episodes") or []
if not eps:
    sys.stderr.write("summary.json has no episodes\n"); sys.exit(2)
e = eps[0]
sys.stderr.write("  episode_id=%s  start_position=%s  stub=%s\n" % (
    e.get("episode_id"), e.get("start_position"), bool(e.get("remembr_stub_mode"))))
if e.get("remembr_stub_mode"):
    sys.stderr.write("STUB caption — captioner did not load (REMEMBR_STRICT should have crashed). ABORT.\n"); sys.exit(3)
c = e.get("remembr_sample_caption")
if not c:
    sys.stderr.write("episode 0 has no remembr_sample_caption\n"); sys.exit(4)
sys.stdout.write(c)
PY
)"
EX=$?
if [ "$EX" -ne 0 ]; then
  echo "RED/ABORT: could not read a real seed caption (extractor rc=$EX). Do NOT trust a verdict."
  exit 1
fi
echo "  SEED caption: $CAP"

banner "[3/3] TIER-3 — check_seed_not_los (DECISIVE)"
python embodied_memory/scripts/check_seed_not_los.py \
    --goal "$CATEGORY" --cos-bar "$COS_BAR" --caption "$CAP"
T3=$?

echo
if [ "$T3" -eq 0 ]; then
  echo "RESULT: TIER-3 GREEN — vision does NOT map '$CATEGORY' at the non-LOS seed."
  echo "  → The write-ON vs write-OFF A/B (oracle upper bound) is worth queuing."
  echo "  → NOTE: race-audiogoal-lifelong.sh as-written would DISCARD this non-LOS seed"
  echo "    (rebuilds an LOS seed + re-renders at a random source). It needs a"
  echo "    --reuse-nonlos hardening first — do NOT launch the A/B as-is."
else
  echo "RESULT: TIER-3 RED — vision still maps the source at the seed (see reason above)."
  echo "  Decision rule:"
  echo "   • names '$CATEGORY' (token) OR cos >= 0.30 → relocate source behind a doorway:"
  echo "       nrun bash scripts/race-audiogoal.sh --scene $SCENE --class $CLASS \\"
  echo "         --category $CATEGORY --source=<x,y,z> --lifelong --tag nonlos-doorway"
  echo "   • cos in [0.23,0.30) and no token → densify for a higher-detour seed:"
  echo "       bash scripts/race-rerender-grid.sh --n-cells 32 $RIR_GRID"
  echo "       then re-run race-nonlos-seed-gate.sh"
  echo "   • both fail → accept the redundant-with-vision boundary (document, do NOT spend)."
fi
exit "$T3"
