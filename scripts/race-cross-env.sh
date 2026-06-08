#!/bin/bash
# scripts/race-cross-env.sh — one-shot RACE driver for the CROSS-ENVIRONMENT
# revisit ablation (step 2 of the diagnose-first program).
#
# The same-scene revisit eval (race-revisit.sh) tests recall of a past sighting
# WITHIN one scene (+0.24 warm soft-SPL, reproduced 8x). The proposal's actual
# thesis is broader: reuse across environments (跨环境). This driver builds a
# dataset where the cold sighting accumulates in a HOME scene and the warm visit
# is queried in a DIFFERENT AWAY scene, then measures whether the LTM transfers
# across scenes (paired warm soft-SPL S3-S1) with the cross-scene seam enabled.
#
# IMPORTANT — read before interpreting the result. The cross-scene seam
# (LTM_CROSS_SCENE) is geometrically honest: a scene-A sighting is RECALLED in
# scene B (counter n_cross_scene_recall > 0) but its stored agent_position is in
# scene A's coordinate frame, so it is NOT injected as a waypoint in scene B.
# So S3 behaves like S1 in the away scene and the expected result is a STRUCTURAL
# NULL (S3-S1 ~= 0): the architecture has no cross-scene geometry to act on the
# recall. That null is the informative answer to step 2's decision rule —
# positive cross-env transfer requires the coarse-affordance mechanism (step 4),
# NOT a raw fine-layer filter relaxation. The n_cross_scene_recall counter proves
# the memory DOES recall the cross-scene sighting (so the null is "no mechanism",
# not "empty memory").
#
# Mirrors race-revisit.sh (pull -> setup -> pre-verify -> build -> run -> analyze).
# EXECUTE it (do NOT source) — it activates conda in its own process:
#
#   bash scripts/race-cross-env.sh --tag crossenv-1
#
# A bare invocation uses the two val_mini scenes x {chair, bed}, n-warm 3. The
# HOME (cold sighting) scene is the alphabetically-FIRST of the two — the runner
# processes scenes in sorted() order with group_by_scene=True, so the home cold
# episodes must precede the away warm episodes for the LTM to be warm when the
# away visits run.
#
# Invariants carried over from race-revisit.sh (each cost a re-run before):
#   * --backbone remembr     — omitting it silently uses the 'frontier' stub.
#   * REMEMBR_STRICT=1        — a missing-weights/stub fallback CRASHES.
#   * S1/S3 in SEPARATE processes/out-dirs — the LTM persists within a process.
#   * --scene all + pinned order — home (cold) precedes away (warm).
#   * --target any            — runs all dataset episodes.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults ---
SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair bed"
NWARM="3"
TAG="crossenv-1"
N_EPISODES=""
TARGET="any"

while [ $# -gt 0 ]; do
  case "$1" in
    --scenes|--scene)    SCENES="$2"; shift 2 ;;
    --categories|--cats) CATS="$2"; shift 2 ;;
    --n-warm)            NWARM="$2"; shift 2 ;;
    --tag)               TAG="$2"; shift 2 ;;
    --n-episodes)        N_EPISODES="$2"; shift 2 ;;
    --target)            TARGET="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"
SCENES="${SCENES//,/ }"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alphanumeric/dash/underscore (got '$TAG')"; exit 1; }

# HOME (cold sighting) = the scene the runner processes FIRST, so its cold seed is
# indexed before the AWAY warm visits run. The runner orders scenes with Python's
# sorted() (run_hm3d_pol.py:153 — codepoint order). Compute HOME/AWAY with the SAME
# sorted() (not shell `sort`, which is locale-aware and disagrees, e.g. 'T' vs 'w',
# and would silently run the away scene first, leaving the LTM cold).
ORDER="$(python3 -c "
import sys
s = sorted(sys.argv[1].split())
if len(s) != 2:
    sys.exit('need exactly TWO scenes, got %r' % (s,))
print(s[0], s[1])
" "$SCENES")" || { echo "FATAL: cross-env requires exactly TWO --scenes (got '$SCENES')"; exit 1; }
read -r HOME_SCENE AWAY_SCENE <<EOF
$ORDER
EOF

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/crossenv_${TAG}"
NAME="crossenv_${TAG}"
DS="${DS_DIR}/${NAME}.json.gz"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/6] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup ---
banner "[2/6] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify (free; aborts before any paid run if broken) ---
banner "[3/6] pre-test code verify (builder + cross-scene seam + analyzer + instance diag)"
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed."; exit 1; }
python embodied_memory/scripts/test_cross_scene_propose.py \
  || { echo "FATAL: cross_scene_propose sanity suite failed."; exit 1; }
python embodied_memory/scripts/test_diagnose_sbert_cosines.py \
  || { echo "FATAL: diagnose_sbert_cosines sanity suite failed."; exit 1; }
python embodied_memory/scripts/test_analyze_ablation.py \
  || { echo "FATAL: analyze_ablation --revisit dispatch sanity suite failed."; exit 1; }

# --- 4. build the cross-env dataset (home cold + away warm) into one shared dir ---
banner "[4/6] build cross-env dataset: HOME(cold)=$HOME_SCENE  AWAY(warm)=$AWAY_SCENE  cats=[$CATS] n-warm=$NWARM"
rm -rf "$DS_DIR"
HOME_SRC="${VALMINI}/${HOME_SCENE}.json.gz"
AWAY_SRC="${VALMINI}/${AWAY_SCENE}.json.gz"
[ -f "$HOME_SRC" ] || { echo "FATAL: home source missing: $HOME_SRC"; exit 1; }
[ -f "$AWAY_SRC" ] || { echo "FATAL: away source missing: $AWAY_SRC"; exit 1; }
# shellcheck disable=SC2086
python embodied_memory/scripts/make_revisit_smoke.py --cross-env \
    --home-src "$HOME_SRC" --home-scene "$HOME_SCENE" \
    --away-src "$AWAY_SRC" --away-scene "$AWAY_SCENE" \
    --categories $CATS --n-warm "$NWARM" --out-dir "$DS_DIR" \
  || { echo "FATAL: cross-env dataset build failed."; exit 1; }
[ -f "$DS" ] || { echo "FATAL: expected top-level dataset not written: $DS"; exit 1; }

if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes."; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (home cold + away warm, one pass)"
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: episode count '$N_EPISODES' invalid."; exit 1; }

# --- 5. run S1 (memory off) and S3 (full + cross-scene seam) ---
OUT_DIRS=""
for S in 1 3; do
  out_dir="runs/${TAG}-s$S"
  banner "[5/6] run: setting=$S backbone=remembr scenes=all LTM_CROSS_SCENE=1 -> $out_dir"
  # LTM_CROSS_SCENE is a no-op for S1 (memory off) and enables the cross-scene
  # recall counter for S3; safe to export for both.
  REMEMBR_STRICT=1 LTM_CROSS_SCENE=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --episodes-path "$DS" \
      --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
      --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- 6. analysis: paired warm soft-SPL + cross-scene recall proof ---
# analyze_ablation --revisit pairs warm episodes across S1/S3 by (scene, episode)
# key. Caveat: it labels each scene's first visit "cold" by within-scene visit
# order, so the away scene's FIRST warm episode is dropped as "cold" (conservative
# — it never inflates the delta). The headline is still the warm S3-S1 delta on
# the remaining away visits, expected ~0 (the structural null).
banner "[6/6] paired warm soft-SPL: analyze_ablation.py --revisit$OUT_DIRS"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS

# Cross-scene recall PROOF: the away warm episodes must show n_cross_scene_recall
# > 0 in S3 (memory recalled the home sighting) even though soft-SPL didn't move.
banner "cross-scene recall counter (S3 away warm episodes' bridge_stats_after)"
python - "runs/${TAG}-s3" <<'PY'
import glob, json, os, sys
run = sys.argv[1]
total = 0
rows = []
for f in sorted(glob.glob(os.path.join(run, "episode_*.json"))):
    d = json.load(open(f))
    eid = str(d.get("episode_id", ""))
    if "warm-away" not in eid:
        continue
    n = (d.get("bridge_stats_after") or {}).get("n_cross_scene_recall", 0)
    total += n
    rows.append((eid, n, d.get("soft_spl")))
for eid, n, sspl in rows:
    print(f"  {eid:<22} n_cross_scene_recall={n:<4} soft_spl={sspl}")
print(f"  --> total cross-scene recalls on away warm episodes = {total}")
print("  (>0 confirms the memory RECALLED the home sighting; if soft-SPL S3-S1~=0,")
print("   the null is 'no cross-scene geometry mechanism', not 'empty memory' -> step 4.)")
PY

banner "DONE — paste the Gate block + the cross-scene recall counter"
