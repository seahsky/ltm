#!/bin/bash
# scripts/race-cross-env.sh — one-shot RACE driver for the CROSS-ENVIRONMENT
# revisit ablation (step 2 of the diagnose-first program).
#
# The same-scene revisit eval (race-revisit.sh) tests recall of a past sighting
# WITHIN one scene (+0.24 warm soft-SPL, reproduced 8x). The proposal's actual
# thesis is broader: reuse across environments (跨环境). This driver builds a
# dataset where the cold sighting accumulates in a HOME scene and the SAME
# category is then queried ONCE in a DIFFERENT AWAY scene, and measures cross-env
# transfer with a role-based analyzer (analyze_cross_env.py).
#
# REDESIGN (2026-06-08, after crossenv-1): the first version used n-warm=3 +
# analyze_ablation --revisit. That CONFOUNDED the result — multiple away visits
# create WITHIN-away-scene revisit, which the visit-order analyzer measured as
# the "warm" effect (+0.1675), NOT cross-env transfer; and the recall counter
# read 0 only because Habitat renumbers episode_id so the "warm-away" filter
# matched nothing. Fixes: (1) n-warm=1 -> exactly ONE away visit/category, no
# within-away revisit; (2) analyze_cross_env.py labels by scene ROLE (away=query,
# home=source) and reads the recall counter by scene_id; (3) more categories for
# power instead of more visits.
#
# IMPORTANT — interpreting the result. The cross-scene seam (LTM_CROSS_SCENE) is
# geometrically honest: a scene-A sighting is RECALLED in scene B (counter
# n_cross_scene_recall > 0) but its stored position is in scene A's frame, so it
# is NOT injected as a waypoint. So the PRIMARY evidence is the recall counter:
# counter>0 proves the LTM recalls the cross-scene sighting, while the away
# soft-SPL S3-S1 cannot be cross-env transfer (counted-not-injected) — it is
# within-episode same-scene memory at most. Positive cross-env transfer requires
# the coarse-affordance mechanism (step 4), not a fine-layer relaxation.
#
# Mirrors race-revisit.sh (pull -> setup -> pre-verify -> build -> run -> analyze).
# EXECUTE it (do NOT source) — it activates conda in its own process:
#
#   bash scripts/race-cross-env.sh --tag crossenv-2
#   bash scripts/race-cross-env.sh --tag crossenv-3 --isolate   # rigor pass (see below)
#
# --isolate freezes the AWAY-scene LTM writes (LTM_FREEZE_SCENE) so each away
# episode queries ONLY the earlier home sightings — stripping the within-away
# cross-episode accumulation that inflated crossenv-2's +0.1695. Expected:
# recall counter still >0, away S3-S1 -> ~0 (STRENGTHENS the no-transfer headline).
#
# A bare invocation uses the two val_mini scenes x the 4 SHARED categories
# {chair, bed, sofa, toilet}, n-warm 1 (one away visit/category -> n=4 away
# pairs). The HOME (cold sighting) scene is the codepoint-FIRST of the two — the
# runner processes scenes in sorted() order with group_by_scene=True, so the home
# cold episodes precede the away query for the LTM to be warm when it runs.
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
CATS="chair bed sofa toilet"   # the 4 categories shared by both val_mini scenes
NWARM="1"                      # ONE away visit/category -> no within-away revisit
TAG="crossenv-2"
N_EPISODES=""
TARGET="any"
ISOLATE=0   # --isolate: freeze away-scene LTM writes (rigor pass, see below)
COARSE=0    # --coarse: enable the step-4 coarse-affordance head (LTM_COARSE_AFFORDANCE)
NO_ROOM_CLIP=0  # --no-room-clip: caption-only coarse grounding (LTM_COARSE_ROOM_CLIP=0) — the A/B baseline

while [ $# -gt 0 ]; do
  case "$1" in
    --scenes|--scene)    SCENES="$2"; shift 2 ;;
    --categories|--cats) CATS="$2"; shift 2 ;;
    --n-warm)            NWARM="$2"; shift 2 ;;
    --tag)               TAG="$2"; shift 2 ;;
    --n-episodes)        N_EPISODES="$2"; shift 2 ;;
    --target)            TARGET="$2"; shift 2 ;;
    --isolate)           ISOLATE=1; shift ;;
    --coarse)            COARSE=1; shift ;;
    --no-room-clip)      NO_ROOM_CLIP=1; shift ;;
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
python embodied_memory/scripts/test_analyze_cross_env.py \
  || { echo "FATAL: analyze_cross_env sanity suite failed."; exit 1; }
python embodied_memory/scripts/test_memory_bridge_consolidate.py \
  || { echo "FATAL: memory_bridge_consolidate (incl --isolate freeze) sanity suite failed."; exit 1; }
python embodied_memory/scripts/test_room_resolver.py \
  || { echo "FATAL: room_resolver sanity suite failed."; exit 1; }
python embodied_memory/scripts/test_room_classifier.py \
  || { echo "FATAL: room_classifier (CLIP zero-shot) sanity suite failed."; exit 1; }
python embodied_memory/scripts/test_room_clip_wiring.py \
  || { echo "FATAL: room-CLIP wiring sanity suite failed."; exit 1; }
python embodied_memory/scripts/test_coarse_propose.py \
  || { echo "FATAL: coarse-affordance proposer sanity suite failed."; exit 1; }

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
# --isolate (rigor pass): freeze the AWAY-scene LTM writes so each away/query
# episode sees ONLY the earlier home sightings — no within-away cross-episode
# accumulation. This strips the confound that inflated crossenv-2's +0.1695, so
# the away S3-S1 isolates the (zero) home->away contribution. Expected: recall
# counter still >0, away S3-S1 drops toward ~0 -> STRENGTHENS the no-transfer
# headline. Safe for S1 (memory off -> nothing to freeze).
# --isolate: EXPORT the freeze var so the python child inherits it. (A bash
# `$VAR_ASSIGN cmd` env-prefix only works for LITERAL `K=V` tokens — an EXPANDED
# `$FREEZE_ENV` is parsed as the command word, not an assignment, which is the
# rc=127 'LTM_FREEZE_SCENE=...: command not found' bug. Exporting is robust.)
if [ "$ISOLATE" = "1" ]; then
  export LTM_FREEZE_SCENE="$AWAY_SCENE"
  echo "  --isolate ON: exported LTM_FREEZE_SCENE=$AWAY_SCENE (freezing away-scene LTM writes)"
fi
# --coarse: enable the step-4 coarse-affordance head. It fires only in the AWAY
# scene (where the fine layer has no same-scene hit -> mem_cands empty), grounding
# the goal category's static room prior to the away scene's own observations.
# analyze_cross_env reports away coarse_chosen and AND-gates the verdict GREEN.
if [ "$COARSE" = "1" ]; then
  export LTM_COARSE_AFFORDANCE=1
  echo "  --coarse ON: exported LTM_COARSE_AFFORDANCE=1 (step-4 coarse-affordance head)"
  # CLIP zero-shot room classifier (Stage 5) is the DEFAULT dense room signal for
  # the coarse head. --no-room-clip is the A/B baseline (caption-keyword grounding
  # only) — the coarse head then fires only when a Qwen-VL caption names the goal's
  # affordant room (the coarse-1/2 sparse signal that ~never fired). Thresholds are
  # taken from the env (LTM_ROOM_CLIP_MIN_COS/_MARGIN) if exported by a calibration
  # step, else the conservative in-code defaults (0.25 / 0.02).
  if [ "$NO_ROOM_CLIP" = "1" ]; then
    export LTM_COARSE_ROOM_CLIP=0
    echo "  --no-room-clip ON: exported LTM_COARSE_ROOM_CLIP=0 (caption-only A/B baseline)"
  else
    echo "  room-CLIP ON (default): min_cos=${LTM_ROOM_CLIP_MIN_COS:-0.25} margin=${LTM_ROOM_CLIP_MARGIN:-0.02}"
  fi
fi
OUT_DIRS=""
for S in 1 3; do
  out_dir="runs/${TAG}-s$S"
  banner "[5/6] run: setting=$S backbone=remembr scenes=all LTM_CROSS_SCENE=1${LTM_FREEZE_SCENE:+ LTM_FREEZE_SCENE=$LTM_FREEZE_SCENE}${LTM_COARSE_AFFORDANCE:+ LTM_COARSE_AFFORDANCE=1} -> $out_dir"
  # LTM_CROSS_SCENE is a no-op for S1 (memory off) and enables the cross-scene
  # recall counter for S3; safe for both. LTM_FREEZE_SCENE (if --isolate) is
  # already exported above, so the child inherits it — no command-line prefix.
  REMEMBR_STRICT=1 LTM_CROSS_SCENE=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --episodes-path "$DS" \
      --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
      --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  rc=${PIPESTATUS[0]}
  # Fail-fast on a GENUINE crash, but NOT on rc!=0 alone: run_hm3d_pol returns
  # nonzero when full-system pass_conditions fail (run_hm3d_pol.py:499-502), which
  # is EXPECTED for memory-off S1 (it deliberately can't satisfy fine_layer_nonempty
  # / all_four_modules_invoked). So judge a real failure by episodes WRITTEN, not rc:
  # 0 completed = a true crash (GPU OOM / missing ReMEmbR weights under REMEMBR_STRICT
  # / sim init) -> abort before spending on the next setting and the analysis.
  completed="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${out_dir}/summary.json" 2>/dev/null || echo 0)"
  # `completed` is always numeric (the `|| echo 0` fallback guarantees "0" on any
  # parse failure), so a plain -lt 1 test is safe and avoids `!`-negation pitfalls.
  if [ "${completed:-0}" -lt 1 ] 2>/dev/null; then
    echo "FATAL: setting=$S completed 0/${N_EPISODES} episodes (rc=$rc) — genuine run failure (OOM / missing weights / sim init); aborting before analysis."
    exit 1
  fi
  [ "$completed" = "$N_EPISODES" ] || echo "WARN: setting=$S completed ${completed}/${N_EPISODES} (rc=$rc) — partial run; warm pairs may be lower than expected."
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- 6. cross-environment transfer analysis (role-based, recall by scene_id) ---
# analyze_cross_env.py labels by scene ROLE (away=query, home=source) instead of
# visit-order, pairs the away episodes S3-vs-S1, and reads the cross-scene recall
# counter by scene_id (robust to Habitat renumbering episode_id). PRIMARY evidence
# is the recall counter (>0 = the home sighting is recalled in the away scene);
# the away soft-SPL delta cannot be cross-env transfer (counted-not-injected).
banner "[6/6] cross-env transfer: analyze_cross_env.py$OUT_DIRS --away-scene $AWAY_SCENE"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_cross_env.py $OUT_DIRS --away-scene "$AWAY_SCENE"

banner "DONE — paste the cross-env transfer block (away S3-S1 + recall counter + verdict)"
