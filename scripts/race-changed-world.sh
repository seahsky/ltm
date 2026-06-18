#!/bin/bash
# scripts/race-changed-world.sh — Lever-2 RACE driver: the CHANGED-WORLD eval,
# the regime the M4 temporal-context head was actually designed for.
#
# WHY: the revisit eval's +0.24 warm gain could be purely SPATIAL (any prior goal
# sighting helps) rather than TEMPORAL. Changed-world disambiguates: the cold pass
# starts AT instance A (seeds it) but the warm goal is a DIFFERENT instance B, so
# the cold sighting is STALE. Two questions:
#   Q1 (does stale memory HURT?)  — S1 (mem-off) vs S3 (full LTM). If S3 < S1, the
#      LTM blindly chases the stale sighting; if S3 >= S1, recall is robust.
#   Q2 (does recency MITIGATE?)   — S3 baseline vs S3 + LTM_TEMPORAL_CONTEXT. This
#      is the temporal head's home turf: prefer the FRESH sighting once B is seen.
#
# Reuses make_revisit_smoke --changed-world (keys success to B, cold-starts at A,
# marks info['goal_changed']) + the standard analyze_ablation --revisit + the
# analyze_revisit --compare-a/-b A/B (verbatim). NO runtime/analyzer change — the
# changed-world dataset is just instance-keyed-to-B with the cold start on A.
#
# EXECUTE it (do NOT source):
#
#   bash scripts/race-changed-world.sh --tag cw-1
#
# Needs >=2 instances per (scene,category); single-instance categories are SKIPPED
# by the builder, and the driver ABORTS if that leaves zero episodes.
#
# Critical invariants:
#   * --backbone remembr + REMEMBR_STRICT=1
#   * S1/S2/S3 + the S3-temporal arm in SEPARATE processes/out-dirs (LTM persists
#     within a process; mixing settings would corrupt it)
#   * the S3-temporal arm REUSES the baseline dataset (same episodes → the
#     (scene,category,visit_order) pairing matches for the A/B compare)
#   * --scene all + shuffle=False (pinned via episode_order)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair"          # chairs most reliably have >=2 instances per scene
NWARM="3"
TAG=""
N_EPISODES=""
TARGET="any"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --categories) CATS="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"
SCENES="${SCENES//,/ }"
[ -z "$TAG" ] && { echo "FATAL: --tag <name> required"; exit 1; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/cw_${TAG}"
NAME="cw_${TAG}"
DS="${DS_DIR}/${NAME}.json.gz"

banner() { printf '\n========== %s ==========\n' "$1"; }

banner "[1/7] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

banner "[2/7] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/7] pre-test code verify (builder changed-world + analyzer + temporal head + caption rerank)"
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_analyze_revisit.py \
  || { echo "FATAL: analyze_revisit sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_temporal_context.py \
  || { echo "FATAL: temporal_context sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_analyze_ablation.py \
  || { echo "FATAL: analyze_ablation --revisit dispatch sanity suite failed"; exit 1; }

banner "[4/7] build CHANGED-WORLD dataset: scenes=[$SCENES] cats=[$CATS] n-warm=$NWARM -> $DS_DIR"
rm -rf "$DS_DIR"
for SCENE in $SCENES; do
  SRC="${VALMINI}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_revisit_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories $CATS --n-warm "$NWARM" \
      --changed-world --out-dir "$DS_DIR" \
    || echo "  NOTE: $SCENE produced no changed-world episodes (a category may have <2 instances) — continuing."
done
[ -f "$DS" ] || { echo "FATAL: top-level dataset not written: $DS (no scene yielded >=2-instance categories — try --categories with multiple instances)"; exit 1; }
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes"; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass over all built scenes)"
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: 0 changed-world episodes built (every category had <2 instances). Pick scenes/categories with multiple instances."; exit 1; }

# --- 5. run S1/S2/S3 baseline (Q1: does stale memory hurt?) ---
OUT_DIRS=""
for S in 1 2 3; do
  out_dir="runs/${TAG}-s${S}"
  banner "[5/7] run baseline: setting=$S (changed-world) -> $out_dir"
  # shellcheck disable=SC2086
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --episodes-path "$DS" \
      --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
      --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  completed="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${out_dir}/summary.json" 2>/dev/null || echo 0)"
  [ "$completed" = "$N_EPISODES" ] || echo "WARN: setting $S completed ${completed}/${N_EPISODES} — partial."
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- 6. run S3 + temporal head (Q2: does recency mitigate stale-chasing?) ---
S3T_DIR="runs/${TAG}t-s3"
banner "[6/7] run S3 + LTM_TEMPORAL_CONTEXT (changed-world; the temporal head's design regime) -> $S3T_DIR"
# shellcheck disable=SC2086
LTM_TEMPORAL_CONTEXT=1 REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --setting 3 --episodes-path "$DS" \
    --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
    --out-dir "$S3T_DIR" 2>&1 | tee "${S3T_DIR}.log"
[ -f "$S3T_DIR/summary.json" ] || { echo "FATAL: S3-temporal arm wrote no summary."; exit 1; }

# --- 7. analysis ---
banner "[7/7] Q1 — baseline S1/S2/S3 ablation on changed-world (does stale memory HURT?)"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS \
    2>&1 | tee "runs/${TAG}-cw-ablation.log"

banner "[7/7] Q2 — temporal A/B on changed-world (S3-temporal B − S3-baseline A)"
python embodied_memory/scripts/analyze_revisit.py \
    --compare-a "runs/${TAG}-s3" --compare-b "$S3T_DIR" \
    2>&1 | tee "runs/${TAG}-cw-temporal-compare.log"

echo
echo "DONE. Changed-world eval (goal moves between cold-map and warm-visit)."
echo "  Q1: warm S3-S1 < 0 ⇒ LTM chases the STALE sighting (memory hurts when the world changed);"
echo "      warm S3-S1 >= 0 ⇒ recall is robust to the move."
echo "  Q2: warm B-A > 0 (p<0.1) ⇒ the temporal/recency head MITIGATES stale-chasing (its design regime);"
echo "      <= 0 ⇒ recency doesn't help even here (honest negative)."
