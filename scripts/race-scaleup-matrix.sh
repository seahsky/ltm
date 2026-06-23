#!/bin/bash
# scripts/race-scaleup-matrix.sh — scale the AudioGoal revisit ablation from the
# 2-scene val_mini smoke to the full ~100-cell matrix: 20 HM3D val scenes × the
# goal categories each scene actually contains (97 cells achievable; chair/bed
# 20/20, sofa/toilet 19/20, tv_monitor 14/20, plant 5/20).
#
# A *cell* = one (scene, goal_category). The anomaly source is co-located with the
# category's goal, so the RIR grid + the retrieval target are CATEGORY-keyed; the
# anomaly CLASS is decorative for retrieval (onset-trigger framing) and is round-
# robined across each scene's categories for trigger diversity. Because only 3
# classes exist but a scene holds 5–6 categories, classes REUSE — which is why each
# cell passes --cell-tag <category> to race-audiogoal.sh (so two categories sharing
# a class don't collide on the same grid/out-dir).
#
# This wrapper adds the three things the existing race-audiogoal-matrix.sh lacks for
# a full-val run, and REUSES the tested single-cell race-audiogoal.sh per cell:
#   [4] MESH gate (--download fetches the 18 missing val meshes; else skip+warn)
#   [5] per-scene CELL PLAN from categories actually present (plan_scaleup_cells.py)
#   [6] CONTINUE-ON-FAILURE loop (one unreachable scene can't kill 97 cells) + resume
#   [7] POOLED cross-scene verdict (analyze_ablation --revisit over every cell)
#
# Pipeline (mirrors race-audiogoal-matrix.sh; children inherit RACE_SKIP_PULL=1):
#   [1] git pull ONCE   [2] conda setup ONCE   [3] pre-verify (free)
#   [4] mesh gate   [5] plan cells   [6] run cells (resume)   [7] pooled analyze
#
# Resumable: a cell whose settings already wrote summary.json is SKIPPED. A FAILED
# cell is RECORDED and the batch CONTINUES (unlike race-audiogoal-matrix.sh which
# aborts) — so the final verdict pools whatever completed. Re-run to fill gaps.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull
#   # 1) STAGE a 5-cell smoke (~1.5–2 h) to validate mechanics on val_mini meshes:
#   nrun bash scripts/race-scaleup-matrix.sh --split val_mini --max-cells 5 --tag-prefix scaleup-smoke
#   # 2) DOWNLOAD the full 20-scene val meshes (token-gated; one-time, ~tens of GB):
#   nrun bash scripts/race-scaleup-matrix.sh --download --max-cells 0   # download then exit
#   # 3) FULL 97-cell matrix (~40–50 h serial; resumable):
#   nrun bash scripts/race-scaleup-matrix.sh --tag-prefix scaleup
#
# EXECUTE (do NOT source) — children switch conda envs in their own processes.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1

SPLIT="val"                       # val (20 scenes) | val_mini (2 scenes)
SCENES=""                         # subset; empty = all scenes in the split
CATEGORIES="chair bed sofa toilet tv_monitor plant"
CLASSES="baby_cry alarm glass_break"
NWARM=3; SETTINGS="1 3"; PREFIX="scaleup"
MAX_CELLS=""                      # truncate the flat cell list (staging); "0" = plan/download then exit
DOWNLOAD=""; FETCH="--fetch-audio"; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --categories) CATEGORIES="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    # default "1 3" = headline S3−S1 only (cheaper). Pass "1 2 3" to add the S2-STM
    # decomposition (attributes the gain to the LTM) at ~1.5× the cost.
    --settings) SETTINGS="$2"; shift 2 ;;
    --tag-prefix) PREFIX="$2"; shift 2 ;;
    --max-cells) MAX_CELLS="$2"; shift 2 ;;
    # Fetch the 18 missing full-val meshes via habitat-sim (Matterport-token gated;
    # reads .env). One-time; then re-run without --download.
    --download) DOWNLOAD=1; shift ;;
    # Use the synthetic burst instead of real ESC-50 clips (default = real audio).
    --synthetic-audio) FETCH=""; shift ;;
    # Pass-through to every cell's race-audiogoal.sh (e.g. "--t-anom 30").
    --extra) EXTRA="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag-prefix must be alnum/dash/underscore"; exit 1; }
case "$SPLIT" in val|val_mini) : ;; *) echo "FATAL: --split must be val or val_mini"; exit 1 ;; esac
# Validate --settings up front: an empty/garbage value would otherwise make every
# cell resume-skip as "complete (0/0)" and the matrix exit 0 having done nothing.
[ -n "$(echo $SETTINGS | tr -d ' ')" ] || { echo "FATAL: --settings must list ≥1 of {1,2,3} (got '$SETTINGS')"; exit 1; }
for s in $SETTINGS; do case "$s" in 1|2|3) : ;; *) echo "FATAL: --settings values must be 1/2/3 (got '$s')"; exit 1 ;; esac; done
CONTENT_DIR="data/hm3d/datasets/objectnav/hm3d/v1/${SPLIT}/content"
MESH_ROOT="data/hm3d"
banner() { printf '\n########## %s ##########\n' "$1"; }

banner "[1/7] git pull --ff-only (ONCE for the whole matrix)"
if [ -n "${RACE_SKIP_PULL:-}" ]; then
  echo "  RACE_SKIP_PULL set — skipping pull"
else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/7] conda setup (source race-setup.sh → ltm-embodied)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
# Children must NOT git-pull mid-matrix (a push during the long run could change
# code between cells). We already pulled once above.
export RACE_SKIP_PULL=1

banner "[3/7] pre-verify (free; abort before any GPU spend)"
for t in test_plan_scaleup_cells test_make_audiogoal_smoke test_make_revisit_smoke test_analyze_revisit; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending on the live run."; exit 1; }
done

banner "[4/7] mesh gate (split=$SPLIT)"
[ -d "$CONTENT_DIR" ] || { echo "FATAL: content dir missing: $CONTENT_DIR (need ObjectNav $SPLIT episodes)"; exit 1; }
N_CONTENT=$(ls "$CONTENT_DIR"/*.json.gz 2>/dev/null | wc -l | tr -d ' ')
echo "  $N_CONTENT scene content files under $CONTENT_DIR"
if [ -n "$DOWNLOAD" ]; then
  banner "[4b/7] download full-val meshes (HM3D_SCENE_GROUP=hm3d_val_full; Matterport-token gated)"
  HM3D_SCENE_GROUP=hm3d_val_full bash embodied_memory/scripts/download_hm3d.sh "$MESH_ROOT" \
    || { echo "FATAL: mesh download failed (check .env MATTERPORT_TOKEN_ID/SECRET + habitat-sim)"; exit 1; }
fi
# Report present vs missing meshes (informational; the planner gates per scene below).
N_MESH=$(python embodied_memory/scripts/plan_scaleup_cells.py --content-dir "$CONTENT_DIR" \
           --categories $CATEGORIES --classes $CLASSES --mesh-root "$MESH_ROOT" --format json 2>/dev/null \
         | python -c "import json,sys; p=json.load(sys.stdin); print(p['n_scenes'])")
echo "  scenes with a usable mesh on disk: $N_MESH / $N_CONTENT"
[ "$N_MESH" -lt "$N_CONTENT" ] && echo "  NOTE: $((N_CONTENT-N_MESH)) scene(s) have no mesh → SKIPPED. Re-run with --download to fetch them."

banner "[5/7] plan cells (per-scene categories present, mesh-gated)"
SCENES_ARG=""; [ -n "$SCENES" ] && SCENES_ARG="--scenes $SCENES"
MAXC_ARG=""; { [ -n "$MAX_CELLS" ] && [ "$MAX_CELLS" != "0" ]; } && MAXC_ARG="--max-cells $MAX_CELLS"
# shellcheck disable=SC2086
python embodied_memory/scripts/plan_scaleup_cells.py --content-dir "$CONTENT_DIR" \
    --categories $CATEGORIES --classes $CLASSES --mesh-root "$MESH_ROOT" $SCENES_ARG $MAXC_ARG --format json \
    > runs/${PREFIX}-plan.json 2>/dev/null \
  || { echo "FATAL: cell planning failed"; exit 1; }
python -c "
import json
p=json.load(open('runs/${PREFIX}-plan.json'))
print(f\"  PLAN: {p['n_cells']} cells across {p['n_scenes']} scenes\")
from collections import Counter
print('  class spread:', dict(Counter(c['anomaly_class'] for c in p['cells'])))
print('  cat spread:  ', dict(Counter(c['category'] for c in p['cells'])))
if p['skipped_no_mesh']:  print('  skipped (no mesh):', ' '.join(p['skipped_no_mesh']))
if p.get('skipped_unreadable'):  print('  skipped (unreadable content):', ' '.join(p['skipped_unreadable']))
"
# Write the cell list to a TSV (scene<TAB>category<TAB>class) and iterate it from a
# FILE (not a pipe) in stage [6] so the loop runs in THIS shell and the failure
# counters accumulate (a `while read` off a pipe runs in a subshell and loses them).
PLAN_TSV="runs/${PREFIX}-plan.tsv"
# shellcheck disable=SC2086
python embodied_memory/scripts/plan_scaleup_cells.py --content-dir "$CONTENT_DIR" \
    --categories $CATEGORIES --classes $CLASSES --mesh-root "$MESH_ROOT" $SCENES_ARG $MAXC_ARG --format lines \
    > "$PLAN_TSV" 2>/dev/null || { echo "FATAL: cell planning (lines) failed"; exit 1; }
N_CELLS=$(grep -c . "$PLAN_TSV" 2>/dev/null || echo 0)
[ "$N_CELLS" -gt 0 ] || { echo "FATAL: 0 runnable cells (no scene has both a mesh AND a target category). Use --download."; exit 1; }
N_SET=$(echo $SETTINGS | wc -w)
EST_EP=$(( N_CELLS * N_SET * (1 + NWARM) ))
printf "  RUN: %d cells × %d settings × (1 cold + %d warm) ≈ %d episodes ≈ %d–%d h serial (resumable)\n" \
  "$N_CELLS" "$N_SET" "$NWARM" "$EST_EP" "$(( EST_EP * 25 / 600 ))" "$(( EST_EP * 35 / 600 ))"

if [ "${MAX_CELLS:-}" = "0" ]; then
  echo
  echo "DONE (--max-cells 0): plan + mesh gate only, no GPU spend. Plan at runs/${PREFIX}-plan.json"
  exit 0
fi

banner "[6/7] run cells (resume on summary.json; CONTINUE on per-cell failure)"
ALL_OUT_DIRS=""; N_DONE=0; N_RAN=0; N_FAIL=0; FAILED_CELLS=""
i=0
while IFS=$'\t' read -r S CAT CLS; do
  [ -n "$S" ] || continue
  i=$((i+1))
  OUT_TAG="${PREFIX}-${S}"
  # out-dirs race-audiogoal.sh will write: runs/<OUT_TAG>-<CAT>-s<N> (CELL_TAG=CAT)
  cell_dirs=""; done_count=0
  for N in $SETTINGS; do
    od="runs/${OUT_TAG}-${CAT}-s${N}"; cell_dirs="$cell_dirs $od"
    [ -f "$od/summary.json" ] && done_count=$((done_count+1))
  done
  ALL_OUT_DIRS="$ALL_OUT_DIRS $cell_dirs"
  if [ "$done_count" -eq "$N_SET" ]; then
    echo "  [$i/$N_CELLS] RESUME: cell ($S,$CAT) complete ($N_SET/$N_SET) — skip"
    N_DONE=$((N_DONE+1)); continue
  fi
  banner "[$i/$N_CELLS] cell: scene=$S category=$CAT class=$CLS  settings=[$SETTINGS] n_warm=$NWARM"
  # shellcheck disable=SC2086
  bash scripts/race-audiogoal.sh --scene "$S" --class "$CLS" --category "$CAT" \
      --cell-tag "$CAT" --src-content-dir "$CONTENT_DIR" \
      --n-warm "$NWARM" --settings "$SETTINGS" \
      --tag "${PREFIX}-${S}-${CAT}" --out-tag "$OUT_TAG" $FETCH $EXTRA
  rc=$?
  if [ "$rc" -eq 0 ]; then
    N_RAN=$((N_RAN+1)); echo "  [$i/$N_CELLS] cell ($S,$CAT) OK"
  else
    N_FAIL=$((N_FAIL+1)); FAILED_CELLS="$FAILED_CELLS ($S,$CAT,rc=$rc)"
    echo "  [$i/$N_CELLS] WARN: cell ($S,$CAT) FAILED rc=$rc — continuing (re-run to retry)."
  fi
done < "$PLAN_TSV"

banner "[7/7] pooled cross-scene verdict (warm S3-S1 + S2 decomposition + cold control)"
# Pool every cell out-dir that actually has a summary.json (a failed/partial cell
# contributes only the settings it completed — the analyzer pairs by (scene,category,
# visit_order) so partial cells can't cross-contaminate).
PRESENT_DIRS=""; N_STALE=0
for d in $ALL_OUT_DIRS; do
  if [ -f "$d/summary.json" ]; then
    PRESENT_DIRS="$PRESENT_DIRS $d"
  elif [ -d "$d" ]; then
    # dir exists but no summary.json = a setting that started then died (OOM/timeout
    # signal) before writing. Excluded from the verdict; surface it so a partial
    # matrix isn't mistaken for a complete one.
    N_STALE=$((N_STALE+1))
  fi
done
N_PRESENT=$(set -- $PRESENT_DIRS; echo $#)
[ "$N_STALE" -gt 0 ] && echo "  NOTE: $N_STALE out-dir(s) exist without summary.json (killed mid-run) — EXCLUDED from the verdict; re-run to complete them."
if [ "$N_SET" -lt 2 ]; then
  echo "  [analyze] skipped: need ≥2 settings for a paired S3−S1 delta (got $N_SET)."
elif [ "$N_PRESENT" -lt 2 ]; then
  echo "  [analyze] skipped: <2 completed out-dirs."
else
  # shellcheck disable=SC2086
  python embodied_memory/scripts/analyze_ablation.py --revisit $PRESENT_DIRS \
      2>&1 | tee "runs/${PREFIX}-matrix-analysis.log"
fi

echo
echo "########## SCALE-UP MATRIX SUMMARY ##########"
echo "  split=$SPLIT  prefix=$PREFIX  settings=[$SETTINGS]  n_warm=$NWARM"
echo "  cells: planned=$N_CELLS  resumed=$N_DONE  ran_ok=$N_RAN  failed=$N_FAIL  stale_dirs=$N_STALE"
[ -n "$FAILED_CELLS" ] && echo "  FAILED cells:$FAILED_CELLS"
echo "  pooled verdict: runs/${PREFIX}-matrix-analysis.log   plan: runs/${PREFIX}-plan.json"
echo "  re-run the SAME command to retry failed/missing cells (completed cells resume)."
# Exit non-zero only if EVERY attempted cell failed (nothing to analyze); a partial
# matrix with some completed cells is a SUCCESS (resumable).
[ "$N_RAN" -eq 0 ] && [ "$N_DONE" -eq 0 ] && { echo "  FATAL: no cell completed."; exit 1; }
exit 0
