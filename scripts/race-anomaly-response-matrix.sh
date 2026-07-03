#!/bin/bash
# scripts/race-anomaly-response-matrix.sh — scale the N3 anomaly-response eval from
# the single-cell smoke to a multi-(scene,category) matrix, for a POWERED systems
# result: the interrupt→investigate→resume→report controller working across a
# representative set + the warm soft-SPL S3−S1 decomposition.
#
# A *cell* = one (scene, primary_category). Unlike the scale-up AudioGoal matrix
# (source==goal), here the anomaly source is DECOUPLED from the goal (a different
# object), so each cell must have >=2 distinct objects; cells that cannot decouple
# fail their construction/feasibility gate and are SKIPPED (continue-on-failure).
#
# REUSES the tested per-cell race-anomaly-response.sh + plan_scaleup_cells.py cell
# discovery; adds continue-on-failure + a POOLED soft-SPL verdict AND a POOLED
# CONTROLLER CENSUS (the systems headline: how many warm episodes complete a full
# investigate+resume across all cells).
#
# Pipeline (children inherit RACE_SKIP_PULL=1):
#   [1] git pull ONCE  [2] setup ONCE  [3] pre-verify  [4] plan cells
#   [5] run cells (resume; CONTINUE on per-cell failure)
#   [6] pooled soft-SPL verdict (analyze_ablation --revisit)
#   [7] POOLED controller census (diagnose_anomaly_controller over every S3 dir)
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull
#   # smoke (2 val_mini scenes, resumable):
#   nrun bash scripts/race-anomaly-response-matrix.sh --split val_mini --tag-prefix anommx
#   # POWERED (20 val scenes). Step 1 — download the 18 missing meshes then exit:
#   nrun bash scripts/race-anomaly-response-matrix.sh --split val --download --max-cells 0 --tag-prefix anommxv
#   # Step 2 — the full matrix (~40-50 h serial, resumable; re-run to fill gaps):
#   nrun bash scripts/race-anomaly-response-matrix.sh --split val --tag-prefix anommxv
#
# EXECUTE (do NOT source) — children switch conda envs in their own processes.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1

SPLIT="val_mini"
SCENES=""
# Full val category set (turnkey for --split val). val_mini simply has fewer of
# these present per scene; plan_scaleup_cells drops absent ones.
CATEGORIES="chair bed sofa toilet tv_monitor plant"
CLASSES="baby_cry alarm glass_break"
NWARM=3; SETTINGS="1 3"; PREFIX="anommx"
MAX_CELLS=""          # "0" = mesh-gate/plan (and --download) then exit, no GPU spend
DOWNLOAD=""           # --download fetches the 18 missing full-val meshes (token-gated)
# race-anomaly-response.sh fetches REAL ESC-50 audio BY DEFAULT (--no-fetch-audio
# disables it) — the OPPOSITE of race-audiogoal.sh. So the matrix passes NOTHING
# by default and only forwards --no-fetch-audio for the synthetic arm.
FETCH=""; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --categories) CATEGORIES="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --settings) SETTINGS="$2"; shift 2 ;;
    --tag-prefix) PREFIX="$2"; shift 2 ;;
    --max-cells) MAX_CELLS="$2"; shift 2 ;;
    # Fetch the 18 missing full-val meshes via habitat-sim (Matterport-token gated,
    # reads .env). One-time; then re-run without --download. Pair with --max-cells 0
    # to download-then-exit before committing to the ~40-50h matrix.
    --download) DOWNLOAD=1; shift ;;
    --synthetic-audio) FETCH="--no-fetch-audio"; shift ;;
    # pass-through to every cell's race-anomaly-response.sh (e.g. "--min-source-sep 4.0"
    # or "--investigate-max-steps 120").
    --extra) EXTRA="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag-prefix must be alnum/dash/underscore"; exit 1; }
case "$SPLIT" in val|val_mini) : ;; *) echo "FATAL: --split must be val or val_mini"; exit 1 ;; esac
[ -n "$(echo $SETTINGS | tr -d ' ')" ] || { echo "FATAL: --settings must list >=1 of {1,2,3}"; exit 1; }
for s in $SETTINGS; do case "$s" in 1|2|3) : ;; *) echo "FATAL: --settings values must be 1/2/3 (got '$s')"; exit 1 ;; esac; done

CONTENT_DIR="data/hm3d/datasets/objectnav/hm3d/v1/${SPLIT}/content"
MESH_ROOT="data/hm3d"
banner() { printf '\n########## %s ##########\n' "$1"; }

banner "[1/7] git pull --ff-only (ONCE)"
if [ -n "${RACE_SKIP_PULL:-}" ]; then echo "  RACE_SKIP_PULL set — skipping"; else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/7] conda setup (source race-setup.sh → ltm-embodied)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export RACE_SKIP_PULL=1   # children must NOT pull mid-matrix

banner "[3/7] pre-verify (free; abort before any GPU spend)"
for t in test_plan_scaleup_cells test_make_anomaly_response_smoke test_diagnose_anomaly_feasibility \
         test_diagnose_anomaly_controller test_make_audiogoal_smoke test_make_revisit_smoke \
         test_analyze_revisit test_anomaly_controller test_anomaly_wiring; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending on the live run."; exit 1; }
done

banner "[4/7] mesh gate + plan cells (per-scene categories present, mesh-gated)"
[ -d "$CONTENT_DIR" ] || { echo "FATAL: content dir missing: $CONTENT_DIR (need ObjectNav $SPLIT episodes)"; exit 1; }
if [ -n "$DOWNLOAD" ]; then
  banner "[4b] download full-val meshes (HM3D_SCENE_GROUP=hm3d_val_full; Matterport-token gated)"
  HM3D_SCENE_GROUP=hm3d_val_full bash embodied_memory/scripts/download_hm3d.sh "$MESH_ROOT" \
    || { echo "FATAL: mesh download failed (check .env MATTERPORT_TOKEN_ID/SECRET + habitat-sim)"; exit 1; }
fi
# Report present vs missing meshes (informational; the planner mesh-gates per scene).
N_CONTENT=$(ls "$CONTENT_DIR"/*.json.gz 2>/dev/null | wc -l | tr -d ' ')
N_MESH=$(python embodied_memory/scripts/plan_scaleup_cells.py --content-dir "$CONTENT_DIR" \
           --categories $CATEGORIES --classes $CLASSES --mesh-root "$MESH_ROOT" --format json 2>/dev/null \
         | python -c "import json,sys; print(json.load(sys.stdin)['n_scenes'])" 2>/dev/null || echo 0)
echo "  scenes: $N_CONTENT content files, $N_MESH with a usable mesh on disk"
[ "$N_MESH" -lt "$N_CONTENT" ] 2>/dev/null && echo "  NOTE: $((N_CONTENT-N_MESH)) scene(s) have no mesh → SKIPPED. Re-run with --download to fetch them."
SCENES_ARG=""; [ -n "$SCENES" ] && SCENES_ARG="--scenes $SCENES"
MAXC_ARG=""; { [ -n "$MAX_CELLS" ] && [ "$MAX_CELLS" != "0" ]; } && MAXC_ARG="--max-cells $MAX_CELLS"
PLAN_TSV="runs/${PREFIX}-plan.tsv"
# shellcheck disable=SC2086
python embodied_memory/scripts/plan_scaleup_cells.py --content-dir "$CONTENT_DIR" \
    --categories $CATEGORIES --classes $CLASSES --mesh-root "$MESH_ROOT" $SCENES_ARG $MAXC_ARG --format lines \
    > "$PLAN_TSV" 2>/dev/null || { echo "FATAL: cell planning failed"; exit 1; }
N_CELLS=$(grep -c . "$PLAN_TSV" 2>/dev/null || echo 0)
[ "$N_CELLS" -gt 0 ] || { echo "FATAL: 0 runnable cells (no scene has both a mesh AND a target category)."; exit 1; }
N_SET=$(echo $SETTINGS | wc -w)
echo "  PLAN: $N_CELLS cells (each needs >=2 distinct objects to decouple a source; un-decouplable cells will SKIP)"
if [ "${MAX_CELLS:-}" = "0" ]; then echo; echo "DONE (--max-cells 0): plan only."; exit 0; fi

banner "[5/7] run cells (resume on summary.json; CONTINUE on per-cell failure)"
ALL_OUT_DIRS=""; S3_DIRS=""; N_DONE=0; N_RAN=0; N_FAIL=0; FAILED_CELLS=""
i=0
while IFS=$'\t' read -r S CAT CLS; do
  [ -n "$S" ] || continue
  i=$((i+1))
  TAG="${PREFIX}-${S}"                       # race-anomaly-response writes runs/<TAG>-<CAT>-s<N>
  cell_dirs=""; done_count=0
  for N in $SETTINGS; do
    od="runs/${TAG}-${CAT}-s${N}"; cell_dirs="$cell_dirs $od"
    [ "$N" = "3" ] && S3_DIRS="$S3_DIRS $od"
    [ -f "$od/summary.json" ] && done_count=$((done_count+1))
  done
  ALL_OUT_DIRS="$ALL_OUT_DIRS $cell_dirs"
  if [ "$done_count" -eq "$N_SET" ]; then
    echo "  [$i/$N_CELLS] RESUME: cell ($S,$CAT) complete ($N_SET/$N_SET) — skip"
    N_DONE=$((N_DONE+1)); continue
  fi
  banner "[$i/$N_CELLS] cell: scene=$S primary=$CAT class=$CLS settings=[$SETTINGS] n_warm=$NWARM"
  # shellcheck disable=SC2086
  bash scripts/race-anomaly-response.sh --scene "$S" --class "$CLS" --category "$CAT" \
      --tag "$TAG" --src-content-dir "$CONTENT_DIR" \
      --n-warm "$NWARM" --settings "$SETTINGS" $FETCH $EXTRA
  rc=$?
  if [ "$rc" -eq 0 ]; then
    N_RAN=$((N_RAN+1)); echo "  [$i/$N_CELLS] cell ($S,$CAT) OK"
  else
    # rc!=0 here includes a legitimate feasibility SKIP (un-decouplable geometry) —
    # NOT a crash. Record + continue so one un-decouplable cell can't kill the batch.
    N_FAIL=$((N_FAIL+1)); FAILED_CELLS="$FAILED_CELLS ($S,$CAT,rc=$rc)"
    echo "  [$i/$N_CELLS] WARN: cell ($S,$CAT) rc=$rc (feasibility SKIP / build fail / crash) — continuing."
  fi
done < "$PLAN_TSV"

banner "[6/7] pooled soft-SPL verdict (warm S3-S1 + cold control)"
PRESENT_DIRS=""
for d in $ALL_OUT_DIRS; do [ -f "$d/summary.json" ] && PRESENT_DIRS="$PRESENT_DIRS $d"; done
N_PRESENT=$(set -- $PRESENT_DIRS; echo $#)
if [ "$N_SET" -lt 2 ] || [ "$N_PRESENT" -lt 2 ]; then
  echo "  [analyze] skipped: need >=2 settings and >=2 completed out-dirs (got settings=$N_SET, dirs=$N_PRESENT)."
else
  # shellcheck disable=SC2086
  python embodied_memory/scripts/analyze_ablation.py --revisit $PRESENT_DIRS \
      2>&1 | tee "runs/${PREFIX}-matrix-analysis.log"
fi

banner "[7/7] POOLED controller census (the systems headline: full investigate+resume across cells)"
PRESENT_S3=""
for d in $S3_DIRS; do [ -f "$d/summary.json" ] && PRESENT_S3="$PRESENT_S3 $d"; done
N_S3=$(set -- $PRESENT_S3; echo $#)
if [ "$N_S3" -ge 1 ]; then
  # shellcheck disable=SC2086
  python embodied_memory/scripts/diagnose_anomaly_controller.py $PRESENT_S3 \
      2>&1 | tee "runs/${PREFIX}-matrix-controller.log"
else
  echo "  [controller] skipped: no completed S3 out-dir."
fi

echo
echo "########## ANOMALY-RESPONSE MATRIX SUMMARY ##########"
echo "  split=$SPLIT  prefix=$PREFIX  settings=[$SETTINGS]  n_warm=$NWARM"
echo "  cells: planned=$N_CELLS  resumed=$N_DONE  ran_ok=$N_RAN  failed/skip=$N_FAIL"
[ -n "$FAILED_CELLS" ] && echo "  failed/skipped cells:$FAILED_CELLS"
echo "  soft-SPL: runs/${PREFIX}-matrix-analysis.log   controller: runs/${PREFIX}-matrix-controller.log"
echo "  grep POOLED_CONTROLLER_VERDICT in the controller log = the systems headline."
echo "  re-run the SAME command to retry failed/missing cells (completed cells resume)."
[ "$N_RAN" -eq 0 ] && [ "$N_DONE" -eq 0 ] && { echo "  FATAL: no cell completed."; exit 1; }
exit 0
