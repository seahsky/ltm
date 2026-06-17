#!/bin/bash
# scripts/race-audiogoal.sh — M3 one-shot RACE driver for the AudioGoal warm
# S1-vs-S3 measurement. Mirrors race-revisit.sh (pull → setup → pre-verify →
# build → run → analyze) with one inserted RENDER-GRIDS stage.
#
# MVP = ONE (scene, anomaly_class) per invocation → ONE RIR grid (habitat_env
# loads a single --rir-grid path and validates scene_id). Multi-(scene,class)
# per run is a deferred follow-up (key the grid cache by (scene,class)).
#
# Pipeline:
#   [1] git pull   [2] source race-setup (ltm-embodied)   [3] pre-verify tests
#   [4] BUILD audiogoal dataset (make_audiogoal_smoke → source_manifest.json)
#   [5] RENDER the RIR grid at the manifest source (soundspaces-spike env)
#   [6] RUN S1 + S3 (--task audiogoal --backbone remembr --rir-grid <grid>)
#   [7] ANALYZE: analyze_ablation --revisit (warm paired soft-SPL S3-S1 + Gate A)
#
#   cd ~/ltm && git checkout lifelong-revisit-eval \
#       && nrun bash scripts/race-audiogoal.sh --scene TEEsavR23oF --class alarm --category bed
#
# EXECUTE (do NOT source) — it switches conda envs in its own process.
# Aborts before the paid LLM run if pull / setup / pre-tests / build / render fail.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
MINICONDA="${HOME}/miniconda3"; SS_ENV="soundspaces-spike"; LTM_ENV="ltm-embodied"

SCENE="TEEsavR23oF"; CLASS="alarm"; CATEGORY="bed"
NWARM=3; SETTINGS="1 3"; TAG="audiogoal"; T_ANOM_WARM=30; SOURCE_OVERRIDE=""; REUSE_DS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --settings) SETTINGS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --t-anom) T_ANOM_WARM="$2"; shift 2 ;;
    --source) SOURCE_OVERRIDE="$2"; shift 2 ;;
    --reuse-dataset) REUSE_DS=1; shift ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/audiogoal_${TAG}"
NAME="audiogoal_${TAG}"; DS="${DS_DIR}/${NAME}.json.gz"
MANIFEST="${DS_DIR}/source_manifest.json"
GRID="runs/audiogoal/${SCENE}_${CLASS}_rir_grid.npz"
banner() { printf '\n========== %s ==========\n' "$1"; }
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }

banner "[1/7] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

banner "[2/7] conda setup (source scripts/race-setup.sh → $LTM_ENV)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/7] pre-verify (free; abort before spend)"
for t in test_make_audiogoal_smoke test_audio_task test_make_revisit_smoke test_analyze_revisit; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending on the live run."; exit 1; }
done

banner "[4/7] build audiogoal dataset: scene=$SCENE class=$CLASS cat=$CATEGORY n-warm=$NWARM"
if [ -n "$REUSE_DS" ] && [ -f "$DS" ] && [ -f "$MANIFEST" ]; then
  echo "  REUSE existing dataset: $DS"
else
  SRC="${VALMINI}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  rm -rf "$DS_DIR"
  # NOTE: '=' form (not a space) — HM3D source coords start with '-' and contain
  # commas, so they don't match argparse's negative-number regex; passed with a
  # space, argparse mistakes the value for an option flag ("expected one argument").
  SRC_ARG=""; [ -n "$SOURCE_OVERRIDE" ] && SRC_ARG="--source-position=$SOURCE_OVERRIDE"
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_audiogoal_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories "$CATEGORY" --n-warm "$NWARM" \
      --anomaly-class "$CLASS" --name "$NAME" --t-anom-warm "$T_ANOM_WARM" \
      --out-dir "$DS_DIR" --source-manifest "$MANIFEST" $SRC_ARG \
    || { echo "FATAL: dataset build failed."; exit 1; }
fi
[ -f "$DS" ] && [ -f "$MANIFEST" ] || { echo "FATAL: dataset or manifest missing"; exit 1; }

# Pull the source xyz for THIS (scene,class) out of the manifest (single source of truth).
SRC_XYZ="$(python -c "
import json,sys
m=json.load(open('$MANIFEST'))
hit=[e for e in m if e['anomaly_class']=='$CLASS' and (e['scene_id'] or '').endswith('$SCENE')]
assert hit and hit[0]['source_position'], 'no source for ($SCENE,$CLASS) in manifest'
print(','.join('%.6f'%v for v in hit[0]['source_position']))
")" || { echo "FATAL: could not read source from manifest"; exit 1; }
echo "  source for ($SCENE,$CLASS) = $SRC_XYZ"

banner "[5/7] render RIR grid at the manifest source ($SS_ENV)"
GLB="$(find data/hm3d -name "${SCENE}.basis.glb" 2>/dev/null | head -1)"
[ -n "$GLB" ] || GLB="$(find data/hm3d -name "*${SCENE}*.glb" 2>/dev/null | grep -v semantic | head -1)"
[ -n "$GLB" ] || { echo "FATAL: no .glb for $SCENE"; exit 1; }
mkdir -p "$(dirname "$GRID")"
set +u; conda activate "$SS_ENV" || { echo "FATAL: activate $SS_ENV failed (build it: scripts/race-soundspaces-spike.sh)"; exit 1; }; set -u
# '=' form: $SRC_XYZ starts with '-' (HM3D coords) and has commas, so a
# space-separated value is misread by argparse as an option flag.
python embodied_memory/scripts/render_rir_grid.py \
    --scene "$GLB" --source="$SRC_XYZ" --out "$GRID" --n-cells 24 \
    2>&1 | tee "${DS_DIR}/render.log"
rc=${PIPESTATUS[0]}
set +u; conda activate "$LTM_ENV" || { echo "FATAL: re-activate $LTM_ENV failed"; exit 1; }; set -u
if [ "$rc" -ne 0 ] || [ ! -f "$GRID" ]; then
  # Disambiguate the failure so the operator isn't sent down the wrong path:
  #  - render_rir_grid prints a distinctive "RED: ... reachable cells / non-zero IRs"
  #    line (and exits 1) ONLY for a genuine off-navmesh / too-few-cells source.
  #  - an UNCAUGHT exception (habitat_sim/EGL/OOM crash) ALSO exits 1 but bypasses
  #    sys.exit(main()) — no RED line; exit 2 is argparse/CLI/setup/scene-data.
  # Only blame the source position when the off-navmesh signature is actually present.
  if [ "$rc" -eq 1 ] && grep -qE "reachable cells|non-zero IRs" "${DS_DIR}/render.log" 2>/dev/null; then
    echo "FATAL: RIR render found too few reachable cells (rc=1) — off-navmesh source; re-pick with --source x,y,z. See ${DS_DIR}/render.log. NOT spending on the LLM run."
  else
    echo "FATAL: RIR render failed (rc=$rc) — CLI/setup/import/runtime/scene-data error, NOT necessarily the source position; check ${DS_DIR}/render.log for the traceback. NOT spending on the LLM run."
  fi
  exit 1
fi

N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" || N_EPISODES=0
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: episode count '$N_EPISODES' <= 0"; exit 1; }
echo "  n-episodes = $N_EPISODES"

banner "[6/7] run settings [$SETTINGS] (--task audiogoal --backbone remembr)"
OUT_DIRS=""
for S in $SETTINGS; do
  out_dir="runs/${TAG}-${CLASS}-s$S"
  banner "run: setting=$S -> $out_dir"
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --task audiogoal \
      --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_WARM" \
      --episodes-path "$DS" --scene "$SCENE" --target any \
      --n-episodes "$N_EPISODES" --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

banner "[7/7] Gate-A verdict (warm paired soft-SPL S3-S1 + S2 decomposition)"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS 2>&1 | tee "runs/${TAG}-${CLASS}-analysis.log"
echo
echo "DONE. AudioGoal warm S1-vs-S3 for ($SCENE,$CLASS). Cold-silent (t_anom high)"
echo "seeds the LTM; warm episodes fire the anomaly. Onset lines: grep '\\[audio\\]' the run logs."
