#!/bin/bash
# scripts/race-anomaly-response.sh — N3 driver for the REAL anomaly-response eval:
# a primary find-task with an anomaly-sound INTERRUPT from a source DECOUPLED from
# the goal (unlike the audiogoal smoke where source==goal). 2-PHASE build→render
# + a $0 FEASIBILITY GATE that aborts BEFORE any paid LLM run if the decoupled
# geometry has no audible-not-loud search start.
#
# Pipeline:
#   [1] git pull   [2] race-setup (ltm-embodied)   [3] pre-verify tests
#   [4] BUILD the DECOUPLED-source dataset (make_anomaly_response_smoke → manifest)
#   [5] RENDER the RIR grid at the DECOUPLED source (soundspaces-spike env)
#   [5c] FEASIBILITY GATE (diagnose_anomaly_feasibility): does an audible-not-loud
#        warm start exist near the decoupled source? SKIP → abort before spend.
#   [5d] stage anomaly + benign clips; onset calibration
#   [6] RUN S1 + S3 (--task anomaly_response, mixture bed ON, gate ON w/ recal delta)
#   [7] ANALYZE: analyze_ablation --revisit + the controller census
#
# The mixture bed is ON by default (every scene = anomaly + background bed, the
# project direction); the CLAP anomaly gate is FORCED ON for anomaly_response and
# DEFAULTS to the Gate-0b convolved recal (-0.2557/0.0341) via run_hm3d_pol.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull --ff-only \
#       && nrun bash scripts/race-anomaly-response.sh --scene TEEsavR23oF --class alarm --category bed
#
# (Brand-new driver: the first run needs the manual `git pull` above — the driver
# self-pulls via its own open fd. See the RACE-testing memory.)
# EXECUTE (do NOT source) — it switches conda envs in its own process.
# Aborts before the paid LLM run if pull/setup/tests/build/render/FEASIBILITY fail.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
MINICONDA="${HOME}/miniconda3"; SS_ENV="soundspaces-spike"; LTM_ENV="ltm-embodied"

SCENE="TEEsavR23oF"; CLASS="alarm"; CATEGORY="bed"
NWARM=3; SETTINGS="1 3"; TAG="anomresp"; T_ANOM_WARM=30
MIN_SOURCE_SEP="3.0"; ONSET_TARGET_DIST="4.0"; ONSET_RMS_OVERRIDE=""
# N3's source is a REAL detour (decoupled from the goal), so the investigate budget
# must be larger than the audiogoal default 40 (tuned for the degenerate 0.5 m
# source==goal case, where the first real run aborted every detour at 40 steps →
# investigated=0). extend-budget ON so the detour doesn't starve the primary find-task.
INVESTIGATE_MAX_STEPS=100; INVESTIGATE_EXTEND=1
FETCH_AUDIO=1                         # N3 wants REAL audio by default
BG_GAIN="1.0"; BG_CLASS="vacuum"      # mixture ON by default (every scene = anomaly + bed)
# Feasibility gate band (grid-relative cell energy). coverage empty => the gate
# DERIVES it from the grid's actual cell spacing (a fixed radius false-rejects a
# sparse grid); --feas-coverage-m overrides.
FEAS_AUDIBLE_FRAC="0.02"; FEAS_LOUD_FRAC="0.5"; FEAS_COVERAGE_M=""
SRC_CONTENT_DIR=""
# Query-side instance fix A/B (Stage-1). When set to prf|caption, this run adds
# ONE MORE S3 arm on the SAME dataset/grid with LTM_QUERY_EXPANSION exported
# (out runs/<tag>-<cat>-s3qx), then a paired compare of that arm (B) vs the
# baseline S3 (A). Default empty → byte-identical to the current driver. The $0
# encoder-swap gate (GATE_RESULT=GO-QUERY) motivates this: the bare category
# query "there is a {cat}" collapses instance signal; querying with the recalled
# prior-sighting captions (prf keeps the category anchor; caption is pure
# centroid) may recover the instance gap that the powered null lost to
# wrong-instance over-fire. prf is the conservative default.
QUEREXP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --settings) SETTINGS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --t-anom) T_ANOM_WARM="$2"; shift 2 ;;
    --min-source-sep) MIN_SOURCE_SEP="$2"; shift 2 ;;
    --investigate-max-steps) INVESTIGATE_MAX_STEPS="$2"; shift 2 ;;
    --no-extend-budget) INVESTIGATE_EXTEND=""; shift ;;   # detour counts against the primary budget
    --onset-target-dist) ONSET_TARGET_DIST="$2"; shift 2 ;;
    --onset-rms) ONSET_RMS_OVERRIDE="$2"; shift 2 ;;
    --no-fetch-audio) FETCH_AUDIO=""; shift ;;
    --no-bg) BG_GAIN=""; shift ;;               # disable the bed (geometry-only smoke)
    --bg-gain) BG_GAIN="$2"; shift 2 ;;
    --bg-class) BG_CLASS="$2"; shift 2 ;;
    --feas-audible-frac) FEAS_AUDIBLE_FRAC="$2"; shift 2 ;;
    --feas-loud-frac) FEAS_LOUD_FRAC="$2"; shift 2 ;;
    --feas-coverage-m) FEAS_COVERAGE_M="$2"; shift 2 ;;
    --src-content-dir) SRC_CONTENT_DIR="$2"; shift 2 ;;
    --query-expansion) QUEREXP="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }
case "$CLASS" in baby_cry|alarm|glass_break) ;; *) echo "FATAL: --class must be baby_cry|alarm|glass_break"; exit 1 ;; esac
if [ -n "$QUEREXP" ]; then
  case "$QUEREXP" in prf|caption) ;; *) echo "FATAL: --query-expansion must be prf|caption (got '$QUEREXP')"; exit 1 ;; esac
  case " $SETTINGS " in *" 3 "*) ;; *) echo "FATAL: --query-expansion needs setting 3 in --settings (got '$SETTINGS')"; exit 1 ;; esac
fi

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
SRC_CONTENT_DIR="${SRC_CONTENT_DIR:-$VALMINI}"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/anomaly_response_${TAG}"
NAME="anomaly_response_${TAG}"; DS="${DS_DIR}/${NAME}.json.gz"
MANIFEST="${DS_DIR}/source_manifest.json"
# Grid keyed by TAG + CATEGORY (the RIR depends on the decoupled source; keying by
# $TAG too keeps two experiments on the same (scene,category) from sharing a grid).
GRID="runs/audiogoal/${SCENE}_${TAG}-${CATEGORY}_rir_grid.npz"
banner() { printf '\n========== %s ==========\n' "$1"; }
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }

banner "[1/7] git pull --ff-only"
if [ -n "${RACE_SKIP_PULL:-}" ]; then echo "  RACE_SKIP_PULL set — skipping"; else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/7] conda setup (source scripts/race-setup.sh → $LTM_ENV)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/7] pre-verify (free; abort before spend)"
for t in test_make_anomaly_response_smoke test_diagnose_anomaly_feasibility \
         test_make_audiogoal_smoke test_audio_task test_make_revisit_smoke \
         test_anomaly_controller test_anomaly_wiring test_active_goal_noop \
         test_query_expansion test_summary_query_expanded; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending on the live run."; exit 1; }
done

banner "[4/7] build DECOUPLED-source dataset: scene=$SCENE class=$CLASS cat=$CATEGORY (min-sep=${MIN_SOURCE_SEP}m)"
SRC="${SRC_CONTENT_DIR}/${SCENE}.json.gz"
[ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
rm -rf "$DS_DIR"
python embodied_memory/scripts/make_anomaly_response_smoke.py \
    --src "$SRC" --scene "$SCENE" --categories "$CATEGORY" --n-warm "$NWARM" \
    --anomaly-class "$CLASS" --name "$NAME" --t-anom-warm "$T_ANOM_WARM" \
    --min-source-sep "$MIN_SOURCE_SEP" --out-dir "$DS_DIR" --source-manifest "$MANIFEST" \
    ${BG_GAIN:+--background-class "$BG_CLASS"} \
  || { echo "FATAL: dataset build failed (construction gate FAIL, or no decoupled source in this scene)."; exit 1; }
[ -f "$DS" ] && [ -f "$MANIFEST" ] || { echo "FATAL: dataset or manifest missing"; exit 1; }

# Pull the DECOUPLED source xyz for THIS (scene,class) out of the manifest.
SRC_XYZ="$(python -c "
import json
m=json.load(open('$MANIFEST'))
hit=[e for e in m if e['anomaly_class']=='$CLASS' and (e['scene_id'] or '').endswith('$SCENE')]
assert hit and hit[0]['source_position'], 'no source for ($SCENE,$CLASS) in manifest'
print(','.join('%.6f'%v for v in hit[0]['source_position']))
")" || { echo "FATAL: could not read source from manifest"; exit 1; }
echo "  DECOUPLED source for ($SCENE,$CLASS) = $SRC_XYZ (anomaly object != primary goal '$CATEGORY')"

banner "[5/7] render RIR grid at the DECOUPLED source ($SS_ENV)"
GLB="$(find data/hm3d -name "${SCENE}.basis.glb" 2>/dev/null | head -1)"
[ -n "$GLB" ] || GLB="$(find data/hm3d -name "*${SCENE}*.glb" 2>/dev/null | grep -v semantic | head -1)"
[ -n "$GLB" ] || { echo "FATAL: no .glb for $SCENE"; exit 1; }
mkdir -p "$(dirname "$GRID")"
set +u; conda activate "$SS_ENV" || { echo "FATAL: activate $SS_ENV failed (build it: scripts/race-soundspaces-spike.sh)"; exit 1; }; set -u
# '=' form (HM3D coords start with '-'); -u keeps the render log live for the grep.
python -u embodied_memory/scripts/render_rir_grid.py \
    --scene "$GLB" --source="$SRC_XYZ" --out "$GRID" --n-cells 24 \
    2>&1 | tee "${DS_DIR}/render.log"
rc=${PIPESTATUS[0]}
set +u; conda activate "$LTM_ENV" || { echo "FATAL: re-activate $LTM_ENV failed"; exit 1; }; set -u
if [ "$rc" -ne 0 ] || [ ! -f "$GRID" ]; then
  if [ "$rc" -eq 1 ] && grep -qE "reachable cells|non-zero IRs" "${DS_DIR}/render.log" 2>/dev/null; then
    echo "FATAL: RIR render found too few reachable cells (off-navmesh decoupled source). Re-pick with a larger --min-source-sep or a different --category. See ${DS_DIR}/render.log."
  else
    echo "FATAL: RIR render failed (rc=$rc) — CLI/setup/import/runtime error, see ${DS_DIR}/render.log."
  fi
  exit 1
fi

banner "[5c/7] FEASIBILITY GATE (diagnose_anomaly_feasibility) — does an audible-not-loud start exist?"
FEAS_LOG="${DS_DIR}/feasibility.log"
python embodied_memory/scripts/diagnose_anomaly_feasibility.py \
    --dataset "${DS_DIR}/content/*.json.gz" --grid "$GRID" \
    --audible-frac "$FEAS_AUDIBLE_FRAC" --loud-frac "$FEAS_LOUD_FRAC" \
    ${FEAS_COVERAGE_M:+--coverage-m "$FEAS_COVERAGE_M"} 2>&1 | tee "$FEAS_LOG"
feas_rc=${PIPESTATUS[0]}
FEAS_RESULT="$(grep -oE 'FEASIBILITY_RESULT=[A-Z]+' "$FEAS_LOG" | tail -1 | cut -d= -f2)"
echo "  FEASIBILITY_RESULT=$FEAS_RESULT (rc=$feas_rc)"
# This driver is single-(scene,category), so the only usable verdict is GO. SKIP =
# no audible-not-loud start; PARTIAL would mean a mixed multi-cell run (not this
# driver) — abort on anything other than GO to never spend on an unusable geometry.
if [ "$feas_rc" -ne 0 ] || [ "$FEAS_RESULT" != "GO" ]; then
  echo "FATAL: feasibility gate = ${FEAS_RESULT:-<none>} (need GO) — the decoupled geometry"
  echo "  has NO audible-not-loud warm start (all LOUD = source co-located → step-0 false-fire,"
  echo "  or all QUIET/OUT-OF-COVERAGE = inaudible). Re-pick: adjust --min-source-sep, choose a"
  echo "  different --category, or widen the render radius. NOT spending on the LLM run. See $FEAS_LOG."
  exit 1
fi

# Anomaly + benign clips (real ESC-50). Anomaly auto-resolved by run_hm3d_pol.
ANOMALY_CLIP="data/anomaly_audio/${CLASS}.wav"
if [ -n "$FETCH_AUDIO" ] && [ ! -f "$ANOMALY_CLIP" ]; then
  banner "[stage] fetch ESC-50 anomaly clip ($CLASS)"
  python embodied_memory/scripts/fetch_anomaly_clips.py --classes "$CLASS" \
    || echo "WARN: ESC-50 fetch failed; falling back to synthetic burst"
fi
if [ -f "$ANOMALY_CLIP" ]; then
  echo "  anomaly audio: REAL ESC-50 -> $ANOMALY_CLIP"
elif [ -n "$FETCH_AUDIO" ]; then
  # anomaly_response FORCES the CLAP gate ON with the Gate-0b delta calibrated on
  # REAL convolved ESC-50 audio; a synthetic burst would mis-calibrate the gate and
  # the whole discrimination claim. If the fetch failed, abort before spend.
  echo "FATAL: --fetch-audio requested but $ANOMALY_CLIP is missing (ESC-50 fetch failed)."
  echo "  anomaly_response's gate is calibrated on REAL convolved audio; refusing to spend on"
  echo "  a synthetic burst. Retry the fetch, or pass --no-fetch-audio to accept the synthetic burst."
  exit 1
else
  echo "  anomaly audio: SYNTHETIC burst (--no-fetch-audio)"; ANOMALY_CLIP=""
fi

BG_CLIP=""
if [ -n "$BG_GAIN" ]; then
  BG_CLIP="data/benign_audio/${BG_CLASS}.wav"
  if [ ! -f "$BG_CLIP" ]; then
    banner "[stage] fetch benign background bed ($BG_CLASS)"
    python embodied_memory/scripts/fetch_anomaly_clips.py --include-benign \
      || echo "WARN: benign fetch failed; bed disabled"
  fi
  [ -f "$BG_CLIP" ] && echo "  background bed: REAL ESC-50 benign -> $BG_CLIP (bg_gain=$BG_GAIN)" \
    || { echo "  background bed: NO $BG_CLIP -> bed disabled"; BG_CLIP=""; }
fi

# Onset calibration (audible at ~ONSET_TARGET_DIST m across a room), reads the grid.
if [ -n "$ONSET_RMS_OVERRIDE" ]; then
  ONSET_RMS="$ONSET_RMS_OVERRIDE"; echo "  onset_rms pinned = $ONSET_RMS"
else
  banner "[5d/7] onset calibration (diagnose_onset_calib, target ${ONSET_TARGET_DIST} m)"
  CALIB_LOG="${DS_DIR}/onset_calib.log"
  python embodied_memory/scripts/diagnose_onset_calib.py \
      --grid "$GRID" --target-dist "$ONSET_TARGET_DIST" \
      ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} 2>&1 | tee "$CALIB_LOG"
  ONSET_RMS="$(grep -oE 'RECOMMEND_ONSET_RMS=[0-9.]+' "$CALIB_LOG" | tail -1 | cut -d= -f2)"
  [ -n "$ONSET_RMS" ] || { echo "WARN: no RECOMMEND_ONSET_RMS; default 0.05"; ONSET_RMS="0.05"; }
  echo "  onset_rms (calibrated) = $ONSET_RMS"
fi

N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" || N_EPISODES=0
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: episode count '$N_EPISODES' <= 0"; exit 1; }
echo "  n-episodes = $N_EPISODES"

banner "[6/7] run settings [$SETTINGS] (--task anomaly_response --backbone remembr)"
echo "  investigate: max_steps=$INVESTIGATE_MAX_STEPS extend_budget=$([ -n "$INVESTIGATE_EXTEND" ] && echo ON || echo OFF) (N3 source is a REAL detour → larger than the 40 default)"
OUT_DIRS=""
for S in $SETTINGS; do
  out_dir="runs/${TAG}-${CATEGORY}-s$S"
  banner "run: setting=$S -> $out_dir"
  rm -f "$out_dir/summary.json"
  # anomaly_response FORCES the CLAP gate ON and DEFAULTS the recalibrated delta
  # (-0.2557/0.0341) — no need to pass --audio-anomaly-delta. The mixture bed is
  # passed via --bg-gain/--background-clip.
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --task anomaly_response \
      --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_WARM" \
      --investigate-max-steps "$INVESTIGATE_MAX_STEPS" ${INVESTIGATE_EXTEND:+--investigate-extend-budget} \
      --audio-onset-rms "$ONSET_RMS" ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} \
      ${BG_GAIN:+--bg-gain "$BG_GAIN"} ${BG_CLIP:+--background-clip "$BG_CLIP"} \
      --episodes-path "$DS" --scene "$SCENE" --target any \
      --n-episodes "$N_EPISODES" --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  rc=${PIPESTATUS[0]}
  complete="$(python -c "import json,sys
try:
    s=json.load(open(sys.argv[1])); a=s.get('n_episodes_attempted',0); c=s.get('n_episodes_completed',0)
    print(1 if a>0 and c==a else 0)
except Exception:
    print(0)" "$out_dir/summary.json" 2>/dev/null || echo 0)"
  [ "$complete" = 1 ] \
    || { echo "FATAL: setting=$S run INCOMPLETE at $out_dir (rc=$rc; summary missing or n_completed<n_attempted = a hard crash). See ${out_dir}.log."; exit 1; }
  echo "  setting=$S complete (rc=$rc; rc=1 is normal for S1 — pass_conditions are S3-oriented)"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- Query-side instance fix A/B arm (Stage-1) -------------------------------
# One MORE S3 run on the SAME dataset/grid with LTM_QUERY_EXPANSION exported.
# Distinct out-dir (-s3qx) so the baseline S3 is never clobbered; paired vs it
# below. Guarded: the arm must actually FIRE (n_query_expanded > 0) or the A/B is
# vacuous. Empty QUEREXP → this block is skipped → driver byte-identical.
QX_DIR=""
if [ -n "$QUEREXP" ]; then
  QX_DIR="runs/${TAG}-${CATEGORY}-s3qx"
  banner "[6b/7] query-expansion A/B arm: setting=3 + LTM_QUERY_EXPANSION=$QUEREXP -> $QX_DIR"
  rm -f "$QX_DIR/summary.json"
  LTM_QUERY_EXPANSION="$QUEREXP" REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting 3 --task anomaly_response \
      --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_WARM" \
      --investigate-max-steps "$INVESTIGATE_MAX_STEPS" ${INVESTIGATE_EXTEND:+--investigate-extend-budget} \
      --audio-onset-rms "$ONSET_RMS" ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} \
      ${BG_GAIN:+--bg-gain "$BG_GAIN"} ${BG_CLIP:+--background-clip "$BG_CLIP"} \
      --episodes-path "$DS" --scene "$SCENE" --target any \
      --n-episodes "$N_EPISODES" --out-dir "$QX_DIR" 2>&1 | tee "${QX_DIR}.log"
  qx_rc=${PIPESTATUS[0]}
  # complete AND expansion actually fired (n_query_expanded>0) — else a vacuous A/B.
  qx_ok="$(python -c "import json,sys
try:
    s=json.load(open(sys.argv[1]))
    a=s.get('n_episodes_attempted',0); c=s.get('n_episodes_completed',0)
    print(1 if a>0 and c==a and s.get('n_query_expanded',0)>0 else 0)
except Exception:
    print(0)" "$QX_DIR/summary.json" 2>/dev/null || echo 0)"
  if [ "$qx_ok" != 1 ]; then
    QX_FIRED="$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('n_query_expanded','?'))" "$QX_DIR/summary.json" 2>/dev/null || echo '?')"
    echo "FATAL: query-expansion arm unusable at $QX_DIR (rc=$qx_rc; n_query_expanded=$QX_FIRED)."
    echo "  Either the run crashed, or expansion NEVER FIRED (n_query_expanded=0 → the arm is"
    echo "  byte-identical to baseline S3 → a vacuous A/B). Firing needs first-pass memory hits;"
    echo "  check that this cell recalls at all (baseline S3 n_memory_chosen>0). See ${QX_DIR}.log."
    exit 1
  fi
  echo "  query-expansion arm complete (rc=$qx_rc; expansion fired, n_query_expanded>0)"
fi

banner "[7/7] Gate-A verdict (warm paired soft-SPL S3-S1) + controller census"
N_RUN_DIRS=$(set -- $OUT_DIRS; echo $#)
if [ "$N_RUN_DIRS" -lt 2 ]; then
  echo "  [gate-A] skipped: need >=2 settings, got $N_RUN_DIRS"
else
  # shellcheck disable=SC2086
  python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS 2>&1 | tee "runs/${TAG}-${CATEGORY}-analysis.log"
fi
echo

# Query-fix A/B verdict: paired soft-SPL of the query-expansion S3 (B) vs the
# baseline S3 (A) — the direct test of whether the query change beats the
# wrong-instance over-fire baseline (not S3-vs-S1, which is the LTM effect).
if [ -n "$QX_DIR" ]; then
  BASE_S3="runs/${TAG}-${CATEGORY}-s3"
  banner "[7b/7] query-expansion A/B verdict (paired S3: expansion-on B − baseline-off A)"
  if [ -f "$BASE_S3/summary.json" ]; then
    python embodied_memory/scripts/analyze_revisit.py \
        --compare-a "$BASE_S3" --compare-b "$QX_DIR" \
        2>&1 | tee "runs/${TAG}-${CATEGORY}-queryexp-compare.log"
  else
    echo "  [query-fix A/B] skipped: baseline S3 $BASE_S3/summary.json missing (need setting 3 in --settings)."
  fi
  echo
fi

banner "controller census (did the interrupt→investigate→resume→report machine run?)"
for d in $OUT_DIRS $QX_DIR; do
  echo "  --- $d ---"
  python embodied_memory/scripts/diagnose_anomaly_controller.py "$d" 2>&1 | tail -20 || true
done
echo

echo "DONE. N3 anomaly-response S1-vs-S3 for ($SCENE, primary=$CATEGORY, source=DECOUPLED)."
echo "The cold pass silent-maps; warm passes fire the anomaly from a DIFFERENT object;"
echo "the controller diverts to it and must RESUME the find-task. Onset: grep '\\[audio\\]' the logs."
exit 0
