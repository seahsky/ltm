#!/bin/bash
# scripts/race-demo-video.sh — render a DEMO VIDEO **with sound** of the
# full-system (S3) robot on the AudioGoal (audio+visual) task.
#
# What it produces: runs/demo-video-<class>-s3/demo_with_sound.mp4 — the silent
# first-person clip (HUD overlay incl. the audiogoal audio strip) MUXED with a
# post-hoc, pose-conditioned binaural soundtrack (the alarm/cry swells as the
# agent nears the source). 7B ReMEmbR backbone, real ESC-50 audio.
#
# Pipeline (mirrors race-audiogoal.sh; the soundspaces env is SKIPPED by reusing
# a prior m3 RIR grid when present — the two-env split holds):
#   [1] git pull   [2] source race-setup (ltm-embodied)   [3] pre-verify tests
#   [4] BUILD a 1-episode audiogoal dataset (cold visit, t_anom set to FIRE)
#   [5] REUSE-or-RENDER the RIR grid at the dataset's source
#   [6] RUN S3 (--task audiogoal --backbone remembr --save-video)
#   [7] SOUNDTRACK + MUX (render_demo_audio_track) → demo_with_sound.mp4
#
#   cd ~/ltm && git checkout lifelong-revisit-eval \
#       && nrun bash scripts/race-demo-video.sh --scene wcojb4TFT35 --class baby_cry --category bed
#   (default scene/class/category = the wcojb baby_cry->bed M3 cell whose RIR grid
#    already exists, so the source stays co-located with the goal and no soundspaces
#    render is needed. Override --category ONLY together with a matching grid/source.)
#
# EXECUTE (do NOT source) — it switches conda envs in its own process.
# Aborts before the paid LLM run if pull / setup / pre-tests / build / render fail.
#
# CAVEAT (honest, put in the demo caption): on a COLD/first visit the LTM recalls
# NOTHING (the fine layer is empty → propose_memory_candidates returns [] —
# memory_bridge.py:1042). The full S3 stack is LIVE (STM/consolidation/ReMEmbR/
# rerank/memory-injection seam/audio brain) but the story the clip TELLS is the
# audio→goal→navigate→stop loop, NOT recall. A WARM visit is what showcases LTM
# recall — pass --warm to add warm episodes (then the soundtrack renders the
# LAST, warm, episode). The binaural pan is ILD-only (ITD-stripped by SoundSpaces)
# → a clear loudness gradient, weak left/right stereo image.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
MINICONDA="${HOME}/miniconda3"; SS_ENV="soundspaces-spike"; LTM_ENV="ltm-embodied"

# Default a known-good cell. wcojb4TFT35 + baby_cry→crib / chair is M0c-demo good.
SCENE="wcojb4TFT35"; CLASS="baby_cry"; CATEGORY="bed"
TAG="demo-video"; VIDEO_FPS=8; T_ANOM_FIRE=30; SOURCE_OVERRIDE=""
ONSET_TARGET_DIST="4.0"; ONSET_RMS_OVERRIDE=""; WARM=0; NWARM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    # The onset step for the DEMO episode (cold/first visit fires here). Low so
    # the anomaly sounds early in the clip.
    --t-anom) T_ANOM_FIRE="$2"; shift 2 ;;
    --source) SOURCE_OVERRIDE="$2"; shift 2 ;;
    --onset-target-dist) ONSET_TARGET_DIST="$2"; shift 2 ;;
    --onset-rms) ONSET_RMS_OVERRIDE="$2"; shift 2 ;;
    # --warm: add N warm episodes (default 2) so the FINAL episode is a WARM
    # visit where the LTM actually RECALLS the prior sighting — the recall story.
    # The soundtrack is rendered on that last (warm) episode.
    --warm) WARM=1; NWARM="${2:-2}"; shift 2 ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }
[ "$WARM" = 1 ] || NWARM=0

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/audiogoal_${TAG}"
NAME="audiogoal_${TAG}"; DS="${DS_DIR}/${NAME}.json.gz"
MANIFEST="${DS_DIR}/source_manifest.json"
GRID="runs/audiogoal/${SCENE}_${CLASS}_rir_grid.npz"
OUT_DIR="runs/${TAG}-${CLASS}-s3"
banner() { printf '\n========== %s ==========\n' "$1"; }
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }

banner "[1/7] git pull --ff-only"
if [ -n "${RACE_SKIP_PULL:-}" ]; then
  echo "  RACE_SKIP_PULL set — skipping"
else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/7] conda setup (source scripts/race-setup.sh → $LTM_ENV)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/7] pre-verify (free; abort before spend)"
for t in test_render_demo_audio_track test_make_audiogoal_smoke test_audio_task; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending on the live run."; exit 1; }
done

banner "[4/7] build 1-episode audiogoal demo dataset: scene=$SCENE class=$CLASS cat=$CATEGORY n-warm=$NWARM"
SRC="${VALMINI}/${SCENE}.json.gz"
[ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
rm -rf "$DS_DIR"
# '=' form (not a space): HM3D source coords start with '-' and contain commas,
# so argparse misreads a space-separated value as an option flag.
SRC_ARG=""; [ -n "$SOURCE_OVERRIDE" ] && SRC_ARG="--source-position=$SOURCE_OVERRIDE"
# --t-anom-cold sets the COLD (first/only) episode's onset LOW so the anomaly
# fires on the demo visit (the default cold value is 10000 = silent mapping).
# shellcheck disable=SC2086
python embodied_memory/scripts/make_audiogoal_smoke.py \
    --src "$SRC" --scene "$SCENE" --categories "$CATEGORY" --n-warm "$NWARM" \
    --anomaly-class "$CLASS" --name "$NAME" \
    --t-anom-cold "$T_ANOM_FIRE" --t-anom-warm "$T_ANOM_FIRE" \
    --out-dir "$DS_DIR" --source-manifest "$MANIFEST" $SRC_ARG \
  || { echo "FATAL: dataset build failed."; exit 1; }
[ -f "$DS" ] && [ -f "$MANIFEST" ] || { echo "FATAL: dataset or manifest missing"; exit 1; }

SRC_XYZ="$(python -c "
import json
m=json.load(open('$MANIFEST'))
hit=[e for e in m if e['anomaly_class']=='$CLASS' and (e['scene_id'] or '').endswith('$SCENE')]
assert hit and hit[0]['source_position'], 'no source for ($SCENE,$CLASS) in manifest'
print(','.join('%.6f'%v for v in hit[0]['source_position']))
")" || { echo "FATAL: could not read source from manifest"; exit 1; }
echo "  source for ($SCENE,$CLASS) = $SRC_XYZ"

banner "[5/7] RIR grid (reuse prior m3 grid if present → skip soundspaces env)"
if [ -f "$GRID" ]; then
  echo "  REUSE existing grid: $GRID (two-env split: no soundspaces render needed)"
else
  echo "  no prior grid at $GRID — rendering one ($SS_ENV)"
  GLB="$(find data/hm3d -name "${SCENE}.basis.glb" 2>/dev/null | head -1)"
  [ -n "$GLB" ] || GLB="$(find data/hm3d -name "*${SCENE}*.glb" 2>/dev/null | grep -v semantic | head -1)"
  [ -n "$GLB" ] || { echo "FATAL: no .glb for $SCENE"; exit 1; }
  mkdir -p "$(dirname "$GRID")"
  set +u; conda activate "$SS_ENV" || { echo "FATAL: activate $SS_ENV failed (build it: scripts/race-soundspaces-spike.sh)"; exit 1; }; set -u
  python embodied_memory/scripts/render_rir_grid.py \
      --scene "$GLB" --source="$SRC_XYZ" --out "$GRID" --n-cells 24 \
      2>&1 | tee "${DS_DIR}/render.log"
  rc=${PIPESTATUS[0]}
  set +u; conda activate "$LTM_ENV" || { echo "FATAL: re-activate $LTM_ENV failed"; exit 1; }; set -u
  { [ "$rc" -eq 0 ] && [ -f "$GRID" ]; } \
    || { echo "FATAL: RIR render failed (rc=$rc) — see ${DS_DIR}/render.log. NOT spending on the LLM run."; exit 1; }
fi

# Anomaly audio: stage the real ESC-50 clip; fall back to synthetic burst.
ANOMALY_CLIP="data/anomaly_audio/${CLASS}.wav"
if [ ! -f "$ANOMALY_CLIP" ]; then
  banner "[stage] fetch ESC-50 anomaly clip ($CLASS)"
  python embodied_memory/scripts/fetch_anomaly_clips.py --classes "$CLASS" \
    || echo "WARN: ESC-50 fetch failed; falling back to synthetic burst"
fi
if [ -f "$ANOMALY_CLIP" ]; then
  echo "  anomaly audio: REAL ESC-50 clip -> $ANOMALY_CLIP"
else
  echo "  anomaly audio: SYNTHETIC burst (no $ANOMALY_CLIP)"; ANOMALY_CLIP=""
fi

# Onset calibration so the anomaly is audible ~ONSET_TARGET_DIST m out.
if [ -n "$ONSET_RMS_OVERRIDE" ]; then
  ONSET_RMS="$ONSET_RMS_OVERRIDE"; echo "  onset_rms pinned = $ONSET_RMS"
else
  banner "[5b/7] onset calibration (target ${ONSET_TARGET_DIST} m)"
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

banner "[6/7] run S3 (--task audiogoal --backbone remembr --save-video, 7B planner)"
rm -f "$OUT_DIR/summary.json"
# shellcheck disable=SC2086
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --setting 3 --task audiogoal \
    --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_FIRE" \
    --audio-onset-rms "$ONSET_RMS" ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} \
    --episodes-path "$DS" --scene "$SCENE" --target any \
    --n-episodes "$N_EPISODES" --out-dir "$OUT_DIR" \
    --save-video --video-fps "$VIDEO_FPS" 2>&1 | tee "${OUT_DIR}.log"
rc=${PIPESTATUS[0]}
# Gate on COMPLETION, not rc (run_hm3d_pol returns 1 on S3-oriented pass-condition
# fails too — only NO summary.json / n_completed<n_attempted is a hard crash).
complete="$(python -c "import json,sys
try:
    s=json.load(open(sys.argv[1])); a=s.get('n_episodes_attempted',0); c=s.get('n_episodes_completed',0)
    print(1 if a>0 and c==a else 0)
except Exception:
    print(0)" "$OUT_DIR/summary.json" 2>/dev/null || echo 0)"
[ "$complete" = 1 ] \
  || { echo "FATAL: S3 run INCOMPLETE at $OUT_DIR (rc=$rc; summary missing or n_completed<n_attempted = hard crash). See ${OUT_DIR}.log."; exit 1; }
echo "  S3 complete (rc=$rc; rc=1 can be normal — pass_conditions are S3-oriented)"

banner "[7/7] post-hoc soundtrack + ffmpeg mux → demo_with_sound.mp4"
# Render the soundtrack for the LAST episode (the warm/recall one if --warm, else
# the single cold demo episode). Pick the highest-numbered episode_NNN.json that
# actually got a video.
LAST_EP="$(python -c "
import os,json,glob
best=None
for f in sorted(glob.glob('$OUT_DIR/episode_*.json')):
    if '_error' in f: continue
    try: ep=json.load(open(f))
    except Exception: continue
    if ep.get('video_path'): best=f
print(best or '')
")"
[ -n "$LAST_EP" ] || { echo "FATAL: no episode_NNN.json with a video_path in $OUT_DIR — was --save-video honoured? See ${OUT_DIR}.log."; exit 1; }
echo "  soundtrack episode = $LAST_EP"
# shellcheck disable=SC2086
python embodied_memory/scripts/render_demo_audio_track.py \
    --run-dir "$OUT_DIR" --episode-json "$LAST_EP" \
    --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_FIRE" \
    --fps "$VIDEO_FPS" ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} \
  || { echo "FATAL: soundtrack render/mux failed."; exit 1; }

FINAL="$OUT_DIR/demo_with_sound.mp4"
banner "DONE"
if [ -f "$FINAL" ]; then
  echo "  DEMO VIDEO WITH SOUND: $FINAL"
else
  echo "  ffmpeg absent — soundtrack written separately. WAV: $OUT_DIR/demo_track.wav"
  echo "  Silent clip: $(python -c "import json;print('$OUT_DIR/'+ (json.load(open('$LAST_EP')).get('video_path') or 'video/episode_000.mp4'))")"
  echo "  Mux manually with the printed [demo-audio] command above."
fi
