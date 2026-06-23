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
# SHOWCASE-BY-DEFAULT: this driver defaults to several WARM visits (--warm 4)
# because a COLD audiogoal episode starts the robot AT the goal view-point
# (pick_cold_pose) → it barely moves → a ~2 s clip that ALSO ends before the onset
# step (so its whole soundtrack is silence), AND the LTM recalls nothing (empty fine
# layer, memory_bridge.py:1042). A WARM visit starts the robot FAR from the goal: it
# hears the alarm, navigates in, and the LTM RECALLS the prior sighting — a clip that
# actually shows memory working. ACT 2 is picked (in the TESTED picker
# make_two_act_demo.pick_two_acts) as the BEST warm RECALL episode — one that ARRIVED
# (success_1m / distance_to_goal<1m) AND whose memory FIRED (n_memory_chosen>=1) AND
# ended in a STOP — preferring the SHORTEST such (a crisp arrival), NOT the longest.
# This fixes the old "longest-wins" bug where a 250-step timeout (the wander that
# never found the bed) was chosen and "the video just stopped" mid-wander. Warm
# arrival is ~0.67, so 4 tries make a clean arrival near-certain (P(all fail)~1.6%).
# If NO warm episode arrives, ACT 2 falls back to the longest-past-onset episode
# (so the soundtrack is non-silent) with a LOUD banner that the demo shows
# exploration, not arrival. (Every field the picker reads — success_1m,
# n_memory_chosen, n_arrival_stop, n_stop_signals, distance_to_goal, n_steps — is in
# episode_NNN.json/ep_log; the dead n_audio_onset_fired summary-only field is gone.)
# The render step additionally REFUSES to mux a silent track.
# Force the literal cold (short) demo with --cold.
# Length levers (defaults chosen so a 30-60 step episode is >=8-10s):
#   --min-dist (default 5 m → robot starts ~2 rooms away so it actually travels)
#   --keyframe-every (default 1 → a frame every sim-step, dense+smooth)
#   --video-fps (default 4 → slower playback, longer clip)
#   --t-anom (default 5 → anomaly fires near the start, audible most of the clip)
# Duration ≈ (n_steps / keyframe_every + 1) / video_fps, reported at the end.
# render_demo_audio_track REFUSES to mux a silent track (errors non-zero) and the
# driver prints the soundtrack RMS/duration + ffprobe-confirms a non-silent audio
# stream after the mux, so the run log PROVES audio is present.
# Binaural pan is ILD-only (ITD-stripped by SoundSpaces) → a clear loudness
# gradient, weak left/right stereo image.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
MINICONDA="${HOME}/miniconda3"; SS_ENV="soundspaces-spike"; LTM_ENV="ltm-embodied"

# Default a known-good cell. wcojb4TFT35 + baby_cry→crib / chair is M0c-demo good.
SCENE="wcojb4TFT35"; CLASS="baby_cry"; CATEGORY="bed"
# Demo length/onset defaults tuned so a 30-60 step episode renders >=8-10s with
# audible sound (see [7/7] formula): keyframe_every=1 (a frame EVERY sim-step →
# dense, slow playback) + video_fps=4 (slower → longer clip) + min_dist=5 (more
# travel). T_ANOM_FIRE=5 fires the anomaly NEAR THE START so it plays for almost
# the whole clip AND even short episodes capture the onset. All overridable.
TAG="demo-video"; VIDEO_FPS=4; T_ANOM_FIRE=5; SOURCE_OVERRIDE=""
ONSET_TARGET_DIST="4.0"; ONSET_RMS_OVERRIDE=""; KEYFRAME_EVERY=1; MIN_DIST="3.0"
# DEFAULT = warm showcase (robot travels + recalls). --cold forces the short
# at-the-goal cold demo. NWARM=4 (not 2): warm arrival rate is ~0.67, so with 4
# tries P(no warm episode arrives) ≈ 0.33^4 ≈ 1.6% — the picker almost always has a
# real recalled arrival to choose for ACT 2 (vs ~11% with 2 tries). Override --warm N.
WARM=1; NWARM=4
while [ $# -gt 0 ]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --keyframe-every) KEYFRAME_EVERY="$2"; shift 2 ;;
    --min-dist) MIN_DIST="$2"; shift 2 ;;
    --cold) WARM=0; NWARM=0; shift 1 ;;
    # The onset step for the DEMO episode (cold/first visit fires here). Low so
    # the anomaly sounds early in the clip.
    --t-anom) T_ANOM_FIRE="$2"; shift 2 ;;
    --source) SOURCE_OVERRIDE="$2"; shift 2 ;;
    --onset-target-dist) ONSET_TARGET_DIST="$2"; shift 2 ;;
    --onset-rms) ONSET_RMS_OVERRIDE="$2"; shift 2 ;;
    # --warm: build N warm episodes (default 4) so the picker has several chances
    # at a WARM episode where the LTM actually RECALLS the prior sighting and the
    # robot ARRIVES — ACT 2 is then chosen as the cleanest such recall story.
    --warm) WARM=1; NWARM="${2:-4}"; shift 2 ;;
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
# Build the dataset. A two-act demo NEEDS a warm (recall) episode, but a too-large
# --min-dist on a small cell yields ONLY the cold episode (no navigable warm start
# survives the filter) → the run produces just episode_000 → the short single clip.
# So if NWARM>0 and no warm episode is built, RETRY with a smaller min_dist, and
# FAIL LOUDLY if none can be built (rather than silently shipping the cold clip).
# --t-anom-cold/-warm both = T_ANOM_FIRE so BOTH acts carry audio.
N_BUILT=0; MD_USED=""
for MD in "$MIN_DIST" 2.5 2.0 1.5 1.0; do
  rm -rf "$DS_DIR"
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_audiogoal_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories "$CATEGORY" --n-warm "$NWARM" \
      --anomaly-class "$CLASS" --name "$NAME" --min-dist "$MD" \
      --t-anom-cold "$T_ANOM_FIRE" --t-anom-warm "$T_ANOM_FIRE" \
      --out-dir "$DS_DIR" --source-manifest "$MANIFEST" $SRC_ARG \
    || { echo "FATAL: dataset build failed."; exit 1; }
  [ -f "$DS" ] && [ -f "$MANIFEST" ] || { echo "FATAL: dataset or manifest missing"; exit 1; }
  N_BUILT="$(python -c "import gzip,json,glob; print(sum(len(json.load(gzip.open(f))['episodes']) for f in glob.glob('$DS_DIR/content/*.json.gz')))" 2>/dev/null || echo 0)"
  MD_USED="$MD"
  echo "  min_dist=$MD → $N_BUILT episode(s) (1 cold + $((N_BUILT-1)) warm)"
  # cold-only (--cold, NWARM=0) is fine with 1; a two-act needs >=2 (a warm).
  { [ "$NWARM" -eq 0 ] || [ "${N_BUILT:-0}" -ge 2 ]; } && break
  echo "  ↳ no warm start at min_dist=$MD — retrying smaller so the two-act has a recall episode…"
done
if [ "$NWARM" -gt 0 ] && [ "${N_BUILT:-0}" -lt 2 ]; then
  echo "FATAL: no WARM (recall) episode could be built for ($SCENE,$CATEGORY) at any min_dist."
  echo "       This cell's candidate start poses are all near the goal. Try a different"
  echo "       --scene/--category, or run with --cold for the short single-visit clip."
  exit 1
fi
echo "  built $N_BUILT episode(s) at min_dist=$MD_USED"

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

[[ "$KEYFRAME_EVERY" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: --keyframe-every must be a positive integer"; exit 1; }

banner "[5c/7] ensure ffmpeg (needed for the audio mux)"
if command -v ffmpeg >/dev/null 2>&1; then
  echo "  system ffmpeg: $(command -v ffmpeg)"
elif python -c "import imageio_ffmpeg,os,sys; sys.exit(0 if os.path.exists(imageio_ffmpeg.get_ffmpeg_exe()) else 1)" 2>/dev/null; then
  echo "  bundled ffmpeg (imageio-ffmpeg): $(python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')"
else
  echo "  no ffmpeg found — installing a bundled static binary (imageio-ffmpeg)…"
  if pip install -q imageio-ffmpeg 2>/dev/null \
       && python -c "import imageio_ffmpeg,os,sys; sys.exit(0 if os.path.exists(imageio_ffmpeg.get_ffmpeg_exe()) else 1)" 2>/dev/null; then
    echo "  installed imageio-ffmpeg: $(python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')"
  elif conda install -y -c conda-forge ffmpeg >/dev/null 2>&1 && command -v ffmpeg >/dev/null 2>&1; then
    echo "  installed system ffmpeg via conda: $(command -v ffmpeg)"
  else
    echo "  WARN: ffmpeg install failed — the run still proceeds, but the mux will be"
    echo "        skipped and the soundtrack written as a separate .wav (mux manually)."
  fi
fi

banner "[6/7] run S3 (--task audiogoal --backbone remembr --save-video, 7B planner)"
rm -f "$OUT_DIR/summary.json"
# shellcheck disable=SC2086
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --setting 3 --task audiogoal \
    --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_FIRE" \
    --audio-onset-rms "$ONSET_RMS" ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} \
    --episodes-path "$DS" --scene "$SCENE" --target any \
    --n-episodes "$N_EPISODES" --out-dir "$OUT_DIR" \
    --save-video --video-fps "$VIDEO_FPS" --keyframe-every "$KEYFRAME_EVERY" 2>&1 | tee "${OUT_DIR}.log"
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

banner "[7/7] post-hoc soundtrack + ffmpeg mux → demo video"
# ============================================================================
# TWO-ACT path (default; NWARM>0). Stitch ONE story from two episodes of this run:
#   ACT 1 = the COLD seed (smallest episode_idx with a video — idx 0; the robot is
#           spawned at the bed, observes & memorizes it).
#   ACT 2 = the BEST warm RECALL episode (TESTED picker pick_two_acts): one that
#           ARRIVED (success_1m / distance_to_goal<1m) AND whose memory FIRED
#           (n_memory_chosen>=1) AND ended in a STOP, preferring the SHORTEST such
#           (crisp arrival) — NOT the longest (the old bug picked the 250-step
#           timeout wander). Falls back to longest-past-onset only if no warm
#           episode arrived (loud banner). Both acts fire the alarm (driver passes
#           t_anom to every episode) so both have audio → concat into demo_two_act.mp4.
# The single-clip path (below, NWARM==0 / --cold) is the fallback when there is no
# warm episode to play ACT 2.
# ============================================================================
if [ "$NWARM" -gt 0 ] 2>/dev/null; then
  # Pick ACT1 (cold seed) and ACT2 (the BEST warm RECALL episode) via the TESTED
  # picker in make_two_act_demo.py (`--pick`). It ranks ACT2 recall-story-first —
  # ARRIVED (success_1m / d2g<1m) AND memory FIRED (n_memory_chosen>=1) AND ended in
  # a STOP > merely ARRIVED > (last resort) longest-past-onset — then SHORTEST within
  # a tier, so a clean arrival wins over a 250-step timeout (the old longest-wins
  # picker's bug). It also prints a per-warm-episode diagnostic table (steps /
  # success_1m / mem_chosen / remembr_chosen / min_d2g) so recall-fired-or-not is
  # VISIBLE, and a loud banner when NO warm episode arrived. All fields read are in
  # episode_NNN.json (ep_log) — no summary.json, no dead n_audio_onset_fired field.
  PICK2_OUT="$(python embodied_memory/scripts/make_two_act_demo.py --pick \
      --run-dir "$OUT_DIR" --t-anom "$T_ANOM_FIRE" 2>&1)"
  # Echo the diagnostic table + any warning banner into the run log.
  printf '%s\n' "$PICK2_OUT" | grep -v $'^PICK\t'
  PICK2_LINE="$(printf '%s\n' "$PICK2_OUT" | grep $'^PICK\t' | tail -1)"
  P2_STATUS="$(printf '%s' "$PICK2_LINE" | cut -f2)"
  ACT1_EP="$(printf '%s' "$PICK2_LINE" | cut -f3)"
  ACT2_EP="$(printf '%s' "$PICK2_LINE" | cut -f4)"
  if [ -z "$P2_STATUS" ] || [ "$P2_STATUS" = "NONE" ]; then
    echo "FATAL: no episode_NNN.json with a video_path in $OUT_DIR — was --save-video honoured? See ${OUT_DIR}.log."; exit 1
  elif [ "$P2_STATUS" = "SOLO" ]; then
    echo "  ⚠ only ONE episode has a video — cannot build two acts; falling back to single-clip on it."
    LAST_EP="$ACT1_EP"; NWARM=0   # fall through to the single-clip path below
  else
    # status ∈ {OK, ARRIVED, NOFIRE}; the picker already printed the per-warm
    # diagnostic table above. OK = arrived + memory fired (the clean recall story);
    # ARRIVED = arrived but recall didn't visibly drive it; NOFIRE = no warm episode
    # reached the bed → best-effort exploration clip (loud honest banner).
    [ "$P2_STATUS" = "NOFIRE" ] && {
      echo "  ════════════════════════════════════════════════════════════════"
      echo "  ⚠ no warm episode reached the bed (recall did not pay off in this"
      echo "    cell) — ACT 2 shows EXPLORATION, not arrival. The demo still"
      echo "    renders on the best-effort (longest-past-onset) episode so it is"
      echo "    not silent, but it does NOT show a recalled arrival."
      echo "    Try --warm 6 or a different --scene/--class/--category."
      echo "  ════════════════════════════════════════════════════════════════"; }
    A1_STEPS="$(python -c "import json;print(json.load(open('$ACT1_EP')).get('n_steps',0))" 2>/dev/null || echo 0)"
    A2_STEPS="$(python -c "import json;print(json.load(open('$ACT2_EP')).get('n_steps',0))" 2>/dev/null || echo 0)"
    A1_DUR="$(python -c "print('%.1f' % ((($A1_STEPS/$KEYFRAME_EVERY)+1)/$VIDEO_FPS))" 2>/dev/null || echo '?')"
    A2_DUR="$(python -c "print('%.1f' % ((($A2_STEPS/$KEYFRAME_EVERY)+1)/$VIDEO_FPS))" 2>/dev/null || echo '?')"
    echo "  ACT 1 (cold seed)   = $ACT1_EP  (n_steps=$A1_STEPS → ~${A1_DUR}s)"
    echo "  ACT 2 (warm recall) = $ACT2_EP  (n_steps=$A2_STEPS → ~${A2_DUR}s)"
    # make_two_act_demo banners each act + muxes its soundtrack + concats → demo_two_act.mp4.
    # It REFUSES a silent act (exits non-zero) so a silent demo can never ship.
    # shellcheck disable=SC2086
    TWO_LOG="$(python embodied_memory/scripts/make_two_act_demo.py \
        --run-dir "$OUT_DIR" --act1-episode "$ACT1_EP" --act2-episode "$ACT2_EP" \
        --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_FIRE" \
        --fps "$VIDEO_FPS" --out-name demo_two_act.mp4 \
        ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} 2>&1)"
    TWO_RC=$?
    echo "$TWO_LOG"
    [ "$TWO_RC" -eq 0 ] || { echo "FATAL: two-act stitch failed (rc=$TWO_RC) — see above. A demo act was SILENT or empty; refusing to ship it."; exit 1; }
    TWO_SECS="$(printf '%s\n' "$TWO_LOG" | grep -oE 'TWO_ACT_SECONDS=[0-9.]+' | tail -1 | cut -d= -f2)"
    FINAL="$OUT_DIR/demo_two_act.mp4"
    banner "DONE"
    if [ -f "$FINAL" ]; then
      if command -v ffprobe >/dev/null 2>&1; then
        ASTREAM="$(ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "$FINAL" 2>/dev/null | head -1)"
        echo "  ffprobe audio stream in $FINAL: ${ASTREAM:-<none>}"
        [ "$ASTREAM" = "audio" ] || echo "  ⚠ ffprobe found NO audio stream in the two-act mp4 (expected one)."
      fi
      echo "  TWO-ACT DEMO VIDEO: $FINAL"
      echo "    ACT 1 (cold seed):   n_steps=$A1_STEPS (~${A1_DUR}s)"
      echo "    ACT 2 (warm recall): n_steps=$A2_STEPS (~${A2_DUR}s)"
      echo "    total ~${TWO_SECS:-?}s, fps=$VIDEO_FPS"
    else
      echo "  ffmpeg absent — per-act mp4s written (act1.mp4 / act2.mp4); concat them manually (see [two-act] above)."
    fi
    exit 0
  fi
fi

# ============================================================================
# SINGLE-CLIP path (NWARM==0 / --cold, or the SOLO degrade above).
# ============================================================================
# PICK THE SOUNDTRACK EPISODE: the LONGEST episode that RAN PAST THE ONSET STEP
# (n_steps >= t_anom+3). This guarantees (a) the anomaly had time to fire → the
# soundtrack is non-silent (backstopped by the render's SilentSoundtrackError) and
# (b) length-first picks the travel+recall warm showcase, not the short cold-at-goal
# pass. We DROP the old "prefer success" rule: a cold at-goal episode trivially
# "succeeds" in ~22 steps and was winning the picker, yielding the 2 s silent clip.
# (NB the earlier "n_audio_onset_fired>=1" attempt read a field absent from the
# per-episode log → always 0 → dead no-op; n_steps is the real, log-present signal.)
# Loud-warn-and-fall-back only if NOTHING ran past onset (build
# is broken — lower --t-anom / raise --min-dist).
# (A SOLO two-act fallback above already set $LAST_EP; honour it.)
if [ -z "${LAST_EP:-}" ]; then
PICK="$(python -c "
import json,glob
T_ANOM=int($T_ANOM_FIRE)
viable=[]; any_video=[]
for f in sorted(glob.glob('$OUT_DIR/episode_*.json')):
    if '_error' in f: continue
    try: ep=json.load(open(f))
    except Exception: continue
    if not ep.get('video_path'): continue
    n=int(ep.get('n_steps',0) or 0)
    any_video.append((n,f))
    # 'Ran past the onset step' is the robust filter for 'the anomaly had time to
    # fire'. Use n_steps — it IS a real episode_NNN.json (ep_log) field, whereas
    # n_audio_onset_fired lives ONLY in summary/metrics and is NEVER in ep_log
    # (reading it returned 0 for every episode = a dead no-op). The render-side
    # RMS guard backstops the rare ran-past-onset-but-silent case.
    if n >= T_ANOM + 3: viable.append((n,f))
if viable:
    viable.sort(key=lambda c: c[0], reverse=True)
    print('OK\t'+viable[0][1])
elif any_video:
    any_video.sort(key=lambda c: c[0], reverse=True)
    print('NOFIRE\t'+any_video[0][1])
else:
    print('NONE\t')
")"
PICK_STATUS="${PICK%%$'\t'*}"; LAST_EP="${PICK#*$'\t'}"
if [ "$PICK_STATUS" = "NOFIRE" ]; then
  echo "  ⚠⚠ WARNING: NO episode ran past the onset step (every episode < t_anom=$T_ANOM_FIRE + 3 steps)."
  echo "     The build is BROKEN for a demo — the anomaly never had time to fire."
  echo "     LOWER --t-anom (already $T_ANOM_FIRE; try 3) and/or RAISE --min-dist so episodes run longer."
  echo "     Falling back to the longest episode anyway; the soundtrack render WILL likely error"
  echo "     (refuses to mux silence) — that is the correct, loud failure."
elif [ "$PICK_STATUS" = "NONE" ]; then
  LAST_EP=""
fi
fi  # end: if [ -z "${LAST_EP:-}" ] — SOLO two-act fallback skips the picker
[ -n "$LAST_EP" ] || { echo "FATAL: no episode_NNN.json with a video_path in $OUT_DIR — was --save-video honoured? See ${OUT_DIR}.log."; exit 1; }
EP_STEPS="$(python -c "import json;print(json.load(open('$LAST_EP')).get('n_steps',0))" 2>/dev/null || echo 0)"
EST_DUR="$(python -c "print('%.1f' % ((($EP_STEPS/$KEYFRAME_EVERY)+1)/$VIDEO_FPS))" 2>/dev/null || echo '?')"
echo "  soundtrack episode = $LAST_EP  (n_steps=$EP_STEPS → ~${EST_DUR}s @ keyframe_every=$KEYFRAME_EVERY, fps=$VIDEO_FPS)"
SHORT="$(python -c "print(1 if ((($EP_STEPS/$KEYFRAME_EVERY)+1)/$VIDEO_FPS) < 6 else 0)" 2>/dev/null || echo 0)"
[ "$SHORT" = 1 ] && echo "  ⚠ short clip (~${EST_DUR}s) — for longer: --min-dist 6, --keyframe-every 1, or --warm 3 (re-run)."
# render_demo_audio_track REFUSES to mux a silent track (exits non-zero) → a silent
# demo can no longer ship undetected. Capture its stdout so we can echo the proven
# soundtrack RMS/duration into the run log.
# shellcheck disable=SC2086
RENDER_LOG="$(python embodied_memory/scripts/render_demo_audio_track.py \
    --run-dir "$OUT_DIR" --episode-json "$LAST_EP" \
    --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_FIRE" \
    --fps "$VIDEO_FPS" ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} 2>&1)"
RENDER_RC=$?
echo "$RENDER_LOG"
[ "$RENDER_RC" -eq 0 ] || { echo "FATAL: soundtrack render/mux failed (rc=$RENDER_RC) — see above. The demo would have been SILENT or empty; refusing to ship it."; exit 1; }
DEMO_RMS="$(printf '%s\n' "$RENDER_LOG" | grep -oE 'DEMO_AUDIO_RMS=[0-9.]+' | tail -1 | cut -d= -f2)"
DEMO_SECS="$(printf '%s\n' "$RENDER_LOG" | grep -oE 'DEMO_AUDIO_SECONDS=[0-9.]+' | tail -1 | cut -d= -f2)"
echo "  AUDIO CONFIRMED: soundtrack RMS=${DEMO_RMS:-?} (>0 = audible), duration=${DEMO_SECS:-?}s"

FINAL="$OUT_DIR/demo_with_sound.mp4"
banner "DONE"
if [ -f "$FINAL" ]; then
  # PROVE the muxed mp4 actually has a non-silent audio stream (ffprobe if present).
  if command -v ffprobe >/dev/null 2>&1; then
    ASTREAM="$(ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "$FINAL" 2>/dev/null | head -1)"
    echo "  ffprobe audio stream in $FINAL: ${ASTREAM:-<none>}"
    [ "$ASTREAM" = "audio" ] || echo "  ⚠ ffprobe found NO audio stream in the muxed mp4 (expected one)."
  else
    echo "  (ffprobe not on PATH — soundtrack RMS=${DEMO_RMS:-?} above already proves audible audio was muxed)"
  fi
  echo "  DEMO VIDEO WITH SOUND: $FINAL  (~${EST_DUR}s, $EP_STEPS steps, fps=$VIDEO_FPS, audio RMS=${DEMO_RMS:-?})"
else
  echo "  ffmpeg absent — soundtrack written separately. WAV: $OUT_DIR/demo_track.wav"
  echo "  Silent clip: $(python -c "import json;print('$OUT_DIR/'+ (json.load(open('$LAST_EP')).get('video_path') or 'video/episode_000.mp4'))")"
  echo "  Mux manually with the printed [demo-audio] command above."
fi
