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
NWARM=3; SETTINGS="1 3"; TAG="audiogoal"; OUT_TAG=""; T_ANOM_WARM=30; SOURCE_OVERRIDE=""; REUSE_DS=""
ONSET_TARGET_DIST="4.0"; ONSET_RMS_OVERRIDE=""; FETCH_AUDIO=""; AUDIO_WRITE=""; LIFELONG=""
CELL_TAG=""; SRC_CONTENT_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    # Lifelong cross-visit (oracle upper bound): build the dataset with INVERTED
    # t_anom polarity (seed FIRES + writes, recall SILENT + far). The builder runs
    # its $0 construction gate and refuses to build on a FAIL.
    --lifelong) LIFELONG=1; shift ;;
    --scene) SCENE="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --category) CATEGORY="$2"; shift 2 ;;
    # --cell-tag keys the GRID / out-dir / analysis-log NAMES (default = $CLASS,
    # byte-identical legacy behaviour). The scale-up driver passes the CATEGORY so
    # that two categories sharing an anomaly class (only 3 classes exist, but a
    # scene can hold 5–6 categories) don't collide on the same grid/out-dir. The
    # real anomaly CLASS (the sound + CLAP label + --anomaly-class) is unchanged.
    --cell-tag) CELL_TAG="$2"; shift 2 ;;
    # --src-content-dir overrides where the builder reads <scene>.json.gz from
    # (default val_mini/content). Point it at .../v1/val/content to build the 18
    # full-val scenes that are not in the 2-scene val_mini split.
    --src-content-dir) SRC_CONTENT_DIR="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --settings) SETTINGS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    # --out-tag decouples the OUTPUT dir tag from the DATASET tag, so a temporal
    # A/B arm can REUSE the baseline's dataset (--tag m3-<scene> --reuse-dataset)
    # while writing to distinct out-dirs (--out-tag m3t-<scene>) — never clobbering
    # the baseline run. Defaults to --tag (unchanged behaviour for every caller).
    --out-tag) OUT_TAG="$2"; shift 2 ;;
    --t-anom) T_ANOM_WARM="$2"; shift 2 ;;
    --source) SOURCE_OVERRIDE="$2"; shift 2 ;;
    --reuse-dataset) REUSE_DS=1; shift ;;
    # Onset calibration: auto-set --audio-onset-rms so the anomaly is audible at
    # ~ONSET_TARGET_DIST m (default 4 m, across a room) instead of point-blank
    # (the step-130 finding). --onset-rms pins an explicit value and skips calib.
    --onset-target-dist) ONSET_TARGET_DIST="$2"; shift 2 ;;
    --onset-rms) ONSET_RMS_OVERRIDE="$2"; shift 2 ;;
    # Stage real ESC-50 anomaly audio (data/anomaly_audio/<class>.wav) before the
    # run, instead of the synthetic burst. Auto-resolved by run_hm3d_pol; passed
    # to the onset calibration too so the energy scale matches.
    --fetch-audio) FETCH_AUDIO=1; shift ;;
    # Step 2 (audio→LTM write): export LTM_AUDIO_WRITE=1 INSIDE the driver so it
    # reaches `python -m embodied_memory.run_hm3d_pol` WITHOUT relying on ambient
    # env inheritance through `nrun` (the documented robust pattern — see
    # race-cross-env.sh: an ambient prefix once mis-fired as rc=127; exporting is
    # robust). episode_runner.py:1168 reads os.environ.get("LTM_AUDIO_WRITE").
    --audio-write) AUDIO_WRITE=1; shift ;;
    # --task selects the runner task. Default audiogoal (sound==goal, legacy,
    # byte-identical). anomaly_response = the interrupt-investigate-resume
    # controller; on THIS dataset the source is co-located with the goal, so it
    # is a WIRING smoke (onset->divert->CHECK->RESUME->complete->report, no crash),
    # NOT the real eval (which needs a non-LOS source + benign distractor — N3).
    --task) TASK="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }
OUT_TAG="${OUT_TAG:-$TAG}"
[[ "$OUT_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --out-tag must be alnum/dash/underscore"; exit 1; }
CELL_TAG="${CELL_TAG:-$CLASS}"   # default = class → legacy out-dir/grid names unchanged
[[ "$CELL_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --cell-tag must be alnum/dash/underscore"; exit 1; }
TASK="${TASK:-audiogoal}"        # default = audiogoal → legacy run line byte-identical
case "$TASK" in audiogoal|anomaly_response) ;; *) echo "FATAL: --task must be audiogoal|anomaly_response"; exit 1 ;; esac
# --audio-write: thread the Step-2 lever as a first-class flag (export INSIDE the
# driver, so it never depends on ambient inheritance through nrun). The env-var
# form (export LTM_AUDIO_WRITE=1 before nrun) still works and is honoured below.
[ -n "$AUDIO_WRITE" ] && export LTM_AUDIO_WRITE=1
[ -n "${LTM_TEMPORAL_CONTEXT:-}" ] && echo "  [temporal] LTM_TEMPORAL_CONTEXT=$LTM_TEMPORAL_CONTEXT (weight=${LTM_TEMPORAL_WEIGHT:-0.05}) — M4 temporal-context head ON for this run"
[ -n "${LTM_AUDIO_DOA:-}" ] && echo "  [audio-doa] LTM_AUDIO_DOA=$LTM_AUDIO_DOA — S1 onset-gate ON (suppress memory injection until the anomaly is heard → audio causally necessary)"
[ -n "${LTM_AUDIO_DOA_HEAD:-}" ] && echo "  [audio-doa] LTM_AUDIO_DOA_HEAD=$LTM_AUDIO_DOA_HEAD — S2 head ON (weight=${LTM_AUDIO_DOA_WEIGHT:-0.05}): boost the same-category memory candidate whose WORLD direction matches the heard ILD sign (world frame, INVERTED convention)"
[ -n "${LTM_AUDIO_WRITE:-}" ] && echo "  [audio-write] LTM_AUDIO_WRITE=$LTM_AUDIO_WRITE — Step-2 audio→LTM write ON: on onset, persist a fine-layer item AT THE SOURCE xyz (grep '[audio-write]' in the logs to confirm onset→write→recall fires)"

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
SRC_CONTENT_DIR="${SRC_CONTENT_DIR:-$VALMINI}"   # default = val_mini; --src-content-dir for full val
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/audiogoal_${TAG}"
NAME="audiogoal_${TAG}"; DS="${DS_DIR}/${NAME}.json.gz"
MANIFEST="${DS_DIR}/source_manifest.json"
# Grid is keyed by the CELL_TAG (category for the scale-up; class for legacy single
# cells) because the RIR depends on the SOURCE = the goal-category position, not the
# decorative anomaly class. Two categories sharing a class get distinct grids.
GRID="runs/audiogoal/${SCENE}_${CELL_TAG}_rir_grid.npz"
banner() { printf '\n========== %s ==========\n' "$1"; }
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }

banner "[1/7] git pull --ff-only"
if [ -n "${RACE_SKIP_PULL:-}" ]; then
  echo "  RACE_SKIP_PULL set — skipping (matrix driver already pulled once up front)"
else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/7] conda setup (source scripts/race-setup.sh → $LTM_ENV)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/7] pre-verify (free; abort before spend)"
for t in test_make_audiogoal_smoke test_audio_task test_make_revisit_smoke test_analyze_revisit test_planner_decision_kind test_anomaly_controller test_anomaly_wiring test_active_goal_noop; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending on the live run."; exit 1; }
done

banner "[4/7] build audiogoal dataset: scene=$SCENE class=$CLASS cat=$CATEGORY n-warm=$NWARM"
if [ -n "$REUSE_DS" ] && [ -f "$DS" ] && [ -f "$MANIFEST" ]; then
  echo "  REUSE existing dataset: $DS"
else
  SRC="${SRC_CONTENT_DIR}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  rm -rf "$DS_DIR"
  # NOTE: '=' form (not a space) — HM3D source coords start with '-' and contain
  # commas, so they don't match argparse's negative-number regex; passed with a
  # space, argparse mistakes the value for an option flag ("expected one argument").
  SRC_ARG=""; [ -n "$SOURCE_OVERRIDE" ] && SRC_ARG="--source-position=$SOURCE_OVERRIDE"
  LIFELONG_ARG=""; [ -n "$LIFELONG" ] && LIFELONG_ARG="--lifelong"
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_audiogoal_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories "$CATEGORY" --n-warm "$NWARM" \
      --anomaly-class "$CLASS" --name "$NAME" --t-anom-warm "$T_ANOM_WARM" \
      --out-dir "$DS_DIR" --source-manifest "$MANIFEST" $SRC_ARG $LIFELONG_ARG \
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
if [ -n "$REUSE_DS" ] && [ -f "$GRID" ]; then
  echo "  REUSE existing grid: $GRID (A/B arm B reuses arm A's grid — skip render + soundspaces env)"
else
GLB="$(find data/hm3d -name "${SCENE}.basis.glb" 2>/dev/null | head -1)"
[ -n "$GLB" ] || GLB="$(find data/hm3d -name "*${SCENE}*.glb" 2>/dev/null | grep -v semantic | head -1)"
[ -n "$GLB" ] || { echo "FATAL: no .glb for $SCENE"; exit 1; }
mkdir -p "$(dirname "$GRID")"
set +u; conda activate "$SS_ENV" || { echo "FATAL: activate $SS_ENV failed (build it: scripts/race-soundspaces-spike.sh)"; exit 1; }; set -u
# '=' form: $SRC_XYZ starts with '-' (HM3D coords) and has commas, so a
# space-separated value is misread by argparse as an option flag.
# -u (unbuffered): the render output is piped through tee, so Python would
# otherwise full-buffer stdout and a hard-killed render (CUDA OOM / timeout) would
# never flush its "RED: ... reachable cells" off-navmesh line to render.log — the
# downstream grep (below) would then mis-blame the source. -u keeps the log live.
python -u embodied_memory/scripts/render_rir_grid.py \
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
fi  # end render-or-reuse-grid

# Anomaly audio: prefer the staged real ESC-50 clip data/anomaly_audio/<class>.wav
# (run_hm3d_pol auto-resolves the same path; we pass it explicitly so the onset
# calibration uses the SAME clip energy scale). --fetch-audio stages it first.
ANOMALY_CLIP="data/anomaly_audio/${CLASS}.wav"
if [ -n "$FETCH_AUDIO" ] && [ ! -f "$ANOMALY_CLIP" ]; then
  banner "[stage] fetch ESC-50 anomaly clip ($CLASS)"
  python embodied_memory/scripts/fetch_anomaly_clips.py --classes "$CLASS" \
    || echo "WARN: ESC-50 fetch failed; falling back to synthetic burst"
fi
if [ -f "$ANOMALY_CLIP" ]; then
  echo "  anomaly audio: REAL ESC-50 clip -> $ANOMALY_CLIP"
else
  echo "  anomaly audio: SYNTHETIC burst (no $ANOMALY_CLIP; pass --fetch-audio to stage real ESC-50 audio)"
  ANOMALY_CLIP=""
fi

# Onset calibration: set --audio-onset-rms so the anomaly is first audible at
# ~ONSET_TARGET_DIST m (across a room) rather than point-blank — the step-130
# finding (default onset_rms=0.05 < far-cell energy ~0.046, so onset only fired
# when the agent was on top of the source). $0; reads the rendered grid.
if [ -n "$ONSET_RMS_OVERRIDE" ]; then
  ONSET_RMS="$ONSET_RMS_OVERRIDE"
  echo "  onset_rms pinned (override) = $ONSET_RMS"
else
  banner "[5b/7] onset calibration (diagnose_onset_calib, target ${ONSET_TARGET_DIST} m)"
  CALIB_LOG="${DS_DIR}/onset_calib.log"
  python embodied_memory/scripts/diagnose_onset_calib.py \
      --grid "$GRID" --target-dist "$ONSET_TARGET_DIST" \
      ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} 2>&1 | tee "$CALIB_LOG"
  ONSET_RMS="$(grep -oE 'RECOMMEND_ONSET_RMS=[0-9.]+' "$CALIB_LOG" | tail -1 | cut -d= -f2)"
  [ -n "$ONSET_RMS" ] || { echo "WARN: no RECOMMEND_ONSET_RMS; falling back to default 0.05"; ONSET_RMS="0.05"; }
  echo "  onset_rms (calibrated for ${ONSET_TARGET_DIST} m audible radius) = $ONSET_RMS"
fi

N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" || N_EPISODES=0
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: episode count '$N_EPISODES' <= 0"; exit 1; }
echo "  n-episodes = $N_EPISODES"

banner "[6/7] run settings [$SETTINGS] (--task $TASK --backbone remembr)"
OUT_DIRS=""
for S in $SETTINGS; do
  out_dir="runs/${OUT_TAG}-${CELL_TAG}-s$S"
  banner "run: setting=$S -> $out_dir"
  # Clear any STALE summary first, so the completion gate below can't be fooled
  # by a previous attempt's summary.json if THIS run hard-crashes before writing.
  rm -f "$out_dir/summary.json"
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --task "$TASK" \
      --rir-grid "$GRID" --anomaly-class "$CLASS" --t-anom "$T_ANOM_WARM" \
      --audio-onset-rms "$ONSET_RMS" ${ANOMALY_CLIP:+--anomaly-clip "$ANOMALY_CLIP"} \
      --episodes-path "$DS" --scene "$SCENE" --target any \
      --n-episodes "$N_EPISODES" --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  rc=${PIPESTATUS[0]}
  # Gate on COMPLETION, not rc. run_hm3d_pol returns 1 whenever pass_conditions
  # fail (run_hm3d_pol.py:568-571) — which is EXPECTED for S1 (memory-off) and S2
  # (STM-only), since those gates are S3-oriented. So rc!=0 is NORMAL there, NOT a
  # crash. A hard crash (CUDA OOM / sim or model load) instead leaves NO
  # summary.json (written only after the full episode loop) or
  # n_completed < n_attempted. Gate on exactly that.
  complete="$(python -c "import json,sys
try:
    s=json.load(open(sys.argv[1])); a=s.get('n_episodes_attempted',0); c=s.get('n_episodes_completed',0)
    print(1 if a>0 and c==a else 0)
except Exception:
    print(0)" "$out_dir/summary.json" 2>/dev/null || echo 0)"
  [ "$complete" = 1 ] \
    || { echo "FATAL: setting=$S run INCOMPLETE at $out_dir (rc=$rc; summary missing or n_completed<n_attempted = a hard crash, NOT a pass-condition exit). See ${out_dir}.log."; exit 1; }
  echo "  setting=$S complete (rc=$rc; rc=1 is normal for S1/S2 — pass_conditions are S3-oriented)"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

banner "[7/7] Gate-A verdict (warm paired soft-SPL S3-S1 + S2 decomposition)"
# The Gate-A analyzer is a CROSS-SETTING paired delta (S3-S1) and REQUIRES >=2 run
# dirs (analyze_ablation.py:410 -> parser.error, nonzero exit). A single-setting
# smoke (e.g. --settings 3, for plumbing checks like LTM_AUDIO_WRITE) has only one
# run dir, so skip the analyzer and exit 0 — the per-setting run already completed
# (gated above). Multi-setting behaviour is byte-identical.
N_RUN_DIRS=$(set -- $OUT_DIRS; echo $#)
if [ "$N_RUN_DIRS" -lt 2 ]; then
  echo "  [gate-A] skipped: need >=2 settings, got $N_RUN_DIRS (single-setting smoke -> no cross-setting paired delta)"
else
  # shellcheck disable=SC2086
  python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS 2>&1 | tee "runs/${OUT_TAG}-${CELL_TAG}-analysis.log"
fi
echo

# Planner-decision census (S3): WHY the LLM planner's own pick never wins the
# rerank (n_remembr_chosen). goto = it grounded a remembered waypoint (then lost
# the rerank to the injected memory candidate for the same spot); explore = it
# said nothing relevant; retrieve_calls=0 == it never queried memory at all.
banner "planner decision census (S3 — is the LLM 'too dumb to recall'?)"
S3_SUM="runs/${OUT_TAG}-${CELL_TAG}-s3/summary.json"
if [ -f "$S3_SUM" ]; then
  python - "$S3_SUM" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
g = lambda k: s.get(k, 0)
print("  proposals=%d  goto=%d  explore=%d  grounding_rejected=%d  budget_defer=%d  stop=%d  retrieve_calls=%d"
      % (g("n_planner_proposals"), g("n_planner_goto"), g("n_planner_explore"),
         g("n_planner_grounding_rejected"), g("n_planner_budget_defer"),
         g("n_planner_stop"), g("n_planner_retrieve_calls")))
print("  reranks won: remembr(LLM)=%d  memory(injected)=%d  frontier=%d"
      % (g("n_remembr_chosen"), g("n_memory_chosen"), g("n_frontier_chosen")))
verdict = ("NEVER TRIED to recall (lazy/too-dumb)" if g("n_planner_retrieve_calls") == 0
           else "TRIED to recall (goto=%d) but its pick lost the rerank to injected memory" % g("n_planner_goto")
                if g("n_planner_goto") > 0
                else "queried memory but always answered explore (found nothing relevant)")
print("  -> %s" % verdict)
PYEOF
else
  echo "  (no S3 summary at $S3_SUM)"
fi
echo

# S0 audio-DOA pre-flight gate: measure (on THIS run's instrumented logs) whether
# the S2 audio-DOA disambiguation head can help at all — recall presence, heard-
# sign vs source-bearing frame agreement, lateral separation → RECOMMEND verdict
# (GO / RECALL-GAP / FRAME-BROKEN / CO-LINEAR / INSUFFICIENT-DATA). GT source/goal
# positions are OFFLINE labels here only; the live head uses agent-estimable cues.
banner "[S0] audio-DOA calibration gate (diagnose_audio_doa_calib)"
# shellcheck disable=SC2086
python embodied_memory/scripts/diagnose_audio_doa_calib.py $OUT_DIRS 2>&1 \
  | tee "runs/${OUT_TAG}-${CELL_TAG}-audiodoa-calib.log" || true
echo

echo "DONE. AudioGoal warm S1-vs-S3 for ($SCENE,$CLASS). Cold-silent (t_anom high)"
echo "seeds the LTM; warm episodes fire the anomaly. Onset lines: grep '\\[audio\\]' the run logs."
[ -n "${LTM_AUDIO_DOA:-}" ] && echo "S1 onset-gate was ON (LTM_AUDIO_DOA=$LTM_AUDIO_DOA): memory injection suppressed until the anomaly was heard."
# Every real failure above is gated with an inline `exit 1`; reaching here = success.
# Without this, the trailing `[ ... ] && echo` short-circuits to status 1 when
# LTM_AUDIO_DOA is unset (set -uo pipefail, no set -e) -> spurious run-level exit 1.
exit 0
