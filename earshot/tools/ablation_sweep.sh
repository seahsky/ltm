#!/bin/bash
# earshot/tools/ablation_sweep.sh — the ablation table, and the baseline row it needs.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/ablation_sweep.sh --tag abl-1
#
# TWO DELIVERABLES, ONE SWEEP, because they are the same runs read two ways.
#
#   1. THE BASELINE, ON HM3D. The `full` arm is the complete system on the ADR-0017
#      windowed task, and it is the number every other row in the paper is quoted
#      against. It is an HM3D number by decision (ADR-0021): SAVN-CE is MP3D-bound —
#      its config is `magnet/config/mp3d/rgbd_ddppo_clean.yaml`, its episodes are MP3D
#      episodes and its checkpoint was trained on MP3D over 4xA800 for 14 days — so it
#      cannot be re-pointed at HM3D by a flag, and ADR-0015 forbids subtracting it from
#      an earshot number in any case. The baseline this paper compares against is
#      therefore INTERNAL and lives in this sweep.
#
#   2. THE ABLATION. Four arms, one component removed each, every one of them landed as
#      a typed `RunConfig` enum in commit 511b52f and never yet run. Until now the
#      ablation table in the paper was four empty rows.
#
# THE ARMS PAIR BY EPISODE, which is the only reason this sweep has any power. Same
# scenes, same `--seed`, same `--n-episodes`, same `--max-steps`, same class, same
# window, so episode k is the SAME TASK in all five arms and `tools/episode_diff.py`
# runs an exact McNemar over the pairs. `repeat-1` measured a 16.2% outcome flip rate on
# BYTE-IDENTICAL reruns and a net of +11 episodes on nothing at all, so an unpaired
# reading of this sweep would be measuring the apparatus.
#
# WHAT IT CAN AND CANNOT RESOLVE, before it is started rather than after. At the default
# 15 episodes per scene per arm over the 19 scenes that yield anything, each arm holds
# 285 episodes and `tools/power.py` prices the paired MDE at 6.68 points (80% power,
# alpha 0.05, the measured flip rate). 20 episodes a scene buys 5.78 points and costs
# 12.8 hours instead of 9.6. Anything smaller than about 7 points this sweep cannot see,
# and saying so here is cheaper than discovering it in the readout.
#
# THE SCENE-LEVEL TEST IS THE ONE THAT WILL DISAGREE. `sign_test_threshold(19)` is 15:
# an arm has to move 15 of 19 scenes the same way to clear a scene-level sign test, and
# `yield-2`->`arrive-2` already produced a mechanism that was green and exact at the
# episode level and null at the scene level. Both tests are reported. Neither is the
# tie-break for the other.
#
# THE CONTROL ARM RUNS FIRST so a crash in the ablation arms still leaves on disk the one
# arm every comparison needs.
#
# WHY THESE FOUR COMPONENTS. Each is a claim the paper makes about how the agent finds a
# source, and each has an arm because ADR-0014 requires a capability to be exercised
# rather than proxied:
#
#   no-climb   `--climb-rule off`. The energy climb never steers INVESTIGATE, so the
#              agent runs the scan/cast cycle alone. This is the largest expected effect
#              and the one that says whether live audio is doing any work at all.
#   no-cue     `--lateral-cue off`. The interaural sign is treated as ambiguous, so the
#              turn decision falls to its zero/absent default. Isolates BINAURAL
#              localization from mere loudness.
#   scan-only  `--cast-policy scan_only`. Every dead step turns instead of walking a leg.
#              This is the pre-`eps-1` control, and `eps-1`->`cast-1` is the comparison
#              that was DEMOTED from confirmed to "direction consistent, magnitude
#              unresolved" (McNemar p=0.18) for want of exactly this sweep's n.
#   anechoic   `--ir-policy anechoic`. Every rendered IR is replaced by a flat,
#              reverberation-free stand-in at all three render sites. This is row R5 and
#              it exists because `pilot-2` found win-alarm's silent phase audible in
#              356 of 356 episodes and win-burst's in 0 of 356, at SWS 0.115 against
#              0.112 — a well-powered null whose comparison is CONFOUNDED BY SOUND CLASS.
#              This arm asks the same question with class held fixed.
#
# `oracle-loc` AND `oracle-loc-matched` ARE NOT ABLATIONS, and they are in this list so
# nobody reads them as ones. Every arm above REMOVES a component. These two ADD the source
# coordinate, so they are CEILINGS: what the navmesh follower and the scan/cast sweep
# reach when localization costs nothing. Neither is `full` minus anything and neither
# belongs in the ablation table.
#
#   THE ARRIVAL CAVEAT, AND `oracle-1` MEASURED IT RATHER THAN ARGUING IT. The two arms
#   do not share an arrival test. `oracle-loc` arrives when it is inside
#   `ControllerConfig.investigate_arrive_radius_m` (1.5 m, horizontal, `agent/config.py`);
#   the realizable arm arrives on the detector's confirm, which under `Detector.ORACLE` is
#   a GEODESIC 1.0 m — Find-SR's own primary ring. Both land on
#   `FunnelStage.SOURCE_REACHED`.
#
#   `oracle-1` (2026-09-19, 19 val scenes, n=282/arm) is what that costs. Source-reached:
#   `full` 93/282 = 33.0%, `oracle-loc` 266/282 = 94.3%, +173 and 0 lost, exact McNemar
#   p = 0.0000, gains in 19 of 19 scenes. AND FIND-SR@1m INVERTS: 93 of 270 for `full`
#   against 2 of 270 for `oracle-loc`, source SPL 0.251 against 0.007. An agent that stops
#   at 1.5 m has not entered the ring the metric scores. So the +173 is arrival criterion
#   in a proportion that run CANNOT separate from information, and no "localization is
#   worth N points" claim comes off it.
#
#   `oracle-loc-matched` is that arm with the confound removed (ADR-0028): the same point
#   goal, routed to through the same pool, arriving on `visual_confirm` — the SAME
#   expression the realizable arm STOPs on, off the same detector query. It differs from
#   `full` in INFORMATION ALONE, so it is the arm to difference against `full` and the
#   only one of the two that is a ceiling on Find-SR@1m.
#
#   RUN `full` AND `oracle-loc-matched` TOGETHER. `episode_diff` can pair across runs, but
#   `repeat-1` measured 16.2% of outcomes flipping on byte-identical reruns, so the
#   decisive contrast gets its control in the same night. `oracle-loc` beside them is
#   optional and prices the ring alone; `oracle-1`'s arms already pair with these.
#
#   WHAT SURVIVES `oracle-1` UNCONFOUNDED, because it rests on the oracle arm alone:
#   NAVIGATION IS NOT THE LIMITER. Of its 16 abandoned episodes, TWELVE had no navmesh
#   route to the source at any step. With the coordinate handed over and a route
#   available the agent reaches 1.5 m in 266 of 270 = 98.5%, four routable failures in the
#   whole sweep. The follower, the step budget and the navmesh are not what holds `full`
#   at 33.0%. Those same 12 cap ANY Find-SR at 95.7% and are unwinnable under any
#   controller — the detector's view-point list for the anomaly object is seeded with the
#   source position alone, and a `None` distance reads as not-detected.
#
#   `oracle-pilot` (2026-09-19, one scene, `4ok3usBNeis`) was the prior and is superseded
#   by the sweep above. It measured 84% of its detour steps plateaued, so the cue was not
#   what found the source in the arm that found it every time. That number rests on a
#   reconstruction NEITHER oracle arm can validate (`tools/detour_report.py` says so now),
#   which is its own reason to run these arms BESIDE `full` rather than alone.
#
# NO MEMORY ARM IS IN THIS SWEEP, deliberately. ADR-0018's four cells need the stores
# wired into the runner and a prior pass that has run; neither exists yet, and four
# identical arms named after four conditions is worse than no table. This sweep is the
# ablation and the baseline. The matrix is a separate run.
#
# CONTINUE-ON-FAILURE, with the two kinds of failure told apart. A scene that yields ZERO
# EPISODES is a measured fact about that scene and not a broken run — `mL8ThkuaVTM` has
# posed no episode in any sweep this repo has ever run, and it is why `pilot-2` exited 1
# with 1095 good episodes on disk. It is now counted separately and printed by name. A
# run that failed for any OTHER reason is still red.
#
# Flags: --tag T (required in practice; one directory is one run), --n-episodes N
#        (default 15, PER SCENE PER ARM), --max-steps M (default 250), --scenes "a b c"
#        (default: every scene with a mesh), --limit N (default 0 = no limit),
#        --sounding-steps N (default 60), --anomaly-class C (default alarm),
#        --seed S, --out-dir DIR, --arms "a b" (default: every arm in ARM_NAMES),
#        --no-pull, --force,
#        --resume.
#
# --resume PICKS UP A KILLED SWEEP at the scene grain. It implies --force (it reuses the
# directory on purpose). A finished scene -- one that wrote a summary.json, including a
# zero-yield one -- is skipped; an unfinished one is CLEARED and re-run, because
# `write_episode` refuses to overwrite and half a scene is debris rather than a record.
# For the chained `dream` arm it refuses outright unless the memory file's own scene list
# matches the scenes about to be skipped: a chain that silently lost a house cannot be
# detected afterwards. The summary says how many cells were resumed, because the tag's
# records then come from more than one invocation.
#
# THE `dream` ARM IS NOT IN THE DEFAULT FIVE and has to be asked for by name. It is not an
# ablation of the anomaly-response system -- it is an ADDITION to it, so it belongs in the
# same sweep only because it must pair by episode against `full` for the comparison to
# mean anything. Ask for it with:
#
#   nrun bash earshot/tools/ablation_sweep.sh --tag dream-1 --arms "full dream"
#
# and read it with `tools/episode_diff.py runs/dream-1/full runs/dream-1/dream`, which is
# an exact McNemar over the paired episodes. `repeat-1` measured a 16.2% outcome flip rate
# on byte-identical reruns, so the unpaired difference of two arms is not a result.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
TAG="abl-$(date +%Y%m%d-%H%M%S)"
N_EPISODES=15
MAX_STEPS=250
SOUNDING_STEPS=60
ANOMALY_CLASS=alarm
SEED=20260805
SCENES=""
LIMIT=0
OUT_DIR=""
WANTED_ARMS=""
NO_PULL=0
FORCE=0
RESUME=0
DREAM_ETA=2.0
DREAM_MAX_RETAINED=12
ORIGINAL_ARGS="$*"

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)             need_value $# "$1"; TAG="$2";             shift 2 ;;
    --n-episodes)      need_value $# "$1"; N_EPISODES="$2";      shift 2 ;;
    --max-steps)       need_value $# "$1"; MAX_STEPS="$2";       shift 2 ;;
    --sounding-steps)  need_value $# "$1"; SOUNDING_STEPS="$2";  shift 2 ;;
    --anomaly-class)   need_value $# "$1"; ANOMALY_CLASS="$2";   shift 2 ;;
    --seed)            need_value $# "$1"; SEED="$2";            shift 2 ;;
    --scenes)          need_value $# "$1"; SCENES="$2";          shift 2 ;;
    --limit)           need_value $# "$1"; LIMIT="$2";           shift 2 ;;
    --arms)            need_value $# "$1"; WANTED_ARMS="$2";     shift 2 ;;
    --out-dir)         need_value $# "$1"; OUT_DIR="$2";         shift 2 ;;
    --no-pull)         NO_PULL=1;                                shift ;;
    --force)           FORCE=1;                                  shift ;;
    --resume)          RESUME=1; FORCE=1;                        shift ;;
    --dream-eta)       need_value $# "$1"; DREAM_ETA="$2";       shift 2 ;;
    --dream-max-retained) need_value $# "$1"; DREAM_MAX_RETAINED="$2"; shift 2 ;;
    -h|--help) sed -n '2,79p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done
OUT_DIR="${OUT_DIR:-runs/$TAG}"

banner() { printf '\n========== %s ==========\n' "$1"; }

# A ZERO-YIELD cell is a scene that placed no episode: a measured fact about HM3D, not a
# broken run. `runner.run` writes `summary.json` and THEN re-raises on EmptyDatasetError
# precisely so the answer lives in the artefact rather than in a log line, which is what
# lets both the run loop and the readout loop below ask the same question of the same file.
# `abl-1` is why they must: the run loop skipped `mL8ThkuaVTM` correctly and the readout
# loop then judged it anyway, `smoke` returned 2 for NOT_RUN as it should, and a sweep with
# 1410 good episodes and five complete arms reported RED over five empty directories.
is_zero_yield() {
  [ -f "$1/summary.json" ] || return 1
  python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['n_episodes']==0 else 1)" \
    "$1/summary.json" 2>/dev/null
}

# --- RESUMING A KILLED SWEEP ----------------------------------------------
# A nineteen-scene two-arm sweep is 4h45m in one process tree, and a box that drops it at
# hour four has cost a night and produced nothing readable. These make the unit of loss
# ONE SCENE rather than one sweep. The sweep was already chunked -- one `python -m earshot`
# per (arm, scene) -- so all that was missing was the ability to not redo the finished ones.
#
# A scene is finished iff it wrote a `summary.json` that parses and carries `n_episodes`.
# That is the LAST artefact `run()` writes, so its presence means the episodes and the
# memory dump are both done. A ZERO-YIELD scene writes one too, with `n_episodes: 0`, and
# so counts as finished -- it is a scene that cannot pose the task, not work left to do.
is_finished_scene() {
  [ -f "$1/summary.json" ] || return 1
  python -c "import json,sys; json.load(open(sys.argv[1]))['n_episodes']; sys.exit(0)" \
    "$1/summary.json" 2>/dev/null
}

# A scene that is NOT finished has debris: episodes 0..k on disk from the attempt that
# died. `write_episode` REFUSES to overwrite, so re-running into it fails on the first
# existing file -- and `--overwrite` would be the wrong fix for `matrix_sweep.sh`'s
# reason: it replaces what the new run writes and leaves whatever it does not, which is
# two invocations in one directory with nothing saying so. Clearing is the honest version.
#
# The path is checked before anything is removed. `$OUT_DIR` is operator-supplied.
clear_scene_dir() {
  [ -n "$1" ] || { echo "FATAL: clear_scene_dir got an empty path"; exit 1; }
  case "$1" in
    *..*) echo "FATAL: refusing to clear '$1' — it contains '..'"; exit 1 ;;
  esac
  case "$1" in
    "$OUT_DIR"/*/*) ;;
    *) echo "FATAL: refusing to clear '$1' — not an <arm>/<scene> under $OUT_DIR"
       exit 1 ;;
  esac
  [ -d "$1" ] || return 0
  echo "      clearing $(find "$1" -type f 2>/dev/null | wc -l | tr -d ' ') file(s) left by an attempt that did not finish"
  rm -rf "${1:?}"
}

# What one scene's smoke gate means for the sweep. `gate_verdict <arm> <rc> <output>`:
#   0  the gate passed
#   1  criterion 5 ALONE, in an ablation arm: 0 episodes in this scene closed the loop.
#      A measurement, not a failure -- see the readout loop for why.
#   2  red.
#
# This is a function so the decision can be exercised rather than described: a rule this
# permissive, asserted only by reading the script's text, is a rule nothing checks.
gate_verdict() {
  local arm="$1" rc="$2" out="$3"
  [ "$rc" -eq 0 ] && return 0
  # A glob on the captured string, never `| grep -q`: under pipefail a matching grep
  # exits early and SIGPIPEs its producer, which is the footgun line 182 already names.
  # The match is ANCHORED at the end, so "criteria 5, 7" is red -- an arm that also
  # failed the audio or hermeticity criteria must not ride through on this allowance.
  if [ "$arm" != "full" ] && [[ "$out" == *"criteria 5" ]]; then
    return 1
  fi
  return 2
}

# Which arm every other one is PAIRED AGAINST. `reference_arm <name>...`, in the order the
# operator asked for them, prints the one to quote against.
#
# `full` is the reference whenever the sweep ran it: it is the baseline of record
# (ADR-0021) and its position in `--arms` is not meant to matter. Otherwise the FIRST arm
# asked for is the reference.
#
# THE SECOND RULE EXISTS BECAUSE THE FIRST ONE CALLED A GOOD RUN A FAILURE. `dream-4` ran
# `--arms "dream dream-nomem"` -- the two-arm repeat pair ADR-0025 pre-registered by name
# -- for 5h40m, wrote all 564 episodes, passed every gate, and exited 1 because no
# directory called `full` was on disk. A sweep that did not ASK for the baseline is not a
# sweep missing it. Red is for an arm that was requested and did not run; what this is, is
# a contrast between two arms, and the reader prints which one it chose so nothing is ever
# quoted against an arm the operator did not expect.
reference_arm() {
  local first="$1"
  local arm
  for arm in "$@"; do
    [ "$arm" = "full" ] && { echo "full"; return 0; }
  done
  echo "$first"
}

# --- ONE DIRECTORY IS ONE RUN, before anything expensive ------------------
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  if [ "$FORCE" = 0 ]; then
    echo "FATAL: $OUT_DIR already exists and is not empty."
    echo "       One directory is one run. Re-using it pools two sweeps into one"
    echo "       aggregate with nothing on disk saying so. Pass a fresh --tag."
    exit 1
  fi
  if [ "$RESUME" = 1 ]; then
    echo "  --resume: reusing $OUT_DIR on purpose. Finished scenes are skipped and"
    echo "  unfinished ones are cleared and re-run, so the invariant still holds at the"
    echo "  grain it can: one scene directory, one invocation."
  else
    echo "WARN: --force — reusing a non-empty $OUT_DIR. Earlier records will be pooled"
    echo "      into the aggregates below and cannot be told apart."
  fi
fi

# A RESUME MUST NOT CHANGE THE KNOBS MID-SWEEP. The scenes already on disk ran at the
# eta this file recorded; finishing the rest at a different one produces ONE tag holding
# two configurations, which `dream_report` would later call a mixed arm -- after the
# night, and only if someone read section F. `--dream-eta` is a flag precisely so it can
# be forgotten on the resume invocation, so it is checked here instead.
check_resumed_knobs() {
  local record="$OUT_DIR/provenance.txt"
  [ -f "$record" ] || return 0
  local was_eta was_cap
  was_eta="$(awk '/^dream_eta:/ {print $2}' "$record")"
  was_cap="$(awk '/^dream_max_ret:/ {print $2}' "$record")"
  [ -n "$was_eta" ] || return 0   # a tag from before these were recorded; nothing to check
  if [ "$was_eta" != "$DREAM_ETA" ] || [ "$was_cap" != "$DREAM_MAX_RETAINED" ]; then
    echo "FATAL: this resume would change the DREAM knobs mid-sweep."
    echo "       on disk: --dream-eta $was_eta --dream-max-retained $was_cap"
    echo "       now:     --dream-eta $DREAM_ETA --dream-max-retained $DREAM_MAX_RETAINED"
    echo "       The scenes already finished ran at the first pair. One tag holding two"
    echo "       configurations is not one run. Pass the same knobs, or a fresh --tag."
    exit 2
  fi
  echo "  --resume: knobs match the record (eta $was_eta, cap $was_cap)"
}
[ "$RESUME" = 1 ] && check_resumed_knobs

# --- 1. self-update by re-exec (bash runs the body it loaded, not the file) -
if [ "$NO_PULL" = 0 ]; then
  banner "[1/5] git pull --ff-only"
  _self_before="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
  _self_after="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  if [ -n "$_self_before" ] && [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
    echo "  this script changed in the pull — re-execing the new body"
    export _REEXEC=1
    _force_flag=""
    [ "$FORCE" = 1 ] && _force_flag="--force"
    # --resume HAS TO SURVIVE THE RE-EXEC. Without it the new body sees --force alone,
    # re-runs the scenes already on disk, and dies on the first episode file
    # `write_episode` refuses to overwrite -- after the pull, with the operator asleep.
    _resume_flag=""
    [ "$RESUME" = 1 ] && _resume_flag="--resume"
    exec bash "$0" --tag "$TAG" --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
         --sounding-steps "$SOUNDING_STEPS" --anomaly-class "$ANOMALY_CLASS" \
         --seed "$SEED" --limit "$LIMIT" ${SCENES:+--scenes "$SCENES"} \
         ${WANTED_ARMS:+--arms "$WANTED_ARMS"} --out-dir "$OUT_DIR" \
         --dream-eta "$DREAM_ETA" --dream-max-retained "$DREAM_MAX_RETAINED" \
         ${_force_flag:+--force} ${_resume_flag:+--resume}
  fi
else
  banner "[1/5] git pull SKIPPED (--no-pull)"
fi
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  commit: $COMMIT"

# --- 2. the env -----------------------------------------------------------
banner "[2/5] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
# A DIRECTORY check, never `conda env list | grep -q`: under pipefail a matching grep
# exits early, SIGPIPEs conda, and turns found-it into a pipeline failure.
[ -d "$MINICONDA/envs/$ENV_NAME" ] || { echo "FATAL: env '$ENV_NAME' missing — run bootstrap_ss2.sh"; exit 1; }
set +u
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "  python: $(python -V 2>&1)"

[ -d "data/anomaly_audio" ] || {
  echo "FATAL: no data/anomaly_audio — stage the ESC-50 recordings once:"
  echo "       python -m earshot.audio.clips --out-dir data/anomaly_audio"
  exit 1
}

# --- 3. scenes ------------------------------------------------------------
banner "[3/5] scenes"
if [ -z "$SCENES" ]; then
  SCENES="$(python - <<'PY'
# The same check task/runner._pick_scene makes: load the content file, then test that its
# resolved mesh is on this box. A content file without its .glb fails at load, which is
# not the same fact as a scene that cannot pose the task.
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene
split_dir = find_split_dir("val", root=".")
scenes_dir = find_scenes_dir(root=".")
found = []
for label in available_scenes(split_dir):
    try:
        load_scene(split_dir, label, scenes_dir=scenes_dir)
    except Exception:
        continue
    found.append(label)
print(" ".join(found))
PY
)" || { echo "FATAL: scene discovery failed"; exit 1; }
fi
# shellcheck disable=SC2206
SCENE_LIST=($SCENES)
if [ "$LIMIT" -gt 0 ] && [ "${#SCENE_LIST[@]}" -gt "$LIMIT" ]; then
  SCENE_LIST=("${SCENE_LIST[@]:0:$LIMIT}")
fi
[ "${#SCENE_LIST[@]}" -gt 0 ] || { echo "FATAL: no scene with a mesh on this box"; exit 1; }

# --- 4. the arms ----------------------------------------------------------
# Parallel arrays rather than an associative array so the ORDER is fixed and the control
# is first. `ARM_FLAGS` is a single string per arm, word-split at the call site: every
# entry is a literal flag and value with no spaces inside either, so the split is safe
# and `shellcheck` is told so once, there.
# DREAM's fifteen knobs, in one place. `--dream` refuses to run without every one of
# them (the paper values none, so there is nothing to default to) and each is recorded on
# every episode's audit by `DreamKnobs.as_metrics`, so this line and the artefact cannot
# disagree. The values are FIRST CHOICES and this run is what prices them:
#
#   stm-horizon 8 / stm-decay 0.8 / present-weight 0.7 -- eq. 4 and 19. Eight steps is
#     about 8 s of a 250-step episode at the shipped 1.0 s step.
#   coherence 0.99 -- eq. 9, and the one number the box has actually measured. Real
#     consecutive `z^av` sat at min 0.9256 median 0.9765 max 0.9905 over 19 pairs, so
#     0.99 cuts near the median rather than never (a value above 0.9905 never cuts) or
#     always (below 0.9256). min/max segment 3/12 bound it either side.
#   alpha/beta/gamma 1 -- eq. 10, equal weight until a run says otherwise.
#   eta 2.0 / max-retained 8 -- eq. 13, AND BOTH ARE UNPRICED. `dream-2` set eta 0.5
#     against the box's empty-memory `I_j` of 1.0000-1.2901, which is a regime that
#     occurs ONCE per chain: with M^E non-empty, max I_j was about 0.38 and the gate
#     never opened again (275 of 282 episodes retained nothing, 45 rows all night, 38 of
#     them from the one empty-memory episode). ADR-0024 fixed the cause -- C_j and U_j
#     were shares carrying a hidden 1/J while N_j is a cosine -- so `mean(C_j + U_j)` is
#     now exactly 2.0 for ANY J and any memory state. eta 2.0 therefore reads as "this
#     segment carried more than an average segment's worth", which is stationary in a
#     way 0.5 never was. IT IS STILL A GUESS ABOUT A DISTRIBUTION NOBODY HAS MEASURED.
#     PRICE IT BEFORE THE NEXT HEADLINE ARM: run one scene, read `dream_segments_over_eta`
#     from `dream_report`, and set eta so its median sits BELOW max-retained. If the cap
#     binds on most episodes then the cap is the retention rule and eta is decoration,
#     which is the top-k deviation ADR-0024 declined to ship.
#   max-retained 12 -- the cap on eq. 13, argued at `consolidate.retain`. It bounds the
#     empty-memory episode, which clears any eta below 1 outright because N_j is 1.0 for
#     every segment there: dream-2's first episode wrote 38 rows that way and this would
#     have held it to 12. Below the 21-84 segments a 250-step episode produces, so it
#     bites a flood and is mostly absent on an episode whose gate is working.
#     A SYNTHETIC PROBE IS WHY IT IS 12 AND NOT 8. Over 250-step trajectories differing
#     only in whether progress arrives evenly or in bursts, the count clearing eta 2.0
#     ran 9 to 14 of 21, and eta 3.0 ran 0 to 3 of the same fixtures. That spread over
#     fixtures this similar is the finding: THE DISTRIBUTION IS NOT PREDICTABLE OFF-BOX,
#     a cap of 8 would have bound on every synthetic episode tried, and a cap that always
#     binds IS the retention rule with eta as decoration.
#   min-support 2 -- eq. 16. A regularity that happened once is not one.
#   k 3/2/1, temperature 0.5 -- eq. 20-23.
#   lambda 1 / 0.5 / 0.5 -- eq. 26. S_plan keeps unit weight so the arm stays anchored to
#     the ranking ADR-0008 froze; memory and feasibility are additions to it, not
#     replacements for it.
DREAM_KNOBS="--dream \
  --dream-stm-horizon 8 --dream-stm-decay 0.8 --dream-present-weight 0.7 \
  --dream-coherence 0.99 --dream-min-segment 3 --dream-max-segment 12 \
  --dream-alpha 1.0 --dream-beta 1.0 --dream-gamma 1.0 \
  --dream-eta $DREAM_ETA --dream-max-retained $DREAM_MAX_RETAINED \
  --dream-min-support 2 \
  --dream-k-experience 3 --dream-k-pattern 2 --dream-k-knowledge 1 \
  --dream-temperature 0.5 \
  --dream-lambda-plan 1.0 --dream-lambda-memory 0.5 --dream-lambda-feasibility 0.5"

# THE CONTROL ADR-0024 ASKED FOR, and the only thing that makes a DREAM arm readable.
# `dream-2` was a TWO-VARIABLE contrast: `runner.py`'s claim that `pick_plan` reduces to
# `pick_waypoint` on an empty memory is FALSE at `lambda_feasibility = 0.5`, so `full` vs
# `dream` differenced the memory term AND the feasibility term at once, and with the
# memory near-empty the likelier reading is that it measured feasibility at about zero.
#
# `dream-nomem` is the same arm with `lambda_memory = 0.0` and EVERYTHING ELSE identical:
# same segmentation, same consolidation, same retrieval, same feasibility weight. So
#   dream       vs dream-nomem  = the memory term, alone, which is eq. 26's whole claim
#   dream-nomem vs full         = the feasibility term, alone, which dream-2 confounded
# It is built by substitution rather than by a second literal so the two arms CANNOT
# drift apart in any other knob, and the substitution is checked: a silent no-op here
# would run the control as a duplicate of the treatment and difference an arm with itself.
DREAM_KNOBS_NOMEM="${DREAM_KNOBS/--dream-lambda-memory 0.5/--dream-lambda-memory 0.0}"
if [ "$DREAM_KNOBS_NOMEM" = "$DREAM_KNOBS" ]; then
  echo "FATAL: the dream-nomem substitution matched nothing, so the control arm is"
  echo "       byte-identical to the treatment arm. Someone changed --dream-lambda-memory"
  echo "       in DREAM_KNOBS without updating the substitution above. A sweep run this"
  echo "       way would difference an arm against itself and report a clean null."
  exit 2
fi

ARM_NAMES=(full no-climb no-cue scan-only anechoic oracle-loc oracle-loc-matched dream dream-nomem)
ARM_FLAGS=(
  ""
  "--climb-rule off"
  "--lateral-cue off"
  "--cast-policy scan_only"
  "--ir-policy anechoic"
  # THE ONLY ARM THAT ADDS INFORMATION INSTEAD OF REMOVING A COMPONENT, and the only one
  # whose flag is already on the command line above. `--localization realizable` is
  # passed explicitly at the invocation and `${ARM_FLAGS[$i]}` is word-split AFTER it, so
  # argparse's last-wins gives this arm `oracle`. That ordering is load-bearing: moving
  # `ARM_FLAGS` before the explicit flags would silently run this arm realizable and
  # report a null.
  "--localization oracle"
  # THE SAME CEILING ON THE BASELINE'S OWN CRITERION (ADR-0028). Same override, same
  # last-wins ordering; what differs from the line above is which arrival test the arm is
  # scored on, and that is a `Localization` value rather than a radius. `oracle-1`
  # measured why the two are not interchangeable: 94.3% source-reached and 0.7%
  # Find-SR@1m out of the SAME arm.
  "--localization oracle_matched"
  "--clap $DREAM_KNOBS"
  "--clap $DREAM_KNOBS_NOMEM"
)
ARM_WHY=(
  "the BASELINE: the complete system, and the row every other one is quoted against"
  "R1 the energy climb never steers — does live audio do any work at all"
  "R2 the interaural sign is ambiguous — loudness without binaural localization"
  "R3 every dead step turns instead of walking a leg — the pre-eps-1 control"
  "R5 flat IRs at all three render sites — does the reverb tail buy any SWS"
  "THE CEILING ON ITS OWN CRITERION, not an ablation: HANDED the source coordinate and arriving at 1.5 m. oracle-1 measured 94.3% source-reached and 2 of 270 Find-SR@1m out of this arm, so DO NOT difference it against full — read the arrival caveat in the header and use oracle-loc-matched"
  "THE CEILING ON THE BASELINE'S CRITERION (ADR-0028): handed the source coordinate and arriving on the realizable arm's own detector confirm, so it differs from full in INFORMATION ALONE. This is the arm a localization ceiling is quoted from"
  "DREAM: M^S, consolidation, a three-level M^L that GROWS across this arm's episodes, and memory-weighted planning"
  "THE CONTROL for the arm above: identical in every knob but lambda_memory = 0.0, so the difference is eq. 26's memory term and nothing else"
)

if [ -n "$WANTED_ARMS" ]; then
  # shellcheck disable=SC2206
  _wanted=($WANTED_ARMS)
  _names=(); _flags=(); _why=()
  for want in "${_wanted[@]}"; do
    _hit=0
    for i in "${!ARM_NAMES[@]}"; do
      if [ "${ARM_NAMES[$i]}" = "$want" ]; then
        _names+=("${ARM_NAMES[$i]}"); _flags+=("${ARM_FLAGS[$i]}"); _why+=("${ARM_WHY[$i]}")
        _hit=1
      fi
    done
    [ "$_hit" = 1 ] || { echo "FATAL: unknown arm '$want'. Known: ${ARM_NAMES[*]}"; exit 2; }
  done
  ARM_NAMES=("${_names[@]}"); ARM_FLAGS=("${_flags[@]}"); ARM_WHY=("${_why[@]}")
fi

N_SCENES="${#SCENE_LIST[@]}"
N_ARMS="${#ARM_NAMES[@]}"
TOTAL_EPISODES=$((N_SCENES * N_ARMS * N_EPISODES))
# 24.2 s/episode all-in is the MEASURED figure from `pilot-2`: 26516 s of wall clock over
# 1095 episodes, including scene loads, calibration and the smoke gates. It is not the
# per-step render cost and it is not an extrapolation from the anomaly-response task.
# DREAM's arm costs more per episode and the estimate has to say so, or a reader sizes
# an overnight run from a number that is wrong for one arm in it. The box measured the
# SUSTAINED extra at the median DREAM step; at 250 steps that is the per-episode addition.
# 0.057 s/step is the first measurement (f_u 0.0443 + observe 0.0123, both means) and it
# is an estimate rather than a promise -- `dream_step_s_mean` on every episode's audit is
# what the run itself reports.
_N_DREAM_ARMS=0
for _a in "${ARM_NAMES[@]}"; do [ "$_a" = "dream" ] && _N_DREAM_ARMS=$((_N_DREAM_ARMS + 1)); done
_DREAM_EXTRA=$(awk "BEGIN{printf \"%d\", $_N_DREAM_ARMS * $N_SCENES * $N_EPISODES * $MAX_STEPS * 0.057}")
EST_SECONDS=$(awk "BEGIN{printf \"%d\", $TOTAL_EPISODES * 24.2 + $_DREAM_EXTRA}")
EST_HOURS=$(awk "BEGIN{printf \"%.1f\", $EST_SECONDS / 3600.0}")
PER_ARM=$((N_SCENES * N_EPISODES))
echo "  $N_SCENES scene(s): ${SCENE_LIST[*]}"
echo "  $N_ARMS arm(s): ${ARM_NAMES[*]}"
echo "  $N_EPISODES episodes per scene per arm -> $PER_ARM per arm, $TOTAL_EPISODES total"
echo "  estimated wall clock: ${EST_HOURS} h at the measured 24.2 s/episode"
if [ "$_N_DREAM_ARMS" -gt 0 ]; then
  echo "    ... of which $(awk "BEGIN{printf \"%.1f\", $_DREAM_EXTRA / 3600.0}") h is DREAM's per-step encoders"
  echo "    (f_v + f_u run on EVERY step; criterion 7 does not audit them, so this cost"
  echo "     appears only in the wall clock and in each episode's dream_step_s_* metrics)"
fi
echo ""
echo "  what this n can resolve, at 80% power and alpha 0.05. Read the PAIRED block:"
echo "  the arms share episodes, so the unpaired table is the wrong column for them."
python -m earshot.tools.power \
    --n-per-cell "$PER_ARM" --paired-n "$PER_ARM" --n-scenes "$N_SCENES" \
    2>/dev/null | sed 's/^/    /' \
  || echo "    (power.py did not run — the MDE is UNKNOWN for this sweep)"

mkdir -p "$OUT_DIR"
{
  echo "tag:            $TAG"
  echo "commit:         $COMMIT"
  echo "args:           $ORIGINAL_ARGS"
  echo "scenes:         ${SCENE_LIST[*]}"
  echo "resume:         $RESUME"
  echo "arms:           ${ARM_NAMES[*]}"
  echo "n_episodes:     $N_EPISODES (per scene, per arm) -> $PER_ARM per arm"
  echo "max_steps:      $MAX_STEPS"
  echo "sounding_steps: $SOUNDING_STEPS (fixed_steps, ADR-0017)"
  echo "anomaly_class:  $ANOMALY_CLASS"
  echo "seed:           $SEED"
  echo "dream_eta:      $DREAM_ETA"
  echo "dream_max_ret:  $DREAM_MAX_RETAINED"
  echo "started:        $(date -Is)"
} > "$OUT_DIR/provenance.txt"

# --- criterion 9's evidence, armed once around the whole sweep ------------
# `pilot-1` is why this is not optional: criterion 9 was structurally NOT_RUN on all
# twelve gates, NOT_RUN is red, and the driver exited 1 over twelve smoke gates whose
# other eight criteria were green. A criterion that is red on every ordinary run is one
# the reader learns to skip, which is how a never-armed canary read as a pass.
HERM_BEFORE="$OUT_DIR/.hermeticity-before.json"
if ! python -m earshot.tools.reset_manifest --verify-absent --when before > "$HERM_BEFORE"; then
  echo "WARN: could not record the pre-run hermeticity check — criterion 9 will be NOT_RUN"
  rm -f "$HERM_BEFORE"
fi

FAILED_RUNS=0
ZERO_YIELD=""
RESUMED=0
banner "[4/5] $N_ARMS arms x $N_SCENES scenes"
for i in "${!ARM_NAMES[@]}"; do
  arm="${ARM_NAMES[$i]}"
  echo ""
  echo "  --- arm $arm: ${ARM_WHY[$i]} ---"
  # THE DREAM ARM CHAINS ITS MEMORY ACROSS SCENES; every other arm has none to chain.
  # `dream-1` built nineteen memories and threw each away (`dream_report` section D), so
  # `M^P` -- the only level `omega_t` can weigh against, per
  # `tests/mac/test_dream_omega_reach.py` -- almost never had two successes sharing a
  # concept triple. One file, appended to by each scene in turn, is what gives `abstract`
  # a corpus to find a regularity IN.
  #
  # SCENE ORDER IS NOW PART OF THE RESULT for this arm. `SCENE_LIST` is built once and
  # sorted above, so it is fixed and a re-run repeats it; but episode k of the last scene
  # now depends on every scene before it, and a sweep that changed `--scenes` would not
  # be comparing the same thing. `episode_diff` still pairs by (scene, index) and is
  # unaffected.
  MEMORY_FILE=""
  if [ "$arm" = "dream" ]; then
    MEMORY_FILE="$OUT_DIR/$arm/memory.json"
    echo "      chaining M^E across scenes through $MEMORY_FILE"
    # A RESUME CAN SILENTLY SHORTEN THE CHAIN. A scene whose episodes finished but whose
    # memory dump never landed leaves a finished directory and a memory file that never
    # saw it; skipping that scene drops a house from `M^E` for the whole rest of the
    # sweep, and no audit afterwards can tell. The memory file records its own scene
    # list, so the disagreement is cheap to catch -- and it is caught BEFORE the first
    # episode of the resumed arm, not after the night.
    if [ "$RESUME" = 1 ]; then
      _finished=""
      for _scene in "${SCENE_LIST[@]}"; do
        is_finished_scene "$OUT_DIR/$arm/$_scene" || break
        _finished="$_finished $_scene"
      done
      python -m earshot.tools.chain_check "$MEMORY_FILE" --expect "$_finished" || {
        echo "FATAL: the chained M^E does not match the scenes this resume would skip."
        exit 1
      }
    fi
  fi
  for scene in "${SCENE_LIST[@]}"; do
    run_dir="$OUT_DIR/$arm/$scene"
    if [ "$RESUME" = 1 ] && is_finished_scene "$run_dir"; then
      RESUMED=$((RESUMED + 1))
      echo "    $arm / $scene   already finished, skipped (--resume)"
      continue
    fi
    [ "$RESUME" = 1 ] && clear_scene_dir "$run_dir"
    echo "    $arm / $scene   ($(date +%H:%M:%S))"
    # `--dream-memory-in` only once the file EXISTS. The first scene of the chain has no
    # memory to restore, and `run()` treats a missing in-path as an error rather than as
    # an empty memory on purpose -- a typo that silently started from nothing would
    # reproduce exactly the defect this chain removes.
    MEMORY_FLAGS=""
    if [ -n "$MEMORY_FILE" ]; then
      MEMORY_FLAGS="--dream-memory-out $MEMORY_FILE"
      if [ -f "$MEMORY_FILE" ]; then
        MEMORY_FLAGS="--dream-memory-in $MEMORY_FILE $MEMORY_FLAGS"
      fi
    fi
    # `--detector oracle` and `--localization realizable` are the defaults and are passed
    # explicitly: the ORACLE STOP deletes the stop_miss half of the failure mass, so
    # these find numbers are an upper bound and the command line should say so rather
    # than the reader having to know the default.
    # shellcheck disable=SC2086
    python -m earshot \
      --run-dir "$run_dir" \
      --scene "$scene" \
      --n-episodes "$N_EPISODES" \
      --max-steps "$MAX_STEPS" \
      --seed "$SEED" \
      --localization realizable \
      --detector oracle \
      --anomaly-class "$ANOMALY_CLASS" \
      --sounding-policy fixed_steps \
      --sounding-steps "$SOUNDING_STEPS" \
      ${ARM_FLAGS[$i]} \
      ${MEMORY_FLAGS} \
      > "$OUT_DIR/$arm-$scene.log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then
      # Anything that is not zero yield is red.
      if is_zero_yield "$run_dir"; then
        echo "      ZERO YIELD — this scene placed no episode. Recorded, not a failure."
        ZERO_YIELD="$ZERO_YIELD $arm/$scene"
        continue
      fi
      echo "      FAILED (exit $status) — tail:"
      tail -n 12 "$OUT_DIR/$arm-$scene.log" | sed 's/^/        /'
      FAILED_RUNS=$((FAILED_RUNS + 1))
      continue
    fi
    # stderr is NOT suppressed: `--verify-absent` prints "STILL PRESENT: <paths>" there
    # and that list is the entire diagnostic.
    if [ -f "$HERM_BEFORE" ]; then
      python -m earshot.tools.reset_manifest --verify-absent --when after \
          > "$run_dir/.hermeticity-after.json" \
        && python -m earshot.tools.reset_manifest --write-record \
             --run-dir "$run_dir" --before "$HERM_BEFORE" \
             --after "$run_dir/.hermeticity-after.json" --commit "$COMMIT" \
             >/dev/null \
        || echo "      WARN: hermeticity incomplete — criterion 9 will not be green"
    fi
  done
done

# --- 5. read it back ------------------------------------------------------
banner "[5/5] the ablation table"
GATE_FAILED=0
READ_FAILED=0
# Criterion 5 is a RATE criterion: green iff at least one episode reached CHECK and
# RESUME. `tally` still fails it at 0/n, and that floor exists to catch ADR-0014's
# VACUOUS ARM -- a loop that never once ran is a loop that is not wired.
#
# `abl-1` showed the floor is at the wrong GRAIN for an ablation. Ten scene/arm cells
# came back `RED -- criteria 5` and every one of them was in an arm built to cripple the
# controller: `scan-only` in seven scenes, one each in `no-climb`, `no-cue` and
# `anechoic`, and NONE in `full`. Identical episodes, identical harness, and the baseline
# completed the loop in all nineteen scenes -- so a crippled arm failing to close the loop
# in one room is the ablation working, not the loop being unwired, and it is a
# MEASUREMENT. Asking per scene turns the strongest result in the table into a red banner.
#
# So the floor moves to the arm: an ablation arm is vacuous only if it completes the loop
# in NO scene at all, which is still exactly the condition ADR-0014 wants caught. `full`
# is the baseline of record and keeps the per-scene bar, because there a zero IS a bug.
VACUOUS_CELLS=""
VACUOUS_BY_ARM=""
for arm in "${ARM_NAMES[@]}"; do
  arm_green=0
  arm_vacuous=0
  for scene in "${SCENE_LIST[@]}"; do
    [ -d "$OUT_DIR/$arm/$scene" ] || continue
    if is_zero_yield "$OUT_DIR/$arm/$scene"; then
      echo ""
      echo "  --- smoke gate: $arm / $scene --- SKIPPED, zero yield (see the list below)"
      continue
    fi
    echo ""
    echo "  --- smoke gate: $arm / $scene ---"
    gate_out="$(python -m earshot.task.smoke --run-dir "$OUT_DIR/$arm/$scene" 2>&1)"
    gate_rc=$?
    echo "$gate_out"
    gate_verdict "$arm" "$gate_rc" "$gate_out"
    case $? in
      0) arm_green=$((arm_green + 1)) ;;
      1) echo "      criterion 5 alone, and this is an ablation arm: 0 of the episodes in"
         echo "      this scene closed the loop. Recorded as a measurement, not a failure."
         arm_vacuous=$((arm_vacuous + 1))
         VACUOUS_CELLS="$VACUOUS_CELLS $arm/$scene" ;;
      *) GATE_FAILED=1 ;;
    esac
  done
  VACUOUS_BY_ARM="$VACUOUS_BY_ARM $arm=$arm_vacuous"
  if [ "$arm_green" -eq 0 ]; then
    echo ""
    echo "  RED: arm $arm closed the loop in NO scene. That is the vacuous arm ADR-0014"
    echo "       exists to catch, and no per-scene allowance covers it."
    GATE_FAILED=1
  fi
done

echo ""
echo "  --- every arm, side by side ---"
# The reader is `tools/window_report.py` and NOT a heredoc: `pilot-1` ran 120 episodes,
# wrote all of them, and printed "NO EPISODES ON DISK" three times because its reader was
# forty lines of Python inside a bash string and no test in the tree could see it. The
# layout here — <tag>/<arm>/<scene>/ — is the one that module already reads.
python -m earshot.tools.window_report "$OUT_DIR" --arms "${ARM_NAMES[*]}" || READ_FAILED=1

REFERENCE_ARM="$(reference_arm "${ARM_NAMES[@]}")"
REFERENCE_DIR="$OUT_DIR/$REFERENCE_ARM"
echo ""
echo "  --- each arm against $REFERENCE_ARM, PAIRED BY EPISODE ---"
echo "  Exact McNemar over the pairs. Report the scene-level sign test with it:"
echo "  a mechanism that was green and exact at the episode level has already come"
echo "  back null at the scene level in this repo, and neither test breaks the other."
if [ ! -d "$REFERENCE_DIR" ]; then
  echo "  SKIPPED: the reference arm '$REFERENCE_ARM' has no episodes on disk. Every row"
  echo "           below would be quoted against nothing, which is not a smaller result."
  echo "           It is no result."
  READ_FAILED=1
elif [ "${#ARM_NAMES[@]}" -lt 2 ]; then
  echo "  SKIPPED: one arm. There is nothing to pair it against, which is a property of"
  echo "           what was asked for and not a failure. A single arm is a pilot: it times"
  echo "           the episode and exposes mechanism failures, and resolves no difference."
else
  for arm in "${ARM_NAMES[@]}"; do
    [ "$arm" = "$REFERENCE_ARM" ] && continue
    [ -d "$OUT_DIR/$arm" ] || continue
    echo ""
    echo "  === $REFERENCE_ARM -> $arm ==="
    python -m earshot.tools.episode_diff "$REFERENCE_DIR" "$OUT_DIR/$arm" 2>&1 | tail -n 30
  done
fi

echo ""
echo "  logs:       $OUT_DIR/<arm>-<scene>.log"
echo "  provenance: $OUT_DIR/provenance.txt"
echo "  audits:     $OUT_DIR/<arm>/<scene>/"
if [ -n "$ZERO_YIELD" ]; then
  echo ""
  echo "  zero-yield scene/arm cells (measured, not failures):$ZERO_YIELD"
fi
if [ -n "$VACUOUS_CELLS" ]; then
  echo ""
  echo "  scenes where an ablation arm closed the loop ZERO times, by arm:$VACUOUS_BY_ARM"
  echo "  This is a second ordering of the same table and it belongs beside the SR column:"
  echo "  the baseline is absent from it by construction, and an arm that appears often is"
  echo "  an arm that stops working in whole rooms rather than losing episodes evenly."
  echo " $VACUOUS_CELLS"
fi

if [ "$RESUMED" -gt 0 ]; then
  # Said out loud, because the numbers above then come from more than one invocation on
  # more than one day, and a reader comparing wall clocks or quoting "one run" needs to
  # know. The EPISODES are unaffected: each scene directory still holds exactly one
  # invocation's work, which is what `episode_diff` pairs on.
  echo ""
  echo "  RESUMED: $RESUMED scene/arm cell(s) were already finished and were NOT re-run."
  echo "           This tag's records come from more than one invocation of this script."
fi

if [ "$FAILED_RUNS" -ne 0 ]; then
  echo ""
  echo "RED: $FAILED_RUNS run(s) failed for a reason other than zero yield. A sweep"
  echo "     missing runs is NOT_RUN for those cells, and NOT_RUN is red. The aggregates"
  echo "     above are over what survived."
  exit 1
fi
if [ "$READ_FAILED" -ne 0 ]; then
  echo ""
  echo "RED: the readout found no episode under an arm this sweep was ASKED to run."
  echo "     Runs that produced nothing and a reader that cannot find what they produced"
  echo "     look identical from here: check $OUT_DIR/<arm>/<scene>/episodes/."
  echo "     An arm that was never requested is NOT this: --arms without \`full\` is a"
  echo "     contrast between the arms named, quoted against the first of them."
  exit 1
fi
if [ "$GATE_FAILED" -ne 0 ]; then
  echo ""
  echo "RED: at least one smoke gate did not pass. The measurements above stand as"
  echo "     measurements; what fails is the claim that the loop behaved."
  exit 1
fi
echo ""
echo "GREEN — every arm ran and every gate passed."
echo "        arms: ${ARM_NAMES[*]}"
echo "        Every row above is quoted against '$REFERENCE_ARM'."
echo "        This is the ablation table. It is NOT ADR-0018's generalization matrix,"
echo "        which is a different driver (matrix_sweep.sh) over different cells."
