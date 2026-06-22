#!/bin/bash
# scripts/race-partb-overnight.sh — ONE gated overnight RACE run, priority-ordered
# by EV, that (1) tests the CORRECTED root-cause fix and (2) finishes Part B (the
# multi-instance disambiguation harness) only AS FAR AS the evidence justifies.
#
# Run it (NOTE the self-pull gotcha — this driver is NEW; bash runs the OPEN fd, so
# a brand-new driver must be pulled MANUALLY before the first nrun, else nrun runs
# whatever is on disk now — which is fine the FIRST time, but a later code change
# needs a 2nd invocation):
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull --ff-only \
#       && nrun bash scripts/race-partb-overnight.sh
#
# The CHAIN (each step FAIL-ISOLATED via runsec()/RUNSEC_RC — a failure in one step
# never aborts the others; the emailed SUMMARY greps the headline lines):
#
#   STEP 1  [HIGHEST EV — the ACTUAL bottleneck]  consume A/B on the AudioGoal warm
#           matrix.  The corrected root cause is NAVIGATION/TERMINATION thrash:
#           REMEMBR_CONSUME_SINGLEGOAL is default-OFF, so a reached memory waypoint
#           is re-picked every step (n_memory_consumed=0, oscillation) — alarm cells
#           mem_chosen 133/137 + worst soft-SPL = the smoking gun. This re-runs the
#           m3 baseline matrix's S3 with consume ON (out-prefix m3k-*), then a POOLED
#           paired compare B(consume-ON S3) vs A(baseline m3-*-s3). REQUIRES the
#           baseline matrix (m3-*) on RACE; if absent we BUILD it first (so the A arm
#           exists), then the consume arm. NEEDS a one-line --consume arm in
#           race-audiogoal-matrix.sh (see CONSUME_ARM_PRESENT pre-flight below).
#
#   STEP 2  build the instance-keyed (Part B) dataset for the feasible cells (the
#           builder ALREADY produces disambiguation-forcing warm starts — see the
#           validity note below; NO adversarial-placement code is needed/added).
#
#   STEP 3  the metric-validity GATE: check_instance_keyed_validity --use-pathfinder
#           (true geodesics on RACE). Captures the per-cell VALID/DEGENERATE verdict.
#
#   STEP 4  [CONDITIONAL — only if STEP 3 finds >=1 VALID cell] the paid instance-
#           keyed S1/S2/S3 matrix on the VALID cells, analyzed + the wrong-instance-
#           recall readout. If NONE valid: SKIP and report "harness degenerate".
#
# IMPORTANT HONESTY NOTE wired into STEP 4: the consume fix (_consume_memory_applies)
# ungates ONLY for task==audiogoal; the Part B matrix is task==objectnav, so
# REMEMBR_CONSUME_SINGLEGOAL is INERT there (byte-identical to default). We DO export
# it (harmless, future-proof) but the step's value is the wrong-instance readout +
# the VALID/DEGENERATE verdict, NOT a consume A/B. The consume A/B lives in STEP 1.
#
# EXECUTE (do NOT source). Children switch conda envs in their own processes; this
# wrapper only orchestrates + greps. Run via `nrun` (the email/detach wrapper) — do
# NOT nrun the children.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
mkdir -p runs

# ---- knobs ----
# Feasible cells (Part-A GREEN, $0 check_multi_instance_feasible): chair+bed both
# scenes, TEEsav sofa, wcojb toilet. DISTINCT categories per scene so the analyzer's
# (scene,category) pairing never cross-collides. Euclidean validity proxy is GREEN on
# {chair (8-11 distractors), wcojb bed, wcojb toilet, TEEsav sofa}; the RACE geodesic
# gate (STEP 3) is the real verdict.
PARTB_SCENES="${PARTB_SCENES:-wcojb4TFT35 TEEsavR23oF}"
PARTB_CATS="${PARTB_CATS:-chair bed sofa toilet}"
PARTB_NWARM="${PARTB_NWARM:-8}"
PARTB_TAG="partb-ik"
PARTB_DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/${PARTB_TAG}"
# AudioGoal consume A/B (STEP 1): the m3 baseline matrix prefix + the consume arm.
M3_PREFIX="${M3_PREFIX:-m3}"
M3_NWARM="${M3_NWARM:-16}"
# RUN_STEP1_BUILD_BASELINE=1 forces (re)building the m3 baseline if its S3 dirs are
# absent on RACE; default off (the baseline is expected to already exist — STEP 1
# fails LOUD if it doesn't, rather than silently spending ~13 h rebuilding it).
RUN_STEP1_BUILD_BASELINE="${RUN_STEP1_BUILD_BASELINE:-}"

banner() { printf '\n\n############### %s ###############\n\n' "$1"; }
RUNSEC_RC=0
runsec() {  # name  logfile  cmd...
  local name="$1" log="$2"; shift 2
  banner "START  $name"
  echo "  \$ $*"
  ( "$@" ) 2>&1 | tee "$log"
  RUNSEC_RC=${PIPESTATUS[0]}
  banner "END    $name (rc=$RUNSEC_RC)"
}

banner "race-partb-overnight @ $(hostname)  HEAD=$(git rev-parse --short HEAD)  branch=$(git rev-parse --abbrev-ref HEAD)"
command -v nvidia-smi >/dev/null 2>&1 && \
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

# ── one-time setup so the wrapper's own python (validity gate, builds) works ─────
banner "[setup] source race-setup.sh -> ltm-embodied"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Does race-audiogoal-matrix.sh have the --consume variant arm? (a 4th arm beside
# --temporal/--caption-rerank/--query-expansion: export REMEMBR_CONSUME_SINGLEGOAL=1,
# OUT_PREFIX=${PREFIX}k, S3-only, --reuse-dataset, pooled compare B vs baseline S3.)
CONSUME_ARM_PRESENT=0
grep -q -- "--consume)" scripts/race-audiogoal-matrix.sh && CONSUME_ARM_PRESENT=1

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — consume A/B on the AudioGoal warm matrix (the corrected bottleneck).
# ════════════════════════════════════════════════════════════════════════════════
RC_STEP1="skipped"
VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
# A arm = baseline m3-*-s3. Verify it exists (or build it if asked); else STEP 1 is
# a no-op compare and we say so rather than spend on a B arm with no A.
_base_missing=""
for S in $PARTB_SCENES; do
  # m3 default cells = baby_cry:bed alarm:sofa glass_break:chair (matrix default).
  for cls in baby_cry alarm glass_break; do
    d="runs/${M3_PREFIX}-${S}-${cls}-s3"
    [ -f "$d/summary.json" ] || _base_missing="$_base_missing $d"
  done
done
if [ "$CONSUME_ARM_PRESENT" != 1 ]; then
  banner "STEP 1 SKIPPED — race-audiogoal-matrix.sh has NO --consume arm yet"
  echo "  Add the 4th variant arm (mirror --temporal: export REMEMBR_CONSUME_SINGLEGOAL=1,"
  echo "  OUT_PREFIX=\${PREFIX}k, S3-only, --reuse-dataset, pooled compare B vs baseline S3),"
  echo "  commit to lifelong-revisit-eval, then re-run. STEP 1 is the HIGHEST-EV step."
  RC_STEP1="needs --consume arm"
elif [ -n "$_base_missing" ] && [ -z "$RUN_STEP1_BUILD_BASELINE" ]; then
  banner "STEP 1 SKIPPED — baseline m3 S3 dirs missing (no A arm to pair against)"
  echo "  missing:$_base_missing"
  echo "  Run the baseline matrix first:  nrun bash scripts/race-audiogoal-matrix.sh --tag-prefix $M3_PREFIX --n-warm $M3_NWARM"
  echo "  or set RUN_STEP1_BUILD_BASELINE=1 to build it here (adds ~13-15 h)."
  RC_STEP1="needs baseline m3-*"
else
  if [ -n "$_base_missing" ]; then
    runsec "STEP1a baseline m3 matrix (A arm)" runs/partb-step1-baseline.log \
      bash scripts/race-audiogoal-matrix.sh --tag-prefix "$M3_PREFIX" --n-warm "$M3_NWARM"
  fi
  runsec "STEP1b consume A/B (B=consume-ON S3 vs A=baseline m3 S3)" runs/partb-step1-consume.log \
    bash scripts/race-audiogoal-matrix.sh --tag-prefix "$M3_PREFIX" --n-warm "$M3_NWARM" --consume
  RC_STEP1=$RUNSEC_RC
fi

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — build the instance-keyed (Part B) dataset for the feasible cells.
# The builder ALREADY emits disambiguation-forcing warm starts (warm starts are
# drawn from the ObjectNav category source starts, which scatter across the scene's
# MANY same-category instances; with the goal keyed to ONE instance a distractor is
# frequently nearer). NO adversarial-placement code is added — it would be premature.
# ════════════════════════════════════════════════════════════════════════════════
runsec "STEP2 build instance-keyed dataset ($PARTB_CATS @ $PARTB_SCENES, n_warm=$PARTB_NWARM)" \
  runs/partb-step2-build.log bash -c '
    set -uo pipefail
    rm -rf "'"$PARTB_DS_DIR"'"
    for SCENE in '"$PARTB_SCENES"'; do
      SRC="'"$VALMINI"'/${SCENE}.json.gz"
      [ -f "$SRC" ] || { echo "FATAL: missing source $SRC"; exit 1; }
      python embodied_memory/scripts/make_revisit_smoke.py \
          --src "$SRC" --scene "$SCENE" --categories '"$PARTB_CATS"' \
          --n-warm '"$PARTB_NWARM"' --instance-keyed --out-dir "'"$PARTB_DS_DIR"'" \
        || { echo "FATAL: instance-keyed build failed for $SCENE"; exit 1; }
    done
    echo "built content:"; ls -la "'"$PARTB_DS_DIR"'/content/" 2>/dev/null
  '
RC_STEP2=$RUNSEC_RC

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — the metric-validity GATE (true geodesics, --use-pathfinder).
# Captures per-cell VALID/DEGENERATE; rc 0 = >=1 VALID (GREEN), 2 = none (RED).
# ════════════════════════════════════════════════════════════════════════════════
RC_STEP3=2
if [ "$RC_STEP2" -eq 0 ]; then
  runsec "STEP3 validity gate (--use-pathfinder geodesics)" runs/partb-step3-validity.log \
    python embodied_memory/scripts/check_instance_keyed_validity.py \
      "$PARTB_DS_DIR"/content/*.json.gz --use-pathfinder
  RC_STEP3=$RUNSEC_RC
else
  banner "STEP 3 SKIPPED — STEP 2 build failed (rc=$RC_STEP2)"
fi
# Parse the VALID cells (scene:category) from the gate log for the conditional STEP 4.
VALID_CELLS="$(grep -E '\bVALID\b' runs/partb-step3-validity.log 2>/dev/null \
  | awk '{print $1":"$2}' | sort -u | tr '\n' ' ')"

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — CONDITIONAL paid instance-keyed S1/S2/S3 matrix on the VALID cells.
# ════════════════════════════════════════════════════════════════════════════════
RC_STEP4="skipped"
if [ "$RC_STEP3" -ne 0 ] || [ -z "$VALID_CELLS" ]; then
  banner "STEP 4 SKIPPED — harness DEGENERATE (no VALID cell on the geodesic gate)"
  echo "  Adversarial placement was INSUFFICIENT on val_mini (or all cells DEGENERATE/UNREACHABLE)."
  echo "  Do NOT spend the paid matrix. STEP 3 verdict above is the deliverable."
  RC_STEP4="degenerate — skipped paid matrix"
else
  banner "STEP 4 — paid instance-keyed matrix on VALID cells: $VALID_CELLS"
  # Restrict the run to the VALID cells. Rebuild ONLY those (scene,category) cells so
  # the runner never wastes GPU on DEGENERATE ones. Group categories by scene; build a
  # per-scene content with only that scene's VALID categories.
  # NOTE: REMEMBR_CONSUME_SINGLEGOAL is INERT for task==objectnav (the Part B task);
  # we export it for future-proofing only — STEP 4's value is the wrong-instance
  # readout + the VALID verdict, NOT a consume A/B (that is STEP 1).
  export REMEMBR_CONSUME_SINGLEGOAL=1
  IK_DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/${PARTB_TAG}-valid"
  rm -rf "$IK_DS_DIR"
  for SCENE in $PARTB_SCENES; do
    # categories that are VALID in THIS scene
    cats=""
    for vc in $VALID_CELLS; do
      vs="${vc%%:*}"; vcat="${vc#*:}"
      # the gate prints the scene as the glb path basename-ish; match on substring
      case "$vs" in *"$SCENE"*) cats="$cats $vcat" ;; esac
    done
    [ -z "$cats" ] && { echo "  (no VALID category for $SCENE — skip its build)"; continue; }
    SRC="$VALMINI/${SCENE}.json.gz"
    echo "  building VALID cells for $SCENE:$cats"
    # shellcheck disable=SC2086
    python embodied_memory/scripts/make_revisit_smoke.py \
        --src "$SRC" --scene "$SCENE" --categories $cats \
        --n-warm "$PARTB_NWARM" --instance-keyed --out-dir "$IK_DS_DIR" \
      || { echo "FATAL: VALID-cell build failed for $SCENE"; }
  done
  IK_TOP="$IK_DS_DIR/$(basename "$IK_DS_DIR").json.gz"
  if [ ! -f "$IK_TOP" ]; then
    banner "STEP 4 ABORTED — VALID-cell rebuild produced no dataset"
    RC_STEP4="build-failed"
  else
    N_EP="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "$IK_DS_DIR/content/*.json.gz" 2>/dev/null || echo 0)"
    IK_OUT_DIRS=""
    for S in 1 2 3; do
      od="runs/${PARTB_TAG}-s$S"
      runsec "STEP4 instance-keyed setting=$S" "runs/partb-step4-s$S.log" \
        env REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
          --backbone remembr --setting "$S" --episodes-path "$IK_TOP" \
          --scene all --target any --n-episodes "$N_EP" --out-dir "$od"
      [ -f "$od/summary.json" ] && IK_OUT_DIRS="$IK_OUT_DIRS $od"
    done
    RC_STEP4=$RUNSEC_RC
    banner "STEP 4 analysis — Gate-A (paired soft-SPL S3-S1 + S2 decomposition)"
    # shellcheck disable=SC2086
    [ -n "$IK_OUT_DIRS" ] && python embodied_memory/scripts/analyze_ablation.py --revisit $IK_OUT_DIRS \
      2>&1 | tee runs/partb-step4-gateA.log
    banner "STEP 4 wrong-instance recall readout (Part B's whole point)"
    # shellcheck disable=SC2086
    python embodied_memory/scripts/diagnose_goal_anchored_recall.py runs/${PARTB_TAG}-s3 \
        --content-dir "$IK_DS_DIR/content" 2>&1 | tee runs/partb-step4-wronginst.log || true
  fi
fi

# ── SUMMARY (headline lines for the emailed report) ─────────────────────────────
banner "PART-B OVERNIGHT SUMMARY  (step1=$RC_STEP1  step2=$RC_STEP2  step3=$RC_STEP3  step4=$RC_STEP4)"
echo "== STEP 1  consume A/B (B=consume-ON vs A=baseline) — does damping re-pick thrash help warm? =="
grep -hE "WARM B|B−A|B-A|90% CI|REDUNDANT|HELPS|HURTS|honest negative|mem_chosen|replan_stuck" runs/partb-step1-consume.log 2>/dev/null \
  | sed 's/^/  /' | tail -n 20 || echo "  (STEP 1 not run — see banner reason above)"
echo
echo "== STEP 3  metric-validity gate (geodesic) — is the harness non-degenerate? =="
grep -hE "verdict|VALID|DEGENERATE|UNREACHABLE|GREEN|RED" runs/partb-step3-validity.log 2>/dev/null \
  | sed 's/^/  /' | tail -n 16 || echo "  (no validity output)"
echo "  VALID cells: ${VALID_CELLS:-<none>}"
echo
echo "== STEP 4  instance-keyed matrix (only if GREEN) — Gate-A + wrong-instance recall =="
grep -hE "WARM S3 - S1|COLD S3 - S1|warm .*S2|Gate A verdict|wrong-instance recall|RECALL-GAP|ANCHOR" \
  runs/partb-step4-gateA.log runs/partb-step4-wronginst.log 2>/dev/null \
  | sed 's/^/  /' | tail -n 16 || echo "  (STEP 4 skipped — degenerate or build failed)"
echo
echo "Full logs: runs/partb-step{1-consume,2-build,3-validity,4-*}.log"
banner "race-partb-overnight DONE"
