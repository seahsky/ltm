#!/bin/bash
# scripts/race-audiogoal-matrix.sh — M3 full AudioGoal ablation MATRIX
# (onset-trigger framing) on the real ReMEmbR backbone.
#
# Runs the S1/S2/S3 revisit ablation across N scenes × M anomaly classes, each
# class co-located with a DISTINCT goal category (so the analyzer's
# (scene_id, target_category) pairing keys never collide across cells), then a
# SINGLE combined warm S3-S1 verdict (+ S2 decomposition + cold control) pooled
# across all cells.
#
# Onset-trigger framing (user choice, 2026-06-18): the CLAP class is the TRIGGER
# identity (onset = when-to-act + energy/DOA homing); retrieval stays keyed to
# the co-located goal category (anomaly_object), NOT the class affordance. So
# each (class -> category) pairing is for trigger diversity; the category must
# exist in the scene (pre-flighted below).
#
# Per-cell work REUSES the tested single-cell driver scripts/race-audiogoal.sh
# (build -> render -> run S1/S2/S3 -> per-cell analyze). This wrapper:
#   [1] pulls ONCE   [2] sets up conda ONCE   [3] PRE-FLIGHTS category
#       availability for every (scene,category) -> abort before ANY GPU spend
#   [4] loops cells (children run with RACE_SKIP_PULL=1 so the matrix code can't
#       change mid-run)   [5] COMBINED analyze over all cells' out-dirs
#       (analyze_revisit pools dirs per setting).
#
# Resumable: a cell whose settings' out-dirs already hold summary.json is
# SKIPPED — re-run after a crash to continue.
#
# Runtime ~ cells × |settings| × (1 + n_warm) × ~2.7 min/ep. The default
# 2 scenes × 3 classes × {S1,S2,S3} × n_warm=16 is ~13-15 h. STAGE a first pass
# with --scenes <one> --n-warm 8 (~3-4 h) to validate mechanics before the full
# matrix.
#
#   cd ~/ltm && git pull            # REQUIRED: this driver is new / may change
#   nrun bash scripts/race-audiogoal-matrix.sh                         # full matrix
#   nrun bash scripts/race-audiogoal-matrix.sh --scenes TEEsavR23oF --n-warm 8  # stage
#
# EXECUTE (do NOT source). Children switch conda envs in their own processes.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1

SCENES="TEEsavR23oF wcojb4TFT35"
# class:category cells. Categories MUST be DISTINCT (analyzer pairs by
# (scene,category)) and present in every scene (pre-flighted). chair/sofa/bed are
# present in both default val_mini scenes; the class->category mapping is for
# trigger diversity (onset-trigger framing — class is decorative for retrieval).
CELLS="baby_cry:bed alarm:sofa glass_break:chair"
NWARM=16; SETTINGS="1 2 3"; PREFIX="m3"; TEMPORAL=""; CAPTION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scenes) SCENES="$2"; shift 2 ;;
    --cells) CELLS="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --settings) SETTINGS="$2"; shift 2 ;;
    --tag-prefix) PREFIX="$2"; shift 2 ;;
    # A/B-VARIANT arms. Each toggles ONE default-OFF S3 memory head and A/B's it
    # against the baseline S3. Only S3 calls the memory head (S1=mem-off,
    # S2=STM-only), so a variant runs ONLY S3, REUSING each cell's baseline
    # dataset/grid, into a distinct out-prefix (so baseline S3 is never
    # clobbered), then a POOLED paired compare of variant S3 (B) vs baseline S3
    # (A). REQUIRES the baseline matrix (<prefix>-*) to have run first.
    #   --temporal       → M4 recency head  (LTM_TEMPORAL_CONTEXT, out <prefix>t-*)
    #   --caption-rerank → Lever-1 caption-to-caption head (LTM_CAPTION_RERANK, out <prefix>c-*)
    --temporal) TEMPORAL=1; shift ;;
    --caption-rerank) CAPTION=1; shift ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag-prefix must be alnum/dash/underscore"; exit 1; }
[ -n "$TEMPORAL" ] && [ -n "$CAPTION" ] && { echo "FATAL: pass at most one A/B variant (--temporal XOR --caption-rerank)"; exit 1; }

# A/B-variant setup: force S3-only, turn the chosen head on for every child run,
# and write to a distinct out-prefix so the baseline S3 dirs are never clobbered.
OUT_PREFIX="$PREFIX"; VARIANT_ON=""; VARIANT_LABEL=""
if [ -n "$TEMPORAL" ]; then
  VARIANT_ON=1; VARIANT_LABEL="temporal"; OUT_PREFIX="${PREFIX}t"; export LTM_TEMPORAL_CONTEXT=1
elif [ -n "$CAPTION" ]; then
  VARIANT_ON=1; VARIANT_LABEL="caption-rerank"; OUT_PREFIX="${PREFIX}c"; export LTM_CAPTION_RERANK=1
fi
if [ -n "$VARIANT_ON" ]; then
  SETTINGS="3"
  echo "  [$VARIANT_LABEL] A/B arm: S3-only, out-prefix=$OUT_PREFIX (baseline=$PREFIX reused)"
fi

# Distinct-category guard (a shared category would cross-pair classes in the analyzer).
_cats="$(for c in $CELLS; do echo "${c#*:}"; done | sort)"
_ucats="$(echo "$_cats" | uniq)"
[ "$(echo "$_cats" | wc -l)" -eq "$(echo "$_ucats" | wc -l)" ] \
  || { echo "FATAL: --cells categories must be DISTINCT (analyzer pairs by (scene,category)): $CELLS"; exit 1; }

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
banner() { printf '\n########## %s ##########\n' "$1"; }

banner "[1/5] git pull --ff-only (ONCE for the whole matrix)"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

banner "[2/5] conda setup (source race-setup.sh -> ltm-embodied)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
# Children must NOT git-pull mid-matrix (a push during the long run could change
# code between cells). The matrix already pulled once above.
export RACE_SKIP_PULL=1

banner "[3/5] pre-flight: category availability per (scene,cell) — abort before GPU"
python - "$VALMINI" "$SCENES" "$CELLS" <<'PYEOF'
import gzip, json, os, sys
valmini, scenes, cells = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
cats = [c.split(":", 1)[1] for c in cells]
missing = 0
for s in scenes:
    src = os.path.join(valmini, f"{s}.json.gz")
    if not os.path.isfile(src):
        print(f"  {s:14s} (no content file: {src})"); missing = 1; continue
    have = {e.get("object_category") for e in json.load(gzip.open(src))["episodes"]}
    for cat in cats:
        ok = cat in have
        print(f"  {s:14s} {cat:12s} {'OK' if ok else 'MISSING (cell would be EMPTY)'}")
        missing |= (0 if ok else 1)
sys.exit(missing)
PYEOF
rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: a (scene,category) cell is empty — fix --cells; NOT spending on GPU."; exit 1; }

N_CELLS=$(echo $CELLS | wc -w); N_SCENES=$(echo $SCENES | wc -w); N_SET=$(echo $SETTINGS | wc -w)
banner "[4/5] run matrix: $N_SCENES scenes × $N_CELLS classes × settings [$SETTINGS], n_warm=$NWARM${VARIANT_ON:+ ($VARIANT_LABEL A/B arm)}"
OUT_DIRS=""; BASE_S3_DIRS=""; VAR_S3_DIRS=""
for S in $SCENES; do
  for cell in $CELLS; do
    C="${cell%%:*}"; CAT="${cell#*:}"
    # DATASET tag — CLASS-keyed so each cell owns its dataset dir
    # (audiogoal_<prefix>-<scene>-<class>). Scene-only keying made all 3 classes
    # share one dir → the baseline rebuilds it in place per cell (last class wins),
    # so temporal --reuse-dataset would read the WRONG class's manifest and FATAL.
    BASE_TAG="${PREFIX}-${S}-${C}"
    # OUTPUT tag — scene-keyed; the child appends -<class>, so out-dirs are
    # runs/<out_prefix>-<scene>-<class>-s<N> (baseline m3-*, temporal m3t-*). The
    # baseline OUTPUT is byte-identical to before (only the internal dataset dir name changed).
    OUT_TAG="${OUT_PREFIX}-${S}"
    cell_dirs=""; done_count=0
    for N in $SETTINGS; do
      od="runs/${OUT_TAG}-${C}-s${N}"; cell_dirs="$cell_dirs $od"
      [ -f "$od/summary.json" ] && done_count=$((done_count+1))
    done
    # Track the S3 dir pair for the variant pooled compare (baseline A vs variant B).
    BASE_S3_DIRS="$BASE_S3_DIRS runs/${PREFIX}-${S}-${C}-s3"
    VAR_S3_DIRS="$VAR_S3_DIRS runs/${OUT_TAG}-${C}-s3"
    if [ "$done_count" -eq "$N_SET" ]; then
      echo "  RESUME: cell ($S,$C->$CAT) already complete ($N_SET/$N_SET out-dirs) — skipping"
      OUT_DIRS="$OUT_DIRS $cell_dirs"; continue
    fi
    banner "cell: scene=$S class=$C category=$CAT  n_warm=$NWARM settings=[$SETTINGS] out_tag=$OUT_TAG"
    # --out-tag == --tag in baseline mode (unchanged); in a variant arm it diverges so
    # the baseline S3 dirs are never clobbered, and --reuse-dataset reuses the baseline
    # build (same episodes → the (scene,category,visit_order) pairing matches).
    bash scripts/race-audiogoal.sh --scene "$S" --class "$C" --category "$CAT" \
        --n-warm "$NWARM" --settings "$SETTINGS" --tag "$BASE_TAG" --out-tag "$OUT_TAG" \
        ${VARIANT_ON:+--reuse-dataset} \
      || { echo "FATAL: cell ($S,$C->$CAT) failed — fix + re-run (completed cells resume)."; exit 1; }
    # Belt-and-suspenders: don't trust the child's exit code alone — verify every
    # setting wrote a summary.json before counting the cell done, so a silent
    # partial cell can never feed the combined analyze under-powered data.
    for od in $cell_dirs; do
      [ -f "$od/summary.json" ] \
        || { echo "FATAL: cell ($S,$C->$CAT) returned OK but $od/summary.json is missing — refusing to analyze incomplete data."; exit 1; }
    done
    OUT_DIRS="$OUT_DIRS $cell_dirs"
  done
done

if [ -n "$VARIANT_ON" ]; then
  banner "[5/5] $VARIANT_LABEL A/B verdict (pooled paired S3: variant-on B − baseline-off A)"
  # The A arm is the BASELINE matrix's S3 dirs — require them (the variant arm only
  # ran/reused S3; it never produced the baseline-off S3). Fail loud if absent.
  _missing=""
  for d in $BASE_S3_DIRS; do [ -f "$d/summary.json" ] || _missing="$_missing $d"; done
  [ -z "$_missing" ] || { echo "FATAL: baseline S3 dir(s) missing — run the baseline matrix (no variant flag) first:$_missing"; exit 1; }
  # shellcheck disable=SC2086
  python embodied_memory/scripts/analyze_revisit.py \
      --compare-a $BASE_S3_DIRS --compare-b $VAR_S3_DIRS \
      2>&1 | tee "runs/${OUT_PREFIX}-${VARIANT_LABEL}-compare.log"
  echo
  echo "DONE. $VARIANT_LABEL A/B (pooled paired S3 variant-on vs baseline-off)."
  echo "  A (baseline S3):$BASE_S3_DIRS"
  echo "  B ($VARIANT_LABEL S3):$VAR_S3_DIRS"
  echo "  WARM B−A > 0 (p<0.1) ⇒ $VARIANT_LABEL head helps; ≤ 0 / mem over-fires ⇒ honest negative."
else
  banner "[5/5] COMBINED matrix verdict (pooled across cells: warm S3-S1 + S2 decomposition + cold control)"
  # shellcheck disable=SC2086
  python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS \
      2>&1 | tee "runs/${PREFIX}-matrix-analysis.log"
  echo
  echo "DONE. M3 AudioGoal matrix (onset-trigger framing)."
  echo "  Per-cell Gate-A verdicts + planner census: in each cell's run log above."
  echo "  COMBINED matrix verdict (pooled by setting, paired by (scene,category)): [5/5] block."
  echo "  Out-dirs:$OUT_DIRS"
fi