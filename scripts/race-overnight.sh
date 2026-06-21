#!/bin/bash
# scripts/race-overnight.sh — ONE overnight RACE run chaining the two compute tracks
# the user picked, FAIL-ISOLATED (a failure in one section never aborts the other),
# with a final SUMMARY that greps the headline lines so the emailed report shows the
# numbers up top.
#
#   ② CORRECTED PRIMARY HEADLINE — wide revisit matrix under the shuffle fix
#      (the citable +0.115 n=26; pure vision; ~6-8h). Re-runs fresh (rm -rf rebuild)
#      so the now-pinned shuffle=False ordering is exercised from a clean dataset.
#   ① MAKE-OR-BREAK AUDIO — non-LOS write-ON vs write-OFF A/B (--reuse-nonlos)
#      consuming the gate's away-facing non-LOS seed + source-targeted grid for BOTH
#      arms (no rebuild / no random re-render; REMEMBR_CONSUME_SINGLEGOAL fixed on
#      both; LTM_AUDIO_WRITE arm-B only). ~1.5-2h.
#
# Default = ② then ① (~8-10h, overnight-sized). Set RUN_M3=1 to ALSO re-run the
# full M3 AudioGoal matrix (~13-15h MORE — a weekend-sized total, not overnight).
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull --ff-only \
#     && nrun bash scripts/race-overnight.sh
#
# Requires (for ①) the $0 gate outputs already on RACE (present; Tier-3 GREEN):
#   runs/nonlos-gate/{audiogoal.json.gz, content/wcojb4TFT35.json.gz, source_manifest.json}
#   runs/audiogoal/wcojb4TFT35_glass_break_rir_grid.npz
# The A/B section's own [2/4] pre-flight HARD-FAILS (no spend) if any are missing.
#
# Each child driver does its own git-pull + conda setup in a subshell; this wrapper
# only orchestrates + greps. Run it via `nrun` (the outer email/detach wrapper) —
# do NOT nrun the children (this calls them with plain bash).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
mkdir -p runs

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

banner "race-overnight @ $(hostname)  HEAD=$(git rev-parse --short HEAD)  branch=$(git rev-parse --abbrev-ref HEAD)"
command -v nvidia-smi >/dev/null 2>&1 && \
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
echo "RUN_M3=${RUN_M3:-<unset, M3 matrix skipped>}"

# ── ② corrected primary headline: WIDE revisit matrix (~6-8h) ──────────────────
runsec "wide-revisit (corrected +0.115 n=26 headline)" runs/overnight-wide.log \
  bash scripts/race-revisit.sh --tag revisit-shuffix-wide \
    --scenes "wcojb4TFT35 TEEsavR23oF" \
    --categories "chair bed sofa plant toilet tv_monitor" --n-warm 3
RC_WIDE=$RUNSEC_RC

# ── ① make-or-break audio: non-LOS write-ON/OFF A/B (~1.5-2h) ───────────────────
runsec "nonlos-audio-AB (--reuse-nonlos, sound-on vs sound-off)" runs/overnight-ab.log \
  bash scripts/race-audiogoal-lifelong.sh --reuse-nonlos \
    --cells "wcojb4TFT35:glass_break:chair"
RC_AB=$RUNSEC_RC

# ── (optional) M3 AudioGoal matrix re-run (~13-15h) — only with RUN_M3=1 ────────
RC_M3="skipped"
if [ -n "${RUN_M3:-}" ]; then
  runsec "m3-audiogoal-matrix (corrected +0.171 n=18, ~13-15h)" runs/overnight-m3.log \
    bash scripts/race-audiogoal-matrix.sh --tag-prefix m3fix --n-warm 16
  RC_M3=$RUNSEC_RC
fi

# ── SUMMARY (headline lines for the emailed report) ────────────────────────────
banner "OVERNIGHT SUMMARY  (exit codes: wide=$RC_WIDE  ab=$RC_AB  m3=$RC_M3)"
echo "== ②  WIDE revisit — corrected warm paired soft-SPL S3-S1  (archived was +0.115, p=0.005) =="
grep -hE "WARM S3 - S1|COLD S3 - S1|warm .*S2|Gate A verdict|outcome:" runs/overnight-wide.log 2>/dev/null \
  | sed 's/^/  /' | tail -n 24 || echo "  (no wide-matrix summary captured — check runs/overnight-wide.log)"
echo
echo "== ①  non-LOS audio write-ON vs write-OFF A/B  (HELPS => sound causally adds nav info) =="
grep -hE "REDUNDANT|HELPS|HURTS|NO-WRITE|WRITE-NOT-RECALLED|B-A|B−A|90% CI|EXCLUDES 0|straddle|leave-one|seed_writes|recall_recalled|n_audio_writes" runs/overnight-ab.log 2>/dev/null \
  | sed 's/^/  /' | tail -n 28 || echo "  (no A/B summary captured — check runs/overnight-ab.log)"
if [ -n "${RUN_M3:-}" ]; then
  echo; echo "== (opt) M3 matrix — corrected warm S3-S1 (archived was +0.171) =="
  grep -hE "WARM S3 - S1|COLD S3 - S1|outcome:" runs/overnight-m3.log 2>/dev/null \
    | sed 's/^/  /' | tail -n 14 || echo "  (no M3 summary)"
fi
echo
echo "Full logs: runs/overnight-{wide,ab$([ -n "${RUN_M3:-}" ] && echo ,m3)}.log"
echo "Compare to archived headline on RACE (cross-run, not paired):"
echo "  python embodied_memory/scripts/analyze_ablation.py --revisit runs/scorer-d3-s1 runs/wide-s2 runs/scorer-d3-s3-heur"
banner "race-overnight DONE"
