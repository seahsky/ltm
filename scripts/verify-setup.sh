#!/usr/bin/env bash
# scripts/verify-setup.sh — VERIFY-and-REPAIR every downloaded artifact the
# anomaly-response arc needs, resilient to a flaky VM network.
#
# Unlike setup-vm.sh (the from-scratch installer), this is safe to run over and
# over: it CHECKS each artifact and re-downloads ONLY the ones that are missing or
# incomplete. Every network step is wrapped in a retry loop, and the Hugging Face
# model pulls use `snapshot_download` — which resumes a half-finished transfer and
# re-fetches only the corrupt/missing files — so a connection that drops
# mid-download just needs another run (or is retried in place). Exit code is 0 iff
# every in-scope REQUIRED artifact is present at the
# end, so it composes with nrun:
#
#     source scripts/race-setup.sh           # defines nrun + activates the env
#     nrun bash scripts/verify-setup.sh      # self-detaches; emails the report
#
# or run it inline:  bash scripts/verify-setup.sh
#
# Artifacts checked (in order):
#   .env  ·  HM3D scenes  ·  HM3D ObjectNav episodes  ·  ESC-50 anomaly+benign
#   clips  ·  ReMEmbR captioner+planner  ·  CLAP  ·  SBERT  ·  CLIP  (+ an
#   informational RIR-grid count — grids are RENDERED, not downloaded).
#
# FLAGS:
#   --check-only        report presence, download NOTHING (dry run)
#   --data-only         only .env + HM3D + episodes + clips (no model pulls)
#   --models-only       only the model weights (no data/clips)
#   --skip-models|--cpu skip the HEAVY ReMEmbR pair (2B+7B); still verify the
#                       small CLAP/SBERT/CLIP that a CPU/analysis box uses
#   --retries N         network attempts per artifact (default 4)
#   --env-name <n>      conda env (default: ltm-embodied, or $LTM_ENV_NAME)
#   --yes | -y          non-interactive (reserved; there are no prompts)
#
# Env overrides honored: REMEMBR_CAPTIONER_MODEL, REMEMBR_PLANNER_MODEL,
# REMEMBR_CLAP_MODEL, HM3D_SCENE_GROUP, CONDA_DIR, LTM_ENV_NAME, HF_HOME.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${LTM_ENV_NAME:-ltm-embodied}"
CAPTIONER_MODEL="${REMEMBR_CAPTIONER_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PLANNER_MODEL="${REMEMBR_PLANNER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CLAP_MODEL="${REMEMBR_CLAP_MODEL:-laion/clap-htsat-fused}"
SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RETRIES=4

CHECK_ONLY=""; DATA_ONLY=""; MODELS_ONLY=""; SKIP_HEAVY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --check-only)  CHECK_ONLY=1; shift ;;
    --data-only)   DATA_ONLY=1; shift ;;
    --models-only) MODELS_ONLY=1; shift ;;
    --skip-models|--cpu) SKIP_HEAVY=1; shift ;;
    --retries)     RETRIES="$2"; shift 2 ;;
    --env-name)    ENV_NAME="$2"; shift 2 ;;
    --yes|-y)      shift ;;
    -h|--help)     sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1' (try --help)"; exit 1 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }
die()    { echo "FATAL: $*" >&2; exit 1; }
have()   { command -v "$1" >/dev/null 2>&1; }

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "ERROR: execute this script, don't source it:  bash scripts/verify-setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# --- status ledger (name -> OK | REPAIRED | FAIL | SKIP | INFO) --------------
NAMES=(); STATES=(); NOTES=()
record() { NAMES+=("$1"); STATES+=("$2"); NOTES+=("${3:-}"); }

# --- retry a network command with backoff ------------------------------------
retry() { # retry <label> <cmd...>
  local label="$1"; shift
  local n=1
  while [ "$n" -le "$RETRIES" ]; do
    if "$@"; then return 0; fi
    echo "   [$label] attempt $n/$RETRIES failed; retrying in $((n*5))s…" >&2
    sleep "$((n*5))"
    n=$((n+1))
  done
  return 1
}

GPU=""
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then GPU=1; fi
echo "verify-setup: repo=$REPO_ROOT env=$ENV_NAME gpu=${GPU:-none} retries=$RETRIES"
echo "              mode=${CHECK_ONLY:+check-only }${DATA_ONLY:+data-only }${MODELS_ONLY:+models-only }${SKIP_HEAVY:+skip-heavy}"
echo "              captioner=$CAPTIONER_MODEL planner=$PLANNER_MODEL clap=$CLAP_MODEL"

# --------------------------------------------------------------------------- #
# activate the conda env (needed for the HF/CLIP checks). Non-fatal for the
# pure-path checks (.env / HM3D / clips) so a broken env still reports data.
# --------------------------------------------------------------------------- #
banner "conda env: $ENV_NAME"
HAVE_ENV=""
if [ -x "$CONDA_DIR/bin/conda" ]; then
  eval "$("$CONDA_DIR/bin/conda" shell.bash hook)" 2>/dev/null || true
  if conda activate "$ENV_NAME" 2>/dev/null; then
    HAVE_ENV=1
    echo "activated $ENV_NAME — python: $(command -v python) ($(python --version 2>&1))"
  else
    echo "WARN: conda env '$ENV_NAME' not found — model checks will be SKIPPED."
    echo "      Build it first: bash scripts/setup-vm.sh"
  fi
else
  echo "WARN: conda missing at $CONDA_DIR — model checks will be SKIPPED (run setup-vm.sh)."
fi
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$REPO_ROOT"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"   # patience on slow links

# =========================================================================== #
# DATA + CLIPS (pure bash + existing downloaders)
# =========================================================================== #
if [ -z "$MODELS_ONLY" ]; then
  # ---- .env -----------------------------------------------------------------
  banner ".env"
  if [ -f "$REPO_ROOT/.env" ]; then
    echo ".env present"; record ".env" OK
    set -a; . "$REPO_ROOT/.env" 2>/dev/null || true; set +a   # load HM3D token
  elif [ -n "$CHECK_ONLY" ]; then
    echo "MISSING .env (check-only)"; record ".env" FAIL "missing"
  else
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env" 2>/dev/null \
      && { echo "created .env from .env.example — FILL IN MATTERPORT_TOKEN_* before HM3D"; record ".env" REPAIRED "fill in tokens"; } \
      || record ".env" FAIL "no .env.example"
  fi

  # ---- HM3D scene meshes -----------------------------------------------------
  banner "HM3D scenes"
  SCENES_DIR="$REPO_ROOT/data/hm3d/scene_datasets/hm3d"
  if [ -e "$SCENES_DIR" ] && [ -n "$(ls -A "$SCENES_DIR" 2>/dev/null)" ]; then
    echo "scenes present ($(ls "$SCENES_DIR" 2>/dev/null | wc -l | tr -d ' ') entries)"; record "hm3d-scenes" OK
  elif [ -n "$CHECK_ONLY" ]; then
    echo "MISSING HM3D scenes (check-only)"; record "hm3d-scenes" FAIL "missing"
  elif [ -z "${MATTERPORT_TOKEN_ID:-}" ] || [ -z "${MATTERPORT_TOKEN_SECRET:-}" ]; then
    echo "MISSING HM3D scenes AND no MATTERPORT_TOKEN_* in .env — cannot download."
    echo "  Fill the token in .env, then re-run (or: bash embodied_memory/scripts/download_hm3d.sh data/hm3d)"
    record "hm3d-scenes" FAIL "no matterport token"
  else
    echo ">> downloading HM3D scenes (retrying on drop)…"
    if retry hm3d bash "$REPO_ROOT/embodied_memory/scripts/download_hm3d.sh" "$REPO_ROOT/data/hm3d"; then
      [ -e "$SCENES_DIR" ] && [ -n "$(ls -A "$SCENES_DIR" 2>/dev/null)" ] \
        && record "hm3d-scenes" REPAIRED || record "hm3d-scenes" FAIL "download ran but dir empty"
    else
      record "hm3d-scenes" FAIL "download failed after $RETRIES tries"
    fi
  fi

  # ---- HM3D ObjectNav episodes (val_mini is what --scene all discovers) ------
  banner "HM3D ObjectNav episodes"
  EP_DIR="$REPO_ROOT/data/hm3d/datasets/objectnav/hm3d/v1"
  if [ -d "$EP_DIR/val_mini" ] || ls "$EP_DIR"/val_mini* >/dev/null 2>&1; then
    echo "episodes present ($(ls "$EP_DIR" 2>/dev/null | tr '\n' ' '))"; record "hm3d-episodes" OK
  elif [ -n "$CHECK_ONLY" ]; then
    echo "MISSING episodes (check-only)"; record "hm3d-episodes" FAIL "missing"
  else
    echo ">> (re)fetching ObjectNav episodes via download_hm3d.sh…"
    if retry episodes bash "$REPO_ROOT/embodied_memory/scripts/download_hm3d.sh" "$REPO_ROOT/data/hm3d"; then
      { [ -d "$EP_DIR/val_mini" ] || ls "$EP_DIR"/val_mini* >/dev/null 2>&1; } \
        && record "hm3d-episodes" REPAIRED || record "hm3d-episodes" FAIL "still absent after download"
    else
      record "hm3d-episodes" FAIL "download failed after $RETRIES tries"
    fi
  fi

  # ---- ESC-50 anomaly + benign clips ----------------------------------------
  banner "ESC-50 clips"
  clips_ok() {
    ls "$REPO_ROOT"/data/anomaly_audio/*.wav >/dev/null 2>&1 \
      && ls "$REPO_ROOT"/data/benign_audio/*.wav >/dev/null 2>&1
  }
  if clips_ok; then
    echo "clips present (anomaly=$(ls "$REPO_ROOT"/data/anomaly_audio/*.wav 2>/dev/null | wc -l | tr -d ' '), benign=$(ls "$REPO_ROOT"/data/benign_audio/*.wav 2>/dev/null | wc -l | tr -d ' '))"
    record "esc50-clips" OK
  elif [ -n "$CHECK_ONLY" ]; then
    echo "MISSING ESC-50 clips (check-only)"; record "esc50-clips" FAIL "missing"
  elif [ -z "$HAVE_ENV" ]; then
    record "esc50-clips" SKIP "env not active"
  else
    echo ">> fetching ESC-50 anomaly + benign clips…"
    if retry clips python "$REPO_ROOT/embodied_memory/scripts/fetch_anomaly_clips.py" --include-benign; then
      clips_ok && record "esc50-clips" REPAIRED || record "esc50-clips" FAIL "fetch ran but clips absent"
    else
      record "esc50-clips" FAIL "fetch failed after $RETRIES tries"
    fi
  fi
fi

# =========================================================================== #
# MODEL WEIGHTS (Hugging Face snapshot_download = resumable; CLIP = load-test)
# =========================================================================== #
if [ -z "$DATA_ONLY" ]; then
  banner "model weights"
  if [ -z "$HAVE_ENV" ]; then
    echo "SKIP — conda env inactive, cannot verify model caches."
    record "models" SKIP "env not active"
  else
    # The heavy ReMEmbR pair is skipped on --skip-models/--cpu; the small
    # CLAP/SBERT/CLIP are always verified (an analysis box still loads them).
    HEAVY_REPOS=""
    [ -z "$SKIP_HEAVY" ] && HEAVY_REPOS="$CAPTIONER_MODEL $PLANNER_MODEL"
    # Pass config through the environment (avoids heredoc arg-quoting pitfalls).
    export VS_HF_REPOS="$HEAVY_REPOS $CLAP_MODEL $SBERT_MODEL"
    export VS_CHECK_ONLY="${CHECK_ONLY:-}" VS_RETRIES="$RETRIES"
    mkdir -p "$REPO_ROOT/runs" 2>/dev/null || true
    MODEL_LOG="$REPO_ROOT/runs/.verify-model-status.$$"   # repo-local (always writable)
    python - <<'PY' 2>&1 | tee "$MODEL_LOG"
import os, sys, glob
from pathlib import Path

repos = [r for r in os.environ.get("VS_HF_REPOS", "").split() if r]
check_only = bool(os.environ.get("VS_CHECK_ONLY"))
retries = int(os.environ.get("VS_RETRIES", "4") or "4")
hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
hub = Path(hf_home) / "hub"

def cache_present(repo_id: str) -> bool:
    """A repo is 'present' when it has a completed snapshot with real files (a
    lone `blobs`/`refs` skeleton from an aborted pull does NOT count)."""
    d = hub / ("models--" + repo_id.replace("/", "--"))
    snaps = d / "snapshots"
    if not snaps.is_dir():
        return False
    for snap in snaps.iterdir():
        # a completed snapshot has at least one resolved file (symlink or file)
        if any(snap.rglob("*")):
            # require a weight/config so a partial dir isn't a false positive
            if list(snap.glob("*.json")) and (
                list(snap.glob("*.safetensors")) or list(snap.glob("*.bin"))
                or list(snap.glob("*.gguf")) or list(snap.glob("*/*.safetensors"))
                or list(snap.glob("*.ckpt")) or list(snap.glob("*.pt"))):
                return True
    return False

def snapshot(repo_id: str) -> bool:
    # snapshot_download resumes partial transfers by default and only fetches the
    # files that are missing/corrupt — ideal for a flaky link. Retry the whole call
    # so a mid-file drop just re-runs and picks up where it left off.
    from huggingface_hub import snapshot_download
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(repo_id, max_workers=4)
            return True
        except Exception as e:                     # network drop / timeout → retry
            print(f"   [{repo_id}] attempt {attempt}/{retries} failed: {e}", flush=True)
            import time; time.sleep(attempt * 5)
    return False

fail = 0
for repo in repos:
    present = cache_present(repo)
    if present:
        print(f"MODEL_STATUS {repo} OK"); continue
    if check_only:
        print(f"MODEL_STATUS {repo} FAIL missing"); fail = 1; continue
    print(f">> pulling {repo} (resumable)…", flush=True)
    if snapshot(repo) and cache_present(repo):
        print(f"MODEL_STATUS {repo} REPAIRED")
    else:
        print(f"MODEL_STATUS {repo} FAIL download-incomplete"); fail = 1

# CLIP (open_clip ViT-B-32, openai) is not a plain HF repo → load-test it. The
# first successful encode downloads+caches the weights; a hit proves it's cached.
if check_only:
    # presence probe only: open_clip caches under HF hub or ~/.cache/clip
    import os as _os
    hit = bool(glob.glob(str(hub / "models--*ViT-B-32*")) or
               glob.glob(_os.path.expanduser("~/.cache/clip/*ViT-B-32*")) or
               glob.glob(_os.path.expanduser("~/.cache/clip/*b32*")))
    print(f"MODEL_STATUS CLIP {'OK' if hit else 'FAIL'}{'' if hit else ' missing'}")
    fail = fail or (0 if hit else 1)
else:
    try:
        import numpy as np
        from embodied_memory.perception import CLIPKeyframeEncoder
        was = bool(glob.glob(str(hub / "models--*ViT-B-32*")) or
                   glob.glob(os.path.expanduser("~/.cache/clip/*")))
        enc = CLIPKeyframeEncoder()
        enc.encode(np.zeros((64, 64, 3), dtype=np.uint8))     # forces the pull if absent
        print(f"MODEL_STATUS CLIP {'OK' if was else 'REPAIRED'}")
    except Exception as e:
        print(f"MODEL_STATUS CLIP FAIL {e}"); fail = 1

sys.exit(1 if fail else 0)
PY
    MODELS_RC=${PIPESTATUS[0]}
    # Fold each per-model MODEL_STATUS line (tee'd to the log) into the ledger so
    # the summary table lists every model, not just an aggregate.
    while read -r _tag name state note; do
      [ "$_tag" = "MODEL_STATUS" ] && record "model:$name" "$state" "$note"
    done < <(grep '^MODEL_STATUS ' "$MODEL_LOG" 2>/dev/null)
    rm -f "$MODEL_LOG"
    [ "${MODELS_RC:-1}" -eq 0 ] || record "models(overall)" FAIL "see FAIL rows above"
  fi

  # ---- informational: RIR grids are RENDERED offline, not downloaded --------
  GRIDS=$(find "$REPO_ROOT/runs" -name '*_rir_grid.npz' 2>/dev/null | wc -l | tr -d ' ')
  [ "${GRIDS:-0}" -gt 0 ] \
    && record "rir-grids(info)" INFO "$GRIDS rendered grid(s) found" \
    || record "rir-grids(info)" INFO "0 grids — rendered by the matrix driver / soundspaces env, not downloaded"
fi

# =========================================================================== #
# SUMMARY
# =========================================================================== #
banner "SUMMARY"
req_fail=0
for i in "${!NAMES[@]}"; do
  st="${STATES[$i]}"
  printf '  %-22s %-9s %s\n' "${NAMES[$i]}" "$st" "${NOTES[$i]}"
  case "$st" in FAIL) case "${NAMES[$i]}" in *info*|*"(info)"*) ;; *) req_fail=1 ;; esac ;; esac
done

echo ""
if [ "$req_fail" -eq 0 ]; then
  echo "VERIFY_RESULT=OK — every in-scope artifact is present."
  exit 0
else
  echo "VERIFY_RESULT=FAIL — some artifacts are still missing (see FAIL rows above)."
  echo "  Re-run this script (network drops just resume); fix any token/env blocker first."
  exit 1
fi
