#!/usr/bin/env bash
# scripts/setup-anomaly-vm.sh — FRESH-GPU-VM bootstrap for the ANOMALY-RESPONSE arc.
#
# The anomaly-response work lives on `main` (the primary branch) and needs an
# audio layer on top of the base ObjectNav/ReMEmbR stack. This script is a thin
# orchestrator: it reuses `scripts/setup-vm.sh` for the heavy base install
# (Miniconda + the `ltm-embodied` conda env + HM3D + ReMEmbR weights) and then adds
# only what the anomaly-response task needs:
#
#   - the `lifelong-revisit-eval` branch (base setup targets `main`)
#   - the CLAP anomaly-detector weights (laion/clap-htsat-fused, via transformers)
#   - the real ESC-50 anomaly + benign clips (fetch_anomaly_clips.py; gitignored)
#   - optionally the one-time SoundSpaces RIR-render env (--with-render)
#
# Every step is idempotent — it checks for its artifact and skips if present, so
# re-running is safe and cheap. Audio at RUNTIME is pre-rendered RIR convolved in
# O(1) (scipy) — SoundSpaces is NOT needed unless you render NEW grids.
#
# EXECUTE it (do NOT source). On a truly cold VM, clone first:
#
#   git clone https://github.com/seahsky/ltm.git && cd ltm
#   bash scripts/setup-anomaly-vm.sh --yes
#
# FLAGS:
#   --with-render     also build the `soundspaces-spike` env (~1h; ONLY needed to
#                     render new RIR grids — pre-rendered grids don't need it)
#   --skip-data       forward to setup-vm.sh: don't download HM3D
#   --skip-models     forward: don't download ReMEmbR/CLAP weights (CPU/analysis box)
#   --cpu             forward: CPU mode (implies --skip-models)
#   --yes | -y        non-interactive (assume yes to Miniconda install)
#   --env-name <n>    conda env name (default: ltm-embodied, or $LTM_ENV_NAME)
#   --branch <n>      branch to run on (default: main)
#
# Env overrides: LTM_ENV_NAME, REMEMBR_CLAP_MODEL, CONDA_DIR.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${LTM_ENV_NAME:-ltm-embodied}"
BRANCH="main"
CLAP_MODEL="${REMEMBR_CLAP_MODEL:-laion/clap-htsat-fused}"

WITH_RENDER=""
FORWARD=()            # flags passed through to setup-vm.sh
SKIP_MODELS=""
ASSUME_YES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --with-render) WITH_RENDER=1; shift ;;
    --skip-data)   FORWARD+=("--skip-data"); shift ;;
    --skip-models) SKIP_MODELS=1; FORWARD+=("--skip-models"); shift ;;
    --cpu)         SKIP_MODELS=1; FORWARD+=("--cpu"); shift ;;
    --yes|-y)      ASSUME_YES=1; FORWARD+=("--yes"); shift ;;
    --env-name)    ENV_NAME="$2"; FORWARD+=("--env-name" "$2"); shift 2 ;;
    --branch)      BRANCH="$2"; shift 2 ;;
    -h|--help)     sed -n '2,38p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1' (try --help)"; exit 1 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }
die()    { echo "FATAL: $*" >&2; exit 1; }
have()   { command -v "$1" >/dev/null 2>&1; }

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "ERROR: execute this script, don't source it:  bash scripts/setup-anomaly-vm.sh" >&2
  return 1 2>/dev/null || exit 1
fi

GPU=""
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then GPU=1; fi
echo "setup-anomaly-vm: repo=$REPO_ROOT branch=$BRANCH env=$ENV_NAME gpu=${GPU:-none}"
echo "                  clap=$CLAP_MODEL render_env=${WITH_RENDER:+yes}"

# --------------------------------------------------------------------------- #
# [1/6] ensure the anomaly-response branch
# --------------------------------------------------------------------------- #
banner "[1/6] branch: $BRANCH"
if have git && [ -d "$REPO_ROOT/.git" ]; then
  cur="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  if [ "$cur" = "$BRANCH" ]; then
    echo "already on $BRANCH"
  else
    git fetch --all --quiet || echo "WARN: git fetch failed (offline?) — continuing"
    git checkout "$BRANCH" || die "could not checkout '$BRANCH' (commit/stash local changes first)"
  fi
  git pull --ff-only 2>/dev/null || echo "WARN: git pull --ff-only skipped/failed — continuing with local"
else
  echo "WARN: not a git checkout — skipping branch step (assuming code is already at $BRANCH)"
fi

# --------------------------------------------------------------------------- #
# [2/6] base install (delegates to setup-vm.sh — conda, env, HM3D, ReMEmbR weights)
# --------------------------------------------------------------------------- #
banner "[2/6] base env via setup-vm.sh"
# NOTE: no --branch here (step 1 already put us on $BRANCH; setup-vm.sh operates
# in-place). setup-vm.sh is itself idempotent (skips an existing env/data/weights).
bash "$REPO_ROOT/scripts/setup-vm.sh" "${FORWARD[@]}" \
  || die "base setup-vm.sh failed — fix it before the audio layer (its output is above)"

# Load the conda hook + activate so the audio steps below run inside the env.
[ -x "$CONDA_DIR/bin/conda" ] || die "conda missing at $CONDA_DIR after base setup"
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)" || die "could not load conda hook"
conda activate "$ENV_NAME" || die "could not activate env '$ENV_NAME'"
PY="$(command -v python)"
echo "python: $PY ($(python --version 2>&1))"

# --------------------------------------------------------------------------- #
# [3/6] audio runtime deps (CLAP=transformers already in env; verify scipy WAV/O(1))
# --------------------------------------------------------------------------- #
banner "[3/6] audio runtime deps"
# scipy.signal.fftconvolve = the O(1) live convolution; scipy.io.wavfile = clip I/O.
# It is usually pulled transitively, but install it explicitly if the env lacks it.
if python -c "import scipy.signal, scipy.io.wavfile" 2>/dev/null; then
  echo "scipy present ($(python -c 'import scipy; print(scipy.__version__)')) — OK"
else
  echo ">> installing scipy (fftconvolve + wavfile)"
  pip install "scipy" || die "scipy install failed"
fi
# CLAP tower ships with transformers (perception.CLAPAudioEncoder -> ClapModel).
python -c "from transformers import ClapModel, ClapProcessor" 2>/dev/null \
  && echo "transformers CLAP classes import — OK" \
  || die "transformers too old for CLAP — check environment.yml (needs transformers>=4.40)"

# --------------------------------------------------------------------------- #
# [4/6] CLAP anomaly-detector weights (HF cache; ~2 GB)
# --------------------------------------------------------------------------- #
banner "[4/6] CLAP detector weights ($CLAP_MODEL)"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
CACHE_TAG="models--$(echo "$CLAP_MODEL" | tr '/' '-')"
if [ -n "$SKIP_MODELS" ]; then
  echo "--skip-models/--cpu — skipping CLAP weights."
  echo "  Pull later on a GPU box: python -c \"from transformers import ClapModel,ClapProcessor as P; ClapModel.from_pretrained('$CLAP_MODEL'); P.from_pretrained('$CLAP_MODEL')\""
elif ls -d "$HF_HOME_DIR"/hub/"$CACHE_TAG" >/dev/null 2>&1; then
  echo "CLAP weights already cached ($CACHE_TAG) — skipping"
else
  echo ">> downloading CLAP tower to the HF cache (no GPU needed for the pull)"
  python - "$CLAP_MODEL" <<'PY' || echo "WARN: CLAP download failed — re-run step 4 after checking network"
import sys
from transformers import ClapModel, ClapProcessor
m = sys.argv[1]
ClapModel.from_pretrained(m)
ClapProcessor.from_pretrained(m)
print(f"CLAP cached: {m}")
PY
fi

# --------------------------------------------------------------------------- #
# [5/6] ESC-50 anomaly + benign clips (real recordings; gitignored)
# --------------------------------------------------------------------------- #
banner "[5/6] ESC-50 clips"
if ls "$REPO_ROOT"/data/anomaly_audio/*.wav >/dev/null 2>&1 \
   && ls "$REPO_ROOT"/data/benign_audio/*.wav >/dev/null 2>&1; then
  echo "anomaly + benign clips already staged — skipping"
else
  echo ">> fetching ESC-50 anomaly + benign clips from GitHub (no GPU)"
  python "$REPO_ROOT/embodied_memory/scripts/fetch_anomaly_clips.py" --include-benign \
    && echo "clips staged into data/anomaly_audio + data/benign_audio" \
    || echo "WARN: clip fetch failed — re-run embodied_memory/scripts/fetch_anomaly_clips.py --include-benign"
fi

# --------------------------------------------------------------------------- #
# [6/6] OPTIONAL: SoundSpaces RIR-render env (one-time, ~1h) — only to RENDER grids
# --------------------------------------------------------------------------- #
banner "[6/6] SoundSpaces render env"
if [ -z "$WITH_RENDER" ]; then
  echo "skipped (default). The eval runs on PRE-RENDERED grids; the render env is"
  echo "only needed to render NEW RIR grids at a new source. Build it later with:"
  echo "    bash scripts/race-soundspaces-spike.sh          # (or re-run with --with-render)"
elif [ -z "$GPU" ]; then
  echo "WARN: --with-render but no GPU — SoundSpaces rendering needs one. Skipping."
else
  echo ">> building the soundspaces-spike env (isolated; never touches $ENV_NAME)"
  bash "$REPO_ROOT/scripts/race-soundspaces-spike.sh" \
    || echo "WARN: soundspaces spike failed — see its blocker list above"
fi

# --------------------------------------------------------------------------- #
# verify — free (no GPU/data): the anomaly-response pure logic + wiring
# --------------------------------------------------------------------------- #
banner "verify (free anomaly-response suite)"
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$REPO_ROOT"
vfail=0
for t in test_diagnose_energy_gradient test_diagnose_room_gate \
         test_summary_anomaly_report test_anomaly_controller test_anomaly_wiring; do
  if python "$REPO_ROOT/embodied_memory/scripts/$t.py" >/tmp/ltm_$t.log 2>&1; then
    echo "  PASS  $t"
  else
    echo "  FAIL  $t  (see /tmp/ltm_$t.log)"; vfail=1
  fi
done
[ "$vfail" -eq 0 ] && echo "anomaly-response sanity suite PASSED" \
  || echo "WARN: some checks failed — inspect the env before any paid run"

# --------------------------------------------------------------------------- #
banner "DONE"
cat <<EOF
Verify every download survived a flaky link (re-pulls only what's missing):

    nrun bash scripts/verify-setup.sh          # or: bash scripts/verify-setup.sh --check-only

Re-bootstrap after every restart:

    source scripts/race-setup.sh

Cheap wiring smoke (real ReMEmbR, 1 scene, needs GPU + weights):

    nrun bash scripts/race-anomaly-response.sh --scene TEEsavR23oF --class alarm --category bed

Phase-0 \$0 gates (before any paid matrix):
    python embodied_memory/scripts/diagnose_energy_gradient.py --grid runs/<tag>/<grid>.npz   # G0.4 climbability
    python embodied_memory/scripts/diagnose_room_clip_cosines.py --scene all                  # G0.1 room cosines
    nrun bash scripts/race-convolved-anomaly-gate.sh --device cuda                             # G0.2/G0.3 gate calib

Powered anomaly-response matrix (the systems + memory headline):
    nrun bash scripts/race-anomaly-response-matrix.sh --split val_mini --tag-prefix anomvm

Query-fix A/B (validate the encoder-gate's GO-QUERY lever):
    nrun bash scripts/race-anomaly-response-matrix.sh --split val_mini --query-expansion prf --tag-prefix qfix
EOF
[ -n "$GPU" ] || echo "(reminder: no GPU here — the ReMEmbR/matrix runs need one)"
ls "$REPO_ROOT"/data/anomaly_audio/*.wav >/dev/null 2>&1 || echo "(reminder: ESC-50 clips missing — re-run step 5)"
