#!/usr/bin/env bash
# scripts/setup-vm.sh — FIRST-TIME, from-scratch environment bootstrap for a
# fresh VM (RACE pod, cloud GPU box, or a clean laptop).
#
# This is the one-time installer. Its sibling `scripts/race-setup.sh` is only a
# per-session re-bootstrap (it assumes ~/miniconda3 + the conda env already
# exist). Run THIS first on a brand-new machine, then `source race-setup.sh`
# every session/restart after.
#
# What it does (each step is idempotent — safe to re-run):
#   [1/8] system prerequisites      (git, curl, unzip, build tools — best effort)
#   [2/8] Miniconda / Miniforge     (installs into ~/miniconda3 if missing)
#   [3/8] conda env                 (create ltm-embodied from environment.yml)
#   [4/8] .env scaffold             (copy from .env.example if absent)
#   [5/8] persist REMEMBR env vars  (guarded block appended to ~/.bashrc)
#   [6/8] HM3D dataset              (download_hm3d.sh — needs Matterport token)
#   [7/8] ReMEmbR model weights     (Qwen pair — needs a GPU; auto-skipped on CPU)
#   [8/8] verify                    (free sanity suite, no GPU/data required)
#
# EXECUTE it (do NOT source) — it activates conda in its own process:
#
#   # On a truly cold VM, clone first (chicken-and-egg: the script lives in-repo):
#   git clone https://github.com/seahsky/ltm.git && cd ltm
#   bash scripts/setup-vm.sh                   # main is the single source of truth
#
# Flags:
#   --branch <name>   git checkout <name> before installing (default: leave as-is)
#   --env-name <name> conda env name (default: ltm-embodied; or $LTM_ENV_NAME)
#   --skip-data       don't download HM3D (do it later with download_hm3d.sh)
#   --skip-models     don't download the 7B/2B weights (e.g. CPU/analysis box)
#   --cpu             force CPU mode (skips model download regardless of GPU)
#   --yes             non-interactive; assume "yes" to the Miniconda install
#
# Env overrides honored: LTM_ENV_NAME, REMEMBR_CAPTIONER_MODEL,
# REMEMBR_PLANNER_MODEL, HM3D_SCENE_GROUP, CONDA_DIR.

set -uo pipefail

# --------------------------------------------------------------------------- #
# config + arg parse
# --------------------------------------------------------------------------- #
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${LTM_ENV_NAME:-ltm-embodied}"
ENV_YML="$REPO_ROOT/embodied_memory/environment.yml"

# Validated lightweight pair (fits one 24 GB GPU). Matches race-setup.sh +
# CLAUDE.md. Override via env to swap models without editing this file.
CAPTIONER_MODEL="${REMEMBR_CAPTIONER_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PLANNER_MODEL="${REMEMBR_PLANNER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

BRANCH=""
SKIP_DATA=""
SKIP_MODELS=""
FORCE_CPU=""
ASSUME_YES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --branch)    BRANCH="$2"; shift 2 ;;
    --env-name)  ENV_NAME="$2"; shift 2 ;;
    --skip-data)   SKIP_DATA=1; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --cpu)         FORCE_CPU=1; SKIP_MODELS=1; shift ;;
    --yes|-y)      ASSUME_YES=1; shift ;;
    -h|--help)   sed -n '2,46p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1' (try --help)"; exit 1 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }
die()    { echo "FATAL: $*" >&2; exit 1; }
have()   { command -v "$1" >/dev/null 2>&1; }

# Refuse to be sourced — conda activate side effects would leak into the
# caller's shell and the `exit`s would kill their session.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "ERROR: execute this script, don't source it:  bash scripts/setup-vm.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# OS / arch detection (for the Miniforge installer + GPU gating).
OS="$(uname -s)"; ARCH="$(uname -m)"
GPU=""
if [ -z "$FORCE_CPU" ] && have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
  GPU=1
fi

echo "setup-vm: repo=$REPO_ROOT env=$ENV_NAME conda=$CONDA_DIR"
echo "          os=$OS arch=$ARCH gpu=${GPU:-none}"
echo "          captioner=$CAPTIONER_MODEL"
echo "          planner=$PLANNER_MODEL"

# --------------------------------------------------------------------------- #
# optional: checkout requested branch
# --------------------------------------------------------------------------- #
if [ -n "$BRANCH" ]; then
  banner "git checkout $BRANCH"
  git fetch --all --quiet || echo "WARN: git fetch failed (offline?) — continuing"
  git checkout "$BRANCH" || die "could not checkout branch '$BRANCH'"
  git pull --ff-only || echo "WARN: git pull --ff-only failed — continuing with local"
fi

# --------------------------------------------------------------------------- #
# [1/8] system prerequisites (best effort — needs sudo; skipped if unavailable)
# --------------------------------------------------------------------------- #
banner "[1/8] system prerequisites"
NEED_PKGS=()
for c in git curl unzip; do have "$c" || NEED_PKGS+=("$c"); done
if [ "${#NEED_PKGS[@]}" -gt 0 ]; then
  echo "missing: ${NEED_PKGS[*]}"
  if have apt-get && have sudo; then
    sudo apt-get update -qq && \
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        git curl unzip build-essential ca-certificates \
      && echo "installed via apt-get" \
      || echo "WARN: apt-get install failed — install ${NEED_PKGS[*]} manually"
  else
    echo "WARN: no apt-get/sudo — install ${NEED_PKGS[*]} yourself, then re-run"
  fi
else
  echo "git/curl/unzip present — OK"
fi
have git || die "git is required and could not be installed"

# --------------------------------------------------------------------------- #
# [2/8] Miniconda / Miniforge into $CONDA_DIR
# --------------------------------------------------------------------------- #
banner "[2/8] conda ($CONDA_DIR)"
if [ -x "$CONDA_DIR/bin/conda" ]; then
  echo "conda already installed at $CONDA_DIR — skipping"
else
  case "$OS/$ARCH" in
    Linux/x86_64)  MF="Miniforge3-Linux-x86_64.sh" ;;
    Linux/aarch64) MF="Miniforge3-Linux-aarch64.sh" ;;
    Darwin/arm64)  MF="Miniforge3-MacOSX-arm64.sh" ;;
    Darwin/x86_64) MF="Miniforge3-MacOSX-x86_64.sh" ;;
    *) die "unsupported OS/arch '$OS/$ARCH' — install Miniconda to $CONDA_DIR manually" ;;
  esac
  URL="https://github.com/conda-forge/miniforge/releases/latest/download/$MF"
  if [ -z "$ASSUME_YES" ]; then
    read -r -p "Install Miniforge from $URL into $CONDA_DIR? [Y/n] " ans
    case "${ans:-Y}" in [Nn]*) die "declined Miniconda install — install it manually then re-run" ;; esac
  fi
  TMP_SH="$(mktemp -t miniforge.XXXXXX.sh)"
  trap 'rm -f "$TMP_SH"' EXIT
  echo ">> downloading $URL"
  curl -fL --retry 3 -o "$TMP_SH" "$URL" || die "miniforge download failed"
  echo ">> installing (batch) into $CONDA_DIR"
  bash "$TMP_SH" -b -p "$CONDA_DIR" || die "miniforge install failed"
  rm -f "$TMP_SH"; trap - EXIT
  echo "installed."
fi

# Load the conda shell hook so `conda activate` works inside this process.
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)" || die "could not load conda hook"

# --------------------------------------------------------------------------- #
# [3/8] conda env from environment.yml
# --------------------------------------------------------------------------- #
banner "[3/8] conda env: $ENV_NAME"
[ -f "$ENV_YML" ] || die "missing $ENV_YML"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env '$ENV_NAME' already exists — skipping create"
  echo "  (to rebuild: conda env remove -n $ENV_NAME, then re-run)"
else
  echo ">> conda env create -f $ENV_YML  (this is the slow step: habitat-sim + torch)"
  if [ "$ENV_NAME" = "ltm-embodied" ]; then
    conda env create -f "$ENV_YML" || die "conda env create failed"
  else
    conda env create -n "$ENV_NAME" -f "$ENV_YML" || die "conda env create failed"
  fi
fi
conda activate "$ENV_NAME" || die "could not activate env '$ENV_NAME'"
echo "python: $(command -v python)  ($(python --version 2>&1))"

# --------------------------------------------------------------------------- #
# [4/8] .env scaffold
# --------------------------------------------------------------------------- #
banner "[4/8] .env"
if [ -f "$REPO_ROOT/.env" ]; then
  echo ".env already present — leaving it untouched"
else
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  echo "created .env from .env.example — FILL IN before data download:"
  echo "    MATTERPORT_TOKEN_ID / MATTERPORT_TOKEN_SECRET  (HM3D download)"
  echo "    RESEND_API_KEY / NOTIFY_EMAIL_TO               (optional: run emails)"
fi
# Load it so the token is visible to the data download below.
set -a; [ -f "$REPO_ROOT/.env" ] && . "$REPO_ROOT/.env"; set +a

# --------------------------------------------------------------------------- #
# [5/8] persist REMEMBR env vars into ~/.bashrc (guarded, idempotent)
# --------------------------------------------------------------------------- #
banner "[5/8] persist REMEMBR env vars (~/.bashrc)"
MARK="# >>> ltm-embodied REMEMBR vars >>>"
if grep -qF "$MARK" "$HOME/.bashrc" 2>/dev/null; then
  echo "~/.bashrc block already present — skipping"
else
  {
    echo ""
    echo "$MARK"
    echo "export REMEMBR_CAPTIONER_MODEL=$CAPTIONER_MODEL"
    echo "export REMEMBR_PLANNER_MODEL=$PLANNER_MODEL"
    echo "# <<< ltm-embodied REMEMBR vars <<<"
  } >> "$HOME/.bashrc"
  echo "appended REMEMBR_{CAPTIONER,PLANNER}_MODEL to ~/.bashrc"
fi

# --------------------------------------------------------------------------- #
# [6/8] HM3D dataset
# --------------------------------------------------------------------------- #
banner "[6/8] HM3D dataset"
if [ -n "$SKIP_DATA" ]; then
  echo "--skip-data — skipping (run later: bash embodied_memory/scripts/download_hm3d.sh data/hm3d)"
elif [ -d "$REPO_ROOT/data/hm3d/scene_datasets/hm3d" ] && \
     [ -e "$REPO_ROOT/data/hm3d/scene_datasets/hm3d" ]; then
  echo "data/hm3d/scene_datasets/hm3d already present — skipping download"
elif [ -z "${MATTERPORT_TOKEN_ID:-}" ] || [ -z "${MATTERPORT_TOKEN_SECRET:-}" ]; then
  echo "WARN: MATTERPORT_TOKEN_ID / MATTERPORT_TOKEN_SECRET not set in .env — skipping."
  echo "      Sign the agreement at https://matterport.com/habitat-matterport-3d-research-dataset,"
  echo "      fill them into .env, then: bash embodied_memory/scripts/download_hm3d.sh data/hm3d"
else
  echo ">> downloading HM3D minival slice + ObjectNav v1 episodes"
  bash "$REPO_ROOT/embodied_memory/scripts/download_hm3d.sh" "$REPO_ROOT/data/hm3d" \
    || echo "WARN: HM3D download failed — re-run download_hm3d.sh after checking the token"
fi

# --------------------------------------------------------------------------- #
# [7/8] ReMEmbR model weights (GPU only)
# --------------------------------------------------------------------------- #
banner "[7/8] ReMEmbR weights"
if [ -n "$SKIP_MODELS" ]; then
  echo "skipping weights (${FORCE_CPU:+--cpu }${SKIP_MODELS:+--skip-models})."
  echo "  Pull later on a GPU box: REMEMBR_CAPTIONER_MODEL=$CAPTIONER_MODEL \\"
  echo "    REMEMBR_PLANNER_MODEL=$PLANNER_MODEL python models/download_remembr_models.py"
elif [ -z "$GPU" ]; then
  echo "no GPU detected (nvidia-smi absent/failing) — skipping the ~10 GB weight pull."
  echo "  Force on a GPU later with: python models/download_remembr_models.py"
else
  echo ">> pulling captioner + planner (~10 GB for the Qwen pair) to the HF cache"
  REMEMBR_CAPTIONER_MODEL="$CAPTIONER_MODEL" \
  REMEMBR_PLANNER_MODEL="$PLANNER_MODEL" \
    python "$REPO_ROOT/models/download_remembr_models.py" \
    || echo "WARN: model download failed — re-run models/download_remembr_models.py"
fi

# --------------------------------------------------------------------------- #
# [8/8] verify — free, no GPU/data required
# --------------------------------------------------------------------------- #
banner "[8/8] verify (free sanity suite)"
python "$REPO_ROOT/embodied_memory/scripts/test_propose_candidates.py" \
  && echo "sanity suite PASSED" \
  || echo "WARN: sanity suite failed — inspect the env before any paid run"

# --------------------------------------------------------------------------- #
# done
# --------------------------------------------------------------------------- #
banner "DONE"
cat <<EOF
Next session (and after every restart), re-bootstrap with:

    source scripts/race-setup.sh

Then a cheap end-to-end smoke (model-free, both scenes):

    bash scripts/race-smoke.sh --backbone oracle --setting 1 \\
        --scenes "TEEsavR23oF wcojb4TFT35" --n-episodes 2 --target any \\
        --no-strict-pass --tag oracle-smoke

Full-stack smoke (real ReMEmbR, needs GPU + weights from step 7):

    bash scripts/race-smoke.sh --backbone remembr --setting 3 \\
        --scenes wcojb4TFT35 --n-episodes 2 --target any --tag remembr-smoke
EOF
[ -n "$GPU" ] || echo "(reminder: no GPU here — the remembr smoke needs one)"
[ -f "$REPO_ROOT/data/hm3d/scene_datasets/hm3d" ] || \
  echo "(reminder: HM3D not downloaded — fill .env token + run download_hm3d.sh)"
