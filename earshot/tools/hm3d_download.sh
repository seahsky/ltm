#!/bin/bash
# earshot/tools/hm3d_download.sh — HM3D meshes, with the two footguns already disarmed.
#
#   bash earshot/tools/hm3d_download.sh --list
#   nrun bash earshot/tools/hm3d_download.sh --uids hm3d_train_full
#
# WHY THIS EXISTS AND IS NOT A ONE-LINER IN THE RUNBOOK. The documented invocation
#
#     python -m habitat_sim.utils.datasets_download --help
#
# does not print help. It dies:
#
#     free(): invalid pointer
#     Aborted (core dumped)
#
# That is `test_habitat_import_order.py`'s failure, exactly: `import torch` must precede
# `import habitat_sim`, in the same process, every time. Measured on this box 2026-08-05
# by `import_order_ladder.sh` -- the bare import is RED with `free(): invalid pointer`,
# exit 134, and torch-first is the only GREEN case. It is an ABORT, not an exception:
# nothing is raised, nothing can catch it, and there is no Python-level diagnostic at all.
#
# Every run through `earshot/__main__.py` survives this by accident, because `assert_env()`
# imports torch two probes before anything reaches `sim/world.py`. `-m` on habitat-sim's
# own module has no such luck: it imports `habitat_sim` bare and aborts before parsing a
# single flag. So this script does the import in the required order and hands the rest of
# the command line straight through.
#
# THE SECOND FOOTGUN, from the runbook: `--no-replace` is not optional under `nrun`.
# Without it the downloader asks "Replace versioned data?" on stdin, and a detached
# process has none -- `OSError [Errno 9] Bad file descriptor`, about 21 seconds in. It is
# always passed here rather than left to be remembered at 2 a.m.
#
# Credentials come from `.env` at the repo root (MATTERPORT_TOKEN_ID /
# MATTERPORT_TOKEN_SECRET, see .env.example) or from the environment, which is where the
# Matterport academic agreement's token pair lives. `--list` and `--help` need none, so
# they work before the agreement is signed.
#
# SIZE, BEFORE YOU START ONE: `val` is 100 `.basis.glb` at 9.3 G, so about 93 MB a scene.
# A full train download is 800 raw HM3D scenes on that ratio, roughly 74 G, to use the 80
# that ObjectNav v1 publishes episodes for. Check `df -h .` first; the box had 680 G free.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

DATA_PATH="data/hm3d"
PASSTHROUGH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --data-path)
      [ $# -ge 2 ] || { echo "FATAL: --data-path needs a value"; exit 2; }
      DATA_PATH="$2"; shift 2 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

# `.env` is read, never required: `--list` and `--help` are useful without a token, and
# refusing them for want of one would make this script harder to use than the broken
# command it replaces.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

_needs_credentials=1
for arg in ${PASSTHROUGH+"${PASSTHROUGH[@]}"}; do
  case "$arg" in
    --list|--help|-h) _needs_credentials=0 ;;
  esac
done

CREDENTIALS=()
if [ "$_needs_credentials" = 1 ]; then
  if [ -z "${MATTERPORT_TOKEN_ID:-}" ] || [ -z "${MATTERPORT_TOKEN_SECRET:-}" ]; then
    echo "FATAL: MATTERPORT_TOKEN_ID / MATTERPORT_TOKEN_SECRET are not set."
    echo "       They come from the Matterport academic agreement:"
    echo "       https://matterport.com/habitat-matterport-3d-research-dataset"
    echo "       Put them in $REPO_ROOT/.env (see .env.example), or export them."
    echo "       --list and --help need neither and work now."
    exit 2
  fi
  CREDENTIALS=(--username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET")
fi

# THE ORDER IS THE WHOLE POINT. `import torch` first, in this same process, then run
# habitat-sim's downloader as `__main__` so its own argparse and entry point are the ones
# that execute -- this wraps the shipped tool, it does not reimplement it.
python - "${CREDENTIALS[@]+"${CREDENTIALS[@]}"}" \
         ${PASSTHROUGH+"${PASSTHROUGH[@]}"} \
         --no-replace --data-path "$DATA_PATH" <<'PYTHON'
import runpy
import sys

import torch  # noqa: F401  MUST precede habitat_sim -- see this script's header

sys.argv[0] = "datasets_download"
runpy.run_module("habitat_sim.utils.datasets_download", run_name="__main__")
PYTHON
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo ""
  echo "  download exited $STATUS."
  if [ "$STATUS" = 134 ]; then
    echo "  134 is the abort this script exists to prevent — if you see it here, the"
    echo "  torch-first import is no longer sufficient on this env and tests/box owns it."
  fi
fi
exit "$STATUS"
