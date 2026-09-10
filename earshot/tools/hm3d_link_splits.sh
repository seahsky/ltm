#!/bin/bash
# earshot/tools/hm3d_link_splits.sh — one scene_datasets/hm3d that serves every split.
#
#   bash earshot/tools/hm3d_link_splits.sh --check      # report only, nonzero if broken
#   bash earshot/tools/hm3d_link_splits.sh              # (re)build the layout
#   bash earshot/tools/hm3d_link_splits.sh --pin val=hm3d-0.2
#
# THE THIRD SYMLINK TRAP, and the first one a download can spring on its own.
#
# habitat-sim's downloader points `scene_datasets/hm3d` at the versioned tree it just
# filled. One symlink, one tree. The moment two splits live in two versions, the last
# download silently takes the whole path with it:
#
#     scene_datasets/hm3d -> versioned_data/hm3d-1.0/hm3d     # holds ONLY train
#     versioned_data/hm3d-0.2/hm3d                            # holds val, now unreachable
#
# MEASURED 2026-09-10, immediately after `--uids hm3d_train_habitat`: train resolved 73
# usable scenes and val reported `no scene with a mesh on this box` for all 20. That
# breaks the box gate (nine test files), `clap_gate.sh`, `ablation_sweep.sh` and
# `window_pilot.sh`, all of which read val, plus every default `RunConfig.split`.
#
# The fix is that `scene_datasets/hm3d` must be a real DIRECTORY of per-split symlinks
# rather than one symlink to a tree. Episodes hardcode `hm3d/<split>/<scene>/...`, so the
# split is the level the indirection belongs at.
#
# WHY NOT JUST DOWNLOAD val INTO 1.0. Because that silently re-bases a baseline. `abl-2`
# (`full` 35.8%, the baseline of record) and `matrix-1` were rendered against the 0.2
# meshes. Whether 1.0's val meshes are byte-identical is not known here, and "probably the
# same" is not a thing to assume about the substrate under a published number. Each split
# keeps the version it was measured on unless someone says otherwise, and `--pin` is how
# they say it.
#
# DEFAULT: each split comes from the NEWEST versioned tree that has it, and the choice is
# printed for every split. Today that is train from 1.0 and val from 0.2, which is exactly
# right and needs no pin. Pin a split when a version has to be held against a later
# download: `--pin val=hm3d-0.2`.
#
# IDEMPOTENT, and safe to re-run after any download -- which is the point, because the
# next `datasets_download` will move the symlink again.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

DATA_PATH="data/hm3d"
CHECK_ONLY=0
PINS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --data-path) [ $# -ge 2 ] || { echo "FATAL: --data-path needs a value"; exit 2; }
                 DATA_PATH="$2"; shift 2 ;;
    --pin)       [ $# -ge 2 ] || { echo "FATAL: --pin needs split=version"; exit 2; }
                 PINS="$PINS $2"; shift 2 ;;
    --check)     CHECK_ONLY=1; shift ;;
    -h|--help)   sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done

VERSIONED="$DATA_PATH/versioned_data"
TARGET="$DATA_PATH/scene_datasets/hm3d"

[ -d "$VERSIONED" ] || { echo "FATAL: no $VERSIONED — nothing has been downloaded"; exit 1; }

pinned_version_for() {
  for pin in $PINS; do
    case "$pin" in
      "$1"=*) echo "${pin#*=}"; return 0 ;;
    esac
  done
  return 1
}

# Every (split, version) pair on disk, as lines. Plain text rather than an associative
# array on purpose: macOS ships bash 3.2, which has none, and a tool whose whole job is
# surgery on a live data tree should be exercisable wherever the tests run rather than
# only on the box it edits.
#
# `sort -V` puts the newest tree last and `chosen_version_for` takes the LAST match, which
# is the documented "newest that has it" rule, in one place.
#
# A SPLIT IS A DIRECTORY OF SCENES, not any directory. `resolve_scene_path` builds
# `<scenes_dir>/hm3d/<split>/<NNNNN-SCENE>/<SCENE>.basis.glb`, so a candidate is a split
# exactly when it holds at least one `NNNNN-` scene directory.
#
# Measured the hard way: the first version of this linked every child, and the box's own
# `hm3d-0.2/hm3d/` holds `datasets/`, `scene_datasets/` and `versioned_data/` left by an
# earlier copy. It duly created `scene_datasets/hm3d/versioned_data -> .../hm3d/
# versioned_data` and a `scene_datasets/hm3d/scene_datasets`, which the loader ignores --
# it asks for splits by name -- and which anything walking the tree does not.
PAIRS=""
SKIPPED=""
for tree in $(ls -1 "$VERSIONED" 2>/dev/null | sort -V); do
  [ -d "$VERSIONED/$tree/hm3d" ] || continue
  for split_dir in "$VERSIONED/$tree/hm3d"/*/; do
    [ -d "$split_dir" ] || continue
    split="$(basename "$split_dir")"
    if ls -1d "$split_dir"[0-9][0-9][0-9][0-9][0-9]-*/ >/dev/null 2>&1; then
      PAIRS="$PAIRS
$split $tree"
    else
      SKIPPED="$SKIPPED
$split $tree"
    fi
  done
done

SPLITS="$(printf '%s\n' "$PAIRS" | awk 'NF{print $1}' | sort -u)"
chosen_version_for() {
  printf '%s\n' "$PAIRS" | awk -v s="$1" '$1==s{v=$2} END{if (v != "") print v}'
}
offered_versions_for() {
  printf '%s\n' "$PAIRS" | awk -v s="$1" '$1==s{printf "%s ", $2}'
}

version_for() {
  if pin="$(pinned_version_for "$1")"; then
    if [ -d "$VERSIONED/$pin/hm3d/$1" ]; then
      echo "$pin"
      return 0
    fi
    echo "FATAL: --pin $1=$pin, but $VERSIONED/$pin/hm3d/$1 does not exist" >&2
    return 2
  fi
  chosen_version_for "$1"
}

[ -n "$SPLITS" ] || {
  echo "FATAL: no <version>/hm3d/<split>/ directory under $VERSIONED"
  exit 1
}

echo "splits found under $VERSIONED:"
for split in $SPLITS; do
  version="$(version_for "$split")" || { echo "$version"; exit 2; }
  note=""
  if pinned_version_for "$split" >/dev/null; then
    note="  (PINNED)"
  elif [ "$(offered_versions_for "$split" | wc -w)" -gt 1 ]; then
    note="  (newest of: $(offered_versions_for "$split"))"
  fi
  echo "  $split <- $version$note"
done
if [ -n "$(printf '%s\n' "$SKIPPED" | awk 'NF')" ]; then
  echo "not splits (no NNNNN- scene directory inside), left alone:"
  printf '%s\n' "$SKIPPED" | awk 'NF{print "  " $1 "  (" $2 ")"}' | sort -u
fi

if [ "$CHECK_ONLY" = 1 ]; then
  status=0
  echo ""
  echo "checking $TARGET resolves every split:"
  for split in $SPLITS; do
    if [ -d "$TARGET/$split" ]; then
      n="$(ls -1 "$TARGET/$split" 2>/dev/null | wc -l | tr -d ' ')"
      echo "  OK   $TARGET/$split ($n entries)"
    else
      echo "  DEAD $TARGET/$split — a split on disk that nothing can reach"
      status=1
    fi
  done
  if [ "$status" != 0 ]; then
    echo ""
    echo "  re-run this script without --check to rebuild the layout."
  fi
  exit "$status"
fi

# Replace the downloader's single symlink with a real directory of per-split ones. `rm` is
# safe here and only here: a SYMLINK is removed, never a tree. A real directory is kept and
# its entries are refreshed, so nothing anyone put beside the splits is destroyed.
if [ -L "$TARGET" ]; then
  echo ""
  echo "replacing the single symlink $TARGET -> $(readlink "$TARGET")"
  rm "$TARGET"
fi
mkdir -p "$TARGET"

# Clear links this script would no longer make. Only symlinks INTO ../../versioned_data/
# are touched: those are ours to own, and anything else someone put here is theirs. This
# is what un-does the stray `versioned_data` and `scene_datasets` links the first version
# created before it knew what a split was -- a fix that only stopped making them would
# have left the box exactly as it found it.
for existing in "$TARGET"/*; do
  [ -L "$existing" ] || continue
  name="$(basename "$existing")"
  case "$(readlink "$existing")" in
    ../../versioned_data/*) ;;
    *) continue ;;
  esac
  [ "$name" = "hm3d_basis.scene_dataset_config.json" ] && continue
  if ! printf '%s\n' $SPLITS | grep -qx "$name"; then
    echo "  removing $existing — not a split"
    rm "$existing"
  fi
done

for split in $SPLITS; do
  version="$(version_for "$split")" || { echo "$version"; exit 2; }
  link="$TARGET/$split"
  [ -L "$link" ] && rm "$link"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    echo "  SKIP $split — $link is a real directory, not a symlink. Move it aside first."
    continue
  fi
  # RELATIVE, from inside the directory: the runbook's second trap is an ABSOLUTE symlink
  # that an rsync copies and that then resolves nowhere on the other machine.
  ln -sfn "../../versioned_data/$version/hm3d/$split" "$link"
  echo "  $link -> ../../versioned_data/$version/hm3d/$split"
done

# The scene-dataset config, if any tree ships one. The clean room needs no config of any
# kind (`resolve_scene_path`'s docstring has the argument), so this is carried for anything
# else that looks for it rather than because this tree reads it.
for tree in $(ls -1 "$VERSIONED" 2>/dev/null | sort -V); do
  config="$VERSIONED/$tree/hm3d/hm3d_basis.scene_dataset_config.json"
  if [ -f "$config" ]; then
    ln -sfn "../../versioned_data/$tree/hm3d/hm3d_basis.scene_dataset_config.json" \
      "$TARGET/hm3d_basis.scene_dataset_config.json"
  fi
done

echo ""
echo "done. Verify with the real loader, which is the only check that counts:"
echo "  python -m earshot.tools.anchor_yield --split train --classes \"toilet_flush snoring keyboard_typing\""
echo "  python -m earshot.tools.anchor_yield --split val   --classes \"toilet_flush snoring keyboard_typing\""
