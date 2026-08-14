#!/bin/bash
# earshot/tools/arrival_audit.sh — pull, then read the arrival criterion off finished runs.
#
#   bash earshot/tools/arrival_audit.sh                     # cast-1 eps-1 yield-2
#   bash earshot/tools/arrival_audit.sh --tags "cast-1"     # just one
#
# READ-ONLY. It renders nothing, needs no GPU and writes nothing under `runs/`; it reads
# audit records a sweep already wrote. Minutes, not hours.
#
# Two counts, both of which the funnel hides:
#
#   REFUSED ARRIVALS — an episode that stood inside the 1.0 m ring and was scored as
#   never arriving. `visual_confirm` is a pure function of distance, so a recorded
#   distance under the ring means the confirm fired. On a run written BEFORE `a0f4625`
#   the rule STOPped on confirm-and-not-rising, so an abandoned in-ring episode proves
#   `rising` was true at every in-ring step: cast-1's DYehNKdT76V held seven, and the
#   sweep held 23. SINCE `a0f4625` the rule STOPs on the confirm alone, so this count is
#   zero by construction on a routed episode — arrive-2, repeat-1 and both r2500 arms
#   read 0 of 365, which is the mechanism check rather than an absence of evidence.
#
#   UNROUTED SOURCES — an episode whose geodesic to the source was None at EVERY step,
#   meaning `find_path` failed throughout. `runner._make_detector` seeds the anomaly
#   object's view-point list with the source position, so the detector asks that same
#   pathfinder the same question and gets the same None, which reads as NOT detected.
#   Those episodes cannot reach SOURCE_REACHED under any controller and they are sitting
#   in the headline's denominator.
#
# CONTINUE-ON-FAILURE, and continuing is not passing: a tag that is missing or holds no
# records is reported and the exit code is NONZERO. A run report that could not be
# produced must never read as a clean one (CLAUDE.md).
#
# Flags: --tags "a b c" (default "cast-1 eps-1 yield-2"), --no-pull.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

TAGS="cast-1 eps-1 yield-2"
NO_PULL=0

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --tags)    need_value $# "$1"; TAGS="$2"; shift 2 ;;
    --no-pull) NO_PULL=1;                     shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- self-update by re-exec (bash runs the body it loaded, not the file) ---
# The same footgun `yield_sweep.sh` carries: a pull that changes this file does not change
# the body already in memory, so a fix would silently need a second invocation.
if [ "$NO_PULL" = 0 ]; then
  banner "git pull --ff-only"
  _self_before="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
  _self_after="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  if [ -n "$_self_before" ] && [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
    echo "  this script changed in the pull — re-execing the new body"
    export _REEXEC=1
    exec bash "$0" --tags "$TAGS" --no-pull
  fi
fi

FAILED=""
for tag in $TAGS; do
  banner "runs/$tag"
  if [ ! -d "runs/$tag" ]; then
    echo "MISSING: runs/$tag is not a directory on this box — nothing to read."
    FAILED="$FAILED $tag"
    continue
  fi
  python -m earshot.tools.detour_report "runs/$tag" --across-scenes
  status=$?
  # 2 is the tool's "nothing to count" — an empty tag directory, or one holding no
  # episode records. Reported as a failure of THIS audit, not of the sweep it read.
  [ "$status" -eq 0 ] || FAILED="$FAILED $tag"
done

banner "result"
if [ -n "$FAILED" ]; then
  echo "INCOMPLETE — no count for:$FAILED"
  echo "A tag that could not be read is not a tag with nothing to find."
  exit 1
fi
echo "counted every tag:$(printf ' %s' $TAGS)"
