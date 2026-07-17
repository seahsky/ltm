#!/bin/bash
# scripts/race-r1-preflight.sh — $0 R1 preflight (no GPU spend).
#
# Runs the two pure-scoring test suites added in the 2026-07-17 grilling session
# and answers the one blocking question for R1 / Table 1 (ADR-0005): what success
# RING does the harness score `spl` at, i.e. is the native number already
# comparable to VLFM's SPL 0.304 / VLingNav's 0.429, or is there a 0.1-vs-1.0 m
# ring gap that `spl_1m` must close before the full-val run?
#
#   * success_distance == 1.0  -> native `spl` IS benchmark SPL@1.0 m; the 0.1 m
#                                 in episode_runner.py:2423 is only the OR-fallback,
#                                 not the eval ring. spl_1m becomes a free cross-check.
#   * success_distance == 0.1  -> ring gap confirmed; wire spl_1m (STOP-gated,
#                                 geodesic-weighted) before headlining Table 1.
#
# EXECUTE it (do NOT source):
#   bash scripts/race-r1-preflight.sh
#
# git pull runs first, but a change to THIS script only takes effect on the 2nd
# invocation (the running copy is already in memory).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
MINICONDA="${HOME}/miniconda3"; LTM_ENV="ltm-embodied"

banner() { printf '\n========== %s ==========\n' "$1"; }

banner "[1/4] git pull --ff-only"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
if git rev-parse --git-dir >/dev/null 2>&1; then
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/4] conda setup (source scripts/race-setup.sh -> $LTM_ENV)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/4] pure-scoring tests (free; must pass)"
for t in test_metrics test_pass_conditions; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed."; exit 1; }
done

banner "[4/4] RING VERIFICATION (D3 / ADR-0005): what ring does the harness score at?"
python - <<'PY'
from habitat.config.default import get_config
c = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
succ = c.habitat.task.measurements.success
sd = getattr(succ, "success_distance", None)
print("success measure     :", succ)
print("success_distance (m):", sd if sd is not None else "NOT FOUND")
print("spl measure present :", c.habitat.task.measurements.get("spl", None) is not None)
print()
if sd is None:
    print("VERDICT: could not read success_distance — inspect the measure above by hand.")
elif abs(float(sd) - 1.0) < 1e-6:
    print("VERDICT: ring == 1.0 m. Native `spl` IS benchmark SPL@1.0 m (VLFM-comparable).")
    print("         spl_1m is a free cross-check; R1 can headline native `spl`.")
elif abs(float(sd) - 0.1) < 1e-6:
    print("VERDICT: ring == 0.1 m. Ring gap CONFIRMED — wire spl_1m (STOP-gated,")
    print("         geodesic-weighted) before headlining Table 1 (buildplan staged step 2).")
else:
    print(f"VERDICT: ring == {sd} m (neither 0.1 nor 1.0). Match this ring to VLFM's before quoting.")
PY

echo
echo "DONE (\$0). Paste the VERDICT line to decide the spl_1m wiring."
