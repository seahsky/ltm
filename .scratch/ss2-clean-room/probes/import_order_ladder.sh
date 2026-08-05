#!/usr/bin/env bash
# What makes `import habitat_sim` survive on this box.
#
# `audio_registration_probe.py` aborts the interpreter with `free(): invalid pointer`
# at its `import habitat_sim`, printing nothing further. `python -m earshot` imports the
# same module in the same env and does not. So the crash is not habitat-sim being
# broken; it is something the package's entry path does first that a bare script does
# not, and there are exactly two candidates:
#
#   - `earshot/__init__` runs `pin_habitat_logging()`, setting HABITAT_SIM_LOG before
#     the import. `audio/guard.py:118` reasons from source that an untouched env already
#     defaults to Verbose, "so on an untouched env this pin changes nothing". If case 1
#     is red and case 2 green, that sentence is wrong on this fork and the pin is load-
#     bearing for the import itself.
#   - `assert_env()` imports torch and runs a CUDA op before habitat-sim is ever
#     touched — visible in the run's probe order (numpy, torch, torch_cuda_allocation,
#     habitat_sim). A `free(): invalid pointer` is a textbook symptom of two libraries
#     disagreeing about an allocator, which is what an import-order constraint looks
#     like. If case 4 is the first green, the tree satisfies that constraint only by
#     accident of env_check running first, and any entry point that skips it aborts.
#
# numpy-first is common to BOTH paths (`audio/sensor.py` is imported before
# `sim/world.py` inside `runner.run`), so case 3 is the control that rules it out
# rather than a suspect.
#
# No `set -e`: every case here is allowed to abort, and the abort IS the measurement.
#
#     bash .scratch/ss2-clean-room/probes/import_order_ladder.sh

echo "python      : $(command -v python)"
echo "version     : $(python -c 'import sys; print(sys.version.split()[0])' 2>&1)"
echo "shell env   : HABITAT_SIM_LOG=${HABITAT_SIM_LOG-<unset>}  MAGNUM_LOG=${MAGNUM_LOG-<unset>}"
echo "PYTHONPATH  : ${PYTHONPATH-<unset>}"
echo

green=""

run_case() {
    local label="$1" envs="$2" code="$3"
    local out status

    # $envs is deliberately unquoted so `env` receives it as separate assignments.
    # Every value passed below is space-free; keep it that way.
    out="$(env $envs python -c "$code" 2>&1)"
    status=$?

    if [ "$status" -eq 0 ]; then
        printf '%-14s GREEN  (exit 0)\n' "$label"
        green="$green $label"
    else
        printf '%-14s RED    (exit %s)\n' "$label" "$status"
        printf '%s\n' "$out" | tail -3 | sed 's/^/                 | /'
    fi
}

run_case "0 interpreter" "" "print('OK')"
run_case "1 bare"        "" "import habitat_sim; print('OK')"
run_case "2 pinned"      "HABITAT_SIM_LOG=Sensor,Assets=Debug" "import habitat_sim; print('OK')"
run_case "3 numpy"       "" "import numpy, habitat_sim; print('OK')"
run_case "4 torch"       "" "import torch, habitat_sim; print('OK')"
run_case "5 earshot"     "" "import earshot, habitat_sim; print('OK')"

echo
echo "green:${green:- none}"
echo
# The reading is stated as a rule rather than computed into a single verdict: cases are
# not mutually exclusive (the pin and the torch load could BOTH be required), and a
# script that collapsed that into one line would hide the combination.
echo "reading:"
echo "  1 red, 2 green            -> the HABITAT_SIM_LOG pin is required to IMPORT, not"
echo "                               only to keep guard.py's invariant 2 armed."
echo "  1 and 2 red, 4 green      -> torch must be imported first; the tree only gets"
echo "                               away with it because assert_env() runs first."
echo "  1-4 red, 5 green          -> something else in earshot/__init__'s chain; bisect."
echo "  1 green                   -> the crash is inside the probe, not the import."
