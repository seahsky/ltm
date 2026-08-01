# habitat-sim local patches

Every `*.patch` here is applied, in filename order, to a clean checkout of
`facebookresearch/habitat-sim@RLRAudioPropagationUpdate` before the `--audio`
build runs (`oneenv_gate.sh` step 4).

The gate resets the checkout to `origin/RLRAudioPropagationUpdate` first, applies
each patch with `git apply --check` before `git apply`, and records the applied
list to `$SS2_BUILD_ROOT/applied-patches.txt`.
A patch that does not apply cleanly is fatal — a silently half-patched build is
worse than no build.

Empty is a valid state: with no patches the gate builds the stock branch and says
so.

## Why this directory exists before any patch does

Ticket 02 found habitat-sim hardcodes a **single** audio source (`RLRA_AddSource`
called once, every accessor pinned to index 0) while the engine underneath is
natively multi-source with one IR per source.
Exposing that is roughly 40 lines across `src/esp/sensor/AudioSensor.{h,cpp}` and
`src/esp/bindings/SensorBindings.cpp`, all of which this build already compiles.

Whether we take that patch is **ticket 09's** call, gated on **ticket 06's**
source-count cost sweep.
What ticket 04 owes is a build that can accept a patch and record which ones were
applied, so the box state is reproducible — retrofitting patch support after the
fact is how a build stops being reproducible.

Note the open question this feeds: the map's **Not yet specified** section asks how
far the clean room is willing to fork habitat-sim.
Each patch added here moves the tree closer to owning a fork, with a maintenance
cost and a reproducibility story attached.
Add patches deliberately.
