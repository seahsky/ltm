# 05 — RACE box inventory

Type: task
Status: open
Blocked by: none

## Question

What is actually on the RACE V100 box right now, and what of it is worth keeping through the reset?

## Why it matters

The reset is being executed, not just planned, and the box is now the only execution environment.
Deleting or rebuilding blind is how hours get lost re-downloading 1.2 GB of HM3D or rebuilding habitat-sim for no reason.

It also feeds ticket 04 directly: if an audio-capable habitat-sim build already exists on the box and is sound, that build step is minutes rather than an hour.

## What would resolve it

Record, concretely:
- conda envs present, their Python and numpy versions, and which of them import habitat_sim
- whether any existing habitat-sim build is audio-capable, and on which branch/commit
- disk free, and what the large consumers are
- HM3D copies: where, which splits, whether semantics are present, whether meshes exist for all 20 val scenes (a prior run found only 2 of 20 had meshes)
- model weights already downloaded, and their sizes
- GPU: driver, CUDA version, VRAM free
- CPU core count, since `threadCount` is a free speed knob currently set to 1
- the `soundspaces-spike` build dir state

Deliverable: a short inventory file in the new tree, plus a keep/rebuild/delete call on each item.
