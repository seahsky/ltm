# SAVN-CE, vendored as a submodule

`earshot/reference/savnce/` is a git submodule of
[yichenzeng24/SAVN-CE](https://github.com/yichenzeng24/SAVN-CE), pinned at
`9c7a1a3b1eae95fa027ad5e3b73914595b5b44c9`.
It is licensed **CC-BY-4.0** and is used here with attribution:

> Yichen Zeng, Hebaixu Wang, Meng Liu, Yu Zhou, Kehan Chen, Chen Gao, Gongping Huang.
> *Semantic Audio-Visual Navigation in Continuous Environments.* CVPR 2026.
> arXiv:2603.19660.

## Why it is here

It is the closest published work to this project's simulation regime: binaural audio
rendered **live and continuously** as the agent moves, not looked up from a precomputed
RIR grid.
ADR-0015 stages it as a **reproduced reference** (see `CONTEXT.md`): their method,
re-measured by us, on their benchmark, their data, their metrics.
It is never paired with an earshot number.

## How it differs from `reference/memory/`

`reference/memory/` is vendored **inert and deliberately broken** — nothing may import
it, ever.
This one is **inert to earshot and live to its own environment**.
`earshot/reference/__init__.py` raises `ImportError`, so `earshot.reference.savnce` is
unreachable from the earshot package, and the structural walker, the lint config and the
import guard all exclude `reference/` as before.
But the `savnce` conda env installs it as top-level `savnce` / `habitat` /
`savnce_baselines` packages and runs it directly, which is the whole point.

Two trees, two environments, one rule: **nothing in `earshot/` imports anything in
here, and nothing in here knows `earshot/` exists.**

## Operating it

```bash
bash earshot/tools/savnce_bootstrap.sh                 # build the env, prove it renders
bash earshot/tools/savnce_eval.sh --tag smoke1 --episodes 20
python -m earshot.tools.savnce_gate --run-dir runs/savnce-smoke1
```

The licence step for MP3D is `earshot/tools/savnce_licence_wizard.sh`, and it is the one
part of this no agent can do for you.

## The build tree, and why it is not shared

`savnce_bootstrap.sh` builds in `~/savnce-build/habitat-sim`, seeded by a local copy of
ss2's checkout (no network; the submodules are already there).
The first version shared ss2's tree to save a compile, and two box runs on 2026-08-27
showed why that was wrong: a shared `build/` carries the CMake cache, and the CMake
cache carries the **compiler**.
Wiping it to pick up a newly installed EGL header silently swapped the pinned conda
gcc-10 for Ubuntu 24.04's gcc-13, which cannot compile this 2022 tree at all
(`'std::uint32_t' has not been declared`, the GCC-13 signature).

`earshot/tools/habitat_sim_toolchain.sh` now holds the four things that make the build
work: conda gcc-10, the CPATH include shim, the cmake path hints for GLVND, and
**bounded parallelism** (unbounded `-j` has taken this box down before).
`bootstrap_ss2.sh` still has its own inline copy and is deliberately unchanged until the
savnce build proves the shared file green.

## Two footguns already found, before the first box trip

1. **Their docs and their config disagree about the dataset directory.**
   `INSTALLATION.md` draws the tree with `datasets/savnce-dataset/` (hyphen);
   `configs/savnce/magnet/mp3d/savnce_clean.yaml:72` reads
   `data/datasets/savnce_dataset/...` (underscore).
   The config wins. `savnce_bootstrap.sh` reconciles it and says so.
2. **`EVAL.USE_CKPT_CONFIG` defaults to `True`,** so eval config comes from the
   checkpoint. Command-line overrides still win — `_setup_eval_config` in
   `savnce_baselines/common/base_trainer.py` applies `eval_cmd_opts` last — but only
   because of that ordering, which is worth knowing before trusting a flag.
