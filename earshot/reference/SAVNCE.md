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
