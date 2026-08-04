# 27 — The hermeticity re-run and the phase-3 deletion commit

Type: task
Status: open
Blocked by: 26

## Question

Run ticket 10's phase 2 gate, then execute its phase 3: the one irreversible commit.

This is the map's destination — the smoke green on the box **and** both old trees deleted.

## Phase 2, the gate

The smoke being green (ticket 26) is **not** sufficient. Move `embodied_memory/` and `dialogue_memory/` out of the repo entirely, re-run the smoke, require green, then move them back.

"The smoke is green" and "the smoke is green *without the old trees present*" are different claims, and only the second licenses an irreversible delete. A static grep was considered and rejected: it misses `importlib`, a `sys.path` append, a hardcoded `"embodied_memory/…"` string in a config, or a data path only the old tree knew about.

**If it fails: fix the leak, restore, repeat. It failing is the gate working.**

Prerequisites both landed: the tag `archive/pre-clean-room-2026-08-01` (phase 0) and `docs/race-box-runbook.md` (ticket 14).

## Phase 3 — one atomic commit, on a branch, PR'd

**Delete:** `embodied_memory/` (163 tracked files), `dialogue_memory/` (30 — 15 `.py` and 15 tracked `.pyc`, one of which is a stale artifact of a module that no longer has a `.py`), `scripts/` wholesale except the three notify files already carried, `README_LTM_MSC_EVAL.md`, `README_MSC_EVAL.md`, the three `run_msc_*.sh`, and `data/msc/*.json`.

**Keep untouched:** `docs/adr/0001`–`0013` (0007, 0008 and 0013 are this map's own output), `PHASE2_ABLATION_REPORT.md`, `ICRA2027_PAPER_DRAFT.md`, `CONTEXT.md`, `Research Proposal_Embodied Agent.md`, `MSC_BENCHMARK_REPORT.md`, `models/README.md`, `docs/` in full.

**Rewrite in the same commit:** `CLAUDE.md` — 580 lines → short (mission + new-tree orientation + how to run the smoke + pointers). It is auto-loaded into every session and currently carries 7 references to the deleted trees, a "Repo orientation" section describing them, and a "Running the ablation" section invoking `python -m embodied_memory.run_hm3d_pol`. **The ~450 lines of outcome narrative are not summarised into it** — that history lives in `PHASE2_ABLATION_REPORT.md`, and duplicating it is how the file reached 580 lines. Also `docs/archive/README.md`, which points at a `race-setup.sh` this commit deletes.

**Why one commit:** it makes the whole reset a single revert.

**Rollback:** `git revert <commit>`, or `git checkout archive/pre-clean-room-2026-08-01 -- embodied_memory dialogue_memory scripts`. Both available indefinitely; the tag is pushed.

## The box-side sweep

**Remove, on trust:** the `soundspaces-spike` conda env — ticket 04 refused to trust it for its verdict, its recipe is deleted in this same reset, and it is one `conda activate` from handing a future session a wrong answer about the exact thing this map hangs off.

**Verify, do not delete:** the suspected ~9.3 GB duplicate in `data/` (`val` vs `versioned_data`). Confirm by inode/`-samefile`, record the answer, keep both regardless. 680 GB free means nothing here is a space call.

**Now decidable, if ticket 21 answered it:** the 9.3 GB of HM3D semantic annotations were kept solely because it was unproven whether `objectnav_hm3d` v1 requires `hm3d_annotated_basis.scene_dataset_config.json`. If ticket 21 measured that it does not, this is where that decision can finally be made — deliberately, and still not as a space call.

**Leave, recorded:** `~/soundspaces-build`, the HF and pip caches, `runs/`, and the `ltm-embodied` env (ticket 13 has closed, so its block is lifted — but it remains the only on-box proof that a modern torch runs on this V100).

## Done when

The PR is merged, `CLAUDE.md` describes a tree that exists, and the map's destination sentence is true.
