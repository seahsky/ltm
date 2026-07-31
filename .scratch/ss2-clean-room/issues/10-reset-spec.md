# 10 — The reset spec: what is deleted, what carries, and when

Type: grilling
Status: open
Blocked by: none

## Question

Exactly which files and directories are deleted, exactly what carries into the new tree, and what has to be true before the deletion happens?

## Why it matters

This is the irreversible half of the effort, and it should be an explicit checklist approved before it runs, not a judgement call made at the end of a long session.

Already decided at chart time:
- New top-level package in this repo, built alongside the old tree.
- Old trees deleted only once the smoke is green.
- The STM and LTM calculation carries across as the reference implementation for the follow-on memory effort.

Already established as non-issues:
- `data/`, `runs/` and `models/` are gitignored. Only 8 stray tracked files exist across all three. The reset is a source-tree operation and does not touch 1.2 GB of HM3D or any weights on disk.

Genuinely open:
- **Does "the STM and LTM calculation" mean the consolidation math only** (`I = αR + βU + γN` in `dialogue_memory/consolidation.py`), **or the whole retrieval stack** (`ltm.py`'s three-layer FAISS memory, `memory_bridge.py`'s injection path, `reranking.py`)? The answer sets how much of `dialogue_memory/` survives deletion.
- Does it carry as vendored code in the new tree, or as a documented pointer into git history?
- What happens to `CONTEXT.md`, `docs/adr/0001`–`0006`, `ICRA2027_PAPER_DRAFT.md` and `PHASE2_ABLATION_REPORT.md`. These were not selected when the question was asked, and deleting the ADRs would discard constraints the new tree is actively reasoning against in tickets 07 and 09. Confirm the intent rather than assuming it.
- What happens to `scripts/race-*.sh`. Some encode hard-won box knowledge (the self-update gotcha, the conda `set -u` trap, the apt line that resolves across Ubuntu releases) that is worth extracting before deletion even if the scripts themselves go.
- Whether a tag is cut on `main` before anything is removed.

## What would resolve it

A grilling session producing a literal checklist: paths, order of operations, the precondition for each deletion, and the rollback.
Nothing is deleted until this ticket is resolved and the smoke is green.

Deliverable: the checklist, approved, plus whatever knowledge extraction it identifies as a prerequisite.
