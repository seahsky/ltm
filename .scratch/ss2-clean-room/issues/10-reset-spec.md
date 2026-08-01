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
  (Ticket 05 measured the real figure: `data/` is **19.9 GB**, not 1.2 GB. Does not change the conclusion — still gitignored, still untouched by a source-tree reset — but the number was wrong by 16x wherever it is quoted.)
- **Disk pressure is not a reason to delete anything.** 680.1 GB free of 773.9 GB (ticket 05). Every item below is a tidiness or trust call, never a space call.

Box-side items ticket 05 found and explicitly declined to act on, for this ticket to rule on:
- `soundspaces-spike` (conda env) and `~/soundspaces-build` (5.5 GB). Superseded by `ss2` and `~/ss2-build`. Ticket 04 already refused to trust the spike env for its verdict.
- `ltm-embodied` (conda env). Carries habitat_sim 0.3.3 non-audio, torch 2.8.0, numpy 1.26.4. It is the old stack, but it is also the only on-box proof that a modern torch runs on this V100 — which ticket 13 is now leaning on. Do not delete it before 13 closes.
- `~/.cache/huggingface` 23.9 GB and `~/.cache/pip` 5.7 GB. Caches, safe to clear, and clearing them costs a re-download rather than anything irreversible.
- `runs/` 880.3 MB on the box. Note the audit caveat already on record: the committed `runs/abl-s*-qwen` dirs held provenance-mismatched Run-2 data and were git-removed for exactly that reason. Whatever survives here should survive deliberately.
- **A suspected ~9.3 GB duplicate inside `data/`**: `val` and `versioned_data` each report 9.3 GB at 100 `.basis.glb` / 36 `.semantic.glb`. Inferred from `du` arithmetic (the parent totals ~19.7 GB, so they are not hardlinked), **not verified** — confirm before touching, and note that 20/20 val scenes now have meshes, so whichever copy survives must be the complete one.
- **~24 GB of VRAM is held by something unaccounted for** (8249 MiB free of 32768). Not a deletion item, but it belongs on the same sweep of "what is actually running on this box".

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

## Note added by ticket 08 (resolved 2026-08-01) — the dataset half of the keep list is settled

**HM3D stays; MP3D is out of scope** (`docs/adr/0007-hm3d-stays-mp3d-out-of-scope.md`). So the scene-asset side of this ticket's keep/rebuild/delete list no longer needs deciding:

**KEEP, all of it:**

- HM3D `val` — 100 `.basis.glb` / 36 `.semantic.glb`, 9.3 G
- HM3D `minival` — 10 / 4, 1.1 G
- the ObjectNav episode datasets for both splits

**DOWNLOAD: nothing.** No MP3D, now or later in this effort.

**The semantic annotations are explicitly KEPT, and the reasoning should not be re-derived.** They look like dead weight — materials are off so the audio never reads them, and HM3D's semantic sensor is a known all-zeros dead end. They stay anyway because **it is not established on the box whether the ObjectNav episode dataset loads against `hm3d_basis.scene_dataset_config.json` or requires `hm3d_annotated_basis.scene_dataset_config.json`**. The old `habitat_env.py` reaches for the annotated config, which is suggestive but not proof. 9.3 G against 680 G free is not worth a re-download over ssh if the answer turns out to be "required". The new tree settles it at runtime; until then, deleting them is the only irreversible move available.

This does **not** touch this ticket's other two loose ends from ticket 05 — the suspected ~9.3 G duplicate in `data/`, and the ~24 GB of VRAM held by something unaccounted for. Both are still open here.
