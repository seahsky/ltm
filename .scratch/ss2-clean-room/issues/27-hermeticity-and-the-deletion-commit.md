# 27 — The hermeticity re-run and the phase-3 deletion commit

Type: task
Status: claimed
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

## Comments

### 2026-08-05 — the Mac half is built; the box half is one command away

**Status: still open.** Phase 2 needs the box and no session on this Mac can run it. Everything that could be built, verified and staged from here is done, and the gate is now a single command rather than a hand procedure.

#### The gate is code, and it moves more than this ticket said

`earshot/tools/hermeticity_gate.sh` — moves the delete set out, runs the box suite and one smoke episode with it gone, moves it back, writes the evidence. `earshot/tools/reset_manifest.py` holds the delete set as data, so the same list drives the gate's move, the deletion commit and the criterion.

**The one correction to this ticket: phase 2 moves out all five delete groups, not two.** As written it moves `embodied_memory/` and `dialogue_memory/`; phase 3 also deletes `scripts/`, the MSC readmes and drivers, and `data/msc/`. This ticket's own argument for the gate — a static grep misses `importlib`, a `sys.path` append, a hardcoded path in a config, a data path only the old tree knew about — applies to every group, and the narrow version would have been green over a real leak. See the next section.

#### Found before the box trip: the carried `nrun` was dead, three ways

Ticket 10 carried the notify trio "as-is" to `earshot/tools/notify/`, and verbatim is what broke it. All three self-references were repo-relative, and the files moved from one level deep to three:

| | resolved to | symptom |
|---|---|---|
| `nrun`'s dispatch | `earshot/tools/scripts/notify-run.sh` | printed `[nrun] detached (pid …)`, ran nothing, reason in a detached `.out` nobody reads |
| the emailer | `earshot/tools/scripts/notify_email.py` | swallowed by the `\|\| true` that exists to protect the wrapped exit code |
| `.env` | `earshot/tools/` | a fully configured box reporting "not configured" |

`box_gate.sh:50` documents `source earshot/tools/notify/notify-run.sh && nrun bash earshot/tools/box_gate.sh` — so the box trip this ticket opens with had no working launcher. Every path is derived from the script's own location now, and a missing notifier is loud instead of silent.

**Why nothing caught it:** `earshot/tools/notify/test_notify_email.py` is a standalone assert script that `unittest discover earshot/tests/mac` does not collect. The trio was carried with its tests and its tests were carried out of the suite. Its own wrapper smoke was green throughout, because it asserts the wrapped exit code and the log file — precisely the two things `|| true` protects.

**And the first fix for it was green against the bug.** A behavioural test written inside this tree passes with a stale `$repo_root/scripts/…` reference, because `scripts/` has not been deleted yet: a reference to a doomed path is correct today and broken the instant the commit lands. So `tests/mac/test_notify.py` runs the trio inside a skeleton holding only what survives phase 3 — this ticket's gate in miniature, on the one carried tool that is not Python and is therefore invisible to the layering and import checks. Verified by planting the original bug: the in-repo tests stay green, the hermetic ones go red.

#### Criterion 9 is answerable now, and `NOT_RUN` still means red

`task/smoke.py` reported criterion 9 as structurally `NOT_RUN`, on the reasoning that hermeticity is a property of two runs. That was half right. The comparison is not what makes the claim — **absence during this run** is, and absence is a property of one run's environment that simply was not written down. The gate verifies the delete set is gone immediately before the run and again immediately after, and writes both halves into the run directory as `hermeticity.json`.

An ordinary run has no such file and still reads `NOT_RUN`, so the baseline run cannot read green by being handed to this judge. The judge recomputes `complete` from the two halves (a top-level `true` is one edit away), refuses a record naming another run, and refuses a record that verified a subset — moving out one path and recording that is the obvious way to fake this.

The manifest's *correctness* is a separate question with a separate owner: `tests/mac/test_reset_manifest.py` audits it against `git ls-files`. Restating the delete set inside the judge would be ticket 24's one-rule-in-two-languages, and the copy that drifted would be the one gating the irreversible commit.

#### The delete list is audited, and one line of this ticket was ambiguous

Every entry verified against `git ls-files`, and the counts are pinned so the tree cannot move under the reset silently: `embodied_memory` 163, `dialogue_memory` 30 (15 `.py`, 15 `.pyc`, and `evaluation.cpython-310.pyc` has indeed had no `.py` for a long time), `scripts` 51, the two READMEs, the three `run_msc_*.sh`, `data/msc` 4.

**"`scripts/` wholesale except the three notify files already carried" reads as an exception to the delete if you skim.** It is not — it names why nothing is lost. Taken as an exception it would leave two divergent copies of the trio, and after the fix above the copies under `scripts/` are the stale ones. The manifest deletes all 51 and says so.

#### The restore is our logic, so it is not a box test

The move and the restore need no simulator, and ADR-0014 is explicit that an our-logic assertion reaching for the real artefact is a seam defect. So the script grew `--dry-run` (move, verify, restore, stop) and `tests/mac/test_hermeticity_gate.py` drives it against a scratch git repo. The arm that matters is `--self-test-abort`, which exits *while the repo is taken apart*: a restore that only works on the happy path is worth little, because the failure mode is a gate that dies half way and leaves the tree missing a directory the operator then commits around. Verified non-vacuous by deleting `trap restore EXIT` — two tests go red naming all nine paths.

Three layers of restore: the EXIT trap; a `git status` check afterwards that shouts rather than logs; and — since every moved path is tracked — a `git checkout -- …` recovery line printed into the log *before* anything moves, for the case where a `kill -9` leaves no trap at all.

#### Also landed

- `docs/race-box-runbook.md`: a hermeticity-gate section, and the three live `scripts/notify-*` instructions repointed. The four remaining `scripts/` references there are past-tense history of the extraction and are correct as they stand.
- `earshot/reference/memory/README.md`: `memory_bridge.py` also lazily imports `dialogue_memory.train_scorer` and `train_predictor`, which do not carry and have no vendored copy. Deliberate — the importance-head lever is closed on measurement — but a revival will reach for them, so it is stated.
- `.scratch/ss2-clean-room/claude-md-rewrite.md`: the 580-line `CLAUDE.md` replacement, staged for review rather than written during the irreversible commit. `CLAUDE.md` itself is untouched, because phase 3 rewrites it in the same commit and it is not wrong yet.

695 Mac tests green (662 → +33), ruff clean.

#### What is left, in order

1. **Box, phase 2.** `source earshot/tools/notify/notify-run.sh && nrun bash earshot/tools/hermeticity_gate.sh --tag hermetic-1`, then `python -m earshot.task.smoke --run-dir runs/hermetic-<ts>` after it restores. Nine green is the licence. A red is the gate working.
2. **Box sweep.** Remove the `soundspaces-spike` env. Confirm the suspected `data/` duplicate by inode and record the answer, keeping both. The semantic annotations are now decidable — ticket 21 measured that the clean room needs no scene-dataset config of any kind — and the call is **keep**: 680 GB free means it was never a space question, and the deletion commit removes `scripts/download_hm3d_semantics.sh`, so re-fetching them would be a fresh piece of work.
3. **Phase 3**, on green: the atomic commit, `CLAUDE.md` from the staged draft, `docs/archive/README.md`'s heading, PR.
