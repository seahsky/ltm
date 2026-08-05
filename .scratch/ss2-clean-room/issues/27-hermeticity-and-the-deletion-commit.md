# 27 — The hermeticity re-run and the phase-3 deletion commit

Type: task
Status: resolved
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

### 2026-08-05 — first box run: the gate stopped, and its reason was wrong

`hermetic-1` on the V100, commit `675d8cd`, exit 1 at 1m35s. **The move, the absence check and the restore all worked on the real 244 files** — nine paths moved, all verified absent, all restored, working tree clean again. It stopped at the box suite with

> FATAL: the box suite is red without the old trees — that is a leak

and **it was not a leak.** 43 of 45 box tests passed with the delete set gone. The two failures are both `clap_instantiable`: `ClapModel.from_pretrained("laion/clap-htsat-unfused")` raising transformers' CVE-2025-32434 guard, which refuses `torch.load` on a `.bin` checkpoint under torch < 2.6. That probe touches no path the reset deletes, which is checkable from source rather than by re-running: `env_check.py:457` and `task/models.py:66` load through transformers from the HF cache, and nothing in `embodied_memory/`, `dialogue_memory/`, `scripts/`, `data/msc/` or the MSC files is on that path.

**The defect is the gate's, and it is the shape this map keeps catching: an unsupported causal claim in a message.** A red without the old trees is evidence of a leak only if the same test is green with them, and the run had one arm.

Fixed with a control arm. `earshot/tools/suite_result.py` runs the box suite on either side of the move and `compare()` sorts the difference into three verdicts: **leaks** (green in the control, red without the trees) are fatal and are the only thing that earns the word; **pre_existing** (red on both) is loud, recorded, and explicitly not a hermeticity failure; **recovered** (red only in the control) is reported because a test that passes once its own tree is gone is worth a look. A fourth outcome is separate from all three: if the two runs did not collect the same tests, "no leaks" is an absence of evidence and the gate says so rather than passing. `compare()` is pure, so all of it is Mac-tested with injected results, including the two outcomes the box actually produced.

The comparison goes into `hermeticity.json`, so criterion 9 fails on a leak and passes-but-says-so on a pre-existing failure. The smoke keeps its hard failure, since a smoke that cannot complete blocks the deletion whatever the cause, but its message no longer says "leak": it names the control command that would settle it.

**Not fixed here, and out of this map's scope: CLAP itself.** The smoke runs without `--clap` by design (§4.3: one sound, the anomaly by construction), so nothing on the route to the destination needs it. But it is a real regression against ticket 13's recorded green, and the interesting part is *how* it regressed: **`pinned_versions_match` PASSED — 9 of 9 pins agree with `ss2-constraints.txt`** — so torch and transformers are at the exact versions that produced ticket 13's finite `[1,3]` logit, and the capability is gone anyway. Ticket 17 pinned the inputs; the *checkpoint* is not an input it names. The chain that follows from the error text is that the resolved weights are now a `.bin` rather than safetensors, which points at the HF cache rather than at any version.

That is a hypothesis, not a measurement, and the separating check is one line on the box:

```
ls ~/.cache/huggingface/hub/models--laion--clap-htsat-unfused/snapshots/*/
```

A `model.safetensors` there means something else is choosing the `.bin`; only `pytorch_model.bin` means the cache is the whole story and `from_pretrained(..., use_safetensors=True)` both fixes it and makes the next failure loud instead of a silent fallback. Ticket 09 already leaned on exactly this property for Qwen2-VL-2B. Deliberately not guessed at from here: it is a box capability, and this map's rule is that a capability is exercised, never proxied.

**Re-run:** `nrun bash earshot/tools/hermeticity_gate.sh --tag hermetic-2`. Expect the control arm to record the same two CLAP failures on both sides and report them as pre-existing, then proceed to the smoke.

### 2026-08-05 — `hermetic-2`: phase 2 ran clean, and the gate still owed a verdict

Exit 0 in 3m20s on commit `f9dacf1`, run dir `runs/hermetic-20260805-131808`.

**The control arm worked on its first real use, and the prediction held exactly.** Control 43/45 green, hermetic 43/45 green, `leaks: []`, `pre_existing:` the same two `TestClapIsInstantiableWhenRequested` methods, `comparable: true` on 45 tests both sides. The failure that stopped `hermetic-1` is now correctly classified as a sick environment rather than a leak, by measurement instead of by argument.

**The smoke completed with the delete set gone**, which is the claim phase 2 exists to license:

- `env_check` GREEN, 5 probes, `clap=False` (§8's own configuration — CLAP is not on this path)
- audio context armed at **335,370 vertices** against ticket 12's 10,000 floor, IR `(2, 14227)`
- calibration `onset_rms` 0.0136, **45.29 dB separation over 16 poses**
- onset step 4 → INVESTIGATE step 4 → RESUME step 7 → primary goal reached step 33
- funnel **PRIMARY_RESUMED**, stage 6 of 6
- audio 34 renders, mean 51.6 ms, **max 59.5 ms** against criterion 7's 0.5 s ceiling (8.4x margin)
- `hermeticity.json` written, `complete=True`; restore clean

Ticket 26's builder fix also visibly did its job on the episode that got skipped: *"this episode spans floors and the greedy climb cannot take stairs"* — refused with a reason instead of running as a silent null.

**But the gate exited 0 without producing the nine-point verdict.** It printed `python -m earshot.task.smoke --run-dir …` for the operator to run next, and a green exit that has not evaluated the criteria the deletion is gated on is the same shape as every other defect this ticket has found: a success that does not mean what it says. Nobody had run it yet, so **criteria 1–8 are unverified as of this entry** — everything above is ingredients, not the verdict.

Fixed: the EXIT trap now runs `smoke --run-dir` after the restore, and **its exit code is the gate's**. After the restore, so the log still ends on a clean tree. Armed by `JUDGE_RUN_DIR`, which is set only once the record is written, with a Mac test pinning that `--dry-run` does not claim a verdict about a run that never happened.

**Outstanding, and it is one command on the box** — the run directory already exists, so this needs no re-run:

```
python -m earshot.task.smoke --run-dir runs/hermetic-20260805-131808
```

Nine green licenses phase 3. Anything else and the gate has found something.

## Answer

**SMOKE GREEN, all nine, and the reset is committed. The map's destination sentence is a measured fact.**

```
1. PASS  audio live and every-step      34 renders / 34 loop steps
2. PASS  audio context sound            335370 verts (submitted 335362), canary seen
3. PASS  the IR is real                 shape (2, 14227), peak 0.5949
4. PASS  provenance did not raise       onset step 4, 4 pre-onset readings, t_anom 4
5. PASS  the full loop ran              funnel stage 6 (PRIMARY_RESUMED)
6. PASS  a report was emitted           all 9 of §5.1's keys
7. PASS  audio wall-clock in ceiling    max 0.05953 s, mean 0.05157 s, ceiling 0.5 s
8. PASS  env_check passed               5 probe(s), all pass
9. PASS  hermeticity                    9 paths absent before and after; no leaks
```

Criterion 2's **+8 vertex gap** (335,370 held vs 335,362 submitted) is ticket 16's finding reproducing exactly: the engine holds slightly more geometry than habitat submits, which is the direct evidence that invariant 1 reads the *engine's* mesh rather than the caller's claim about it. Criterion 4's `t_anom 4` is ticket 26's per-episode derivation, not the old `fake` constant. Criterion 7 has 8.4x headroom.

**Phase 3, one commit, 253 tracked files** — `embodied_memory/` 163, `dialogue_memory/` 30, `scripts/` 51 wholesale, `data/msc/` 4, the two MSC readmes, the three `run_msc_*.sh`. `CLAUDE.md` rewritten in the same commit, 580 lines → 110, with the outcome narrative deliberately not carried into it. `docs/archive/README.md`'s two dead pointers fixed. All 14 ADRs, both reports, `CONTEXT.md`, the proposal and `models/README.md` untouched.

### The prerequisite this ticket recorded as landed had not

**The tag `archive/pre-clean-room-2026-08-01` does not exist** — not locally, not on `origin`, not on `upstream`. Phase 0 was two halves and only the box-runbook half (ticket 14) ever happened; this ticket's "Prerequisites both landed" and its rollback line were both written against a tag nobody had made. Found by checking it rather than by needing it, which is the only reason it was not found the hard way.

Created and pushed immediately before the deletion as **`archive/pre-reset-2026-08-06`**, named for what it actually is — the last commit where both old trees exist — rather than for a date and a state it does not have. It points at the merge of PR #36, so it carries the whole clean-room build as well as the old trees; the original name would have misdescribed that too.

Rollback is therefore real in both forms the ticket promised: `git revert <commit>`, or `git checkout archive/pre-reset-2026-08-06 -- embodied_memory dialogue_memory scripts`.

### What the manifest is now

Its job changed rather than ended. The test that asserted every entry **exists** at its audited count — the guard against the delete list widening under a stale tree — now asserts the paths **stay gone**, which catches a partial revert or a directory restored by habit. The gate is kept working deliberately: `git revert` brings back the old trees *and* the machinery that checks them, and a gate deleted alongside its subject would leave that revert unverifiable. It now announces that the reset has landed rather than moving nothing and running a smoke whose green would mean something else.

### The box sweep

Not done, and it is not a gate on anything: `soundspaces-spike` is still on the box, the suspected ~9.3 GB `data/` duplicate is unconfirmed, and the semantic annotations are decided (**keep** — ticket 21 measured that the clean room needs no scene-dataset config, 680 GB free means it was never a space question, and this commit deletes `scripts/download_hm3d_semantics.sh` so re-fetching would be fresh work). Listed on the map as the one loose end that outlives the map.

### Deliberately not done

The **CLAP regression** is out of scope and recorded with its diagnostic: `pinned_versions_match` passes on 9 of 9 pins, so the box is at ticket 13's known-good versions and the capability is gone anyway — ticket 17 pinned the inputs, and the resolved checkout is not an input it names. §8's smoke runs without `--clap` by design, so nothing on the route to the destination needed it.
