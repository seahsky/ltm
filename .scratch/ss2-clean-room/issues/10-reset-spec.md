# 10 — The reset spec: what is deleted, what carries, and when

Type: grilling
Status: resolved
Assignee: Sky
Blocked by: none
Resolved: 2026-08-01 — see the Answer section. Surfaced tickets 14 and 15.

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

## Answer

**The reset is a four-phase operation with one irreversible commit at the end, gated on a hermeticity re-run rather than on the smoke alone.**
The memory stack carries as a **vendored, inert `reference/` directory** — the whole retrieval stack plus the embodied bridge, not just the consolidation math.
Every result and constraint document stays; only how-to-run docs for deleted code go.
`scripts/` is deleted wholesale after its box knowledge is extracted, because the knowledge and the artifacts have come apart.

Nothing below is executed by this ticket. This is the approved checklist; phase 3 is additionally gated on ticket 09 and on the smoke being green.

### The one-line rationale per branch

- **Carry mechanism — vendored, not a git-history pointer.** A pointer costs a `git show` at exactly the moment a follow-on session is least likely to spend one.
- **Vendor line — the stack *and* the bridge.** The map's phrase "the STM and LTM calculation" is not satisfied by the five dialogue modules, because the embodied STM is not in `dialogue_memory/stm.py`. See the correction below.
- **Detector — live port, OWLv2 dropped.** The seam ships with two implementations or it is not a seam.
- **Docs — the record stays, the instructions go, `CLAUDE.md` is a gate.** An auto-loaded file that describes a deleted tree misinforms every session in the window before it is fixed, including the sessions doing the build.
- **Scripts — extract, then delete.** The drivers are bound to a CLI that will not exist; the facts inside them are not.
- **Box — trust calls only, never space calls.** 680 G free of 773.9 G. One footgun removed, everything else recorded.

---

### A correction this ticket owes the map

**`dialogue_memory/stm.py` does not carry, and the map's phrase "the STM and LTM calculation" was half-wrong about where the STM lives.**

`dialogue_memory/stm.py` (80 LOC) is `ShortTermMemory` over `DialogueTurn`s. It is imported only by `agent.py` and `quickstart.py` — both pure MSC path, both out of scope. Nothing embodied has ever imported it.

The **embodied** STM is inside `embodied_memory/memory_bridge.py`: the `_pending` buffer, the event-boundary consolidation flush, and the `disable_stm` / `disable_ltm` toggles that made the S1/S2/S3 ablation possible at all. Anyone reading the map's Notes and reaching for `stm.py` would carry the wrong file and leave the real one behind. This is why the vendor line includes the bridge.

---

### Phase 0 — now, unblocked, reversible

1. **Cut and push the tag.** `archive/pre-clean-room-2026-08-01` on `main`, pushed to `origin`.
   Cut **now**, not at deletion time: the old trees do not change during the clean-room build (it is built alongside them), so the tag's content is already final, and cutting it now means no future session can forget.
   Matches the repo's existing habit (`archive/lifelong-revisit-eval`, `backup/main-pre-lifelong-2026-07-12`).
2. **Extract the box runbook** → `docs/race-box-runbook.md`. This is **ticket 14**, not work for this ticket. It is a hard prerequisite of phase 3: once `scripts/` is gone the facts are only recoverable by reading 8,200 LOC out of a tag.

### Phase 1 — once `<newroot>` is named (post-09), reversible

The new root does not exist yet — the map's fog puts the package layout behind ticket 09 — so nothing here can start today. When it does, the vendoring is a **plain copy from the still-live tree**, not a `git show` against the tag.

**Vendor, inert, to `<newroot>/reference/memory/`** — ~3,400 LOC:

| file | LOC | what it is |
|---|---|---|
| `dialogue_memory/consolidation.py` | 476 | importance `I = αR + βU + γN`, the write gate |
| `dialogue_memory/ltm.py` | 229 | 3-layer FAISS memory (only the fine layer was ever queried) |
| `dialogue_memory/pattern_cluster.py` | 336 | mid layer — `ltm_mid=false` in every run, never enabled |
| `dialogue_memory/reranking.py` | 470 | score fusion |
| `dialogue_memory/encoder.py` | 187 | the SBERT seam — the measured bottleneck |
| `embodied_memory/memory_bridge.py` | 1,690 | embodied STM + both seams |

They vendor cleanly: the only non-stdlib imports across the five dialogue modules are `numpy`, `faiss` (in `ltm.py` alone), and one relative import `consolidation → ltm`. No rewriting beyond that single line.

**`reference/` is excluded from the new tree's lint, test and import surface.** This is a requirement on the package layout, not a suggestion — the vendored code is vendored *broken* (it imports `faiss` and `sentence-transformers`, and `memory_bridge.py`'s interface is built against the deleted `episode_runner` and the env-flag surface ADR-0008 removed). If it can fail CI or be imported by accident, vendoring it was a mistake.

`<newroot>/reference/memory/README.md` must state:

- the write path — keyframe → segment → consolidate → LTM
- the read path — `propose_memory_candidates()`, which **is** ADR-0008's "memory later plugs in as another proposer"
- that the env-flag surface is superseded by ADR-0008
- the closed levers, so no follow-on effort re-runs them: coarse-affordance (proposed but never chosen), R and U importance heads (five formulations, all ≤ heuristic), the M4 temporal head (inert), the S2 audio-DOA head (zero-sum on a single instance), `write_audio_event` (redundant with vision on an LOS seed)
- that the bottleneck is **SBERT instance discrimination on the read side**, measured, not asserted — and that the CapRL gate closed the captioner as a lever for $0

**Port into live code:**

- `embodied_memory/metrics.py` → **near-verbatim**. 55 LOC, one pure function `compute_benchmark_spl`, no simulator. A third Mac-testable layer alongside the anomaly controller and the navmesh reachability filter.
- **the ObjectNav `.json.gz` loader** → **extracted and rewritten**. It is not a file: it is entangled inside `habitat_env.py` (623 LOC, habitat-lab-coupled, does not carry) across the dataset-path search, the scene-label resolution, and the `content/<scene>.json.gz` lazy load. This extraction is where ticket 08's outstanding box-fact gets settled — whether `objectnav_hm3d` v1 loads against `hm3d_basis.scene_dataset_config.json` or requires `hm3d_annotated_basis.scene_dataset_config.json`. Until it is settled, the 9.3 G of semantic annotations stay.
- `embodied_memory/goal_detector.py` → **reshaped behind ADR-0008's `detects(obj)` seam**. Note the existing class exposes `locate(...)`, so a straight port would not have fitted the seam regardless.
  - **KEEP**: `parse_qwen_bbox`, `robust_depth_at_pixel`, `back_project_pinhole`, and the **snap gate with its L3 floor-plane (xz) fix** (`3307f19` / `7fbf370`, plus `DETECTOR_SNAP_FLOOR_EPS`). That fix is a real correctness win — the old gate used a 3D distance while every consumer uses only (x, z), so it would have wrongly rejected genuinely-elevated correct detections.
  - **DROP**: `_ensure_owlv2` / `_infer_owlv2`. Max box score 0.031 (base) and 0.058 (large) on HM3D sim renders is the noise floor; carrying it means carrying an OWLv2 dependency for a measured negative.
  - Ships as `OracleDetector` (what the smoke runs) + `CaptionDetector` (what R2 runs).
  - **Cost, disclosed:** the caption path is live but has no consumer inside this map, so it ships untested until R2. That was chosen over letting the seam ship with one side.
- `scripts/notify_email.py`, `scripts/notify-run.sh`, `scripts/test_notify_email.py` → **as-is**. Stack-agnostic Python; the `nrun` overnight-report path is how RACE results come back at all.

**Already settled by ticket 07, restated so the checklist is complete:** `anomaly_controller.py` (316 LOC) ports near-verbatim, with ticket 09 owning whether its localization policy is amended for continuous receiver positions; `frontier_planner.py` (1,129 LOC) is rewritten to ~300 LOC.

### Phase 2 — the gate

1. **The smoke is green, per ticket 09's criteria.** This ticket deliberately does not define them; the map's fog assigns smoke-green acceptance to 09, and defining it here would decide 09 by accident.
2. **The hermeticity re-run.** Move `embodied_memory/` and `dialogue_memory/` out of the repo entirely, re-run the smoke, require green, then move them back.

The second gate exists because "the smoke is green" and "the smoke is green *without the old trees present*" are different claims, and only the second one licenses an irreversible delete. A static grep was considered and rejected: it misses `importlib`, a `sys.path` append, a hardcoded `"embodied_memory/..."` string in a config, or a data path only the old tree knew about. Cost is one smoke run.

If it fails: fix the leak, restore, repeat. It failing is the gate working.

### Phase 3 — irreversible: one atomic commit, on a branch, PR'd

**Delete:**

- `embodied_memory/` — 163 tracked files, in full.
- `dialogue_memory/` — 30 tracked files: 15 `.py` and **15 tracked `.pyc`**. The `.pyc` tracking is real rot, not cosmetic — one of them, `evaluation.cpython-310.pyc`, is a stale artifact of a module that no longer has a `.py` at all.
- `scripts/` — ~35 `race-*.sh` drivers, `scripts/archive/` (7 more), `setup-vm.sh`, `setup-anomaly-vm.sh`, `verify-setup.sh`, `race-smoke.sh`, `download_hm3d_semantics.sh`, `fix_hm3d_semantics.sh`, `run_phase2_ablation.sh`. Everything except the three notify files carried in phase 1.
- `README_LTM_MSC_EVAL.md` (403), `README_MSC_EVAL.md` (285) — how-to-run for deleted code.
- `run_msc_baseline.sh`, `run_msc_quick_eval.sh`, `run_msc_full_eval.sh`.
- `data/msc/*.json` — the 4 tracked MSC files. Note these are the *only* substantial tracked assets under `data/`; everything else there is gitignored, so the reset does not touch the 19.9 G on disk.

**Keep, untouched:**

- `docs/adr/0001`–`0008`. 0001–0004 are ticket 09's live inputs; 0005–0006 constrain what the paper may claim; **0007 and 0008 are this map's own output** and deleting the sequence would delete the map's product.
- `PHASE2_ABLATION_REPORT.md` (3,486 lines) — the experimental record of everything being deleted, and the paper's evidence base.
- `ICRA2027_PAPER_DRAFT.md`, `CONTEXT.md` (binding per the map's Notes), `Research Proposal_Embodied Agent.md`, `MSC_BENCHMARK_REPORT.md` (24 lines — a result, not an instruction), `models/README.md`, `docs/` in full.

**Rewrite, in the same commit:**

- `CLAUDE.md`, 580 lines → short. It is auto-loaded into every session, carries 7 references to the deleted trees, a "Repo orientation" section describing them, and a "Running the ablation" section invoking `python -m embodied_memory.run_hm3d_pol`. The new version is mission + new-tree orientation + how to run the smoke + pointers. The ~450 lines of outcome narrative are not summarised into it — that history already lives in `PHASE2_ABLATION_REPORT.md`, and duplicating it is how the file got to 580 lines.
- `docs/archive/README.md` — it currently says its runbooks are "superseded by `scripts/race-setup.sh` + CLAUDE.md", and `race-setup.sh` is being deleted in this same commit.

**Why one commit:** it makes the whole reset a single revert. It goes on a branch and through a PR, matching the map's existing habit (`wayfinder/ss2-clean-room-07` → PR #10).

**Rollback:** `git revert <commit>`, or `git checkout archive/pre-clean-room-2026-08-01 -- embodied_memory dialogue_memory scripts`. Both remain available indefinitely; the tag is pushed.

### The box-side sweep

Framed by this ticket's own finding, which holds: **680.1 G free of 773.9 G, so nothing here is a space call.** Only trust and correctness justify acting.

**REMOVE — one item, on trust:**

- the `soundspaces-spike` conda env. Ticket 04 explicitly refused to trust it for its verdict, and it remains one `conda activate` away from handing a future session a wrong answer about the exact thing this map hangs off. Its recipe (`race-soundspaces-spike.sh`) is deleted in the same reset, so the env would otherwise outlive its own provenance: unrebuildable and untrusted.

**VERIFY, DO NOT DELETE:**

- the suspected ~9.3 G duplicate in `data/` (`val` vs `versioned_data`, each reporting 9.3 G at 100 `.basis.glb` / 36 `.semantic.glb`). Confirm by inode/`-samefile`, record the answer, keep both regardless. It is still only **inferred from `du` arithmetic**, and 20/20 val scenes now have meshes so whichever copy survives must be the complete one. Keeping both is free; deleting the wrong one is not.

**LEAVE, recorded:**

- `~/soundspaces-build` (5.5 G) — superseded but not activatable, so not a footgun.
- `~/.cache/huggingface` (23.9 G), `~/.cache/pip` (5.7 G) — caches. Clearing them buys 29 G of a resource that is not scarce and costs re-downloading weights the clean room needs anyway.
- `runs/` (880.3 M on the box) — prune deliberately, later, remembering the audit caveat: the committed `runs/abl-s*-qwen` dirs held provenance-mismatched Run-2 data and were git-removed for exactly that reason.
- the `ltm-embodied` conda env — **blocked by ticket 13**, which is leaning on it as the only on-box proof that a modern torch runs on this V100.

**NOT a cleanup item — promoted to ticket 15:** the ~24 G of VRAM held by something unaccounted for (8,249 MiB free of 32,768). On a 32 G V100 where ADR-0008 is already counting VRAM (dropping the 7B planner to free ~15 G), an unexplained 24 G is a correctness threat to the whole build, and ticket 06 is timing renders underneath it right now.

---

### What this ticket deliberately did not decide

**The audio module carry line is owned by ticket 09.** `audio.py` (719), `audio_task.py` (402) and `perception.py` (506, CLIP) are unresolved here on purpose. The old audio path is the precomputed-grid convolve path this map killed at chart time, but the CLAP anomaly gate, the onset detection and the ESC-50 clip resolution are documented durable wins, and ticket 07 explicitly routed the ADR-0002 CLIP room-classifier question to 09. Ruling on the carry line here would decide 09 by accident — the exact trap 07 avoided. The ordering holds: 09 resolves before the smoke, and the smoke gates phase 3.

**Smoke-green acceptance criteria** are 09's, per the map's fog.

### Risks accepted, named

1. **~3,400 LOC of dead code lands in a clean room.** That is the cost of vendoring over a pointer, and it was taken knowingly. The `reference/` exclusion from lint/test/import is what keeps it from becoming a maintenance liability rather than an archive.
2. **The caption detector ships live and untested** until R2, which is out of this map's scope.
3. **`CLAUDE.md` is rewritten before the new tree is finished**, so it describes a partly-built package. That was preferred over a window in which every agent session — including the ones doing the build — loads a description of a repo that no longer exists.
4. **The box sweep leaves ~29 G of clearable cache and a possible 9.3 G duplicate in place.** Deliberate: at 680 G free, both are trades of certainty for a resource that is not scarce.
