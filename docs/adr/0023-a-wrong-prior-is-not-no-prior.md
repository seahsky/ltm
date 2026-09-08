# A wrong prior is not no prior, and a narrowed list is not a memory

**Status:** proposed (2026-09-08), on the 2026-09-05 review of `matrix-1` and the four read-only answers `tools/matrix_audit.py` returned from it.
**Section E has since run on `matrix-1` and its pre-registered branch fired: SEPARABLE, at the maximum possible margin — every wrong recall below every right one, 194 of 194 dropped for zero correct recalls lost.** The measurement and the one condition that limits what it may claim are recorded under change 2 below. Nothing else in this ADR is decided by it.
Nothing here is implemented beyond section E of that reader.
Two of the four changes cost a night of box time and the third changes what a published number is allowed to say, so they are Sky's to accept, not mine to land.

ADR-0018 built a 2x2 over two memories.
`matrix-1` ran it: 19 scenes, 4 cells, 282 episodes an arm, 7h09m.
The arithmetic reproduces exactly and the heard column moved by 14-16 points.
**Neither axis measured the quantity ADR-0018 names**, and both reasons were found by reading the run rather than by running another one.

## What `matrix-1` measured

| arm | Find-SR@1m (of the 268 routable) |
|---|---|
| `heard_seen` | 86 (32.1%) |
| `heard_unseen` | 91 (34.0%) |
| `not_heard_seen` | 45 (16.8%) |
| `not_heard_unseen` | 48 (17.9%) |

Co-primary B (`heard_unseen` vs `not_heard_unseen`) passes the strict scene-level sign test at 12 of 14 non-tied scenes, p ~= 0.013.
Primary A fails it at 10 of 13, p ~= 0.092.
Both effects are about 2.1-2.4x the 6.7-point MDE, from one run of each arm.

### The heard axis is a right prior against a WRONG one

`without_class` strips the run's own class and leaves the other two.
`_vote` has no abstain: it returns `None` only for an empty store, a zero-norm query, or all-zero-norm entries.
So a `not_heard` episode does not lose its prior — it gets a different one, and walks toward it on every diverting silent step.

`matrix_audit` section C counted this on the real audits, and it is exact:

- **zero `no_prediction` in any of the 1128 episodes.**
- The `not_heard` arms received a confident wrong category on 86-108 episodes each, overwhelmingly `chair` (79 and 102). The wrong prior is a systematic march to the living room, not noise.
- `unreachable` on 68 and 49 more: a recall was made and the navmesh routed to no instance of it, so those episodes got **no divert target at all** and ran unsteered.
- About 46% of episodes never consulted the prior in any arm (consulted 148-157 of 282).

"not_heard" is therefore three behaviours under one label: wrong-prior-steered, unsteered-because-unreachable, and never-consulted.
The 14.2 and 16.0 point effects are `benefit(right prior) + harm(wrong prior)`, unsplit, over a mixture.
`memory_prior.py:84` and `runner.py:1408` described the opposite until PR #83 corrected them.

### The seen axis narrowed a list it added nothing to

The tour's stored point is drawn from the same ObjectNav ground-truth table the unseen cell reads (`prior_driver.py:93` -> `memory_build.py:131`); `walk_tour` discards the agent's actual arrival pose.
`points_by_category_for_cell` then replaces the full tuple with that subset.
A seen cell holds **no location the unseen cell lacks**. It only narrows a candidate set, blind to which instance is sounding, while the unseen cell's nearest-reachable rule is positively correlated with the source.

Section B measured both halves of that, and they disagree in exactly the predicted way:

- **Mechanically live.** Instances differed at a shared voted category on 72 heard pairs (12 scenes) and 70 not_heard pairs (10 scenes) — roughly half the consulted episodes resolved through genuinely different candidate sets.
- **Behaviourally empty.** Outcomes did not move: +5 of 27 discordant (p = 0.44) and +3 of 17 (p = 0.63).

So the null is real and it is a null about a **selection rule**, not about episodic memory.
`matrix-1` says nothing about episodic memory in either direction, and the earlier "well-powered null" reading is withdrawn.

Two scenes were also silently dropped from the store (`p53SfW6mjZe`, `qyAac8rV8Zk`, in no provenance list, 0 episodic rows), so their "seen" cells ran identical to their "unseen" cells.
PR #83's coverage gate stops that happening again.
Why those two tours did not complete is **not recorded** — the store predates `scenes_incomplete` — so the next run has to find out rather than assume the leg budget.

## What changes

### 1. A fifth condition, `none`

`MemoryCondition.NONE` already exists end to end, `stores_for_cell` already returns empty stores for it, and `matrix_sweep.sh` already takes `--conditions`.
**It is still not free, and I said otherwise before checking.** Verified by running it:

```
--memory-condition none --memory-store <path>   ->  {}
(no memory flags at all)                        ->  {}
```

`memory_kwargs_from_args` collapses an explicit `none` into the bare invocation, so `run()` receives no memory keywords, `memory` is `None`, and `runner.py:1851` writes `memory_condition: null`.
That is the value reserved for "this record predates the field" — the precise conflation `MemoryCondition`'s own docstring refuses to allow.
A `none` arm run today would be unreadable by `episode_diff` and by `matrix_audit`.

The change is to distinguish **flag absent** from **flag explicitly `none`**: absent keeps returning `{}` and stays byte-identical to every pre-matrix caller; explicit `none` requires `--memory-store` like every other condition, is emptied by `stores_for_cell`, and is witnessed in the audit as `"none"`.

What the arm buys is the decomposition the current table cannot give:

| contrast | what it isolates |
|---|---|
| `heard_unseen` - `none` | the benefit of a right prior |
| `not_heard_unseen` - `none` | the harm of a wrong one |
| their difference | today's 16.0 points, split into its two parts |

### 2. The abstain floor is measured, not chosen

If a wrong recall scores lower than a right one, `_vote` can decline to answer and `not_heard` becomes an arm with **no** prior rather than an arm with a confidently wrong one.
That is a question about data already on disk, so it is answered before any run: `matrix_audit` section E grades every recall against the sweep's own `assignment.tsv` (a recall is correct when the voted category is the anchor object of the class that scene ran) and reports the largest floor that loses no correct recall, plus the max-J operating point.

Pre-registered, before the number is read:

- **Separable, or a free floor that drops a large share of wrong recalls** -> adopt the floor. `not_heard` becomes an arm with no prior, and the `none` arm is a check on that rather than the only control.
- **Overlap with a small free drop** -> there is no floor. `_vote` keeps no abstain, and the `none` arm is the only clean control for a wrong prior.

Section E's blind spot is stated rather than assumed away: confidence reaches the audit only on a **resolved** prior, because a miss builds no `MemoryPrior`.
The 68/49 `unreachable` episodes and the ~46% never-consulted are invisible to any floor computed from `matrix-1`.
If a floor is adopted, `resolve_prior` must also record the vote's score on `CATEGORY_ABSENT` and `UNREACHABLE` — the vote happened there, only its score was dropped — or the next sweep is blind in the same place.

`test_memory_prior.TestTheNotHeardCellsReceiveAWrongPrior` is annotated to be **rewritten, not reverted**, when a floor lands.

#### Measured on `matrix-1`, 2026-09-08

```
  pooled CORRECT n=272  min/p25/median/p75/max 0.7842/0.9060/0.9370/0.9548/0.9755
  pooled WRONG   n=194  min/p25/median/p75/max 0.1466/0.2291/0.2811/0.3921/0.4467
  free floor 0.7842 (the largest that loses NO correct recall): drops 194 of 194 wrong (100.0%)
  -> SEPARABLE
```

The two distributions do not touch: a gap of **0.3375** between the highest wrong recall and the lowest right one, over 466 graded episodes.
The separable branch fires, and it fires at the ceiling.

**What the two sets actually are, which is narrower than "right recalls against wrong ones".**
Every one of the 272 correct recalls came from a `heard` arm and every one of the 194 wrong ones from a `not_heard` arm.
The heard arms never once voted a wrong category — 136 resolved priors in each, 136 correct in each — so `matrix-1` contains **zero** cases of a full store erring, and section E cannot say what such a case would score.
What is demonstrated is exactly the rule the abstain floor needs: **when the class is absent from the store, the vote scores low, without exception.**
What is not demonstrated is a general is-this-recall-right detector.

**The condition that limits it: both sides render the same recording — and that is ADR-0018's deliberate control, not an oversight.**
`resolve_anomaly_clip` returns `<clip_dir>/<class>.wav`, one file per class, and `fetch_esc50_clips` stages exactly one ESC-50 recording per class.
`prior_driver.py:379` and `runner.py:2135` both resolve it, `matrix_sweep.sh` overrides the clip directory nowhere, so the store entry and the episode's query are **the same waveform through two different impulse responses**.
A median cosine of 0.937 is consistent with that.

ADR-0018's amendment of 2026-08-21 chose this, and gave the reason:

> The heard/not-heard split must control it: the same recordings on both sides, differing only in whether a prior visit stored them.

The reason is measured, not stylistic. Recording difficulty swings enormously inside one class — `water_drops` scored 0.998 anchor recall on ESC-50 clips 0-7 and **0.449** on clips 8-15 — so a heard column drawing different clips from the not-heard column would confound memory with which recordings each side happened to draw.
Holding the recording fixed removes that confound and is the right call.

What it costs is the scope of the claim, and the ADR should say so plainly: **the heard axis measures retrieval of a stored recording, not generalization to a new one.**
That is a real result and it is narrower than "the memory learned the class".
The floor inherits exactly the same scope.

The class-generalization experiment is a **different, never-run** one, and ADR-0018 already promoted it in the same amendment ("the recording-level axis is promoted") and named what a fresh headline needs: `--clip-start 16`.
There is prior evidence it would survive: all three of the matrix's classes — `toilet_flush`, `snoring`, `keyboard_typing` — are in ADR-0018's bank of record, each cleared on two disjoint recording sets, aggregate anchor top-1 0.880 on clips 0-7 against 0.867 on clips 8-15.
So CLAP carries class-level structure for these three; what is unmeasured is whether the *store's* recall and this floor survive it.
The machinery is already there and unused: `clips.stage_corpus` writes `<class>/<NN>.wav`, `clips.clips_for_class` reads them back, and `clips.py --index N` stages a single held-out recording per class into any directory.
The only code missing is a `--clip-dir` on `prior_driver` and a passthrough in `matrix_sweep.sh`.

**Consequences of adopting the floor, stated before it is adopted.**
The 194 wrong priors abstain, and the 68 + 49 `unreachable` recalls very likely abstain with them — *inferred*, not measured, because a miss records no score, which is the same blind spot named above.
That empties most of what steered the `not_heard` arms, so the heard-versus-not-heard effect should **shrink**: today's 14.2 and 16.0 points are `benefit(right prior) + harm(wrong prior)`, and the floor removes the second term.
Adopting the floor is therefore not a free improvement. It is the test that could deflate the headline, and that is the reason to run it.

### 3. The episodic entry carries what the tour saw, not what the annotation says

A requirement, not an implementation: an entry must hold something `category_points(dataset)` does not.
Two candidates, both already produced and then discarded by the tour:

- the agent's own **arrival pose** (`walk_tour` has it),
- which **instance sounded** during the tour.

The test that would have caught the defect is one no current test makes: an entry whose point is **not** in the scene's ObjectNav table.
`test_memory_prior.py:150` certifies divergence on a store shape the prior pass cannot produce, and must be rebuilt against a store it can.

Until an entry carries non-GT information, **no seen/unseen cell is worth box time**, and the seen axis is not reportable in either direction.

### 4. No magnitude from a single run

`repeat-1` measured 16.2% episode flips on byte-identical reruns and a net +11 of 365 (3.0 points) from nothing.
`matrix_audit` section D re-measured it inside `matrix-1` for free, on the two dropped scenes whose seen and unseen cells necessarily ran identical inputs: **0 of 30 flips in the heard pair, 3 of 30 (10%) in the not_heard pair.**

The noise is mode-dependent: a strong prior pins the trajectory, and exploration is where render noise enters.
That also sizes the seen axis's own discordance (27 and 17 pairs) as roughly what the apparatus makes from nothing.

Rule: **a magnitude is published only with a repeat arm of at least one cell.**
A direction may be reported from one run when the pooled McNemar and the scene-level sign test agree, and both tests are always printed.

## What it costs

`matrix-1` was 4 arms x 19 scenes x 15 episodes in 7h09m, so about 1h47m an arm.

| sweep | wall clock |
|---|---|
| the four cells again | ~7h09m |
| five cells (`none` added) | ~8h56m |
| five cells + one repeated cell | ~10h45m |

Both new shapes fit a night.
The episodic redesign adds no arm; it changes what the seen cells mean, and until it lands the honest sweep is three arms (`heard_unseen`, `not_heard_unseen`, `none`) plus a repeat, at about 7h09m.

## What this ADR does not decide

- **Which** non-GT quantity the episodic entry carries. That needs the tour's own record inspected first, and the answer changes what a seen cell claims.
- Whether the prior pass tours **held-out clips**. This is ADR-0018's own promoted recording-level axis, and it needs a `--clip-dir` on `prior_driver` plus a passthrough in `matrix_sweep.sh` — the staging (`clips.py --index 16`) already works. It is the difference between "retrieval of a stored recording" and "generalization to a new one" for the heard axis as a whole, not only for the floor.
- **The exact repeat of `matrix-1` is no longer runnable, and that is correct.** PR #83's coverage gate refuses an assignment the prior pass did not complete, and `--force` does not bypass it, so reproducing `matrix-1` byte for byte would mean reproducing the two mislabelled scenes. Any repeat arm is therefore a repeat of a **corrected** configuration, which does not exist yet — the store changes the moment those two scenes complete, and a changed store changes the vote in all 19. The single-run rule (change 4) applies to whatever configuration is established next, not retroactively to `matrix-1`.
- Whether the assignment keeps `p53SfW6mjZe` and `qyAac8rV8Zk`. The new coverage gate will refuse the sweep until their tours complete or they are dropped **on purpose**, which is the point of the gate.
- ADR-0018 amendment (c), the oracle view-point widening, still unimplemented at `runner.py:766`. It confounded nothing in `matrix-1` because it was absent from every cell, but a pre-registered item is silently missing and that is its own finding.
