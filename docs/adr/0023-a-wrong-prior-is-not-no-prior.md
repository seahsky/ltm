# A wrong prior is not no prior, and a narrowed list is not a memory

**Status:** proposed (2026-09-08), on the 2026-09-05 review of `matrix-1` and the four read-only answers `tools/matrix_audit.py` returned from it.
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
- Whether the assignment keeps `p53SfW6mjZe` and `qyAac8rV8Zk`. The new coverage gate will refuse the sweep until their tours complete or they are dropped **on purpose**, which is the point of the gate.
- ADR-0018 amendment (c), the oracle view-point widening, still unimplemented at `runner.py:766`. It confounded nothing in `matrix-1` because it was absent from every cell, but a pre-registered item is silently missing and that is its own finding.
