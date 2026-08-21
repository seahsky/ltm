# Two memories, a class-level heard axis, and a seen row the field's protocol does not have

**Status:** accepted in shape (2026-08-20, from the same grilling session as ADR-0017), **with two parts open and named as open**: the prior pass is undesigned, and the power question is unresolved.
**Depends on ADR-0017.** Without the sounding window none of this is measurable.
**Reopens the audio axis** that `CONTEXT.md` had recorded as a closed negative, and carries the reason for the reopening rather than the fact of it.

The experiment is a 2×2 over two generalization axes, and the point of the factorial is that the two axes test **two different memories**.

|  | heard sound | not-heard sound |
|---|---|---|
| **seen scene** | episodic + semantic | episodic |
| **unseen scene** | semantic | neither — the baseline |

**Seen/unseen** is over scenes and is answered by an **episodic LTM**, keyed to the scene.
**Heard/not-heard** is over sound CLASSES and is answered by a **semantic LTM**, which is scene-agnostic and is the only store that can return anything useful in a scene the agent has never entered.
**STM** is within-episode and belongs to neither axis: its job is carrying a bearing through the silent phase.

## Why two stores rather than one

The unseen-and-heard cell is the whole reason this is a factorial.
Under a single scene-keyed store that cell returns exactly what the unseen-and-not-heard cell returns, the two are the same experiment run twice, and the 2×2 collapses to three arms and a duplicate.

This is not hypothetical about the code we have.
The archived stack hard-filters its fine layer to the current scene (`reference/memory/memory_bridge.py:829`), so reviving it and running the unseen row would produce a structural null — a null about the store's indexing, reported as a null about memory.
A scene-agnostic semantic store is therefore new code, not a revival, and that is the main build cost this ADR commits to.

## Why the heard axis is class-level, and why the old null does not apply

**Class-level**: heard means the sound CLASS has an entry from a prior visit.
SAVi's own split is recording-level — thirds of an ~81 s clip, all 21 categories present in both settings — and that is meaningful there because a policy memorised waveforms over 300M steps.
Nothing here is trained.
Against a frozen open-set encoder and a retrieval memory, a different recording of a known class is a **CLAP robustness** question, not a memory question, so it is kept as a cheap secondary axis (ESC-50 ships 40 recordings per class) and not as the headline.

**The reopening.** `CONTEXT.md` recorded the audio axis as closed: `write_audio_event` was mechanism-verified and then measured REDUNDANT-WITH-VISION.
That verdict was measured on a single-goal harness where the agent could see the goal, so a stored audio event never carried anything vision was not already supplying.
Under the sounding window the source is silent for most of the episode, and in the silent phase there is nothing left for a stored audio-place association to be redundant *with*.
That is a mechanical difference in the harness, not an appeal to a better implementation, and it is the only thing licensing the reopening.
If a future run finds the same null under silence, the axis closes for good and this paragraph is the record of what was tried.

## The agent is told nothing

The goal class is **inferred** from what the agent heard, by CLAP, rather than given at episode start.
Told the category outright, the sound's identity never matters and the two heard columns score alike by construction.

Inference is also the claim against the prior work.
SAVN-CE welds `num_classes = 21` into its ACCDDOA tensor shapes (`goal_descriptor.py:30`), so a 22nd class costs a retrain — their README reports 14 days on 4×A800 for the clean condition alone.
An open-set text encoder takes one more prompt and no training.
On one V100 that is not a workaround, it is the architecture, and the not-heard column is the experiment that demonstrates it.

**Nothing may read the inferred class until the separation gate has a number.**
`audio/clap.py` ships its thresholds with a caveat in its own source — calibrated on offline-convolved grid audio, carried across on an inference — and the one arc that exercised the gate live had it reject 0 of 8, which is also what a gate that discriminates nothing does.
`earshot/tools/clap_gate.sh` is that measurement and it runs before anything else.

## Placement, and the fence around it

Sources sit at one of HM3D ObjectNav's six goal categories, and the sound class is an ESC-50 recording with a plausible anchor there.
Six categories cannot themselves be split heard-from-not-heard, and four of the six are silent objects, so the vocabulary is built at the class level and anchored to the six rather than being the six.

**The sound-object mapping is placement ground truth and the agent must never read it.**
Handing over `anchor_object` turns the unseen-and-heard cell into a measurement of the author's table rather than the agent's semantic store.
ADR-0013's layer graph already forbids `agent/` importing `audio/`, so the naive leak cannot compile; the leak that can happen is the wiring layer passing it in, and `tests/mac/test_audio_vocabulary.py` holds an allowlist of call sites against exactly that.

**Affinity is the membership test, not CLAP separability.**
Coughing and laughing happen on all six categories, so carrying them at face value asks the semantic store to learn noise.
The grades in `audio/vocabulary.py` are declared judgements so the analysis can ask whether the gate agreed with them.

**A prompt that names its own object is not a leak** and is deliberately allowed.
Language already relates a flush to a toilet, and an agent with only the class name is precisely the not-heard column's intended baseline.
What the heard column then measures is what *experience* adds over the language prior, which is a sharper question than whether the agent knows what a toilet is.

## We chose this over

**Placing sources at arbitrary navigable points with rooms as the semantic target.**
Larger vocabulary (~50 ESC-50 classes), and rejected because it needs a room labeller that does not exist here: `NullRoomLabeler` is what runs, `earshot/vlm.py` was never built, and `Detector.CAPTION` raises.
That is a second unproven capability stacked on CLAP, which is itself unproven on this renderer.

**Switching to MP3D for SAVi's 21 categories.**
Rejected by ADR-0007, which keeps MP3D out of scope.
The cost is real and is owned below.

## Consequences

**The semantic transfer has a six-way ceiling.**
Sound → class → one of six objects → search.
That is testable and it is not a rich semantic space, and the paper must name the six rather than let a reader assume more.
It is the price of ADR-0007.

**Both axes run through one prior pass**, so the matrix is a two-visit design and episode cost roughly doubles.
The prior pass is **not designed yet**: how long it runs, whether it is scripted or agent-driven, and what it is allowed to store are all open.

**The seen row is not in the field's protocol.**
SAVi evaluates with "the test environments always unseen".
A seen scene is the only place episodic memory can pay, so the row is the contribution rather than an oversight, but it has to be argued explicitly or a reviewer reads it as evaluating on train.

**Power is unresolved and is the largest open risk.**
The measured per-episode flip rate on byte-identical reruns is 16.2%, the MDE is 15 episodes or 4.1 points at n=365, and single-run-per-arm comparisons are already recorded as not reportable.
Four cells carved out of one sweep gives ~90 per cell.
No matrix run should be launched before this has its own decision.

---

## Amendment, 2026-08-20: the anchor is a ROOM, not an object

**Status:** accepted, on the `clapsmoke-3` separation gate. The two-store design, the class-level heard axis and the inferred goal class all carry unchanged. Only the anchor taxonomy moves.

Sources still sit at an HM3D ObjectNav object, because that is the only thing the episode builder can place against and the only thing the agent can navigate to. What changes is the level the **semantic LTM learns at**: a room, not an object.

### What forced it

The gate scored the object taxonomy at **0.764** anchor top-1 and `plant` at **0.383**, with 187 of its 480 rows landing on `toilet`. Water sounds predict a room with plumbing, and a houseplant is not one. The `plant` assignments were the author reasoning from what could plausibly happen near an object rather than from what a sound predicts, which is the same error as the affinity grades and the third time the gate corrected that table.

Grouping was then scored against the identical rows, nothing re-rendered:

| taxonomy | anchor top-1 | chance | ratio |
|---|---|---|---|
| object (6) | 0.764 | 0.167 | 4.6x |
| room (4) | 0.779 | 0.250 | 3.1x |

**Rooms are not better at classification and this ADR does not claim they are.** +0.015 absolute, and worse against chance. `greenery` scored the identical 0.383 `plant` did, because the map is one-to-one — which is the useful part of the result: the classes were misassigned, not mis-grouped.

### The actual reason

**Splittability.** A heard/not-heard split needs two or more classes at an anchor, or that anchor appears in one column only and the columns end up confounded with object difficulty rather than memory.

- Under objects: `bed` (5) and `toilet` (3) split. `chair` and `plant` hold one each, `sofa` and `tv_monitor` none. **A 2-way semantic space over 8 classes.**
- Under rooms, with affinity re-graded at the room level: **bathroom 4, bedroom 5, living_room 4. Three splittable rooms over 13 classes.**

Merging also helped where it should: `chair` 0.683 and `sofa` 0.792 became 0.830 together, a weighted +0.036, at the cost of pulling `tv_monitor` down from 0.963.

### Consequences

**The 6-way ceiling this ADR declared was never real.** It was 2-way under the taxonomy as shipped. It is now 3-way. That is still a small semantic space and the paper must name the three rooms rather than let a reader assume more.

**`plant` carries no class.** An object with no room cannot host one, and `SoundClass.__post_init__` now raises on it.

**Three classes changed side rather than being deleted.** `chirping_birds`, `crickets` and `rain` moved to `ABSENT_CLASSES`: an outdoor sound heard indoors is precisely a sound with no room, so they are the forced-failure arm's best negatives. The arm goes from five classes to eight.

**Grades were re-derived at the room level and NOT from the gate's recall.** Fitting ground truth to the classifier would make the whole matrix circular. `mouse_click` keeps a moderate grade at 0.017 recall; the separation gate is what cuts it, and the two cuts stay separate counts.

**`clapsmoke-3`'s numbers do not carry forward.** The prompt bank drops from 20 classes to 17, so chance moves 0.050 to 0.059 and the task is easier by construction. The gate must be re-run before any number here is quoted.

**Read per-anchor accuracy with a caveat**: it is inflated for an anchor whose classes confuse each other and deflated for one whose classes scatter. `tv_monitor`'s 0.963 was two mutually-confusable classes landing on their only shared anchor. It is not a clean per-anchor quality score.

---

## Amendment, 2026-08-21: the separation cut is made at the anchor, not at the class

**Status:** accepted, on `clapgate-1` (a partial run: 48 rows per class over roughly half the scenes, see the caveat below). Nothing about the two stores, the class-level heard axis or the inferred goal class moves. Only the prune criterion.

The amendment above made the anchor the level the semantic store learns at, and said plainly that anchor accuracy "is the number the task rests on". The prune did not follow. It kept cutting on class recall, so the two disagreed, and `clapgate-1` made the disagreement expensive.

### What forced it

`pouring_water` measured **0.354 class recall and 1.000 anchor recall**. Every one of its misses landed on another bathroom class, so the agent walked to the right room on every row and the class bar discarded it anyway. That is a quarter of the bathroom vocabulary, cut for a cost the task never pays.

It is not an isolated case. Under the anchor bar `laughing` gains +0.250, `clock_tick` +0.250, `snoring` +0.250 and `brushing_teeth` +0.167.

### The rule

The separation cut reads **anchor recall**. `separation.prune` now takes a required `recall_level` with no default, because the two bars disagree by design and a run has to say which one it read. Asking for the anchor bar on a report summarised without an anchor map raises rather than passing: NOT_RUN is red.

Both bars stay scoreable and both are printed, so the looser one can always be checked against the stricter. Take the class bar if the claim being defended includes the agent NAMING the sound; this design's claim is that it goes to the right room.

**The affinity cut is unchanged and still independent.** `coughing` scored 1.000 at both levels and is still disqualified: people cough in every room, so there is no association for the semantic store to learn. Anchor recall is a looser separation bar, not a looser gate.

### What this does not license

Anchor recall is **inflated for a room whose classes confuse each other**, which the amendment above already records. `pouring_water` is exactly that case. The defence is that the inflation describes a real property of the task rather than an artefact: the agent's action space is rooms, so a confusion the action space cannot express is not an error the agent makes. Any claim about the agent identifying the sound must quote class recall, and both numbers are in `anchor_report_room.json` for that reason.

### Consequences

**Stage 4 of `tools/clap_gate.sh` no longer implements its own prune.** It calls `earshot.tools.anchor_report`, so the live run and a re-score of it apply the same cut by construction. The duplicate had already diverged: the heredoc cut on class recall and skipped the affinity rule entirely, which is how `clapsmoke-3` kept three weak classes.

**`clapgate-1` is a partial run and none of its numbers are the gate's result.** It was killed four minutes in when an SSH session dropped, with the class loop complete and roughly half the scenes covered. It is quoted here for the confusion STRUCTURE it exposed, which is a property of the vocabulary rather than of the scene sample. The bar of record needs the full run.

---

## Amendment, 2026-08-21: the vocabulary is twelve classes, and the open-set arm fails in a shape

**Status:** accepted, on `clapgate-2`. The full gate: 20 val scenes, 16320 in-vocabulary rows, 7680 absent, 26m39s.

### The numbers of record

**Anchor top-1 0.880** over three rooms against 0.333 chance. Class top-1 0.751 against 0.059. Bathroom 0.984, bedroom 0.869, living_room 0.835, with essentially all of the error mass between bedroom and living_room.

**The distance curve is flat.** 0.751 / 0.753 / 0.746 / 0.753 across 0.8 to 8.1 m, with only the margin decaying (+0.1028 to +0.0954). Class inference does not degrade with range on this renderer. `AudioConfig.audible_band_m` is therefore not a constraint on the sounding window, and the design does not need the agent close to the source to know what it is hearing. This was visible on the `clapsmoke-3` half-run and now holds on 16320 rows.

### The vocabulary

**Twelve classes: bathroom 4, bedroom 5, living_room 3.** All three rooms splittable. The anchor bar rescued exactly what the amendment above predicted it would: `pouring_water` 0.510 class against 1.000 anchor, `snoring` 0.573 against 0.840, `clock_tick` 0.711 against 0.964, `laughing` 0.787 against 1.000.

Two classes were cut for separation. `mouse_click` sent 662 of 960 to `clock_tick`, which is a bedroom class, so the anchor bar could not rescue it (0.308). `drinking_sipping` reached 0.442 at the anchor and fails on affinity anyway.

Three were cut for affinity despite passing recall: `coughing` at 1.000, `vacuum_cleaner` at 1.000, `door_wood_creaks` at 1.000 anchor. That is the affinity rule earning its place. All three are sounds a person makes in any room, and a semantic store cannot learn an association that is not there.

**The gate agreed with the affinity table on the strong end and not on the weak end.** Mean anchor recall was 0.913 strong, 0.870 moderate, 0.860 weak. The weak classes are not harder to recognise; they are unusable for a different reason, which is exactly why the two cuts are counted separately.

### The open-set arm, and a hypothesis this ADR got wrong

EER 0.318, up from `clapsmoke-3`'s 0.232. The obvious suspect was the amendment above, which promoted `rain`, `crickets` and `chirping_birds` into `ABSENT_CLASSES`. **That is refuted.** Those three reject at 0.786, 0.815 and 0.480. The hardest negative is `chainsaw` at 0.351, which had been in the absent set from the start.

`AbsentResult.top_match` was added to say why, and the failure has a shape: **each hard negative has a twin in the prompt bank.** A chainsaw is continuous motor noise and so is a `vacuum_cleaner`. That is a different problem from a rule that discriminates nothing, and it has a different fix.

**It may already be fixed.** `vacuum_cleaner` is cut for affinity, so the shipped bank does not contain `chainsaw`'s twin. `separation.restrict_to` re-scores an existing run against the pruned bank without re-rendering, and `anchor_report` now prints that pass.

**The restricted pass is a direction and not a result.** The bank was chosen using those same rows, so any recall re-measured on them is selection on the outcome. `--clip-start 8` stages ESC-50 recordings 8 to 15 against the 0 to 7 a default run uses, which is the only unbiased measurement available without new audio.

### Consequences

**ADR-0018's assumption that the open-set gate could hard-reject is withdrawn.** At EER 0.318 it cannot, and 0.232 was never good enough either. The inferred class must be usable without a reject decision, or the reject needs a lever this gate has not tested.

**The `n_classes` in every number above is 17, the candidate bank.** The system ships 12. Numbers from the two banks are not comparable and must be labelled.

### Confirmed, same day, on the pruned-bank pass

The twin hypothesis is **measured, both arms**, which is what ADR-0014 asks of a detector.

| | candidate bank (17) | pruned bank (12) |
|---|---|---|
| class top-1 | 0.751 | 0.863 |
| anchor top-1 | 0.880 | **0.959** |
| open-set EER | 0.318 | **0.234** |
| `chainsaw` rejected | 0.351 | 0.759 |

`chainsaw` named `vacuum_cleaner` on 568 of its 960 rows. `vacuum_cleaner` is cut for AFFINITY, having no room of its own, and removing it took chainsaw from the worst negative to a middling one. `rain` went 0.786 to 0.926 and `helicopter`, `airplane` and `rain` had all been pointing at the same sink.

**The affinity cut paid twice.** It was justified purely on the semantic store having nothing to learn from a sound that happens in every room. It also removed the class that was absorbing every broadband negative. The two cuts are independent by design, and this is the first evidence that the independence is doing real work rather than being a bookkeeping nicety.

**The EER never regressed.** 0.234 on the shipped bank against `clapsmoke-3`'s 0.232. The 0.318 was an artefact of scoring a bank nothing will run, and the paragraph above that treated it as a degradation was reasoning from the wrong configuration.

**The withdrawal of the hard-reject assumption stands anyway.** 0.234 is roughly one error in four at the balanced point. The inferred class must be usable without a reject decision. What survives is a *soft* signal: `airplane` and `helicopter` sit at mean decision scores of -0.042 and -0.041 against positives well above the +0.172 threshold, so the far tail is separable even though the middle is not.

**The remaining sink is `toilet_flush`.** `chirping_birds` 0.506 and `sea_waves` 0.577 both name it, and `church_bells` names `clock_alarm` on 927 of 960. Those are the negatives to watch if the arm is ever tightened. They are not, on this evidence, a reason to change the vocabulary: all three are absent classes by construction and the task never asks the agent to hear them.

**None of the pruned-bank numbers are unbiased.** The bank was chosen using those rows. `anchor_report --bank <other run>/pruned_vocabulary.json` scores a run against a bank it had no part in picking, and `--clip-start 8` gives ESC-50 recordings 8 to 15 against a default run's 0 to 7. Both are needed; either alone leaves the circularity in place one step along.

---

## Amendment, 2026-08-21: one run's prune is not the vocabulary, and the held-out run proves it

**Status:** accepted, on `clapgate-2` (ESC-50 clips 0-7) and `clapheld-1` (clips 8-15). The bank of record is the INTERSECTION, and `earshot/tools/bank_intersect.py` computes it.

### The aggregate is stable and the per-class numbers are not

| | `clapgate-2` | `clapheld-1` |
|---|---|---|
| anchor top-1, candidate bank | 0.880 | 0.867 |
| open-set EER, candidate bank | 0.318 | 0.318 |
| distance curve | flat | flat |

Two disjoint recording sets, 0.013 apart on the headline and identical on the EER to three decimals. The flat curve reproduced. `chainsaw` named `vacuum_cleaner` on 960 of 960 rows this time, against 568 before, so the twin is not a tendency.

**Per class it falls apart.** `water_drops` measured 0.998 anchor recall on clips 0-7 and **0.449** on clips 8-15. `mouse_click` went 0.308 to 0.789. Bathroom's room accuracy moved 0.984 to 0.799 on the back of one class. The two runs' prunes agree on 11 of 12 classes and disagree on the twelfth in both directions.

**Eight recordings do not pin a class.** That is a fact about ESC-50 rather than about CLAP, and it was invisible until there were two disjoint sets to compare.

### The bank of record

**The intersection: 11 classes, bathroom 3 / bedroom 5 / living_room 3.** All three rooms splittable. Written out here rather than only in `runs/bank_of_record.json`, because `runs/` is gitignored and this is a decision:

| room | classes |
|---|---|
| bathroom | `brushing_teeth`, `pouring_water`, `toilet_flush` |
| bedroom | `breathing`, `clock_alarm`, `clock_tick`, `crying_baby`, `snoring` |
| living_room | `clapping`, `keyboard_typing`, `laughing` |

The weakest cell of the whole table is `brushing_teeth` at 0.747 on clips 8-15. Everything else clears 0.75 in both columns and most clear 0.95.

Each class in it cleared the bar on two recording sets that share no audio, so every class carries a held-out validation rather than the aggregate carrying one. `water_drops` and `mouse_click` are **disputed** and are cut. A disputed class is not a marginal call to settle by judgement: its recall depends on which recordings it drew, which is the one confound the heard/not-heard column cannot survive.

`bank_intersect.py` reads `clip_start` and `n_per_class` from each run's `provenance.txt` and **raises on overlapping ranges**. A class clearing the bar on shared audio is one observation counted twice, and the whole argument for the intersection is that it is not.

### What is still not measured

**No unbiased number exists for the bank of record.** Both input runs helped derive it, so scoring either against it is selection on the outcome, one step removed. The evidence for the bank is the per-class side-by-side table, not a headline. A fresh headline needs `--clip-start 16`.

**The recording-level axis is promoted.** This ADR filed "a different recording of a known class" as a cheap secondary axis on the grounds that it is a CLAP robustness question rather than a memory question. It is still that, but a 0.55 swing on `water_drops` means recording difficulty is large enough to confound the heard column if the two columns draw different clips. The heard/not-heard split must control it: the same recordings on both sides, differing only in whether a prior visit stored them.

### Consequences

**The three open parts of this ADR are unchanged and are now the whole remaining risk.** The prior pass is undesigned, power at roughly 90 episodes per cell against a measured 16.2% flip rate is unresolved, and the seen-scene row still has to be argued against SAVi's always-unseen protocol. CLAP is no longer on that list.

---

## Amendment, 2026-08-21: the prior pass is a scripted tour, and it does not double the cost

**Status:** accepted. `earshot/task/prior_pass.py`, with the pure planner tested on a Mac. This closes the first of the three parts this ADR named as open.

### The decision

**A scripted navmesh tour of the scene's anchor rooms, source sounding, one stop per room.** Not an agent-driven prior episode, and not an oracle write.

**Against agent-driven.** The matrix carries a measured 16.2% per-episode flip rate and roughly 90 episodes a cell. It cannot afford another variance source. An agent-driven pass gives each seen cell whatever coverage that episode happened to achieve, so a null becomes unreadable: memory failed, or the prior pass never entered the room. A fixed route makes coverage identical across scenes by construction.

**Against an oracle write.** It would cost no episodes and be perfectly controlled, and CLAUDE.md forbids it: a capability is exercised, never proxied. The tour runs the real audio sensor and the real encoder, so what a store receives is what the agent could have perceived.

**One stop per room, not per object.** The semantic store learns at the room level, so a second sofa adds nothing it can learn and does make route length a property of the house rather than of the design.

### The cost claim above was wrong

This ADR said the matrix "is a two-visit design and episode cost roughly doubles". It over-counts, and the correction matters for the power question that is still open.

The episodic store is scene-keyed, so **one prior pass serves every test episode in that scene**. The semantic store is scene-agnostic and class-keyed, so **one pass hearing a class serves every test episode of that class anywhere**. Ten seen scenes and six heard classes is tens of prior episodes against a test set in the hundreds: single-digit percent overhead, not 100%.

### What is guarded

**An incomplete tour is not a seen scene.** `TourRecord.complete` is False unless every planned leg arrived, and an abandoned leg is recorded with its step count, its final geodesic gap and a reason. Silently accepting a partial tour would reintroduce exactly the coverage variance scripting exists to remove.

**A leg has a step budget.** A follower oscillating between two navmesh polygons would otherwise hang a sweep, and the run would report nothing rather than reporting a bad tour. Over budget, the leg is abandoned and the tour continues, because throwing away the rooms that did work helps nobody.

**An unroutable stop is a field, not a warning.** 23 of 365 episodes in the anomaly-response sweep had no navmesh route to their source and nothing counted them until a tool did. `TourPlan.unreachable` carries every dropped candidate with its reason.

**An observation is only taken at a stop the agent REACHED.** An observation from a stop it never arrived at is fabricated audio, which is the failure that invalidated the whole `anommxv` arc.

### What this does not do

**Nothing here writes to a memory store.** The scene-agnostic semantic store is new code this ADR commits to and it does not exist. `walk_tour` takes an `observe` callback and returns a `TourRecord`; that is the seam a store will attach to. Building the tour first means it can be exercised and measured before anything depends on it.

**The tour is an upper bound on episodic memory quality.** A complete route through every anchor room is the best prior map the design can give. That is the right first arm: if memory does not pay with a complete map, it will not pay with a partial one. An agent-driven pass becomes a realism ablation on top of a result rather than the thing the result rests on, and the paper must say so.

---

## Amendment, 2026-08-21: 200 episodes a cell, and this is a PRE-REGISTRATION

**Status:** accepted. Closes the second of the three parts this ADR named as open. `earshot/tools/power.py` computes every number below; none of it is hand arithmetic.

**Written before any matrix episode has run.** This project's history is one of post-hoc reinterpretation, so the analysis is fixed here and dated.

### The size

**200 episodes per cell, 800 total, about six hours at the measured 27 s/episode.**

| n/cell | total | wall | SD(diff) | 2 sigma | MDE @80% |
|---|---|---|---|---|---|
| 90 | 360 | 2.7 h | 7.5 pt | 14.9 pt | **20.9 pt** |
| **200** | **800** | **6.0 h** | **5.0 pt** | 10.0 pt | **14.0 pt** |
| 400 | 1600 | 12.0 h | 3.5 pt | 7.1 pt | 9.9 pt |

The M3 revisit headline this project measured was **+17.1 points**, which clears the 14.0 bar at 200/cell. 135/cell would clear it exactly; 200 buys margin. The 90/cell figure this ADR had been arguing from would have **missed** it.

### Two corrections that changed the arithmetic

**"2 sigma" is not an MDE.** The first table put in front of this decision used a 2.0 multiplier. An effect sitting exactly at 2 sigma is detected half the time. At 80% power the multiplier is 2.80, so that table understated every requirement by about 40%. Both columns are printed side by side now so the gap cannot be silently reintroduced.

**The paired formula does not apply here.** "MDE = 15 episodes = 4.1 points at n=365" comes from `SD = sqrt(flip_rate * n)`, valid because `episode_diff` compares the SAME episode in two arms. Cells of the 2x2 share no episodes. At equal total rendering cost the paired design is **1.76x** more sensitive: 5.9 points against 10.4. That ratio was itself asserted as "roughly three times" before anything computed it, which is the same error one level down, and both are now pinned by tests.

### The pre-registered analysis

**Primary contrast:** seen-heard against unseen-not-heard. Both memories against neither.
**Secondary:** the two main effects, seen-vs-unseen and heard-vs-not-heard.
**Both tests are reported, always** (the `funnel_diff` rule): the episode-level comparison AND the scene-level sign test. Neither alone is the result.
**Base rate assumed 0.5**, the worst case. An MDE computed there cannot be flattered by a lucky base rate.

### The limitation episodes cannot buy

Episodes inside a scene share a room, a source and a renderer, so they are not independent. The scene-level sign test needs:

| scenes | must agree |
|---|---|
| 10 | 9 of 10 |
| 20 | 15 of 20 |
| 40 | 27 of 40 |
| 5 | **impossible at any outcome** |

At ten scenes a side the scene test needs a near-sweep. **No episode count repairs this**, and if `room_yield --split train` shows HM3D train episodes on the box the scene pool should be widened before the matrix runs. If it does not, the seen/unseen axis is bounded at ten scenes each and the paper states that rather than burying it.
