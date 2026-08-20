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
