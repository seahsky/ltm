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
