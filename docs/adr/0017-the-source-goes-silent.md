# The source goes silent, because a sound that never stops leaves memory nothing to do

**Status:** accepted (2026-08-20, from the grilling session on the direction change).
**Retires the anomaly-response loop** as the task: the interrupt, the resume, the CLAP normality gate and the room-normal distractor arm all stop being the thing under test.
**The task spec** `docs/anomaly_response_task_spec.md` is superseded on §1, §4.2 and §4.3; §2, §3 and §5 carry almost unchanged.
**ADR-0009, 0010, 0011, 0015 and 0016 all carry.** The renderer, the single positioned source, the unrendered bed, the same-floor builder policy, the realizable climb and the cast are untouched.

One episode is now: a source sounds for a bounded **sounding window**, goes silent, and the agent must reach it.
The primary find-task is the sound source itself.
There is no interrupt, because there is nothing to interrupt.

## What forced it

The direction changed to sound-source finding with a memory stack, and the first question a memory has to answer is what it is for.

Against a source that sounds continuously, the answer is nothing.
`arrive-2` reached 40.3% of 365 episodes and `yield-2` reached 46.0%, both with **no memory in the tree at all** — the live binaural climb plus the cast does that on its own.
A memory arm added to a continuously-sounding task would be measured against a baseline that already has the cue, in a task where the cue is sufficient.
That is not a hard experiment, it is an unmeasurable one: every cell of the intended generalization matrix would be solved the same way, by climbing.

Silence is what creates the question.
After the offset step there is no gradient, so what the agent does next is entirely a function of what it knows — this scene's layout, or what that class of sound usually comes from.
Those are the two memories ADR-0018 builds, and the silent phase is the only place either of them can pay.

## Why the field's convention rather than ours

SAVi (Chen et al., CVPR 2021) and SAVN-CE (CVPR 2026) both draw an onset and a duration per episode; SAVN-CE uses onset uniform on [0, 5] s and duration Gaussian with mean 15 s.
Adopting it costs nothing and buys **SWS**, "success when silent" — the published fraction of episodes completed after the acoustic event ended.
That is a cross-quotable metric measuring exactly what this decision creates, and re-deriving it under a local name would forfeit the comparison for no gain.

## We chose this over

**A continuously sounding source (AudioGoal / AV-Nav).**
Rejected on the argument above: the existing climb already solves it at 40–46% with no memory, so the matrix has no contrast to show.

**Keeping the ObjectNav primary and treating the sound as a cue.**
Rejected because it leaves two success criteria in one episode.
Every delta then splits across a find and a response, which is the ambiguity `CONTEXT.md` already warns about under **Mission** — "mission complete is not a single number" — and it would put the memory effect on the wrong side of it.

## What it costs, stated rather than discovered later

**Built work is retired.** The interrupt, the resume, the once-per-episode detour budget, the CLAP normality gate and the never-built distractor arm stop being the subject.
The controller's climb and cast survive; the state machine around them does not.

**The renderer needs a reverb tail.** `clips.render_through_ir` convolves the whole clip through the current pose's IR every step, which models a source sustaining forever and is fine while `playing` is monotone.
Cut to a window and the silence arrives as a hard step to the bed with no tail, which is unphysical.
SAVN-CE names this as SoundSpaces 2.0's own shortcoming and fixes it with a preallocated accumulation buffer that adds each step's convolution into a running signal.
About 50 lines, pure NumPy, no Habitat coupling, CC-BY-4.0 and portable by hand.
**This is not yet built** and no sounding-window run may report an SWS before it is.

**The cast becomes the default, not the exception.** ADR-0016's cast fires when the rising predicate reads false.
After the offset step the field collapses to the bed, so the predicate reads false at every remaining step by construction.
Everything ADR-0016 measured about plateau windows was measured on a live cue; none of it transfers to a silent one, and the cast's behaviour in silence is unmeasured.

**"Onset" now means two things.** In this tree `onset_step` is the step the AGENT's RMS crossed threshold; in the literature "onset" is the step the source started.
`CONTEXT.md` carries the tree's meaning and says "sounding window opens" for the other.
The collision is recorded rather than resolved because renaming `onset_step` would invalidate every audit record on disk.
