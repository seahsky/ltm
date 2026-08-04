# Realizable localization on live audio: an agent-frame lateral sign and a measured ceiling

**Status:** accepted (2026-08-04, grilling session on ticket 09 of the `ss2-clean-room` map).
**Supersedes ADR-0001** (realizable anomaly-source localization via energy-gradient climb), whose decision stands and whose stated ceiling and frame convention do not.

Realizable localization stays: the agent reaches the anomaly source from live binaural RMS, the inter-aural level sign, and visual confirmation, never from an oracle source coordinate.
It is now the **default arm and the one the smoke runs**, with the oracle arm retained as a bisection tool.
Three things change under live rendering, and the controller's decision rule is not one of them.

## The frame convention inverts, and the code does not change

This is the most dangerous item in this ADR, because nothing about it is visible at the call site.

`lateral_sign` (`audio.py:595`) is a pure sign on the interaural level difference: it reports which ear is louder and nothing else.
Its *meaning* comes from how the IR was rendered.
The precomputed grid rendered at **identity listener yaw**, so the cue was a **world-frame** bearing, and the audio-visual fusion arc calibrated against that, landing on `heard == -right(world-bearing)`.

Under live rendering the listener transform is the agent's actual pose, so the same function returns an **agent-frame** cue by construction.
Carried across verbatim with the old compensation, the controller turns the wrong way on every stall.
It would not crash and it would not look like a bug; it would look like a mediocre climb.

The convention is therefore **pinned by a test the tree owns**, not by a calibration run and not by a comment.

## The ~1 m ceiling was a property of the grid

ADR-0001 stated the localization ceiling as "roughly one RIR grid cell (~1 m)" and recorded a measured spearman of about −0.45 between energy and distance.
Both numbers are grid artefacts.
Ticket 06 measured the live gradient at rho −0.98/−0.99, and −0.95/−0.98 on the non-line-of-sight portions, on same-floor navmesh walks.
ADR-0010 keeps the task same-floor, so that measurement is in-regime rather than extrapolated.

The gradient is not marginally better than the grid's, it is near-monotone.
So the ceiling stops being asserted and becomes **measured**: the reported quantity is the distribution of distance-at-STOP to the source, and "reach within ~1 m is the honest metric" is replaced by whatever that distribution says.

## Rotation now moves the gradient, and we instrument rather than fix

With a real listener transform, `turn_left` changes measured RMS without changing distance.
So `realizable_investigate_step`'s `rising` test conflates "closer" with "better oriented".
On the grid this could not happen, because every cell was rendered at one yaw.

We do not amend the rule for it.
The per-step record carries measured RMS alongside the action taken, which makes a rotation-driven rise distinguishable from a translation-driven one after the fact.
Changing the rule first would be changing the one module ADR-0008 describes as the paper's single framing-independent positive, on a hypothesis rather than on evidence.

## We chose this over

**Porting verbatim with no amendment**, which ticket 07 deliberately left open rather than deciding.
Rejected because the frame inversion is silent and would corrupt every stall recovery.

**Reworking the climb rule for rotation now**, for example by only comparing energy across translations, or by adding an explicit scan-then-advance phase.
Rejected as speculative: the conflation is real but unmeasured, and this map's record is a long series of levers that looked necessary and measured inert.
The instrumentation makes the amendment decidable later on data.

**Running the oracle arm in the smoke.**
It is far more likely to go green.
Rejected because it leaves the entire live-audio path unexercised in the one episode that exists to prove it, which is the "the sound is just a stopwatch" objection ADR-0001 was written against.

## Consequences

**Arrival stays agent-estimable.**
STOP fires on peak-or-plateau plus visual confirm, unchanged.
The oracle `investigate_arrive_radius_m` of 1.5 m is not an arrival criterion in this arm; it survives only in the oracle arm.

**`sourceIsVisible()` is recorded and never read by the controller.**
It is computed from the ground-truth source position, so feeding it to the decision rule would plant a hidden oracle inside the arm built to avoid one.
This line is stated in the spec because it is the kind of thing crossed by accident later.

**The report cannot contain the source coordinate.**
See ADR-0009's companion decision in `docs/anomaly_response_task_spec.md`: the agent's testimony is constructible from agent-estimable signals only, in both arms, so whether an arm is realizable is checkable by reading the schema.

**The oracle arm survives as a flag**, which ADR-0008 already licensed as one of the two genuine experimental arms.
Its first job is diagnostic: if the smoke fails, running it isolates audio from controller in one step.
