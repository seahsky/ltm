# Live in-sim audio uses one positioned source and an unrendered bed

**Status:** accepted (2026-08-04, grilling session on ticket 09 of the `ss2-clean-room` map).
**Supersedes ADR-0004** (the background bed is a noise floor), whose conclusion stands and whose central argument is retired.

Under SoundSpaces 2.0 rendered live in the simulator at every step, the clean room renders **exactly one positioned source**: the anomaly.
The background bed is a fixed-level diotic signal generated directly, and it never touches the RIR.
habitat-sim is built stock, with no multi-source patch and no fork.

## Why one source is enough

Ticket 02 established that habitat-sim exposes one source (hardcoded index 0) while the engine underneath is natively multi-source with one IR per source, reachable by a ~40-line wrapper patch.
Ticket 06 then measured that the patch is **not budget-gated**: one render at the `cheap_preset` is 27.2 ms, so three sequential renders is roughly 82 ms/step, still inside budget.
So the choice was free, and it is decided on what the task needs rather than on cost.

The task needs one source at a time, for two reasons neither ticket stated.

**ADR-0002's room-normal distractor is not simultaneous.**
Its design is a same-sound / two-rooms A/B *across episodes*: one episode places the sound where it is room-normal and the agent must not interrupt, another where it is room-anomalous and it must.
ADR-0002 says so explicitly, and records that it rejected a simultaneous two-source distractor to preserve the O(1) invariant.
The distractor needs a position, but never at the same instant as the anomaly.

**A bed is diotic by definition.**
`CONTEXT.md` defines the background bed as the continuous diotic noise floor.
Rendering it from a position through the RIR is what it is not, and it is what made ADR-0004's threshold unmeetable.

## Why the unrendered bed retires ADR-0004's argument

ADR-0004 concluded that **no absolute RMS threshold can work**: before `t_anom` the signal is `bed(x)`, after it is `bed(x)+alarm(x)`, roughly a 1.4x temporal step, while position swings energy nearly 8x across the grid.
It rejected a temporal step detector as the technically correct fix, unbuilt and confounded by agent motion.

That argument depends entirely on the bed being rendered.
With a bed that never touches the RIR, it is position-invariant, and the anomaly contributes exactly zero before `t_anom` because it is convolved with silence.
So the pre-onset signal is **flat at the bed level, at every pose, in every scene**.
The 8x spatial swing lives entirely in the post-onset term.
An absolute threshold is well-founded for the first time, and the step detector stays unbuilt because it is no longer needed.

ADR-0004's *conclusion* is unchanged and is now stronger: the bed is a noise floor and never the interrupt trigger.
What changes is that this is structural rather than calibrated.
`bg_gain` is retired outright rather than recalibrated, because there is nothing left to calibrate against.

A second property falls out for free.
`onset_step < t_anom` becomes **structurally impossible**, which is exactly the `anommxv` break where the interrupt fired on the bed at step 0 and was attributed to the alarm.
Ticket 02 argued the multi-source patch was the way to make onset provenance structural; the unrendered bed achieves it without the patch.

## We chose this over

**Patching the wrapper for real multi-source (ticket 02's option 1).**
It buys per-source IRs at zero extra renders, and `RLRA_SetSourceRadius` for a bed that is genuinely a region.
Rejected because the task needs one source at a time, so the patch would buy a property nothing consumes, at the price of owning a habitat-sim fork with a maintenance cost and a reproducibility story.

**Sequential re-render (ticket 02's option 2).**
Also yields per-source IRs if the IRs are kept unsummed, at roughly 82 ms/step, which ticket 06 measured as affordable.
Rejected for the same reason: it solves a problem the task does not have, at 3x the render cost.

## Consequences

**No habitat-sim fork.**
The clean room builds stock from the `RLRAudioPropagationUpdate` branch at the SHA ticket 17 pins.
This also closes ticket 12's second patch candidate, binding the RLRA error channel to Python, which ticket 16 had already downgraded after measuring that `RLRA_SetListenerHRTF` returns `Success` over a failed load.
That patch was only ever worth taking if the multi-source patch was taken anyway, and it is not.

**`RLRA_SetSourceRadius` goes unused**, so a bed that is a region rather than a point is off the table.
This costs nothing, because the bed has no position at all under this decision.

**Making the anomaly and a benign sound audible at the same instant is a rebuild, not a flag.**
If a later experiment needs simultaneous sources, ticket 02 costed the patch and ticket 06 costed the renders, so the path is documented, but it is not a configuration change.

**The onset threshold is now derivable by measurement**, which ADR-0004 explicitly said it was not.
The calibration procedure is in `docs/anomaly_response_task_spec.md`.
