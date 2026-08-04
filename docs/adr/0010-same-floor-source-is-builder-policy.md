# The same-floor anomaly source is builder policy, not an audio-fidelity guard

**Status:** accepted (2026-08-04, grilling session on ticket 09 of the `ss2-clean-room` map).
**Supersedes ADR-0003** (the anomaly source must sit on the primary goal's floor), whose rule survives and whose justification does not.

The anomaly source stays on the primary goal's floor (`|Δy| < ~1.0 m`), checked in the dataset builder.
It is kept as a **task-scope policy** about what the anomaly response is, not as a guard against fabricated audio.
The `RIRGrid.nearest` cross-floor refusal is deleted.

## What live rendering killed

ADR-0003 was a fabrication argument.
The grid was rendered on the source's floor only, `nearest` resolved a listener to a cell by xz with y explicitly ignored, and `pick_anomaly_source` actively *preferred* a cross-floor source because it is xz-near while still clearing the separation bar.
A listener on another floor was silently snapped to a ground-floor cell and handed an impulse response for a position three metres of concrete away.

There is no grid and no `nearest` under live rendering, so this failure mode has no analogue.
Note what that does to the ADR's second requirement, that `nearest` must refuse to snap across floors: it is not retired, it is **unimplementable**, because the thing it guarded does not exist.
With `transmission` measured ON by default (ticket 04), an off-floor source now produces real attenuated audio rather than fabricated audio.

## What live rendering did not touch

ADR-0003's other stated reason survives intact: a cross-floor source "turns the detour into a stair-climb that neither the controller nor the ~100-step investigate budget was designed for".
That is a statement about the controller, not the renderer.

The controller is a greedy energy climb over `move_forward` / `turn_left` / `turn_right` (ADR-0001, ported near-verbatim by ADR-0008).
Across floors it fails in a specific and ugly way: energy rises toward a source through the ceiling while no navmesh path goes that way, so the climb walks into a wall and burns the sub-budget.
The failure looks like a mediocre controller rather than an out-of-scope task.

Ticket 06's gradient measurement does not license relaxing this.
Its rho of −0.98/−0.99, and −0.95/−0.98 on the non-line-of-sight portions, was taken on **same-floor navmesh walks** where direct and transmitted paths dominate.
Ticket 06 said so itself and warned against reading it as evidence that diffraction is dispensable.
Around a corner or a floor away is exactly the untested regime, and it is exactly the regime this constraint excludes.

## We chose this over

**Retiring the constraint outright** and raising the investigate budget.
It tests something more interesting, namely real cross-floor acoustic navigation, and live rendering makes the audio honest there for the first time.
Rejected because it is a capability the ported controller does not have, measured in a regime nothing has measured, and the map's destination is one green episode.

**Keeping ADR-0003 as written.**
Rejected because its justification would then be false, and a constraint kept for a reason that no longer holds is the accretion this rebuild exists to undo.
It also matters practically: someone reading "prevents fabricated audio" would reasonably retire the rule on learning there is no grid.

## Consequences

**The rule moves from invariant to builder policy.**
It lives in the dataset builder as a placement check and nowhere else.
There is no runtime guard, because there is no longer a silent-fabrication failure mode to guard against.

**Relaxing it is gated on a measurement, not on a budget.**
ADR-0003 traded the constraint against render cost, and that trade is gone.
The new gate is measuring the energy gradient in the non-LOS and cross-floor regime, which ticket 06 explicitly did not measure.

**The xz separation rule that decouples source from goal is unaffected** and carries unchanged.

**`sourceIsVisible()` becomes the diagnostic for this boundary.**
It is recorded per step as an analyst annotation (see `docs/anomaly_response_task_spec.md`), and it is what would show a stalled climb reaching toward an unreachable source if the policy were ever relaxed.
