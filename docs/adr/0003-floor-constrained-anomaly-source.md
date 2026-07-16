# The anomaly source must sit on the primary goal's floor

**Status:** accepted (2026-07-16, grilling session)

`pick_anomaly_source` selected the decoupled source by **xz** separation only (`>= min_sep_m`, reachability guarded by "proximity correlates with the same navmesh component"), and `RIRGrid.nearest` resolves a listener to a cell by **xz** distance with y explicitly ignored.
In a multi-floor HM3D scene these two facts compose into fabricated audio: the picker actively *prefers* a cross-floor source (it is xz-near while still clearing the separation bar), the RIR grid is rendered on the source's floor only, and any listener on another floor is silently snapped to a ground-floor cell and handed an impulse response for a position three metres of concrete away.
We decided the source must be **on the goal's floor** (`|source_y - goal_y| < ~1.0 m`, checked in the builder from view_point positions at zero cost, so the two-env split holds), and that `RIRGrid.nearest` must **refuse to snap across floors** rather than fabricate.

This was measured, not theorised.
In `runs/anomresp-bed-s{1,3}` (TEEsavR23oF, primary=bed at y≈3.16, source=chair at y≈0.16, all 24 grid cells at y=1.66) the feasibility gate's verdicts came out exactly inverted: the one warm start on the source's floor read `d2cell=6.76m -> OUT_OF_COVERAGE` and was rejected, while the two starts a floor above read `d2cell=1.68m` and `0.27m -> AUDIBLE` and were greenlit.
Downstream, the two greenlit episodes aborted the investigate on the step budget (the source is a staircase away), and the only episode that completed a full investigate+resume was the one on the source's floor, whose primary find-task then failed at 349 steps.
The "full investigate+resume" set and the "primary completed" set were disjoint for a structural reason.

We chose this over rendering the RIR grid on every floor, which would preserve cross-floor sources but multiplies render cost on the axis that is already the yield bottleneck, requires real 3-D cell resolution, and turns the detour into a stair-climb that neither the controller nor the ~100-step investigate budget was designed for.

**Consequences.**
Yield drops before it improves: cross-floor sources are currently passing the picker, and cells whose only decoupling candidate is on another floor will now SKIP, worsening the already-poor 16/97.
Every `anomresp-*` and `anommxv-*` dataset and grid is superseded and must be rebuilt; the n=64 controller census cannot be salvaged.
The `nearest` floor guard is added as an opt-in `max_dy` parameter defaulting to the current behaviour, so audiogoal / revisit / objectnav paths stay byte-identical and the prior arc stays cross-quotable.
Those prior results are unaffected on their merits regardless: in the audiogoal and scale-up matrices the audio was decorative (anomaly == goal, retrieval visual), so fabricated cross-floor audio could not have moved their soft-SPL.
The bug only bites where audio is load-bearing, which is anomaly-response and nothing else.
