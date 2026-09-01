# The anomaly source is kept away from the agent's start, and every yield before it is dead

**Status:** accepted (2026-08-08, grilling session on the `eps-1` sweep).
**Renumbered from ADR-0015 to ADR-0020 on 2026-09-01.**
Two accepted ADRs had both claimed the number 0015; the other keeps it (`0015-mp3d-on-the-box-for-a-reproduced-reference.md`).
Only the number and the filename moved.
Nothing below changed, and no content here ever referred to its own former number.
**Mirrors the builder's xz decoupling rule** (`min_sep_m = 3.0`), which kept the source away from the *goal* and left the *agent* unguarded.

The dataset builder requires the anomaly source to clear `MIN_SOURCE_START_SEP_M = 2.0` m in `xz` from the episode start, alongside the existing separation from every primary goal view point.
An episode that cannot satisfy both is skipped with a reason rather than run.
The rule costs yield — the 20-scene sweep went from 41% to 36% — and that cost is the point.

## What it fixes

`detour-1` placed a source **0.75 m** from the agent's start.
The anomaly sounded inside the arrival radius before the agent had moved: INVESTIGATE at step 5, RESUME at step 7, two steps of detour, counted as a completed anomaly response.
One of that run's eight successes was a source already at the agent's feet, which makes 8/20 read as 7/20 honestly.

Below the separation bar there is no detour to measure.
The episode is a null dressed as a pass, which is the exact failure mode ADR-0014 exists to refuse.

## Why 2.0 m

The requirement is structural: the source must start **outside the arrival ring**, `DetectorConfig.oracle_radius_m = 1.0` m, or the loop can close without an approach.
`detour-1`'s `d_min` distribution then confirms the choice empirically — the eight reached detours ended at 0.31–0.78 m, the twelve abandoned plateaued at 2.06–9.26 m, and 2.0 sits in the empty band between the ring and the nearest plateau.

That empirical gap is weaker evidence than it first looks, and the ADR records why so nobody re-derives the constant from it alone.
Arrival *requires* `visual_confirm`, which is the oracle detector at 1.0 m, so every reached episode has `d_min ≤ 1.0` m **by construction** — the lower half of the gap is the metric drawing itself.
The load-bearing fact is the structural one: 2.0 > 1.0, with the observed 1.0–2.06 m emptiness as corroboration.

## We chose this over

**Measuring separation against the agent's pose at `t_anom`** rather than at the start.
Rejected because the builder runs before anything is simulated and that pose is not knowable there.
The start is a lower bound in the only direction that matters: an agent that walked away from the source has a real detour either way, and one that walked toward it was heading there anyway.

**Keeping the rule and adjusting the old yields down.**
Rejected — every yield measured before this rule is an overestimate of a *different denominator*, not the same denominator scaled.
They have to be re-measured, and they were: `yield-2` is 365 built / 651 skipped over 20 scenes.

## Consequences

**Every yield measured before this rule is dead**, including the 41% figure. The denominator is `yield-2`'s 36% and nothing earlier is quotable.

**Anomaly-response SR is not comparable across the rule.** `detour-1`'s 8/20 and any post-rule number count different episode populations, so the two cannot be subtracted. A pre/post claim needs both arms built under the same builder.

**Moving `oracle_radius_m` moves this constant.** The two are coupled through the structural requirement above, and widening the ring to raise Anomaly-response SR would both relax the metric and invalidate the placement rule that keeps it honest.

**The rejection counters cannot show what this rule costs on episodes that still built.** `place_anomaly_source` reports its per-rule counts only in the `PlacementError` raised when nothing qualifies, so a scene showing no rejections may still have had a near-start candidate rejected and a farther source substituted. The yield report is blind to that substitution.
