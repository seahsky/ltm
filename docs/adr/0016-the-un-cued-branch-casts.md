# The un-cued branch casts, because the noise it replaced was the exploration

**Status:** accepted (2026-08-08, on the `eps-1` regression), **and its own falsification criterion has since fired** — see the amendment at the foot of this file. The decision stands; the claim made for it was too strong.
**Amends ADR-0011's greedy climb**, whose surge and whose arrival rule carry unchanged. Only the third branch moves.

When the cue is dead and the agent has not arrived, `realizable_investigate_step` no longer answers "turn".
It **scans** for `SCAN_STEPS` turns, then **casts**: one turn, `CAST_STEPS` forwards, repeat, with the legs alternating direction.
Any rise surges immediately and resets the cycle; `visual_confirm` with a dead cue still STOPs, wherever in the cycle the agent stands.

## What forced it

`eps-1` calibrated the climb's threshold — `current > previous + 1e-6` against a renderer scattering 2.8e-3, replaced by a windowed test against the renderer's own measured noise. The estimator got strictly better and **Anomaly-response SR fell from 46.0% to 32.9%**: −48 of 365 paired episodes, 15 of 16 scenes down, sign test p = 0.0005.

The old rule had no branch that advanced a plateaued agent. It survived that on an accident: a threshold three thousand times under the noise it was read through is a coin flip on flat ground, and P(forward) ≈ ½ is a random walk that covers distance. Calibrating it took P(forward) to ≈ ⅒ and the agent stopped moving — 85% of abandoned detour steps plateaued, 87 of 106 abandoned windows `static` with no translation at all.

**The noise was doing the exploring, and a better threshold cannot give that back.** `eps-1` also measured why: one 0.25 m forward buys 0.61–0.86 of the field's own local scatter in every band inside 5 m, so a correctly calibrated single-step rule fires rarely *by construction*. The un-cued case needs a policy.

## The three parts, and why each is there

**Scan before casting.** A mis-oriented agent is not a lost one — facing away from a live source also reads as a dead cue, and consecutive turns fix it because the lateral sign homes onto the bearing in a turn or three. `SCAN_STEPS = 6` is one full sweep at `investigate_probe_turn_deg` of 60°. Only once the cue has stayed dead through a whole revolution is orientation ruled out. This was not in the first version and `TestTheStallTurnsTowardTheSource` failed loudly: the cast interrupted the recovery after one turn and walked the agent away from a source it was about to find.

**Commit a leg.** `CAST_STEPS = 8` is one probe's reach — `investigate_probe_m` 2.0 m over a 0.25 m step. The rule already names a place 2 m out; the agent now walks to it instead of re-deriving a fresh 60° offset every 0.25 m, which is what made the plateau branch an arc.

**Alternate the legs.** Following a stable lateral sign on every leg traces a closed polygon, and a closed polygon around a source is an orbit — which produces exactly the flat field and stable sign it would be orbiting in, so the shape sustains itself. Alternating cannot close: the heading oscillates between two values one turn apart and the sweep drifts along their bisector. The *first* leg still follows the sign, because that is the cue's last useful word.

## We chose this over

**A smaller epsilon.** Rejected on measurement: at 0.61–0.86 of local scatter per step, there is no threshold at which a single-step comparison both fires on the cue and refuses the noise.

**Restoring the coin flip** by loosening the threshold back toward `1e-6`. It is a known 46%, which is better than what we have. Rejected because it is exploration by accident: nothing in the code says it is exploring, the rate is a side effect of an arithmetic error, and it would be re-broken by the next honest fix to the estimator. The same behaviour named and controlled is worth more than a higher number obtained by leaving a bug in.

**Randomising the un-cued action.** Simplest, and it reproduces the coin flip deliberately. Rejected because a random walk has no memory of the directions already swept, and because a stochastic controller makes every future A/B need repeats — this project already has a renderer it cannot seed.

## Consequences

**`ControllerState` carries a plateau counter, and the rule takes it as an argument.** The counter cannot be derived inside the rule: the runner trims `energy_history` to what `is_rising` reads, and a cast leg is longer than that window, so a rule counting its own plateau would restart the cycle at every trim. A replay has the whole series and derives it (`detour_report.plateau_index`), which is what keeps the recorded action checkable against the recomputed one.

**`cast_steps = 0` is exactly the carried rule**, alternation included — that branch is conditional on there being legs — so the control arm is available without a second controller.

**Two invented numbers**, both derived from existing constants rather than chosen, and both module-level. A sweep over leg length is a code edit, deliberately: there is no configuration surface for the controller and ADR-0008 removed the one there was.

**The next result must be read as a three-way.** `yield-2` (46.0%, pre-fix) and `eps-1` (32.9%, post-fix, no cast) are both on disk, so one sweep with the cast completes the comparison. A cast arm that lands between them means the policy is worse than the accident it replaced.

---

## Amendment (2026-08-08, `cast-1`): the criterion above fired

`cast-1` landed **between them**: 134/365 = 36.7%, against 32.9% for `eps-1` and 46.0% for `yield-2`. The sentence directly above is recorded as met rather than reinterpreted.

**What survives.** The cast is a real effect and the exploration diagnosis was right. Isolated against `eps-1` — the only comparison that changes one thing — it recovered **+14 episodes** across 13 of the 16 scenes that moved (sign test p = 0.021), while paying for a *stricter* rising bar than `eps-1` ran. Forwards per episode rose ~45% with displacement to match.

**What does not.** The combination is 34 episodes below the accident (−4.0σ). Since the cast's own contribution is positive, the estimator half carries the whole deficit, and the losses are concentrated in scenes that scored 13/20 under the coin flip. **46.0% remains the best number measured, and it is still produced by an arithmetic error.**

**What this does to the rejections above.** "Restore the coin flip" and "randomise the un-cued action" were rejected on principle — exploration by accident is re-broken by the next honest fix, and a stochastic controller makes every future A/B need repeats against a renderer with no seed. Both objections stand on their own terms. They now have a price: **9.3 points**. That trade is explicit here rather than resolved by assertion, and if a later run cannot close the gap by loosening the rising test, the price is what the choice should be argued against.

**The decision is not reversed**, because nothing measured says the cast is the problem and one thing measured says it is part of the answer. The next arm holds the cast fixed and moves the threshold (`RISING_SIGMAS = 0`), which is the only remaining confound between this controller and the bug it is losing to.
