# R1 shows a weak backbone and an inert semantic frontier; retreat to controller-systems + honest-negative

**Status:** accepted (2026-07-20, grilling session over the `r1spin2` val_mini smoke).

R1's val_mini de-risk smoke (2 scenes, 30 episodes, memory OFF, 7B planner, anti-spin ON, 500-step budget) measured native benchmark SPL ≈ 0.031 (S1) / 0.014 (S1+) against VLFM's 0.304, and the BLIP-2 ITM semantic frontier did **not** beat the geometric frontier (paired SPL −0.0175, soft-SPL +0.010 n.s., succ@1m identical).
We decided the "competitive absolute numbers" ambition added on 2026-07-16 is falsified in expectation and we abandon it, retreating to the stable contribution: the anomaly-response **controller as a working system** plus the **LTM navigation-memory honest negative**, on a deliberately-frozen, honestly-weak backbone whose absolute capability is out of scope.
We also rule out spending on the one measured bottleneck (STOP-localization), because every realizable proxy in the arc went net-zero and the only real fix, an end-to-end ObjectNav policy, emits one action with no candidate pool and would destroy the memory-injection thesis.

## Why the smoke is decisive

The result is a *capability* finding, not an artifact: it ran at the full 500-step benchmark budget with anti-spin ON, so the weakness is the searcher, not a timeout.
Going from 2 to 20 scenes changes the sample, not the per-episode capability (same captioner, planner, follower, and caption-grounding STOP), so full val will not close a 10x gap.
BLIP-2 is the **4th independent non-lift** of a semantic frontier here (CLIP HOLD ×3, BLIP-2-on-Phi flat, now BLIP-2-on-7B flat-to-negative), and the vacuous-arm gate was GREEN (13,405 scores, spread 0.45), so this is "the lever is inert," not "the lever never fired."

## Considered options

- **Keep S1+/S3+ as the headline (the 2026-07-16 plan).** Rejected: the "+" was justified as "characterize the memory delta against a *strong* baseline," and the baseline is not strong.
- **Spend on STOP-localization** (the measured bottleneck; Run-12 oracle-STOP lifts warm succ@0.1m 0.167 → 0.750). Rejected: every realizable proxy failed (arrival-stop, caption-grounding, the L3 detector snaps to the object floor not the view_point), and a real policy backbone destroys the candidate-pool thesis. High cost, sub-even odds, risks the one thing that works.
- **Lead the paper with the negative result** rather than the controller. Rejected: ICRA rewards systems; the controller is the only framing-independent positive.

## Consequences

Full-val R1 still runs, reframed from "prove competitive" to "the honest Table-1 baseline the paper needs"; it keeps **S1 vs S1+** so the BLIP-2 negative is measured with power.
The paper's spine is the **geometric** backbone (S1 / S3), the headline memory delta reverts to **S3 − S1**, and **R2 drops the "+" arms** and runs S1 vs S3 — removing the BLIP-2 VRAM knife-edge on the V100 and keeping cross-quotability to the whole prior arc.
S1+ is retained only as a documented negative: "no cheap semantic explorer, including VLFM's own ITM head, beats a geometric frontier on this renderer," which is the rebuttal to the "re-run S1 with a real explorer and report the delta against *that*" reviewer attack.
The paper **leads with the controller as a system**, so **R2** (re-earning the ADR-0003/0004-invalidated n=64 census on Phase-F-fixed data) is now the single most important run in the project; the LTM negative is done.
A **bounded, diagnose-first spin fix** lands in the frozen R1/R2 backbone before the full-val runs (spin is a fairness and R2-integrity issue, not the ruled-out backbone spend), or the spin rate is disclosed as a known limitation.
The headline positive number is **Find-SR ≈ 0.44**, defended on internal validity and detour-cost, explicitly not on a leaderboard position.
