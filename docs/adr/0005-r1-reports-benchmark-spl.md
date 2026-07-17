# R1 headlines native binary SPL@0.1 m — the benchmark ring is 0.1 m (verified)

**Status:** accepted (2026-07-17, grilling session); verified same day on the V100 (`race-r1-preflight.sh`).

R1 (Table 1) exists to answer one reviewer critique — "44 % Find-SR looks weak" — by putting the backbone's number next to the field's: VLFM's HM3D ObjectNav **SPL 0.304** and VLingNav's **0.429**.
That comparison is only valid if our number and theirs are the same metric at the same success ring.
The ring is now verified: the harness loads the canonical `benchmark/nav/objectnav/objectnav_hm3d.yaml`, whose `Success` measure has **`success_distance: 0.1`** — the standard HM3D ObjectNav ring VLFM and VLingNav report on.
**So R1 headlines the harness's native binary `spl` and SR@0.1 m; they are already cross-quotable to VLFM 0.304 / VLingNav 0.429.**

We first suspected a ring mismatch.
The arc's notes (audit caveat #5; the `episode_runner.py:2495-2498` comment) assert "the benchmark uses 1.0 m", which would have made native `spl@0.1 m` cross-ring and understated.
That belief was **wrong**: the "1.0 m" is the arc's self-invented `success_1m` reach diagnostic (`min_d2g < 1.0` at *any* step, STOP-independent), not the benchmark ring.
The $0 preflight read `success_distance: 0.1` from the composed config and settled it.

**This reverses the earlier plan to add a 1.0 m `spl_1m` headline.**
A 1.0 m SPL is not the benchmark — it is a *relaxed* ring that would **overstate** us against VLFM's 0.1 m number, the mirror image of the error we set out to avoid.
`compute_benchmark_spl` (`embodied_memory/metrics.py`, ring-parameterized) is retained only as an optional, clearly-labelled reach diagnostic, never the R1 headline.

The residual R1 risk is therefore **capability, not metric**.
Native SPL@0.1 m is localization-bound — success requires calling STOP within 0.1 m of a goal viewpoint — and the `r1nav-s1` smoke shows the ReMEmbR backbone lands that precisely only occasionally (native mean SPL ≈ 0.05–0.15, well below VLFM 0.304).
S1+ upgrades *frontier choice*, not STOP-localization, so it may not close the gap.
R1's honest outcome may be "our searcher is weaker than VLFM at the benchmark ring", which D5's interpretation rule already commits us to shipping.

**Consequences.**
R1's Table-1 headline is **native binary SPL@0.1 m + SR@0.1 m**, quoted against VLFM 0.304 / VLingNav 0.429 at the same 0.1 m ring — no metric wiring needed before the spend.
soft-SPL is reported as a ring-independent searcher-quality companion, explicitly **not** VLFM-comparable.
`success_1m` / a 1.0 m `spl` are relaxed reach diagnostics, never quoted as the benchmark SR/SPL.
The rest of the arc keeps soft-SPL primary for the memory delta (S3−S1); the two regimes coexist by design and a future reader should not "fix" R1 onto soft-SPL to match the arc.
