# R1 reports benchmark-standard SPL, not soft-SPL or native 0.1 m SPL

**Status:** accepted (2026-07-17, grilling session).

R1 (Table 1) exists to answer one reviewer critique — "44% Find-SR looks weak" — by putting the backbone's number next to the field's: VLFM's HM3D ObjectNav **SPL 0.304** and VLingNav's **0.429**.
That comparison is only valid if our number and theirs are the same metric at the same success ring.
We decided R1's headline is **benchmark-standard SPL**: binary success = STOP called within the benchmark radius of a goal viewpoint, weighted by the geodesic path ratio, reported at the ring VLFM's 0.304 is measured at.
It is **not** the arc's soft-SPL and **not** the harness's native `spl`.

The harness as wired cannot report this number.
`episode_runner.py:2422-2423` scores `success`/`spl` at the **0.1 m ring** (`success = info["success"] or distance_to_goal < 0.1`), which the code comment at 2495-2498 and audit caveat #5 both call "localization-bound"; `analyze_ablation.py` surfaces that 0.1 m `spl` as the headline, and the driver's DONE line compares it to VLFM 0.304.
The only 1.0 m quantity in the harness is `success_1m` (`min_d2g < 1.0` at *any* step, line 2433) — a **STOP-independent** reach diagnostic that over-counts against a STOP-gated benchmark.
There is no path-weighted SPL at 1.0 m anywhere.
So R1 would compare our SPL@0.1 m (near-zero) against VLFM's benchmark SPL and make the backbone look catastrophically weak — the exact opposite of R1's purpose.

Two steps follow, both blocking the full-val R1 spend: (1) verify `success_distance` in `benchmark/nav/objectnav/objectnav_hm3d.yaml` on the VM and the ring VLFM's 0.304 uses, so both numbers sit at the same ring; (2) if there is a ring gap, add a benchmark SPL to the harness (STOP within the benchmark radius, geodesic-weighted) and report that.

We chose this over (a) reporting **soft-SPL**, which is the arc's primary metric but is not what VLFM/VLingNav publish, so it answers no reviewer; and (b) reporting the **native 0.1 m SPL** unchanged, which is cross-ring and understates the searcher by construction.

**Consequences.**
A metric addition (`spl_1m`, STOP-gated) lands before full-val R1, not after.
`success_1m` stays a diagnostic and is never quoted as an R1 success rate.
The rest of the arc keeps soft-SPL as its primary metric — this ADR scopes the benchmark-SPL headline to R1 (the memory delta S3−S1 is unchanged, still soft-SPL), so the two metric regimes coexist by design and a future reader should not "fix" R1 back onto soft-SPL to match the arc.
