# MP3D lands on the box for a reproduced reference; ADR-0007 is narrowed, not superseded

**Status:** accepted (2026-08-27, grilling session over the request to stand SAVN-CE up as a baseline).

**Narrowed 2026-09-01 by [ADR-0021](0021-the-baseline-is-an-hm3d-internal-arm.md).**
Everything below is unchanged and still binds: SAVN-CE is reproduced wholly inside its own world, and no SAVN-CE number is ever tabled beside an earshot number.
What changed is scheduling, not scope. The paper's baseline is now an internal HM3D arm, so nothing in the results section waits on the Matterport licence and this ADR is off the critical path.

MP3D scene data, the SAVN-CE episode dataset, and a second conda env (`savnce`) are staged on the box for exactly one purpose: to re-measure SAVN-CE's published result **on SAVN-CE's own benchmark**.
The earshot experiment stays on HM3D.
ADR-0007 is **narrowed to what it always decided** — the clean room's scene dataset — and is not superseded.

## Why the question was live

SAVN-CE (CVPR 2026, arXiv 2603.19660) is the closest published work to this project's simulation regime.
It renders binaural audio **live and continuously** as the agent moves, rather than looking it up from a precomputed RIR grid, which is the same structural choice this repo made and the same one that makes both expensive.
Its method, MAGNet, reports SR 37.7 / SPL 32.9 in clean environments; its AV-Nav and SAVi arms report 21.3 and 25.6 SR.
It also reports **SWS, "success when silent"**, the closest published metric to this project's problem of a source that stops giving information.

It is MP3D-only: scenes, the episode dataset, and every config path.
ADR-0007 rules MP3D out of scope **by name**, and ticket 05 measured no MP3D anywhere on the box.
So the request collided with a standing decision, and the collision had to be resolved rather than quietly ignored.

## Why narrowing is the honest reading

ADR-0007's argument stands on three legs.
Acoustic materials buy fidelity **this experiment does not consume**, because every load-bearing term in onset, gradient and provenance is geometric.
The entire prior record is HM3D.
Moving would cost a re-derivation of the episode datasets, not a re-point at data already present.

None of the three is touched by a dataset we never pair against.
A reproduced reference is measured **wholly inside SAVN-CE's world**: their scenes, their episodes, their sensors, their metrics, their checkpoint.
Nothing crosses into an earshot number, so there is no re-derivation, no split record, and no second episode builder.

The line this ADR draws, stated so it can be checked later: **no earshot number is ever computed on MP3D under this decision.**
Running our own controller on their benchmark is a defensible future move and would make their published table our comparison table, but it is a different decision and needs its own ADR.

One piece of independent corroboration arrived with the survey.
SAVN-CE **also disables material-based audio simulation by default**, citing sound-spaces issues #111 and #145, and documents enabling it as an optional extra.
ADR-0007's central claim — that the materials path is off in the reference configuration, not merely off in ours — now has a second, unrelated project agreeing with it on the dataset materials were authored for.

## Considered and rejected

- **Supersede ADR-0007.** Rejected: its reasoning is untouched, and its consequences still bind in full — materials off, HM3D episodes, the smoke on `minival`, no material-database path in the tree.
- **Port SAVN-CE to HM3D so it runs on our episodes.** Rejected: their policies are RL-trained on MP3D with their own sensor suite, and retraining is 240M steps across 4x A800 and a 128-thread dual Xeon over 14 to 23 days. The box is 4 cores and one V100. Zero-shot MP3D-to-HM3D under a different episode definition would produce a near-chance number carrying no information about either method.
- **Cite the paper and run nothing.** Rejected: a number we measured is worth more than one we quoted, and the staging work is the same work a later "our method on their benchmark" effort would need.
- **One conda env for both.** Rejected: SAVN-CE pins `torch==2.7.1+cu126` and `numpy==1.26.0`; `ss2` pins `torch==2.2.2+cu118` and `numpy==1.23.5` **against the same habitat-sim SHA**. Both cannot be right, and reconciling means overriding a pin that one of the two projects actually measured. Two envs, two pin files, no shared assumption.

## Consequences

- **MP3D exists on the box.** A future reader who finds it and reads ADR-0007 should be routed here. ADR-0007 gains a pointer.
- **MP3D requires a signed Matterport Terms of Use**, obtained by a human. `earshot/tools/savnce_licence_wizard.sh` walks that step and records when it was agreed.
- **A second env, a second pin file, and a second install from the same source tree.** `savnce` installs habitat-sim from the existing `~/ss2-build/habitat-sim` checkout, and the multi-audio-sensor patch is applied **only to the installed copy** in the env's `site-packages`. Earshot's pinned source tree is never mutated.
- **The result is a reproduced reference, never a baseline arm** (see `CONTEXT.md`). It is not paired with, subtracted from, or tabled beside any earshot number.
- **Acceptance is pre-registered:** within 2.0 SR points of 37.7, measured over two seeded 1000-episode runs. A miss is reported as a reproducibility finding, not chased; debugging is capped at two box-trips.
- **Acoustic materials stay off on both sides.** SAVN-CE's optional material path is not enabled, so the reproduced number is comparable to their published table, which was produced with it off.
