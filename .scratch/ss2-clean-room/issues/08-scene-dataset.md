# 08 — Scene dataset: stay on HM3D or move?

Type: grilling
Status: open
Blocked by: 03

## Question

Does the clean room stay on HM3D, or is this the moment to move to MP3D (or another dataset) for the sake of the audio simulation?

## Why it matters

HM3D was chosen for the visual/ObjectNav side and predates the audio work entirely.
Everything SoundSpaces-shaped points at MP3D: the audio tutorial runs MP3D, the shipped material config is `mp3d_material_config.json`, and SoundSpaces 1.0's heritage is MP3D and Replica.

Arguments for staying on HM3D:
- 1.2 GB already downloaded, episode datasets already in place, and it is gitignored so the reset does not touch it.
  **Corrected by ticket 05 (2026-08-01): it is 19.9 GB, and the mesh-coverage problem is gone.** The box holds `val` at 100 `.basis.glb` / 36 `.semantic.glb` (9.3G) plus `minival` at 10 / 4 (1.1G), and **val mesh coverage is 20/20**, not the 2/20 that forced the R1 smoke onto `val_mini` and put a mesh preflight in `race-scaleup-matrix.sh`. So the "already downloaded" argument is considerably stronger than this ticket assumed, and the constraint that made full-val work awkward no longer applies.
  Ticket 05 also found **no MP3D anywhere on the box** — every split present is HM3D. Moving to MP3D is a fresh multi-GB download, not a re-point at data already there. Disk is not the obstacle (680 GB free); time and re-derivation are.
- Every prior number in the paper arc is on HM3D. Changing dataset makes the new results non-comparable to the record.
- VLFM 0.304 and VLingNav 0.429, the only external numbers we quote against, are HM3D ObjectNav.

Arguments for moving:
- If ticket 03 finds materials do not resolve on HM3D, the acoustics carry no room character and the realism claim weakens.
- A prior run found meshes present for only 2 of 20 HM3D val scenes, so the HM3D asset situation is already not clean.
- The tree is being rebuilt anyway, which is the cheapest moment this decision will ever be.

Arguments against moving that should be weighed honestly: it also throws away the cross-quotable benchmark, which is the one thing anchoring the work to published baselines.

## What would resolve it

A grilling session after ticket 03 lands, weighing acoustic fidelity against benchmark comparability and asset cost.
Include the option of a split: HM3D for anything quoted against VLFM, MP3D for the audio-realism demonstration.

Deliverable: a decision recorded as an ADR in the new tree, plus the concrete asset work it implies for ticket 05's keep/rebuild list.

## Note added by ticket 03 — this ticket is now unblocked, with its evidence in hand

Ticket 03 resolved, and it sharpens both sides of this decision rather than settling it.

**Against HM3D, on materials specifically.** The material path is a hard no by default and degraded at best.
Three independent gates close it: `enableMaterials` is constructed `false` on this branch, plain HM3D ships no semantic scene, and HM3D-Semantics v0.2's texture-based annotation appears to hand the audio sensor an **empty** mesh (ticket 12).
Even if reachable, HM3D's free-text labels match the MP3D-authored material database on only **51.5 % of object instances**, with 8 ambiguous ties on a single scene.
MP3D feeds mpcat40 names straight in and matches **85.7 %** of its vocabulary, with 1 tie across the whole set, through a mesh representation the audio sensor can actually consume.

**For HM3D, and this is the stronger argument than it looks.** The degraded path is not a corner case, it is the reference configuration.
`sound-spaces` sets `enableMaterials = False` at every entry point, and `PanoIR/render_panoIR.py` has an explicit `hm3d` branch that is deliberately excluded from the material-JSON load.
The SoundSpaces authors render HM3D with no material database.
And ticket 03's physics argument says the loss is realism this experiment does not consume: the energy gradient's load-bearing terms are geometric, so uniform absorption costs contrast, not structure.

**So the honest framing for the grilling is no longer "does HM3D work".** It is: *we are choosing between benchmark comparability on a dataset whose acoustics carry no room character, and acoustic fidelity on a dataset that breaks every external number we quote against.*
Weigh in particular:

- Does the paper make any acoustic-realism claim at all? If it does not, the materials delta costs nothing and HM3D wins on comparability outright.
- Is the split option real? HM3D for anything quoted against VLFM, MP3D for an audio-realism demonstration figure. Ticket 03's evidence makes the MP3D side of that split genuinely better, not just different.
- If HM3D stays, the "room character" limitation must be **stated** in the paper, not discovered by a reviewer. Ticket 03 gives the precise wording: uniform absorption, room-scale RT60 variation preserved via `V/S`, furnishing-dependent variation absent.

One measurement is still outstanding and could move this: ticket 06 now owns a gradient-contrast measurement under HM3D materials-off versus MP3D materials-on.
Decide whether this grilling wants that number first, or whether the comparability argument settles it regardless.
