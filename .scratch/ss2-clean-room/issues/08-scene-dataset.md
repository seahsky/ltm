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
