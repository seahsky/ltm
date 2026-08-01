# The SoundSpaces 2.0 clean room stays on HM3D; MP3D is out of scope

**Status:** accepted (2026-08-01, grilling session over ticket 08 of the `ss2-clean-room` map, with ticket 03's material research in hand).

The clean-room rebuild keeps **HM3D** as its scene dataset, runs the audio simulation with **acoustic materials off**, and rules **MP3D out of scope** for this effort — no fallback, no split, no audio-realism demonstration figure.
The single property MP3D would buy is material-dependent room character, and nothing in the experiment consumes it.

## Why the question was live

SoundSpaces 2.0 assigns acoustic materials by grouping scene triangles by their Habitat semantic-object category name and passing that string to the engine, which picks the material whose `labels` have the most substring matches.
The shipped database is `mp3d_material_config.json` — an mpcat40 lookup table, authored against MP3D's vocabulary.
Ticket 03 quantified the delta at the pinned build commits (`habitat-sim@4f61e321`, `rlr-audio-propagation@4fd446b4`):

| | matched a non-default material | fell to default |
|---|---|---|
| MP3D mpcat40 categories | **36 / 42 (85.7 %)** | 6 |
| HM3D free-text categories (`GLAQ4DNUx5U`) | 64 / 123 (52.0 %) | 59 |
| HM3D object instances | **467 / 907 (51.5 %)** | 440 |

MP3D has one ambiguous tie across its whole vocabulary; HM3D has eight on a single scene.
The representation delta is larger than the vocabulary delta: MP3D carries semantics as per-vertex colours the audio sensor consumes, while HM3D-Semantics v0.2 carries them as **textures**, which route to `GenericMeshData` and appear to fail the `GenericSemanticMeshData` cast inside `joinSemanticHierarchy` — handing the audio context an empty mesh.

On acoustic fidelity alone, MP3D is the better dataset. That was never in dispute.

## Why HM3D wins anyway

**The materials path is off in the reference configuration, not merely off in ours.**
Three independent gates each close it on HM3D: `AudioSensorSpec::enableMaterials_` is constructed `false` on the build branch (measured `False` by ticket 04, not inferred), plain HM3D ships no semantic scene, and the v0.2 texture path appears to break mesh extraction outright.
More decisively, `sound-spaces` itself sets `enableMaterials = False` at every entry point, and `PanoIR/render_panoIR.py` has an explicit `hm3d` branch deliberately excluded from the material-JSON load.
**The SoundSpaces authors render HM3D with no material database.**
The degraded path is the reference case, not a defect we introduced.

**The loss is realism this experiment does not consume.**
The audio channel has to deliver an onset, a climbable received-energy gradient, and onset provenance.
Every load-bearing term in those is geometric — direct-path spreading, occlusion, diffraction — and material-independent.
Uniform absorption costs **contrast, not structure**.
Ticket 04 confirmed the practical end of this on the box: with no material database in the context at all, `minival/00800-TEEsavR23oF` rendered a non-silent IR (`ir_peak_abs` 0.163, `ray_efficiency` 0.548) over 392,356 vertices.

**The cost of moving is the entire prior record.**
Every number in the arc is HM3D.
Ticket 05 found **no MP3D anywhere on the box** — every split present is HM3D — so moving is a fresh multi-GB download and a re-derivation of the episode datasets, not a re-point at data already there.
Disk is not the obstacle (680 GB free); time and re-derivation are.
Moving would also redraw this map's **destination**, which names HM3D in its own text.

**One pro-HM3D argument is weaker than it is usually stated, and we say so.**
The ticket leaned on cross-quotability to VLFM 0.304 and VLingNav 0.429.
ADR-0006 already retired the "competitive absolute numbers" ambition, so that argument no longer carries the weight it once did.
It is not dead: R1 survives as the honest Table-1 baseline, and it is quotable only on HM3D at the 0.1 m ring (ADR-0005).
But the decision rests primarily on the materials delta buying nothing, and only secondarily on comparability.

**Ticket 05 also removed the constraint that made HM3D awkward.**
HM3D `val` mesh coverage is **20/20**, not the 2/20 that forced the R1 smoke onto `val_mini` and put a mesh preflight into `race-scaleup-matrix.sh`.
The "already downloaded" argument is considerably stronger than ticket 08 assumed when it was written.

## Considered and rejected

- **Move to MP3D.** Rejected: buys material fidelity no result consumes, at the cost of the whole prior record, the external comparison, a fresh download, an MP3D licence agreement, and a redrawn destination.
- **Split — HM3D for anything quoted against VLFM, MP3D for an audio-realism figure.** Rejected: the new tree would own two datasets, two scene-dataset configs, two episode loaders and two download stories, to support a claim we do not currently make.
- **HM3D now, MP3D parked as a conditional fallback behind ticket 06's gradient-contrast number.** Rejected deliberately, with the risk named: if ticket 06 reports a flat gradient the map has no pre-agreed answer. We accept that, because a flat gradient would be a **source-placement or gain** problem, not a dataset problem — materials are off in the MP3D reference config too, so switching datasets would not fix it.

## The limitation, stated rather than discovered

If any acoustic-realism claim is ever made, HM3D cannot back it, and the claim must be dropped rather than the dataset changed.
The honest wording, from ticket 03: **uniform absorption; room-scale RT60 variation preserved via `V/S`; furnishing-dependent variation absent.**
This ADR does not assert what the paper currently claims — the draft treats audio as trigger and gate and explicitly disclaims localization — it fixes what the paper is *permitted* to claim on this dataset.

## Consequences

- **Acoustic materials are permanently off.** `enableMaterials` stays `false`; the new tree does not carry a material-database path.
- **Ticket 12 shrinks to its guard half.** The semantic-mesh probe measured a path the clean room now never takes, so it is dropped. What survives is the durable output the ticket already named: a loud assertion at audio-context creation that the mesh is non-empty, that every `RLRA_Error` is checked, and that unknown spec keys are rejected. This matters *more* under this decision, not less — a zero-geometry context still returns plausible audio, which is the failure class that invalidated the `anommxv` headline.
- **Ticket 06 runs the HM3D arm only.** No MP3D materials-on/off arm. Its gradient-contrast measurement stops being a dataset gate and becomes a source-placement gate.
- **HM3D semantic annotations stay on the keep list** (ticket 10). 9.3 GB against 680 GB free, and it is not yet established on the box whether the ObjectNav episode dataset requires `hm3d_annotated_basis.scene_dataset_config.json` or loads against the plain basis config. Keeping them means either answer works; the new tree settles it at runtime.
- **The smoke stays on `minival`**; full `val` is available to later runs at 20/20 mesh coverage, and the `val_mini`-only constraint from earlier work no longer applies.
- **The map's destination is unchanged.** It named HM3D, and it still does.
