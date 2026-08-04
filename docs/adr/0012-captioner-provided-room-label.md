# Scene-conditioned normality keeps its claim, with a captioner-provided room label

**Status:** accepted (2026-08-04, grilling session on ticket 09 of the `ss2-clean-room` map).
**Supersedes ADR-0002** (scene-conditioned anomaly detection), whose claim and behavioural test survive and whose grounding does not.

Whether a heard sound is anomalous still depends on the room it is heard in.
The room label now comes from the **Qwen2-VL-2B captioner the tree already carries**, behind a provider seam.
CLIP does not return, and `perception.py` does not carry into the clean room.

## Why the claim survives

Dropping scene-conditioned normality is not a small edit.
ADR-0004 relocated the **entire** discrimination claim onto ADR-0002's room-normal distractor, on the grounds that the continuous bed never made the agent's INVESTIGATE decision discrimination-dependent.
If scene-conditioned normality dies, the CLAP gate becomes decorative, any onset interrupts, and there is no discrimination claim anywhere in the work.

The behavioural test survives unchanged: same sound, two rooms, one where it is room-normal and must be ignored, one where it is room-anomalous and must interrupt.
ADR-0009 makes that an across-episode arm rather than a simultaneous-source one, which is what ADR-0002 always specified.

## Why not CLIP

Ticket 15 tried to price CLIP and **could not load it at all**.
transformers 4.57.6 refuses `torch.load` below torch 2.6 under CVE-2025-32434, the cached `openai/clip-vit-base-patch32` is a `.bin`, and ticket 13 pinned torch at 2.2.2+cu118 deliberately for this V100.
VRAM is not the constraint; ticket 15 measured 26 GiB of margin with the captioner resident.
The pin is.

That left three options: a safetensors re-fetch of CLIP, which makes the env pin's model layer load-bearing and is ticket 17's problem; a torch bump past 2.6, which ticket 13's pin forbids on this card; or a route that avoids CLIP.

There is a fourth that dominates all three, and neither ticket listed it.
**The tree already carries a VLM.**
ADR-0008's carry list ships Qwen2-VL-2B for the caption-grounded detector, and ticket 15 noted it loads cleanly precisely because it ships safetensors.
A room label derived from a caption of the current view costs no new dependency, no re-fetch, and adds nothing to the env pin's model layer.

It also has the better prior.
CLIP is measured flat on HM3D sim renders across three independent measurements (separation 0.020 against a 0.05 bar), and this project's Phase-3 fix was precisely that VLM captions are discriminative where the cheaper channel was degenerate.
ADR-0008 dropped CLIP and named the room classifier as the only route by which it returns; this closes that route rather than reopening it.

## We chose this over

**Reviving CLIP via a safetensors re-fetch.**
Cheapest in code, and it keeps ADR-0002's grounding literally intact.
Rejected because it makes the model layer of the env pin load-bearing for a component already measured flat three times, to avoid using a model the tree loads anyway.

**Dropping scene-conditioned normality.**
Honest, and it removes the weakest link: `ROOM_PRIOR` is hand-authored, HM3D has no room-type ground truth, and the arm has never been run.
Rejected because it takes the discrimination claim with it, and ADR-0004 has nowhere else to put it.

## Consequences

**`perception.py` (506 LOC) does not carry.**
This discharges one third of the audio carry line ticket 10 left to ticket 09.

**The room label is a provider seam, and the implementation defers to R2.**
The gate takes a label from a provider; what this decision fixes is the seam and the provider, not the code.
The smoke does not exercise it, because the experiment matrix is out of scope for the `ss2-clean-room` map.

**ADR-0002's $0 gate carries across the substitution unchanged.**
ADR-0002 required a room-classifier accuracy check before any live run, because it leaned on CLIP separating two rooms at cosines around 0.30 and called that a robustness risk.
The captioner has never been measured as a room classifier either, so it clears the same bar before anything depends on it.
The cheap form is the CapRL-gate pattern: render frames at known room viewpoints, measure within-versus-between separation.

**ADR-0002's "single RIR grid preserves the O(1) invariant" reasoning is moot**, since there is no grid and ADR-0009 renders one source live.
The conclusion it supported, that the distractor is an across-episode arm, is kept for the reasons above rather than for that one.
