# 15 — What holds 24 GB of VRAM, and what is the clean room's VRAM budget?

Type: task
Status: claimed
Assignee: Sky
Blocked by: none

## Question

On the RACE box, 8,249 MiB is free of 32,768 MiB — roughly 24 GB is held by something nobody has accounted for.
What is holding it, and once it is released, does the clean room's stack fit in 32 GB with margin?

## Why it matters

Ticket 05 found this and explicitly declined to act on it.
Ticket 10 declined it too, but for a different reason: it is not a cleanup item at all, so it does not belong on a reset checklist. It is a correctness threat, and it has two consumers already in flight.

**Ticket 06 is timing audio renders underneath it right now.** If 24 GB is held by a zombie process rather than by anything intentional, 06's numbers are measured under memory pressure, and 06's verdict decides whether the map's destination — audio rendered live at every step — is reachable as named. A cost measurement taken on a contended box is a validity problem, not a rounding error.

**ADR-0008 is already counting VRAM.** Dropping the 7B planner was justified in part as freeing ~15 GB. That arithmetic assumes a known starting point, and 8 GB free is not one. On a 32 GB V100 the clean room has to hold: the audio context and its scene geometry, the Qwen2-VL-2B captioner (kept by ADR-0008 for the caption-grounded detector), and — if ticket 09 rules that scene-conditioned normality survives — CLIP for the room classifier, plus the CLAP anomaly classifier whichever way ticket 13 resolves its torch pin.

Nobody has added those up against a measured free figure.

## What would resolve it

On the box:

1. **`nvidia-smi` with process attribution** — what holds the 24 GB. Expect one of: a stale python from an earlier run, a leaked notebook kernel, or something intentional nobody documented.
2. **Release it and re-measure** the free figure at rest.
3. **Add up the clean room's stack** against that figure, as a table with a measured number per component, not an estimate:
   - the SoundSpaces audio context + scene geometry (ticket 04 loaded a real HM3D scene; 392,356 verts)
   - Qwen2-VL-2B captioner
   - CLAP (whichever pin ticket 13 lands on)
   - CLIP, **conditional on ticket 09** — 07 removed CLIP from the agent, and the ADR-0002 room classifier is the only route by which it returns
4. **State the margin**, and say plainly whether the stack fits with the audio sim co-resident or whether something has to be lazy-loaded or dropped.

Note the precedent: the L3 milestone hit exactly this wall and solved it at the config level — swapping the 7B planner for `microsoft/Phi-3.5-mini-instruct` in one process let OWLv2 base and large both run on cuda co-resident with no OOM. The VRAM fix was the durable win of that milestone even though the detector arc closed as a negative. Same class of problem, and the clean room should not rediscover it during a build.

Two things this ticket is **not**:

- It is not a deletion or tidiness item. Ticket 10's box sweep rules on those, and this was promoted out of it deliberately.
- It does not decide whether CLIP is in the stack. Ticket 09 owns that; this ticket prices both answers.

Deliverable: the process attribution, the released free figure, and a component-by-component VRAM budget with a stated margin.
