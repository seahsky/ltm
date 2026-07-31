# 07 — What is the rebuilt agent?

Type: grilling
Status: open
Blocked by: 04

## Question

In the clean room, what does the agent actually consist of: how does it search, how does it know it found the primary goal, and when does it STOP?

## Why it matters

A clean-room rebuild reopens architecture decisions that were previously settled by accretion rather than by choice.
The prior stack was: geometric frontier planner, ReMEmbR backbone (Qwen2-VL-2B captioner + Qwen2.5-7B planner), SBERT-indexed LTM, anomaly controller on top.
With memory out of scope, most of that is no longer required, and the question is what minimum stack the audio spine needs.

The relevant history, so this is not re-derived from zero:
- ADR-0006 retreated to the **geometric** frontier as the spine. The BLIP-2 semantic frontier (S1+) was the fourth independent non-lift of a semantic frontier and is a documented negative.
- The searcher is weak in absolute terms: SPL 0.031 against VLFM's 0.304, roughly 10x under. That was accepted because the paper leads with the controller, not with absolute navigation numbers.
- Goal detection has been closed twice as a negative: caption-grounding (net-neutral to negative, detector OFF strictly dominates) and OWLv2 on GPU (noise floor on HM3D sim renders).
- Binary SPL at the 0.1 m ring is localization-bound. That finding survives the reset.

So the honest default is: geometric frontier, no detector, oracle-ish or geodesic STOP, and accept the absolute number.
But that default was reached under a *different* environment, and it is worth 20 minutes of grilling to ask whether the clean room should keep it or spend the freed VRAM differently.

The ticket-04 result constrains this hard. If the one env cannot hold torch plus a VLM, the answer is forced.

## What would resolve it

A grilling session covering:
- Search: geometric frontier rebuilt as-is, ported, or replaced by something else.
- Goal detection: none / geodesic oracle / a detector, given both detector arcs closed as negatives.
- STOP policy, and whether the 0.1 m localization bound is accepted again or attacked.
- Whether absolute find-performance matters at all for this map, given the destination is one green episode and the experiment matrix is out of scope.
- Which of these are decisions for *this* map versus decisions that belong to the follow-on memory effort.

Deliverable: a one-page architecture decision recorded as an ADR in the new tree, with the rejected options and why.
