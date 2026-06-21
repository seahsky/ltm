# When Does Memory Help an Embodied Agent? A Lifelong Hierarchical Long‑Term Memory for Audio‑Cued Anomaly Response, and the Boundary of Its Utility

**Authors:** [TBD] · **Target venue:** ICRA 2027 · **Status:** DRAFT v0.1 (auto-generated from `PHASE2_ABLATION_REPORT.md` + run logs; numbers are RACE-verified unless marked [verify])

---

## Abstract

We study a lifelong hierarchical long‑term memory (LTM) for an embodied agent and ask a question that is usually assumed away: *when* does long‑term memory actually help navigation, and when does it not? We build a four‑module LTM — short‑term buffering, bio‑inspired consolidation with an importance score, a three‑layer FAISS memory, and memory‑injected re‑ranking — on top of a vision‑language navigation backbone (a 2B image captioner and a 7B language planner) and evaluate it in the Habitat simulator on HM3D scenes. We instantiate it in an **audio‑cued anomaly‑response** task: the agent first maps a home in silence, then an anomaly sound (alarm, glass break, baby cry) fires from a source co‑located with an object, and the agent must reach it. We report three results and a mechanism. **(1) Memory helps when past observation is *relevant*:** on warm revisits the LTM improves paired soft‑SPL by **+0.115 (n=26, p=0.005)** to **+0.24 (n=12, p=0.008)** and lifts 1.0 m success from 0.33 to 0.67; a setting decomposition attributes essentially all of the gain to the LTM (not the short‑term buffer). The effect **reproduces in the audio task** (+0.171, n=18, p=0.002), giving the first significant strict‑radius success gain we observe. **(2) Memory is cleanly *neutral* when observation is incidental:** on cold multi‑object chains the LTM produces exactly zero compounding (Progress 0.190 with and without memory). **(3) Every *additional* memory head we add — learned importance, room‑affordance priors, a temporal‑recency head, an audio direction‑of‑arrival re‑ranker, and writing the heard anomaly into memory — is inert or redundant,** and we trace all of these to one bottleneck: the caption embedding cannot discriminate object *instances*, so a single‑goal evaluation never rewards the disambiguation these heads provide. Our contribution is therefore not a new state‑of‑the‑art module but a **rigorously‑bounded account of where embodied LTM helps**, validated across two task families with adversarially‑checked ablations, plus a verified open‑set audio anomaly detector and a causal demonstration that the audio onset is *necessary* for the recall. We argue the next gain must come from instance‑level perception or a multi‑goal/changed‑world evaluation — not from more memory heads.

---

## 1. Introduction

A robot deployed in a home for weeks accumulates experience it should be able to reuse: where it saw the kitchen, which room had the crib, where it last heard a window break. This is the promise of **lifelong long‑term memory (LTM)** for embodied agents. Recent vision‑language navigation systems (e.g., ReMEmbR [Anwar et al., ICRA 2025]) caption first‑person observations into a memory and query it at decision time, and a large literature on episodic and semantic memory for agents argues that hierarchy, consolidation, and importance‑weighting should help.

Most of that literature reports *that* memory helps. We instead ask **when**, and we treat the negative cases as first‑class results. We are motivated by a concrete application — **anomaly response**: a home robot silently maps its environment, then an anomalous sound occurs, and the robot must localize and approach it. This setting is attractive because it cleanly separates two regimes that are usually entangled: a **warm** regime, where the agent has *relevant* prior experience of the region the anomaly is in, and a **cold** regime, where its prior experience is merely *incidental*.

We build a four‑module LTM faithful to the bio‑inspired memory proposal — STM → consolidation (importance `I = αR + βU + γN`) → a three‑layer (fine/mid/coarse) FAISS memory → memory‑injected re‑ranking — and run it on the ReMEmbR backbone (a 2B vision‑language captioner and a 7B language planner) in Habitat on HM3D. We render audio offline as binaural room‑impulse responses (SoundSpaces) and convolve it in O(1) at run time, so the simulator stays a standard navigation simulator.

Our findings:

1. **Relevant recall helps.** On warm revisits, the full system improves paired soft‑SPL over a memory‑off baseline by **+0.115–0.24** (two datasets), lifts 1.0 m success from **0.33 → 0.67**, and the gain reproduces in the audio task (**+0.171**). A short‑term‑memory‑only setting captures none of it, so the effect is LTM‑specific.
2. **Incidental observation is neutral.** On cold multi‑object chains, where the agent sees objects only in passing while hunting other goals, the LTM yields **exactly zero** compounding.
3. **Extra memory heads do not move the needle.** Five separately‑motivated additions — a trained importance head, a trained utility/forward‑model head, a CLIP‑grounded room‑affordance prior, a temporal‑recency head, and an audio direction‑of‑arrival re‑ranker — are each *built, correct, and inert*. A sixth, **writing the heard anomaly into memory**, is mechanism‑verified but **redundant with visual recall**. We show all six share one cause: the caption embedding's **instance‑discrimination ceiling**, exposed by a single‑goal evaluation that never requires choosing *which* same‑category object to revisit.

The paper's contribution is the **boundary**: a reproduced, module‑attributed account of the regime where embodied LTM helps, and a mechanistic explanation — backed by adversarially‑verified ablations — of why the obvious extensions do not. As applied contributions we add an **open‑set CLAP anomaly detector** (separates anomaly from benign household sound with zero training, equal‑error‑rate 0.00 on our clips) and a **causal onset‑gate** demonstrating the audio onset is necessary for the recall.

---

## 2. Related Work

**Memory for embodied navigation.** ReMEmbR [Anwar et al., 2025] captions observations with a VLM and answers spatial‑temporal queries from a flat memory; topological and metric memories (e.g., neural topological SLAM, scene‑graph memories) store structure for re‑planning. We extend ReMEmbR with an explicit hierarchical, consolidated memory and study the marginal value of each layer. **Hierarchical / consolidated memory** mirrors complementary‑learning‑systems accounts (fast episodic + slow semantic) and importance‑gated replay; our `I = αR + βU + γN` consolidation follows that lineage. **Audio‑visual navigation** (AudioGoal, SoundSpaces [Chen et al., 2020]; Semantic Audio‑Visual Navigation [Chen et al., 2021]) learns to localize a sounding goal; we instead use audio as an *anomaly trigger and gate* over a pre‑built map, and render it offline so the navigation stack is unchanged. **Anomalous sound detection** (DCASE Task 2; open‑set acoustic novelty) frames the normal‑vs‑anomaly decision; we adopt a zero‑shot contrastive‑prompt variant on a pretrained audio‑text model (CLAP). Our emphasis on *negative, mechanism‑level* results connects to recent calls for rigorous ablation and reproducibility in embodied AI.

---

## 3. System

### 3.1 Backbone and task

The perception/action backbone is **ReMEmbR**: a 2B vision‑language model (Qwen2‑VL‑2B) captions each keyframe; a 7B language model (Qwen2.5‑7B) plans. The agent acts in Habitat on HM3D `val_mini` scenes. A navmesh point‑goal controller (Habitat's `ShortestPathFollower`) executes the agent's self‑chosen waypoint. The three ablation **settings** are: **S1** memory‑off (backbone only), **S2** short‑term‑memory only, **S3** full LTM system.

### 3.2 STM → consolidation → hierarchical LTM

Keyframes are buffered in a short‑term store, then **consolidated** into the LTM by an importance score `I = αR + βU + γN` (relevance, utility, novelty) that gates the top‑*k* writes. The LTM has three FAISS layers — **fine** (individual sightings), **mid** (clustered patterns), **coarse** (category→region priors). Critically, the fine layer is indexed on the **SBERT text embedding of the VLM caption** (not a raw image embedding): we found the image–text cosine flat and non‑discriminative, while the rich VLM caption makes goal‑vs‑caption similarity discriminative. *In the runs reported here, the mid layer is empty and retrieval queries the fine layer only; the coarse layer is seeded but, as §5.5 shows, never chosen. The measured memory effect is therefore "fine layer + memory‑injected re‑ranking," which we state plainly rather than as a working three‑layer hierarchy.*

### 3.3 Memory‑injected re‑ranking

At each decision the planner proposes frontier waypoints; the bridge **injects** LTM‑derived waypoints (the world positions of past sightings that match the goal query `"there is a {category}"` above a cosine gate, scene‑filtered and de‑duplicated against frontiers) into the candidate pool, and a physics‑aware re‑ranker scores the merged pool. Memory is thus *in the action loop*, not merely a re‑ranking of the planner's own list.

### 3.4 Audio channel

Audio is rendered **offline** as a grid of binaural room‑impulse responses (SoundSpaces) at the anomaly source; at run time the agent's pose indexes the nearest cell and the source clip is convolved in **O(1)** (`fftconvolve`). This two‑environment split keeps Habitat a standard navigation simulator. Three audio components:

- **Onset detection.** A calibrated RMS threshold marks the first step the anomaly is audible (calibrated so the source is audible at ≈4 m, not only point‑blank).
- **Open‑set anomaly detection (Step 1).** A pretrained audio‑text model (CLAP) scores the heard clip against an anomaly‑prompt bank *and* a normal/background‑prompt bank; it fires only when the best anomaly cosine beats the best normal cosine by a calibrated margin. This is a real normal‑vs‑anomaly decision (the prior system force‑classified any loud sound into one of three anomaly classes).
- **Onset‑gate.** When enabled, memory retrieval is suppressed until the anomaly is heard, making the audio onset **causally necessary** for warm recall.

---

## 4. Experimental Setup

**Tasks.** (i) **ObjectNav revisit** — the agent maps a scene (cold pass), then revisits it for a goal it has seen (warm pass); the controlled‑start dataset places warm starts at navigable, same‑category‑reachable poses. (ii) **AudioGoal anomaly response** — the cold pass is silent; on the warm pass an anomaly fires from a source co‑located with a captioned object, and the agent must reach it. (iii) **MultiON** — chains of K=3 object goals per episode, to test whether *incidental* cold sightings compound.

**Metrics.** Primary: **soft‑SPL** (graded path efficiency), reported as a warm **paired** S3−S1 delta with a 90% bootstrap CI. Secondary: **success rate at 1.0 m** (the benchmark radius) and **at 0.1 m** (a strict, localization‑bound radius), `min` distance‑to‑goal, and memory‑activity counters. The **module decomposition** S2−S1 (short‑term only) and S3−S2 (consolidation + LTM + re‑rank) attributes the effect.

**Why soft‑SPL is primary.** Strict success at 0.1 m is perception‑bound: caption‑grounded detection cannot localize a stop to 0.1 m, so binary SPL@0.1 m is near‑zero everywhere except where a recalled *view‑point* happens to fall inside the ring. We therefore lead with soft‑SPL and report both success radii.

---

## 5. Results

### 5.1 Memory helps when recall is relevant (warm revisit)

On the warm revisit, the full system improves paired soft‑SPL over the memory‑off baseline:

| Dataset | n (warm pairs) | warm soft‑SPL S3−S1 | 90% CI | p | SR@1.0 m (S1→S3) |
|---|---|---|---|---|---|
| Phase‑C (chair+bed, 2 scenes) | 12 | **+0.240** | [+0.073, +0.417] | 0.008 | 0.333 → 0.667 |
| Wide (6 categories, 2 scenes) | 26 | **+0.115** | — | 0.005 | 0.308 → 0.500 |

The two estimates use different category mixes; we quote both rather than the headline +0.24 alone. Binary SPL@0.1 m rises from 0 to 0.196 on Phase‑C (the recalled view‑point lands inside the ring on some episodes). The result has reproduced across ≈8–12 runs of the development history; treating reruns and re‑analyses as one, it rests on **2–3 distinct datasets**.

### 5.2 The gain is LTM‑specific (module decomposition)

Adding *only* the short‑term buffer does nothing; the gain appears only with consolidation + LTM + re‑ranking:

| Contrast | Phase‑C (n=12) | Wide (n=26) |
|---|---|---|
| S2 − S1 (STM only) | **+0.000** | +0.012 (n.s.) |
| S3 − S2 (consolidation+LTM+rerank) | **+0.240** | **+0.103** (p=0.017) |

So essentially 100% of the soft‑SPL effect and 100% of the binary‑precision effect is attributable to the LTM, not the short‑term memory.

### 5.3 The effect reproduces in the audio task

On the AudioGoal anomaly‑response task (2 scenes × 3 anomaly/object cells, S1/S2/S3), the warm gain reproduces independently:

- Pooled warm soft‑SPL **S3−S1 = +0.171** (n=18, 90% CI [+0.070, +0.277], p=0.002).
- Clean decomposition: S2−S1 not significant ⇒ the LTM‑specific S3−S2 = **+0.172** (p=0.004).
- Cold control ≈ 0 (memory inert without a prior sighting).
- **First significant strict‑radius success gain:** binary SPL@0.1 m **+0.139** (p=0.003), where the cold seed begins at a goal view‑point so the 0.1 m ring is occasionally reachable.

We flag **high heterogeneity**: 4/6 cells win, 2 regress; a leave‑best‑cell‑out estimate softens to +0.095 (p≈0.07). The gain here is **visual** recall — the audio serves as onset trigger and (when gated) causal switch, not as a localizer (§5.6).

### 5.4 Memory is neutral when observation is incidental (cold MultiON)

On cold K=3 chains, where the agent only sees objects in passing while hunting other goals, memory does **nothing**:

| | S1 | S2 | S3 |
|---|---|---|---|
| Progress (n=14/setting) | 0.190 | 0.190 | 0.190 |

S3−S1 = +0.0000; the gap by sub‑goal index is identical at every index — **zero compounding**. The cause is twofold: the backbone's exploration ceiling starves the premise (the agent rarely survives to the recall moment — found‑rate at sub‑goal 0 is 0.5, at index 1 only 0.07), and incidental cold sightings are not warm relevant priors (the caption embedding cannot tell instances apart). The net statement is sharper than either result alone: **the LTM helps when past observation is relevant, and is cleanly neutral when it is incidental.**

### 5.5 What does *not* help: five inert heads and one redundant write

We added six separately‑motivated memory mechanisms. Each was built, made correct, instrumented, and ablated; each is inert or redundant:

| Mechanism | Outcome |
|---|---|
| **Trained importance head** (R, learned on a per‑keyframe goal‑object label) | Recovers to *heuristic‑competitive* on warm (+0.194 vs +0.236, statistical tie) but does **not beat** the hand‑tuned heuristic and hurts cold; head‑to‑head at scale is parity. |
| **Trained utility / forward‑model head** (U, self‑supervised next‑caption surprise) | **Regresses** warm (+0.112 → +0.061) via an over‑fire signature; three U formulations all land at ≈+0.06 with the same over‑fire. |
| **Coarse room‑affordance prior** (CLIP zero‑shot room classifier + category→room prior) | Built, CLIP‑grounded, *proposes* but is **never chosen** at the re‑rank weight; it always loses to concrete sightings. Zero over‑fire, zero gain. |
| **Temporal‑recency head** (M4, additive recency bonus) | **Honest negative:** does not change warm outcomes (B−A = −0.0005, a tie at the floor); fires more but the extra picks are credit re‑attribution, not re‑routing. |
| **Audio direction‑of‑arrival re‑ranker** (S2 head, lateral‑sign bonus) | **Structural negative:** its zero‑sum, mean‑centered bonus is provably 0.0 on a single recalled instance — there is nothing to disambiguate. |
| **Write the heard anomaly to LTM** (Step 2, store source location) | **Mechanism‑verified** (the write fires and is recalled at a distance) but **redundant with visual recall** on a single‑goal line‑of‑sight harness (write‑ON vs write‑OFF paired soft‑SPL ≈ 0 once an over‑fire confound is damped). |

**One cause unifies them.** We measured the caption embedding's instance discrimination directly: within‑instance caption cosine **0.628** vs between‑instance‑same‑category **0.535** — a real **+0.093** separation — but the live category query `"there is a {category}"` collapses instances to a **0.047** rank gap. Every head above either (a) stores or re‑weights *more* goal‑ish frames, which under this ceiling surfaces more *wrong‑instance* candidates → over‑retrieval → worse arrival (the importance/utility heads, the anomaly write), or (b) tries to disambiguate instances that the **single‑goal** evaluation never forces a choice among (the affordance, temporal, and DOA heads). The hand‑tuned, conservative heuristic wins precisely because conservatism is the right bias in an instance‑ambiguous space.

### 5.6 Audio: necessary and detectable, but the gain is visual

We verify three things about the audio path. **(a) Open‑set detection works:** the CLAP contrastive‑prompt gate separates anomaly clips (alarm/glass/cry) from benign household sound (footsteps/coughing/knock/vacuum) with a clean margin — equal‑error‑rate **0.00** on our staged clips, recommended decision margin 0.137. **(b) The onset is causally necessary:** with the onset‑gate enabled, suppressing the sound suppresses the recall (no onset → no memory injection → no warm gain). **(c) But localization is not the source of the gain:** the simulator's binaural cue is inter‑aural‑level only (the time‑difference cue is near‑zero by construction), giving a left/right *sign* but no range; consequently the DOA re‑ranker is inert (§5.5) and the +0.171 warm gain is **visual** recall with audio as trigger and gate. Writing the source location into memory is mechanism‑verified but, on a line‑of‑sight seed where vision already maps the source, redundant.

---

## 6. Analysis: the boundary of LTM utility

Putting the results together, the utility of this LTM is governed by two axes:

1. **Relevance of recall.** Memory helps iff a past observation is *relevant* to the current goal and *navigable‑to* (a stored view‑point). Warm revisits satisfy this (+0.12–0.24); cold/incidental sightings do not (≈0).
2. **Whether the task rewards instance disambiguation.** Every mechanism that could only pay off by choosing the *right* same‑category instance is inert, because (i) the single‑goal evaluation never forces that choice and (ii) the caption embedding cannot make it anyway.

This yields a concrete prediction for where the next gain lies: **not** in additional memory heads, but in **instance‑level perception** (a better instance‑discriminating embedding or a trained detector) and/or an **evaluation that rewards recall** — multi‑instance, changed‑world, or genuinely cross‑environment episodes. We tested the cross‑environment hypothesis directly and found it **structurally absent**: the injector hard‑filters to the current scene, so a home sighting is *recalled* in a new scene (counter > 0) but never *injected* as a waypoint; freezing away‑scene writes collapses the apparent transfer to ≈0 with zero memory chosen. Positive cross‑environment transfer therefore requires a position‑free mechanism (a working coarse‑affordance layer) — which, as §5.5 shows, the current re‑ranker declines to use.

---

## 7. Limitations

- **Scope is within‑scene, same‑category recall**, not cross‑environment reuse; we verify the latter is structurally absent in the current injector.
- **"Hierarchical" is, in the measured path, fine‑layer + re‑ranking** (mid empty, coarse never chosen). We do not claim a working three‑layer hierarchy.
- **Headline magnitude is dataset‑dependent** (+0.24 at n=12 vs +0.115 at n=26); the audio reproduction (+0.171) is heterogeneous across cells and softens to +0.095 leave‑best‑out.
- **Success radius matters:** strict 0.1 m success is perception‑bound; we lead with soft‑SPL and report 1.0 m success (0.33→0.67) as the comparable benchmark number.
- **The anomaly‑write result uses an oracle (ground‑truth) source location** and a line‑of‑sight seed; a non‑oracle, non‑line‑of‑sight evaluation is needed to test whether audio‑written memory can ever beat visual recall, and is left to future work.
- **Backbone variance and small n** (paired bootstrap, n=12–26) bound the precision of every estimate; we report CIs and leave‑one‑out throughout.

---

## 8. Conclusion

We built a faithful lifelong hierarchical LTM for an embodied agent and asked when it helps. The answer is precise and reproduced across object‑ and audio‑goal tasks: **memory helps when past observation is relevant to the current goal (+0.12–0.24 soft‑SPL, 0.33→0.67 success), is cleanly neutral when observation is incidental, and is not further improved by additional importance, affordance, temporal, direction‑of‑arrival, or anomaly‑write heads** — because the caption embedding cannot discriminate instances and the single‑goal evaluation never rewards doing so. For the anomaly‑response application we contribute a verified open‑set audio anomaly detector and a causal onset‑gate, while showing honestly that the navigation gain is visual recall, not audio localization. The practical takeaway for embodied‑memory research is a redirection: the next gain is in instance‑level perception and recall‑rewarding evaluation, not in more memory heads.

---

## Appendix A. Reproducibility

- **Backbone:** Qwen2‑VL‑2B (captioner) + Qwen2.5‑7B (planner); Habitat + HM3D `val_mini`.
- **Audio:** SoundSpaces binaural RIR grid rendered offline; O(1) `fftconvolve` at run time; real ESC‑50 anomaly + benign clips (CC BY‑NC).
- **Ablation:** `S∈{1,2,3}` via `embodied_memory/run_hm3d_pol.py --backbone remembr --setting S`; paired‑bootstrap analyzer `analyze_ablation.py [--revisit|--multion]`.
- **Statistics:** warm‑paired soft‑SPL with 90% bootstrap CIs; module decomposition S2−S1 / S3−S2; leave‑one‑cell‑out robustness.
- **Negative‑result instrumentation:** per‑run counters (`n_memory_candidates`, `n_memory_chosen`, `n_audio_writes`, `n_audio_event_recalled`, write‑seam skip reasons) make every "inert head" claim a checkable number rather than an assertion.
- Code, drivers, and per‑run summaries: `embodied_memory/`, `scripts/race-*.sh`, `runs/*/summary.json`. Full development arc and adversarially‑checked caveats: `PHASE2_ABLATION_REPORT.md`.

## Appendix B. Numbers to finalize before submission [verify]

- Confirm exact n, CI, and p for the wide‑matrix +0.115 (CI not in the current draft).
- Re‑state the MultiON found‑rate (0.5 / 0.07) and Progress (0.190) from `runs/` at submission time.
- Lock the audio M3 per‑cell table (4/6 win) and the leave‑best‑out +0.095/p≈0.07.
- Author list, affiliations, acknowledgements, dataset/clip licensing statement (ESC‑50 CC BY‑NC).
- Figures: (1) system diagram (STM→consolidation→fine/mid/coarse→rerank + audio two‑env split); (2) warm vs cold soft‑SPL bar with CIs; (3) the instance‑cosine separation (0.628/0.535/0.047) as the unifying‑bottleneck figure; (4) audio anomaly‑vs‑benign margin histogram (EER 0.00).
