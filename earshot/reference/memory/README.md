# The memory stack, vendored inert

This directory is an **archive**, not code.
Nothing in the clean room imports it, and both `__init__.py` files raise `ImportError` to keep it that way — an absent `__init__.py` would not, because PEP 420 makes `earshot.reference.memory.ltm` importable from a regular parent package anyway.

It is here so the follow-on memory effort starts from the implementation that produced the measured results, rather than from a description of it.
Ticket 10 decided vendoring over a pointer: ~3,400 LOC of dead code in a clean room is the price of not having to reconstruct the stack out of a git tag.

Six files carry:

| file | LOC | what it is |
|---|---|---|
| `consolidation.py` | 476 | importance `I = αR + βU + γN`, the write gate |
| `ltm.py` | 229 | 3-layer FAISS memory (only the fine layer was ever queried) |
| `pattern_cluster.py` | 336 | mid layer — `ltm_mid=false` in every run, never enabled |
| `reranking.py` | 470 | score fusion |
| `encoder.py` | 187 | the SBERT seam — the measured bottleneck |
| `memory_bridge.py` | 1,690 | embodied STM + both seams |

They are vendored **broken** on purpose.
`ltm.py` imports `faiss`, `encoder.py` reaches for `sentence-transformers`, and `memory_bridge.py` imports `.frontier_planner`, `.perception`, `.room_resolver` and `.text_encode_util` — modules the reset deletes.
If this directory ever becomes importable by accident, vendoring it was a mistake.

## The write path

`keyframe → segment → consolidate → LTM`.

`MemoryBridge.observe_keyframe()` takes a per-step keyframe (caption, embedding, agent pose, scene id) into the STM ring buffer.
`consolidate()` closes a segment at the episode boundary, scores every candidate with `DialogueConsolidation`'s importance `I = αR + βU + γN`, and writes the top-k through the gate into the fine layer of `HierarchicalLTM`.
`consolidate_subgoal_boundary()` is the MultiON variant, gated off for single-goal tasks.
`write_audio_event()` is the one write-side audio lever: on onset it inserts a fine-layer item at the source position so a later visit can recall a waypoint *to the sound*.

The fine layer is indexed on the **VLM caption text**, embedded with SBERT.
That has been true only since the Phase-3 fix; before it, HM3D's all-zero semantic sensor made every keyframe caption `"… room interior"` and the layer carried no discriminative content at all.
Anything in the older reports that reads as "memory is inert" is that bug, not a property of the design.

## The read path

`MemoryBridge.propose_memory_candidates()` — `memory_bridge.py:1044`.

It queries the fine layer with `"a photo of a {target_category}"` in SBERT text space, keeps hits that clear a cosine floor, belong to the current scene, and are not near-duplicates of an already-proposed planner candidate, and materialises each survivor as a navigable candidate at the stored agent position, carrying the cosine through as `raw_score` so the downstream scorer favours semantically strong memories.

**This function is ADR-0008's "memory later plugs in as another proposer".**
The clean-room agent is a candidate-pool frontier explorer: proposers emit waypoint candidates, one scorer picks, a navmesh follower drives.
`propose_memory_candidates` already has exactly that shape — pool in, pool out — so wiring it back is adding a proposer to the list, not reopening the architecture.
Note what the seam does **not** carry: the layer is hard-filtered to the current scene (`memory_bridge.py:829`), so cross-environment reuse is structurally absent and is not a tuning question.

`retrieve()` (`memory_bridge.py:1467`) is the raw three-layer query underneath.
Only the fine layer was ever queried on the action path; `ltm_mid` was `false` in every run and the coarse layer was seeded but never read for a waypoint.
"Hierarchical 3-layer LTM" overstates what was measured — the measured effect is **fine layer plus memory-injected rerank**.

## The env-flag surface is superseded

`memory_bridge.py` reads `LTM_*` from the environment throughout — `LTM_TEMPORAL_CONTEXT`, `LTM_AUDIO_DOA`, `LTM_AUDIO_WRITE`, `LTM_FREEZE_SCENE`, `LTM_SEMANTIC_FRONTIER`, `REMEMBR_CONSUME_SINGLEGOAL` and more.

**ADR-0008 removed that surface.** The clean room carries behaviour, not flags: the two surviving experimental arms are enums on `RunConfig`, and `tests/mac/test_no_env_flags.py` asserts `os.environ` appears nowhere outside `audio/guard.py`'s logging pin and `env_check.py`.
So a revival does not port the flags — it decides which behaviours are real and gives them types.

This is also why the structural walker exempts this directory: `test_no_env_flags.py` would fire on it, and `test_layering.py` would fail on its imports.

## The levers already closed

Do not re-run these.
Each was built, measured, and closed; the reports are `PHASE2_ABLATION_REPORT.md` and the ADRs.

- **Coarse-affordance head** (Run 20). Built end-to-end with a CLIP zero-shot room classifier and frontier grounding. It proposes 1–4 candidates per episode and is **never chosen** — the reranker correctly prefers a concrete sighting to a position-free room prior. Correct, well-instrumented, inert.
- **The `R` importance head, trained** (Run 13). Two labels. Episode soft-SPL is unlearnable for a caption head, and a dead `R` makes memory over-fire and thrash. A per-keyframe `goal_object` label *is* learnable and recovers to heuristic-competitive, but does not beat the heuristic and hurts cold-start.
- **The `U` importance head, trained** (Runs 18–19). Three formulations — next-caption surprise, calibrated surprise, goal proximity — all land at roughly half the heuristic's warm effect with the same over-fire signature. Five angles across `R` and `U`, all ≤ heuristic.
- **The M4 temporal-context head.** A recency bonus on already-recalled same-category sightings. Exercised (selection count rose 25%) yet **bit-identical** on outcomes: the extra picks were credit re-attribution, not re-routing.
- **The S2 audio-DOA rerank head.** Zero-sum by construction — the bonus is mean-centred across candidates, and a single-goal episode recalls one instance clustered in one place, so every candidate sits on the same side and the bonus is exactly `0.0` for any weight. Inert by construction on this eval, not by miscalibration.
- **`write_audio_event`** (AudioGoal Step 2). Mechanism-verified, but redundant with vision on a line-of-sight seed: the write is a duplicate at a position the agent already mapped visually. It also stamps the **ground-truth** source position, so any win would be an oracle upper bound until a DOA-derived write exists.

The pattern across all six is one finding: a single-goal-per-episode eval does not reward recall, so every lever that is *about* disambiguation or timing has no regime in which to act.

## The bottleneck, measured

**SBERT instance discrimination, on the read side.**

Measured, not asserted (`diagnose_sbert_cosines.py`, `runs/diagnose-instance-sep.txt`): within-instance caption cosine **0.628** against between-instance-same-category **0.535**, so the embedding carries **+0.093** of instance signal.
The live query `"there is a {cat}"` then collapses that to a **0.047** rank gap.
The signal exists and the category-shaped query throws it away, which makes query construction the first lever and a trained detector not yet justified.

The captioner is **not** the lever, and that was closed for $0 rather than by a matrix.
A pre-registered 8-minute gate rendered 204 real HM3D keyframes at goal-instance view points, captioned each with both `Qwen2-VL-2B` and `internlm/CapRL-3B`, and measured within-versus-between instance separation per captioner: Qwen +0.146 against CapRL +0.129, i.e. the richer captioner is slightly **worse**, because SBERT mean-pools the longer caption and dilutes exactly the instance-distinctive tokens.
So the read side — the embedder, or the query — is the limiter.
The cheapest next test re-embeds the captions already sitting in `runs/caprl-gate/captions.json` with a stronger or asymmetric text encoder; it needs no new render and no new caption.
