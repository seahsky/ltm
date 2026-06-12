# Memory-grounded ReMEmbR planner waypoint — design

**Date:** 2026-05-25
**Branch:** phase2-readiness
**Status:** approved design → implementation plan

## Problem

In the `remembr-run62` validation smoke (real Qwen2.5-7B planner, `--backbone
remembr` setting 3), the ReMEmbR LLM's own waypoint won **0 of 79** decisions,
and `inspect_memory_rerank.py` showed why: the `remembr`-source `raw_score`
(LLM confidence) is **bimodal and degenerate** — `p50=0.100`, `p75=p90=max=0.632`.

Root cause (read, not guessed): `_llm_propose` (`embodied_memory/remembr_backbone.py`)
asks the 7B to **invent metric `(x,z)` coordinates** from a text-only prompt:

> `Goal: find a chair. Current position: x=1.2, y=0.1, z=3.4. Pick a waypoint (x, z).`

with no map, no frontier list, and no image. A small text LLM cannot do metric
spatial reasoning from that, so it either:

- **echoes the current position** → the regurgitation guard fires → `_stub_propose`
  forward-walk at `raw_score=0.1` (this is the `p50=0.1` cluster, >50% of decisions), or
- emits a **near-trivial low-confidence point** (`confidence=0.632` at `dist≈0.8 m`,
  barely above the 0.5 m regurgitation floor). At `temperature=0.0` the identical
  prompt yields the identical `0.632` every re-proposal — hence the exact cluster,
  not real per-decision confidence.

So the backbone produces no usable waypoints; the occupancy-aware frontier
injection (raw 0.875+) correctly out-scores it every time. This makes setting 3
effectively frontier-planner-driven rather than a faithful test of ReMEmbR.

This is an **input-signal** fix, not a scorer fix: rebalancing the rerank toward
this degenerate signal would chase noise. (The companion CLIP QuickGELU fix,
commit `9672fb7`, addresses the parallel flat-cosine root cause for the memory
signal; this spec addresses the planner.)

## Design

Stop asking the LLM to invent coordinates. The retrieval tool results **already**
surface each remembered observation as `t=<timestep> xz=(x,z) score=... cap="..."`
(`_summarize_hits`). So the LLM references a **retrieved memory by its timestep**,
and the waypoint is that memory's **stored observation position** — never an
invented coordinate. This is ReMEmbR's actual pattern: reason over retrieved
observations, then navigate to a remembered location.

### Protocol change

`_llm_propose` system prompt and `_parse_planner_reply`:

- **Old:** `ANSWER: x=<float>, z=<float>, confidence=<float>`
- **New:**
  - `ANSWER: goto_t=<timestep>, confidence=<float>` — navigate to the position of
    the observation made at `<timestep>`.
  - `ANSWER: explore` — nothing goal-relevant is remembered yet; defer.

The system prompt instructs: each `TOOL_RESULT` lists observations as
`t=<timestep> xz=(x,z) ... cap="..."`; choose `goto_t` from a timestep seen in a
tool result; reply `explore` if nothing relevant. The user prompt keeps the goal
and current position but frames the task as "navigate toward the goal using
remembered observations."

### Grounding

After the tool loop returns an answer:

1. **`goto_t=<t>`** → look up the `MemoryRecord` with that timestep
   (`builder.record_by_timestep(t)`, a new linear-scan helper over `_records`).
   Build a `FrontierCandidate(source="remembr")` at the record's stored `position`
   (xz), with `distance_m` / `bearing_rad` computed from the current agent pose.
2. **`raw_score` = retrieval cosine of the chosen memory vs the goal** — recomputed
   as `cos(text_embed("a photo of a {goal}"), record.caption_embedding)`, clamped
   to `[0, 1]`. The query string is `"a photo of a {goal}"`, matching the bridge's
   `propose_memory_candidates` query so the two memory signals are on the same
   scale. This is the same text-text similarity the builder's `retrieve_from_text`
   computes. The LLM's free-text `confidence` is kept only in the trace for
   logging, not used as `raw_score`.

### Defer → frontier (division of labor)

`explore`, no answer after `max_tool_calls`, a hallucinated `goto_t` not in memory,
or a `goto_t` resolving to ≈ the agent's current pose (the just-ingested keyframe,
zero displacement) all → `_llm_propose` returns **`[]`**.

`_propose_candidates` then yields only the frontier candidates, so the **frontier
planner drives exploration when memory holds nothing relevant**, and the
memory-grounded ReMEmbR pick drives once a relevant observation exists. This is
the intended fallback — frontier winning during early exploration is correct, not
a bug. It also removes the `0.1` stub-forward-walk candidates from the real-LLM
path entirely (the regurgitation guard becomes moot — there are no invented coords
to regurgitate).

### Robustness fallback (format non-compliance)

Small models may ignore the `goto_t` syntax and still emit `x,z`. To avoid
regressing to "no remembr candidate ever," a free-form `(x,z)` answer is **snapped
to the nearest `MemoryRecord` position** within a small radius (reuse
`REMEMBR_MIN_WAYPOINT_DIST`); if none is within radius, defer (`explore`). This
keeps every committed waypoint grounded in a real observation.

### Stub mode unchanged

`stub_mode` (no real LLM) keeps using `_stub_propose`. Only the real-LLM path
changes.

## Components touched

- `remembr_backbone.py`
  - `_llm_propose`: new system/user prompt; parse `goto_t`/`explore`; ground to
    record position; `raw_score` = goal-vs-memory cosine; defer → `[]`.
  - `_parse_planner_reply`: parse `goto_t=<int>`, `explore`, keep `x,z` for the
    snap fallback. Returns `kind ∈ {goto, explore, tool, answer_xy, unparseable}`.
  - `ReMEmbRBuilder.record_by_timestep(t)`: new helper.
  - Regurgitation guard: removed from the real-LLM path (moot); zero-displacement
    handled by the defer rule.
- `_stub_propose`, `_maybe_stop`, the build phase, and the bridge/rerank are
  **unchanged**.

## Testing

Unit tests in `embodied_memory/scripts/test_propose_candidates.py`, stubbing the
LLM `_llm_complete` to return canned replies (no Habitat/model load — same pattern
as existing sanity cases). Pin the builder with a few `MemoryRecord`s at known
positions/captions, then assert:

1. `ANSWER: goto_t=2` → one `remembr` candidate at record 2's position; `raw_score`
   = clamped goal-vs-record-2 cosine; correct `distance_m`/`bearing_rad`.
2. `ANSWER: goto_t=99` (not in memory) → `[]` (defer).
3. `goto_t` resolving to ≈ agent pose → `[]` (defer, zero displacement).
4. `ANSWER: explore` → `[]`.
5. Malformed reply → `[]`.
6. Free-form `ANSWER: x=.., z=..` near a record → snapped to that record's position
   (grounded); far from any record → `[]`.
7. A `TOOL:` reply is dispatched and the loop continues (existing behavior intact).

## Out of scope

- The CLIP QuickGELU fix (already shipped, `9672fb7`) — verified separately.
- Rerank calibration (whether a grounded remembr pick *wins* vs frontier/memory) —
  deliberately deferred; revisit once both input-signal fixes are verified, with
  `inspect_memory_rerank.py` against fresh logs.
- Switching the builder's flat memory from caption text-text to visual embeddings —
  a larger change; the LLM reads caption strings directly, so it is not required
  for grounding.
- Multi-scene / lifelong evaluation.

## Verification

1. Unit suite (`test_propose_candidates.py`) green in the race-setup env.
2. Re-run the cheap `remembr-run62` smoke; with `inspect_memory_rerank.py`, confirm
   the `remembr` `raw_score` distribution is **no longer `0.1`/`0.632` degenerate**
   — grounded picks carry real cosines, and `explore` decisions show no remembr
   candidate (frontier-only), i.e. the `0.1` stub cluster is gone.
3. Per-episode `remembr_stub_mode=false` (backbone still REAL).
