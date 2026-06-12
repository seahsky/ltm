# Memory-grounded ReMEmbR planner waypoint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the ReMEmbR LLM from inventing metric `(x,z)` coordinates (which degenerates: it won 0/79 decisions, conf bimodal 0.1/0.632); instead it references a retrieved memory by timestep and the waypoint becomes that memory's stored observation position, deferring to frontier exploration when nothing relevant is remembered.

**Architecture:** Change the planner's ANSWER protocol from free-form `x,z` to `goto_t=<timestep>` / `explore`. Add a builder lookup (`record_by_timestep`) and a planner grounding method (`_ground_answer`) that materializes a `FrontierCandidate(source="remembr")` at the referenced memory's position, with `raw_score` = the goal-vs-memory CLIP cosine. `explore` / no-answer / hallucinated-id / zero-displacement all return `[]`, so `_propose_candidates` falls through to the frontier candidates.

**Tech Stack:** Python, numpy, open_clip (CLIP text embeddings), HuggingFace transformers (the 7B planner — stubbed in tests). Tests run via the existing `embodied_memory/scripts/test_propose_candidates.py` sanity harness (no Habitat/model load).

**Spec:** `docs/superpowers/specs/2026-05-25-memory-grounded-remembr-planner-design.md`

---

## File structure

- **Modify** `embodied_memory/remembr_backbone.py`:
  - `_parse_planner_reply` (currently lines 885-908) — new grammar.
  - `ReMEmbRBuilder` — add `record_by_timestep` (near the `records` property, ~line 180).
  - `ReMEmbRPlanner` — add `_goal_memory_cosine` and `_ground_answer`; rewrite `_llm_propose` (currently lines 681-789, prompts 714-728, loop 732-758, grounding 760-789); remove the regurgitation guard.
  - Module docstring (~lines 24-26) — update the ANSWER-format description.
- **Modify** `embodied_memory/scripts/test_propose_candidates.py` — add three cases + register them in `main()`.

No other files change. `_stub_propose`, `_maybe_stop`, the build phase, `_dispatch_tool`, `_summarize_hits`, and the bridge/rerank are untouched.

---

### Task 1: New planner-reply grammar (`goto_t` / `explore`)

**Files:**
- Modify: `embodied_memory/remembr_backbone.py:885-908` (`_parse_planner_reply`)
- Test: `embodied_memory/scripts/test_propose_candidates.py` (add `case_remembr_parse` + register in `main`)

- [ ] **Step 1: Write the failing test**

Add this function to `embodied_memory/scripts/test_propose_candidates.py` (after `case_keyword_stop`, before `def main`):

```python
def case_remembr_parse():
    """Run-6.3 planner grammar: ANSWER references a remembered timestep
    (goto_t=) or defers (explore); legacy x,z still parses for the snap
    fallback; TOOL and unparseable unchanged."""
    rb = _load_file_as("embodied_memory._rb_parse",
                       _EMB_DIR / "remembr_backbone.py")
    p = rb._parse_planner_reply

    r = p("ANSWER: goto_t=2, confidence=0.8")
    assert r["kind"] == "goto" and r["timestep"] == 2 and abs(r["conf"] - 0.8) < 1e-6, r

    assert p("ANSWER: explore")["kind"] == "explore"
    assert p("answer: EXPLORE the next room")["kind"] == "explore"  # case-insensitive substring

    r = p("ANSWER: x=1.0, z=2.0, confidence=0.4")
    assert r["kind"] == "answer_xy" and r["xz_conf"] == (1.0, 2.0, 0.4), r

    r = p("TOOL: retrieve_from_text(chair)")
    assert r["kind"] == "tool" and r["tool_name"] == "retrieve_from_text" and r["tool_arg"] == "chair", r

    assert p("blah blah")["kind"] == "unparseable"
    assert p("ANSWER: nonsense")["kind"] == "unparseable"

    print("  case remembr_parse (goto/explore/xy/tool/unparseable): OK")
```

And register it in `main()` (the block of `case_*()` calls, after `case_keyword_stop()`):

```python
    case_keyword_stop()
    case_remembr_parse()
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `python3 embodied_memory/scripts/test_propose_candidates.py`
Expected: FAIL at `case_remembr_parse` — the current parser returns `{"kind": "answer", ...}` (no `goto`/`explore`/`answer_xy`), so the first `assert r["kind"] == "goto"` fails with a `KeyError`/`AssertionError`.

- [ ] **Step 3: Rewrite `_parse_planner_reply`**

Replace the body of `_parse_planner_reply` (lines 885-908) with:

```python
def _parse_planner_reply(reply: str) -> Dict[str, Any]:
    """Permissive parser for the LLM's reply line.

    Accepts ``TOOL: name(arg)`` or one of the ANSWER forms:
      - ``ANSWER: goto_t=<int>, confidence=<float>`` → navigate to a remembered
        observation (the timestep is grounded to its stored position downstream).
      - ``ANSWER: explore`` → nothing goal-relevant remembered yet; defer.
      - ``ANSWER: x=<float>, z=<float>, confidence=<float>`` → legacy free-form
        coordinate, kept only for the snap-to-nearest-memory robustness fallback.
    Returns ``{"kind": "goto"|"explore"|"answer_xy"|"tool"|"unparseable", ...}``.
    """
    s = reply.strip().splitlines()[0] if reply.strip() else ""
    low = s.lower()
    if low.startswith("answer:"):
        if "explore" in low:
            return {"kind": "explore"}
        t = _extract_float(s, "goto_t=")
        if t is not None:
            conf = _extract_float(s, "confidence=", default=0.5)
            return {"kind": "goto", "timestep": int(t), "conf": conf}
        x = _extract_float(s, "x=")
        z = _extract_float(s, "z=")
        if x is not None and z is not None:
            conf = _extract_float(s, "confidence=", default=0.5)
            return {"kind": "answer_xy", "xz_conf": (x, z, conf)}
        return {"kind": "unparseable"}
    if low.startswith("tool:"):
        body = s.split(":", 1)[1].strip()
        if "(" in body and body.endswith(")"):
            name = body.split("(", 1)[0].strip()
            arg = body[body.index("(") + 1 : -1]
            return {"kind": "tool", "tool_name": name, "tool_arg": arg}
        return {"kind": "unparseable"}
    return {"kind": "unparseable"}
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `python3 embodied_memory/scripts/test_propose_candidates.py`
Expected: PASS — `  case remembr_parse (goto/explore/xy/tool/unparseable): OK` and `All cases passed.`

- [ ] **Step 5: Commit**

```bash
git add embodied_memory/remembr_backbone.py embodied_memory/scripts/test_propose_candidates.py
git commit -m "feat(remembr): planner ANSWER grammar — goto_t/explore (grounded waypoints)"
```

---

### Task 2: Builder lookup + grounding method

**Files:**
- Modify: `embodied_memory/remembr_backbone.py` — add `ReMEmbRBuilder.record_by_timestep` (after the `records` property, ~line 181); add `ReMEmbRPlanner._goal_memory_cosine` and `ReMEmbRPlanner._ground_answer` (in the `ReMEmbRPlanner` class, e.g. just before `_llm_propose` at line 681).
- Test: `embodied_memory/scripts/test_propose_candidates.py` (add `case_remembr_grounding` + register)

- [ ] **Step 1: Write the failing test**

Add this function to `embodied_memory/scripts/test_propose_candidates.py` (after `case_remembr_parse`):

```python
def case_remembr_grounding():
    """Run-6.3 grounding: a goto_t answer materializes a remembr candidate at
    the referenced memory's stored position; raw_score = goal-vs-memory cosine;
    unknown timestep / zero-displacement / far free-form → defer (None)."""
    rb = _load_file_as("embodied_memory._rb_ground",
                       _EMB_DIR / "remembr_backbone.py")

    # Hermetic: _ground_answer reads REMEMBR_MIN_WAYPOINT_DIST at call time and
    # race-setup may export a different value. Pin the snap / zero-displacement
    # floor so this logic test is independent of live tuning (cf. case_keyword_stop).
    os.environ["REMEMBR_MIN_WAYPOINT_DIST"] = "0.5"

    def fake_embed(s):
        s = s.lower()
        v = np.zeros(4, dtype=np.float32)
        if "chair" in s:
            v[0] = 1.0
        elif "sofa" in s or "couch" in s:
            v[1] = 1.0
        else:
            v[2] = 1.0
        return v

    builder = rb.ReMEmbRBuilder(rb.ReMEmbRConfig(), text_embed_fn=fake_embed)

    def add(ts, cap, x, z):
        builder._records.append(rb.MemoryRecord(
            timestep=ts, timestamp=float(ts),
            position=np.array([x, 0.0, z], dtype=np.float32),
            caption=cap, caption_embedding=fake_embed(cap)))

    add(1, "a hallway with a window", 0.0, 4.0)     # non-goal, 4 m away
    add(2, "a wooden chair by the desk", 3.0, 0.0)  # the goal object, 3 m away
    add(3, "a window", 0.2, 0.1)                     # ~0.22 m → zero-displacement

    planner = rb.ReMEmbRPlanner(builder, rb.ReMEmbRConfig())
    agent = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    yaw = 0.0
    trace = rb.PlannerTrace(goal="chair")

    # 1. valid goto_t=2 → candidate at (3,0); raw_score = cos(chair, chair) = 1.0
    c = planner._ground_answer("chair", ("goto", 2, 0.8), agent, yaw, trace)
    assert c is not None and c.source == "remembr", c
    assert abs(c.world_xy[0] - 3.0) < 1e-5 and abs(c.world_xy[1] - 0.0) < 1e-5, c.world_xy
    assert abs(c.raw_score - 1.0) < 1e-5, c.raw_score
    assert abs(c.distance_m - 3.0) < 1e-4, c.distance_m
    assert c.metadata["grounded_timestep"] == 2, c.metadata

    # 2. unknown timestep → defer
    assert planner._ground_answer("chair", ("goto", 99, 0.8), agent, yaw, trace) is None

    # 3. zero displacement (record 3 ~0.22 m from agent) → defer
    assert planner._ground_answer("chair", ("goto", 3, 0.8), agent, yaw, trace) is None

    # 4. free-form xy near record 2 (within 0.5 m) → snapped to (3,0)
    c = planner._ground_answer("chair", ("xy", 3.1, 0.0, 0.5), agent, yaw, trace)
    assert c is not None and abs(c.world_xy[0] - 3.0) < 1e-5, c

    # 5. free-form xy far from any record → defer
    assert planner._ground_answer("chair", ("xy", 20.0, 20.0, 0.5), agent, yaw, trace) is None

    # 6. raw_score reflects the goal: goto a non-chair record → cos 0
    c = planner._ground_answer("chair", ("goto", 1, 0.8), agent, yaw, trace)
    assert c is not None and abs(c.raw_score - 0.0) < 1e-5, c.raw_score

    print("  case remembr_grounding (goto/unknown/zero/snap/far/cos): OK")
```

Register in `main()`:

```python
    case_remembr_parse()
    case_remembr_grounding()
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `python3 embodied_memory/scripts/test_propose_candidates.py`
Expected: FAIL at `case_remembr_grounding` with `AttributeError: 'ReMEmbRPlanner' object has no attribute '_ground_answer'`.

- [ ] **Step 3: Add `record_by_timestep` to `ReMEmbRBuilder`**

Insert after the `records` property (after line 181, which is `return self._records`):

```python
    def record_by_timestep(self, timestep: int) -> Optional[MemoryRecord]:
        """Return the flat-memory record observed at ``timestep``, or None.
        Timesteps are unique per ingested keyframe; linear scan (horizon-bounded)."""
        for r in self._records:
            if r.timestep == int(timestep):
                return r
        return None
```

- [ ] **Step 4: Add `_goal_memory_cosine` and `_ground_answer` to `ReMEmbRPlanner`**

Insert these two methods into the `ReMEmbRPlanner` class, immediately before `def _llm_propose` (line 681):

```python
    def _goal_memory_cosine(self, goal: str, record: "MemoryRecord") -> float:
        """CLIP text-text cosine of "a photo of a {goal}" vs the record's caption
        embedding, clamped to [0,1]. Matches the bridge's propose_memory_candidates
        query so remembr- and memory-source raw_scores share a scale."""
        fn = self.builder._text_embed_fn
        if fn is None:
            return 0.0
        q = np.asarray(fn(f"a photo of a {goal}"), dtype=np.float32)
        e = np.asarray(record.caption_embedding, dtype=np.float32)
        nq, ne = float(np.linalg.norm(q)), float(np.linalg.norm(e))
        if nq < 1e-8 or ne < 1e-8:
            return 0.0
        return float(np.clip(float(np.dot(q / nq, e / ne)), 0.0, 1.0))

    def _ground_answer(
        self,
        goal: str,
        answer: tuple,
        agent_pose: np.ndarray,
        agent_yaw: float,
        trace: "PlannerTrace",
    ) -> Optional[FrontierCandidate]:
        """Turn a parsed answer into a grounded waypoint at a remembered position.

        ``answer`` is ("goto", timestep, conf) or ("xy", x, z, conf). Returns a
        FrontierCandidate(source="remembr") at the referenced memory's stored xz,
        or None to defer to frontier exploration (unknown timestep, no nearby
        memory for a free-form xy, or a zero-displacement pick).
        """
        ax, ay, az = float(agent_pose[0]), float(agent_pose[1]), float(agent_pose[2])
        floor = float(os.environ.get("REMEMBR_MIN_WAYPOINT_DIST", "0.5"))

        if answer[0] == "goto":
            _, t, conf = answer
            rec = self.builder.record_by_timestep(int(t))
            if rec is None:
                trace.tool_calls.append({"tool": "goto_rejected_unknown_t", "t": int(t)})
                return None
        else:  # "xy" — snap an invented coordinate to the nearest real observation
            _, x, z, conf = answer
            hits = self.builder.retrieve_from_position(
                np.array([x, ay, z], dtype=np.float32), top_k=1
            )
            if not hits:
                return None
            rec, snap_d = hits[0]
            if snap_d > floor:
                trace.tool_calls.append(
                    {"tool": "answer_xy_rejected_far_from_memory", "snap_d": float(snap_d)})
                return None

        rx, rz = float(rec.position[0]), float(rec.position[2])
        dx, dz = rx - ax, rz - az
        dist = math.hypot(dx, dz)
        if dist < floor:
            trace.tool_calls.append(
                {"tool": "goto_rejected_zero_displacement", "t": int(rec.timestep), "dist": float(dist)})
            return None

        bearing = _rel_bearing(dx, dz, agent_yaw)
        cos = self._goal_memory_cosine(goal, rec)
        self._candidate_counter += 1
        trace.chosen_xyz = [rx, ay, rz]
        trace.confidence = float(cos)
        return FrontierCandidate(
            candidate_id=self._candidate_counter + 50_000,
            world_xy=np.array([rx, rz], dtype=np.float32),
            grid_rc=(-1, -1),
            distance_m=float(dist),
            bearing_rad=float(bearing),
            cluster_size=0,
            raw_score=float(cos),
            source="remembr",
            metadata={
                "grounded_timestep": int(rec.timestep),
                "ground_cos": float(cos),
                "llm_confidence": float(conf),
            },
        )
```

- [ ] **Step 5: Run the test, verify it passes**

Run: `python3 embodied_memory/scripts/test_propose_candidates.py`
Expected: PASS — `  case remembr_grounding (goto/unknown/zero/snap/far/cos): OK` and `All cases passed.`

- [ ] **Step 6: Commit**

```bash
git add embodied_memory/remembr_backbone.py embodied_memory/scripts/test_propose_candidates.py
git commit -m "feat(remembr): record_by_timestep + grounded-waypoint scorer (_ground_answer)"
```

---

### Task 3: Rewire `_llm_propose` to ground answers + defer

**Files:**
- Modify: `embodied_memory/remembr_backbone.py:709-789` (the `_llm_propose` prompt, loop, and grounding tail — drops the regurgitation guard); module docstring ~lines 24-26.
- Test: `embodied_memory/scripts/test_propose_candidates.py` (add `case_remembr_llm_loop` + register)

- [ ] **Step 1: Write the failing test**

Add this function to `embodied_memory/scripts/test_propose_candidates.py` (after `case_remembr_grounding`):

```python
def case_remembr_llm_loop():
    """Run-6.3 loop wiring: a TOOL turn dispatches retrieval, then a goto_t
    answer grounds to that record's position; explore → defer ([]). LLM I/O is
    stubbed (no model load); a dummy torch lets _llm_propose's guard pass."""
    import types
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.modules["torch"] = types.ModuleType("torch")

    rb = _load_file_as("embodied_memory._rb_loop",
                       _EMB_DIR / "remembr_backbone.py")
    os.environ["REMEMBR_MIN_WAYPOINT_DIST"] = "0.5"

    def fake_embed(s):
        s = s.lower()
        v = np.zeros(4, dtype=np.float32)
        if "chair" in s:
            v[0] = 1.0
        else:
            v[2] = 1.0
        return v

    builder = rb.ReMEmbRBuilder(rb.ReMEmbRConfig(), text_embed_fn=fake_embed)
    builder._records.append(rb.MemoryRecord(
        timestep=2, timestamp=2.0, position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        caption="a wooden chair by the desk", caption_embedding=fake_embed("a wooden chair by the desk")))

    planner = rb.ReMEmbRPlanner(builder, rb.ReMEmbRConfig())
    planner._lazy_load_llm = lambda: None
    planner._format_chat = lambda sys_p, usr_p, hist: "prompt"  # skip tokenizer

    agent = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # tool turn → goto answer: grounds to record 2's position (3,0)
    replies = iter(["TOOL: retrieve_from_text(chair)", "ANSWER: goto_t=2, confidence=0.7"])
    planner._llm_complete = lambda prompt: next(replies)
    out = planner._llm_propose("chair", agent, 0.0, 3, rb.PlannerTrace(goal="chair"))
    assert len(out) == 1 and out[0].source == "remembr", out
    assert abs(out[0].world_xy[0] - 3.0) < 1e-5 and abs(out[0].raw_score - 1.0) < 1e-5, out[0]

    # explore → defer to frontier (empty list, NOT a stub forward-walk)
    planner._llm_complete = lambda prompt: "ANSWER: explore"
    assert planner._llm_propose("chair", agent, 0.0, 3, rb.PlannerTrace(goal="chair")) == []

    print("  case remembr_llm_loop (tool->goto grounds; explore->defer): OK")
```

Register in `main()`:

```python
    case_remembr_grounding()
    case_remembr_llm_loop()
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `python3 embodied_memory/scripts/test_propose_candidates.py`
Expected: FAIL at `case_remembr_llm_loop` — the current `_llm_propose` parses `ANSWER: goto_t=2` as unparseable, falls to `_stub_propose`, and returns a forward-walk candidate (not at `(3,0)`), so `abs(out[0].world_xy[0] - 3.0) < 1e-5` fails.

- [ ] **Step 3: Rewrite the `_llm_propose` prompt/loop/tail**

Replace lines 714-789 (from the `sys_prompt = (` assignment through the end of the method — i.e. the old prompt, loop, regurgitation guard, and primary-candidate construction) with:

```python
        sys_prompt = (
            "You are a navigation planner with three retrieval tools:\n"
            "  - retrieve_from_text(<query>): find past observations matching the query\n"
            "  - retrieve_from_position(x,y,z): find past observations near a coordinate\n"
            "  - retrieve_from_time(<t>): find past observations near a timestamp\n"
            "Each TOOL_RESULT lists past observations as: t=<timestep> xz=(x,z) score=.. cap=\"..\".\n"
            "Reply with EXACTLY one of:\n"
            "  TOOL: <name>(<arg>)\n"
            "  ANSWER: goto_t=<timestep>, confidence=<float>   (navigate to that remembered observation)\n"
            "  ANSWER: explore                                  (nothing goal-relevant remembered yet)\n"
            "Choose goto_t from a timestep shown in a TOOL_RESULT whose caption is most relevant to the\n"
            "goal. Reply 'ANSWER: explore' if no remembered observation is relevant. Stop once you ANSWER."
        )
        ax, ay, az = float(agent_pose[0]), float(agent_pose[1]), float(agent_pose[2])
        user_prompt = (
            f"Goal: find a {goal}. Current position: x={ax:.2f}, y={ay:.2f}, z={az:.2f}. "
            f"Use the tools to recall where relevant things were seen, then ANSWER with the "
            f"timestep to navigate toward (or explore)."
        )
        history: List[str] = []
        answer: Optional[tuple] = None

        for _ in range(self.config.max_tool_calls):
            prompt = self._format_chat(sys_prompt, user_prompt, history)
            reply = self._llm_complete(prompt)
            history.append(reply)
            parsed = _parse_planner_reply(reply)
            kind = parsed["kind"]
            if kind == "goto":
                answer = ("goto", parsed["timestep"], parsed["conf"])
                trace.tool_calls.append({"tool": "answer_goto", "t": parsed["timestep"], "reply": reply[:200]})
                break
            if kind == "explore":
                trace.tool_calls.append({"tool": "answer_explore", "reply": reply[:200]})
                return []  # defer to frontier exploration
            if kind == "answer_xy":
                answer = ("xy", parsed["xz_conf"][0], parsed["xz_conf"][1], parsed["xz_conf"][2])
                trace.tool_calls.append({"tool": "answer_xy", "reply": reply[:200]})
                break
            if kind == "tool":
                hits = self._dispatch_tool(parsed["tool_name"], parsed["tool_arg"], agent_pose)
                trace.tool_calls.append(
                    {"tool": parsed["tool_name"], "arg": parsed["tool_arg"], "n_hits": len(hits)})
                history.append(_summarize_hits(hits))
            else:
                trace.tool_calls.append({"tool": "unparseable", "reply": reply[:200]})
                break

        if answer is None:
            return []  # no usable answer after the tool budget → defer to frontier

        cand = self._ground_answer(goal, answer, agent_pose, agent_yaw, trace)
        return [cand] if cand is not None else []
```

Note: this deletes the old free-form-coordinate parsing, the regurgitation guard (`if dist < ...: return self._stub_propose(...)`), and the `primary = FrontierCandidate(...)` block — `_ground_answer` and the `explore`/`None` defers replace all of them.

- [ ] **Step 4: Update the module docstring**

In the module docstring near line 24-26, replace the ANSWER-format description. Change:

```
  The LLM is allowed up to ``max_tool_calls`` rounds, then it commits to a
  target xyz; we wrap that pose as a ``FrontierCandidate(source="remembr")``
```

to:

```
  The LLM is allowed up to ``max_tool_calls`` rounds, then it commits by
  REFERENCING a retrieved observation (``ANSWER: goto_t=<timestep>``) — the
  waypoint is that memory's stored position, never an invented coordinate — or
  defers with ``ANSWER: explore``. The grounded pose is wrapped as a
  ``FrontierCandidate(source="remembr")``
```

- [ ] **Step 5: Run the test, verify it passes**

Run: `python3 embodied_memory/scripts/test_propose_candidates.py`
Expected: PASS — `  case remembr_llm_loop (tool->goto grounds; explore->defer): OK` and `All cases passed.`

- [ ] **Step 6: Commit**

```bash
git add embodied_memory/remembr_backbone.py embodied_memory/scripts/test_propose_candidates.py
git commit -m "feat(remembr): ground LLM waypoints in retrieved memory positions; explore->frontier

LLM answers goto_t=<timestep> (waypoint = that memory's stored position) or
explore; defer (explore/no-answer/unknown-id/zero-displacement) returns [] so
_propose_candidates falls through to frontier exploration. Removes the invented-
coordinate path and its regurgitation guard. Kills the 0.1/0.632 degeneracy."
```

---

### Task 4: Full sanity suite + verification handoff

**Files:** none (verification only).

- [ ] **Step 1: Run the complete sanity suite**

Run: `python3 embodied_memory/scripts/test_propose_candidates.py`
Expected: every case prints `OK`, ending with `All cases passed.` — including the three new cases:
```
  case remembr_parse (goto/explore/xy/tool/unparseable): OK
  case remembr_grounding (goto/unknown/zero/snap/far/cos): OK
  case remembr_llm_loop (tool->goto grounds; explore->defer): OK
All cases passed.
```

- [ ] **Step 2: Push so RACE can pull**

```bash
git push origin phase2-readiness
```

- [ ] **Step 3: Verify on RACE (with the CLIP fix already in place)**

On the RACE host, re-run the cheap smoke and inspect the planner signal:

```bash
cd ~/ltm && git pull --ff-only
bash scripts/race-smoke.sh --backbone remembr --setting 3 \
    --scenes "TEEsavR23oF wcojb4TFT35" --n-episodes 2 --target any --tag remembr-run63
python embodied_memory/scripts/inspect_memory_rerank.py runs/remembr-run63-*
```

Expected:
- `[3/5]` sanity suite reaches the three new cases and prints `All cases passed.`
- In `inspect_memory_rerank.py`: the `remembr raw_score` distribution is **no longer the `0.1`/`0.632` degenerate cluster** — grounded picks carry real goal-vs-memory cosines, and `explore` decisions contribute no `remembr` candidate (the `0.1` stub-forward cluster is gone).
- Per-episode `remembr_stub_mode=false` (backbone still REAL).

This is the end of the planner-grounding plan. The rerank-calibration question (whether a grounded remembr/memory pick should *win* vs frontier) remains deliberately deferred — revisit with `inspect_memory_rerank.py` once both input-signal fixes (CLIP `9672fb7` + this) are verified together.
```
