"""
ReMEmbR backbone — VLM captioner + LLM agent planner.

Drop-in replacement for the proof-of-life ``SemanticCaptioner`` +
``FrontierPlanner.propose`` pair, based on Anwar et al. (NVIDIA ICRA 2025).

Two phases:

- **Build phase**  (``ReMEmbRBuilder``):
  Per keyframe, caption the RGB with a local VLM (default: LLaVA-1.6, with
  VILA as a swap-in) and write
  ``(caption_embedding, position, timestamp, video_chunk_idx)`` to a
  per-episode FAISS index that lives alongside the existing 3-layer LTM.
  This is the **flat** memory the ReMEmbR paper uses; our HierarchicalLTM
  sits on top via ``EmbodiedMemoryBridge``.

- **Query phase**  (``ReMEmbRPlanner``):
  At each decision step a local instruction-tuned LLM (default:
  Mistral-7B-Instruct, Llama-3-8B-Instruct also supported) holds three
  retrieval tools:
      - ``retrieve_from_text(query)``     → top-K caption hits
      - ``retrieve_from_position(xyz)``   → top-K nearest-in-space hits
      - ``retrieve_from_time(t)``         → top-K nearest-in-time hits
  The LLM is allowed up to ``max_tool_calls`` rounds, then it commits by
  REFERENCING a retrieved observation (``ANSWER: goto_t=<timestep>``) — the
  waypoint is that memory's stored position, never an invented coordinate — or
  defers with ``ANSWER: explore``. The grounded pose is wrapped as a
  ``FrontierCandidate(source="remembr")``
  so the existing rerank + memory-injection + low-level controller pipeline
  consumes it unchanged.

Design rules:
- **Lazy load only.** Model weights are loaded on first ``encode()``/
  ``propose()`` so ``import remembr_backbone`` doesn't require torch /
  transformers / GPU and the test suite stays cheap.
- **Stub mode when weights absent.** If the configured weights aren't on
  disk, we don't crash — we return a deterministic placeholder so the
  ``--backbone remembr`` plumbing is testable end-to-end without a download.
  The runner logs a clear ``WARNING(stub_mode=True)`` in the per-episode
  metadata so this is never mistaken for a real run.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# HM3D ObjectNav goal categories → caption synonyms for the keyword STOP. The
# CLIP text-vs-text cosine STOP proved anti-discriminative (full room captions
# score ~0.72-0.77 vs ANY goal word — a "hole in the wall" caption matched
# "plant" at 0.736), so we match the goal object as a whole word in the
# Qwen-VL caption instead. Word-boundary matched so "bedroom" does NOT satisfy
# goal "bed".
_GOAL_SYNONYMS: Dict[str, List[str]] = {
    "chair": ["chair"],
    "bed": ["bed"],
    "plant": ["plant", "potted plant", "houseplant"],
    "toilet": ["toilet"],
    "tv_monitor": ["tv", "television", "monitor", "screen"],
    "sofa": ["sofa", "couch"],
}


def _goal_terms(goal: str) -> List[str]:
    """Goal category + caption synonyms, lowercased."""
    key = str(goal).strip().lower()
    terms = set(_GOAL_SYNONYMS.get(key, []))
    terms.add(key.replace("_", " "))
    return [t for t in terms if t]


def _caption_mentions(caption: str, terms: List[str]) -> Optional[str]:
    """First goal term appearing as a whole word in the caption (case-
    insensitive), else None. Word-boundary so 'bedroom' != goal 'bed'."""
    cap = str(caption).lower()
    for t in terms:
        if re.search(r"\b" + re.escape(t) + r"\b", cap):
            return t
    return None

from .frontier_planner import FrontierCandidate


def _warn_stub(role: str, detail: str) -> None:
    """Loudly announce a stub fallback on stderr. Silent stub fallback hid for
    the entire project once (missing ``accelerate`` → every ``--backbone
    remembr`` run was stub); a visible warning makes that impossible to miss.
    Fires at most once per role per process (lazy-load short-circuits after)."""
    import sys

    print(
        f"WARNING(stub_mode=True): ReMEmbR {role} falling back to STUB — {detail}. "
        f"Captions/proposals will be deterministic placeholders, NOT real model "
        f"output. Set REMEMBR_STRICT=1 to surface the full error.",
        file=sys.stderr,
        flush=True,
    )


# ----------------------------------------------------------------------
# config + records
# ----------------------------------------------------------------------


@dataclass
class ReMEmbRConfig:
    """Captioner + planner model selection. Paths are env-var-overridable so
    a user can point at any local HF snapshot without code changes."""
    captioner_model: str = os.environ.get(
        "REMEMBR_CAPTIONER_MODEL", "llava-hf/llava-v1.6-mistral-7b-hf"
    )
    captioner_dtype: str = os.environ.get("REMEMBR_CAPTIONER_DTYPE", "float16")
    planner_model: str = os.environ.get(
        "REMEMBR_PLANNER_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
    )
    planner_dtype: str = os.environ.get("REMEMBR_PLANNER_DTYPE", "float16")
    device: Optional[str] = os.environ.get("REMEMBR_DEVICE") or None
    max_tool_calls: int = int(os.environ.get("REMEMBR_MAX_TOOL_CALLS", "4"))
    max_caption_tokens: int = int(os.environ.get("REMEMBR_MAX_CAPTION_TOKENS", "64"))
    max_planner_tokens: int = int(os.environ.get("REMEMBR_MAX_PLANNER_TOKENS", "256"))
    # When the requested weights aren't available, fall back to stub mode
    # rather than crashing. Set REMEMBR_STRICT=1 to disable.
    strict: bool = os.environ.get("REMEMBR_STRICT", "0") == "1"


@dataclass
class MemoryRecord:
    """One flat ReMEmbR memory entry — paper-faithful schema."""
    timestep: int
    timestamp: float
    position: np.ndarray              # (3,) world xyz at this keyframe
    caption: str
    caption_embedding: np.ndarray     # text embedding of caption (CLIP-text by default)
    video_chunk_idx: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# build phase
# ----------------------------------------------------------------------


class ReMEmbRBuilder:
    """Per-episode flat memory index + VLM captioner.

    Holds a list of MemoryRecords; their caption embeddings are stacked into
    a numpy array for fast cosine search. We deliberately avoid FAISS for the
    flat index — N is bounded by ``max_steps / keyframe_every`` (≤ 50 records
    per episode), so a numpy matmul is faster and removes a dep at runtime.
    """

    def __init__(
        self,
        config: Optional[ReMEmbRConfig] = None,
        text_embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.config = config or ReMEmbRConfig()
        self._text_embed_fn = text_embed_fn  # required for indexing captions
        self._model = None
        self._processor = None
        self._device = None
        self._stub_mode = False
        self._records: List[MemoryRecord] = []
        self._episode_started_at: Optional[float] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Begin a fresh episode — flush the per-episode flat memory."""
        self._records = []
        self._episode_started_at = time.time()

    @property
    def records(self) -> List[MemoryRecord]:
        return self._records

    def record_by_timestep(self, timestep: int) -> Optional[MemoryRecord]:
        """Return the flat-memory record observed at ``timestep``, or None.
        Timesteps are unique per ingested keyframe; linear scan (horizon-bounded)."""
        for r in self._records:
            if r.timestep == int(timestep):
                return r
        return None

    @property
    def stub_mode(self) -> bool:
        return self._stub_mode

    @property
    def model(self):
        """Read-only handle to the loaded Qwen2-VL model (None until built)."""
        return self._model

    @property
    def processor(self):
        """Read-only handle to the loaded Qwen2-VL processor (None until built)."""
        return self._processor

    @property
    def device(self):
        """Device the model is loaded on (None until built)."""
        return getattr(self, "_device", None)

    def attach_text_embed_fn(self, fn: Callable[[str], np.ndarray]):
        """Late-binding setter — convenient when the embedder is owned by the
        bridge/perception layer and constructed after this object."""
        self._text_embed_fn = fn

    # ------------------------------------------------------------------
    # captioner (lazy)
    # ------------------------------------------------------------------

    def _lazy_load_captioner(self):
        if self._model is not None or self._stub_mode:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError as e:
            if self.config.strict:
                raise RuntimeError(
                    "transformers / torch are required for ReMEmbR captioner"
                ) from e
            _warn_stub("captioner", "transformers/torch not importable")
            self._stub_mode = True
            return

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, self.config.captioner_dtype, torch.float16)
        try:
            self._processor = AutoProcessor.from_pretrained(self.config.captioner_model)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.config.captioner_model, torch_dtype=dtype, device_map=device
            ).eval()
            self._device = device
        except Exception as e:
            if self.config.strict:
                raise RuntimeError(
                    f"Failed to load captioner {self.config.captioner_model}: {e}"
                ) from e
            _warn_stub("captioner", f"{self.config.captioner_model}: {e}")
            self._stub_mode = True

    # ------------------------------------------------------------------
    # caption + index
    # ------------------------------------------------------------------

    def caption_and_index(
        self,
        rgb: np.ndarray,
        agent_position: np.ndarray,
        timestep: int,
        prompt: str = "Describe the scene in one sentence.",
        video_chunk_idx: Optional[int] = None,
    ) -> MemoryRecord:
        """Caption an RGB frame and append it to the flat memory."""
        if self._text_embed_fn is None:
            raise RuntimeError(
                "ReMEmbRBuilder: attach a text_embed_fn before caption_and_index()"
            )

        self._lazy_load_captioner()

        if self._stub_mode:
            # Deterministic captions so paired ablation runs remain comparable
            # even without the real VLM (e.g. an analyzer-side smoke test).
            caption = f"stub-caption step={timestep}"
        else:
            caption = self._caption_rgb(rgb, prompt)

        emb = np.asarray(self._text_embed_fn(caption), dtype=np.float32)
        record = MemoryRecord(
            timestep=int(timestep),
            timestamp=time.time() - (self._episode_started_at or time.time()),
            position=np.asarray(agent_position, dtype=np.float32),
            caption=str(caption),
            caption_embedding=emb,
            video_chunk_idx=video_chunk_idx,
            metadata={"stub_mode": bool(self._stub_mode)},
        )
        self._records.append(record)
        return record

    def _caption_rgb(self, rgb: np.ndarray, prompt: str) -> str:
        """One forward pass of the local VLM. Lazy-loaded; returns a single
        sentence English caption.

        Uses ``processor.apply_chat_template`` so the same code path serves
        any HF VLM that ships a chat template — LLaVA-Next, LLaVA-1.5,
        SmolVLM (Idefics3), Qwen2-VL, etc. The model class is resolved
        via ``AutoModelForVision2Seq``.
        """
        import torch
        from PIL import Image

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        img = Image.fromarray(rgb)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=text, images=img, return_tensors="pt").to(self._device)
        n_input = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_caption_tokens,
                do_sample=False,
            )
        new_tokens = out[0, n_input:]
        decoded = self._processor.batch_decode(
            new_tokens.unsqueeze(0), skip_special_tokens=True
        )[0]
        for term in (". ", "\n"):
            idx = decoded.find(term)
            if idx > 0:
                decoded = decoded[: idx + 1]
                break
        return decoded.strip()

    # ------------------------------------------------------------------
    # retrieval tools (used by the planner)
    # ------------------------------------------------------------------

    def retrieve_from_text(
        self,
        query: str,
        top_k: int = 3,
        min_cosine: float = 0.18,
    ) -> List[Tuple[MemoryRecord, float]]:
        if not self._records or self._text_embed_fn is None:
            return []
        q = np.asarray(self._text_embed_fn(query), dtype=np.float32)
        return self._cosine_top_k(q, top_k, min_cosine)

    def retrieve_from_position(
        self,
        xyz: np.ndarray,
        top_k: int = 3,
        max_distance_m: float = 30.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        if not self._records:
            return []
        target = np.asarray(xyz, dtype=np.float32)
        scored: List[Tuple[MemoryRecord, float]] = []
        for r in self._records:
            d = float(np.linalg.norm(r.position - target))
            if d <= max_distance_m:
                scored.append((r, d))
        scored.sort(key=lambda kv: kv[1])
        return scored[:top_k]

    def retrieve_from_time(
        self,
        t: float,
        top_k: int = 3,
    ) -> List[Tuple[MemoryRecord, float]]:
        if not self._records:
            return []
        scored = [(r, abs(r.timestamp - float(t))) for r in self._records]
        scored.sort(key=lambda kv: kv[1])
        return scored[:top_k]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _cosine_top_k(
        self,
        query: np.ndarray,
        top_k: int,
        min_cosine: float,
    ) -> List[Tuple[MemoryRecord, float]]:
        if not self._records:
            return []
        emb = np.stack([r.caption_embedding for r in self._records], axis=0)
        qn = float(np.linalg.norm(query))
        if qn <= 0.0:
            return []
        sims = emb @ query / (np.linalg.norm(emb, axis=1) * qn + 1e-8)
        order = np.argsort(-sims)
        out: List[Tuple[MemoryRecord, float]] = []
        for idx in order[: max(top_k * 2, top_k)]:
            cos = float(sims[idx])
            if cos < min_cosine:
                continue
            out.append((self._records[int(idx)], cos))
            if len(out) >= top_k:
                break
        return out


# ----------------------------------------------------------------------
# query phase
# ----------------------------------------------------------------------


@dataclass
class PlannerTrace:
    """Per-decision-step diagnostic — tool calls + final pose. Logged into
    the per-episode JSON so we can debug LLM behaviour offline."""
    goal: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    chosen_xyz: Optional[List[float]] = None
    confidence: float = 0.0
    stub_mode: bool = False


class ReMEmbRPlanner:
    """LLM agent + retrieval tools. Returns 1–3 ``FrontierCandidate``s."""

    def __init__(
        self,
        builder: ReMEmbRBuilder,
        config: Optional[ReMEmbRConfig] = None,
    ):
        self.builder = builder
        self.config = config or builder.config
        self._llm = None
        self._tokenizer = None
        self._device = None
        self._stub_mode = False
        self._last_trace: Optional[PlannerTrace] = None
        self._candidate_counter = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Reset per-episode state (not the LLM weights)."""
        self._candidate_counter = 0
        self._last_trace = None

    @property
    def stub_mode(self) -> bool:
        return self._stub_mode or self.builder.stub_mode

    @property
    def last_trace(self) -> Optional[PlannerTrace]:
        return self._last_trace

    # ------------------------------------------------------------------
    # lazy LLM load
    # ------------------------------------------------------------------

    def _lazy_load_llm(self):
        if self._llm is not None or self._stub_mode:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            if self.config.strict:
                raise RuntimeError("transformers / torch are required for ReMEmbR planner") from e
            _warn_stub("planner", "transformers/torch not importable")
            self._stub_mode = True
            return

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, self.config.planner_dtype, torch.float16)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.planner_model)
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.config.planner_model, torch_dtype=dtype, device_map=device
            ).eval()
            self._device = device
        except Exception as e:
            if self.config.strict:
                raise RuntimeError(
                    f"Failed to load planner {self.config.planner_model}: {e}"
                ) from e
            _warn_stub("planner", f"{self.config.planner_model}: {e}")
            self._stub_mode = True

    # ------------------------------------------------------------------
    # propose
    # ------------------------------------------------------------------

    def propose(
        self,
        goal: str,
        agent_pose: np.ndarray,
        agent_yaw: float,
        max_candidates: int = 3,
        current_step: int = 0,
    ) -> List[FrontierCandidate]:
        """Return up to ``max_candidates`` waypoints toward the goal."""
        self._lazy_load_llm()
        trace = PlannerTrace(goal=str(goal), stub_mode=self.stub_mode)

        # Grounded pre-LLM STOP check: if ReMEmbR's flat memory already holds
        # a recent observation whose caption matches the goal AND lies within
        # the success radius of the agent's current xz, short-circuit the LLM
        # and emit a stop candidate. Runner converts this to action=0.
        stop_cand = self._maybe_stop(goal, agent_pose, trace, current_step=current_step)
        if stop_cand is not None:
            self._last_trace = trace
            return [stop_cand]

        if self.stub_mode:
            candidates = self._stub_propose(goal, agent_pose, agent_yaw, max_candidates, trace)
        else:
            candidates = self._llm_propose(goal, agent_pose, agent_yaw, max_candidates, trace)

        self._last_trace = trace
        return candidates

    # ------------------------------------------------------------------
    # grounded STOP check
    # ------------------------------------------------------------------

    # Keyword STOP is the default (REMEMBR_STOP_KEYWORD=1): match the goal
    # object as a whole word in the caption. Set to 0 to fall back to the legacy
    # CLIP text-vs-text cosine path (kept only for comparison — it is anti-
    # discriminative: room captions score ~0.72-0.77 vs any goal word).
    STOP_USE_KEYWORD: bool = os.environ.get("REMEMBR_STOP_KEYWORD", "1") == "1"
    # Cosine threshold for the legacy text-vs-text path (unused in keyword mode).
    STOP_COS_THRESHOLD: float = float(os.environ.get("REMEMBR_STOP_COS", "0.25"))
    # Max xz distance (m) between the matching observation and the agent's
    # current position. HM3D ObjectNav success radius is 1.0 m; add slack
    # for keyframe-position quantisation.
    STOP_DIST_THRESHOLD: float = float(os.environ.get("REMEMBR_STOP_DIST", "1.5"))
    # Minimum step before STOP may fire. The very first keyframe is ingested
    # at the agent's start pose, so the geometric guard above passes trivially
    # — without a step floor, an entry-shot caption mentioning the goal class
    # auto-STOPs the agent at step 0 with success radius irrelevant.
    STOP_MIN_STEP: int = int(os.environ.get("REMEMBR_STOP_MIN_STEP", "8"))

    def _maybe_stop(
        self,
        goal: str,
        agent_pose: np.ndarray,
        trace: "PlannerTrace",
        current_step: int = 0,
    ) -> Optional[FrontierCandidate]:
        """Emit a stop candidate iff a strictly-older observation that names the
        goal object lies within the success radius of the agent's current
        position. Default match is a whole-word caption keyword (discriminative);
        ``REMEMBR_STOP_KEYWORD=0`` falls back to the legacy cosine path."""
        # Step floor: don't allow STOP until the agent has actually explored.
        if current_step < self.STOP_MIN_STEP:
            return None
        ax, _, az = float(agent_pose[0]), float(agent_pose[1]), float(agent_pose[2])
        # best: (record, dist, matched_term_or_None, score)
        best: Optional[Tuple[MemoryRecord, float, Optional[str], float]] = None

        if self.STOP_USE_KEYWORD:
            terms = _goal_terms(goal)
            for rec in self.builder.records:
                # Exclude the current step's just-ingested keyframe (its position
                # equals the agent pose, trivially passing the geometric guard).
                if rec.timestep >= current_step:
                    continue
                term = _caption_mentions(rec.caption, terms)
                if term is None:
                    continue
                rx, _, rz = float(rec.position[0]), float(rec.position[1]), float(rec.position[2])
                dist = math.hypot(rx - ax, rz - az)
                if dist <= self.STOP_DIST_THRESHOLD and (best is None or dist < best[1]):
                    best = (rec, dist, term, 1.0)
        else:
            try:
                hits = self.builder.retrieve_from_text(
                    goal, top_k=5, min_cosine=self.STOP_COS_THRESHOLD
                )
            except Exception:
                return None
            for rec, cos in hits:
                if rec.timestep >= current_step:
                    continue
                rx, _, rz = float(rec.position[0]), float(rec.position[1]), float(rec.position[2])
                dist = math.hypot(rx - ax, rz - az)
                if dist <= self.STOP_DIST_THRESHOLD and (best is None or cos > best[3]):
                    best = (rec, dist, None, cos)

        if best is None:
            return None
        rec, dist, term, score = best
        self._candidate_counter += 1
        trace.tool_calls.append({
            "tool": "stop_check",
            "goal": str(goal),
            "match": term,
            "cos": float(score),
            "dist_m": float(dist),
            "matched_caption": rec.caption[:120],
        })
        trace.chosen_xyz = [ax, float(agent_pose[1]), az]
        trace.confidence = float(score)
        return FrontierCandidate(
            candidate_id=self._candidate_counter + 90_000,
            world_xy=np.array([ax, az], dtype=np.float32),
            grid_rc=(-1, -1),
            distance_m=0.0,
            bearing_rad=0.0,
            cluster_size=0,
            raw_score=float(score),
            source="stop",
            metadata={
                "stop_signal": True,
                "stop_cos": float(score),
                "stop_dist_m": float(dist),
                "stop_match": term,
                "stop_reason": "goal_keyword_near_agent" if term else "goal_cos_near_agent",
                "matched_caption": rec.caption[:120],
            },
        )

    # ------------------------------------------------------------------
    # stub propose — used when weights are absent
    # ------------------------------------------------------------------

    def _stub_propose(
        self,
        goal: str,
        agent_pose: np.ndarray,
        agent_yaw: float,
        max_candidates: int,
        trace: PlannerTrace,
    ) -> List[FrontierCandidate]:
        """Deterministic candidate generator for the no-weights path.

        Strategy: query the flat memory with the goal text; if hits, take the
        best K positions. Otherwise, emit a single forward-walk candidate so
        the runner can still step forward.
        """
        hits = self.builder.retrieve_from_text(goal, top_k=max_candidates)
        trace.tool_calls.append({
            "tool": "retrieve_from_text",
            "query": goal,
            "n_hits": len(hits),
        })

        out: List[FrontierCandidate] = []
        ax, az = float(agent_pose[0]), float(agent_pose[2])
        # Reject records co-located with the agent — those produce zero-
        # displacement candidates that the controller can't act on. The agent
        # also hasn't moved yet if the only records are from steps where it
        # was at the current pose.
        min_waypoint_dist = float(os.environ.get("REMEMBR_MIN_WAYPOINT_DIST", "0.5"))
        for rec, cos in hits:
            xyz = rec.position
            dx = float(xyz[0]) - ax
            dz = float(xyz[2]) - az
            dist = math.hypot(dx, dz)
            if dist < min_waypoint_dist:
                continue
            bearing = _rel_bearing(dx, dz, agent_yaw)
            self._candidate_counter += 1
            out.append(
                FrontierCandidate(
                    candidate_id=self._candidate_counter + 50_000,
                    world_xy=np.array([float(xyz[0]), float(xyz[2])], dtype=np.float32),
                    grid_rc=(-1, -1),
                    distance_m=float(dist),
                    bearing_rad=float(bearing),
                    cluster_size=0,
                    raw_score=float(cos),
                    source="remembr",
                    metadata={"trace_tool": "retrieve_from_text", "cosine": float(cos)},
                )
            )

        if not out:
            # No memory and no model — walk 1.5 m forward so the controller
            # still makes progress (matches FrontierPlanner's fallback).
            forward_x = ax + math.sin(agent_yaw) * 1.5
            forward_z = az + math.cos(agent_yaw) * 1.5
            self._candidate_counter += 1
            out.append(
                FrontierCandidate(
                    candidate_id=self._candidate_counter + 50_000,
                    world_xy=np.array([forward_x, forward_z], dtype=np.float32),
                    grid_rc=(-1, -1),
                    distance_m=1.5,
                    bearing_rad=0.0,
                    cluster_size=0,
                    raw_score=0.1,
                    source="remembr",
                    metadata={"stub_fallback": "forward_walk"},
                )
            )

        trace.chosen_xyz = out[0].world_xy.tolist()
        trace.confidence = float(out[0].raw_score)
        return out

    # ------------------------------------------------------------------
    # grounding helpers (wired into _llm_propose in Task 3)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # real LLM propose
    # ------------------------------------------------------------------

    def _llm_propose(
        self,
        goal: str,
        agent_pose: np.ndarray,
        agent_yaw: float,
        max_candidates: int,
        trace: PlannerTrace,
    ) -> List[FrontierCandidate]:
        """LLM agent loop with three retrieval tools.

        Protocol (mirrors the paper's tool-using LLM pattern but kept
        text-only to avoid an OpenAI-style tool API dep):

        SYSTEM: "You are a navigation planner. Tools:
                 - retrieve_from_text(query)
                 - retrieve_from_position(x,y,z)
                 - retrieve_from_time(t)
                 Reply with a single line in one of these formats:
                     TOOL: <name>(<arg>)
                     ANSWER: goto_t=<timestep>, confidence=<float>
                     ANSWER: explore"

        USER: "Goal: find a <goal>. Current position: <pose>. ANSWER with a timestep to navigate toward, or explore."

        Loop up to ``max_tool_calls`` rounds; a goto_t/x,z answer is grounded to a remembered position via ``_ground_answer``, ``explore`` (or no answer within the budget) returns ``[]`` to defer to frontier exploration. Parsing is permissive — small LLMs emit slightly malformed lines.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            return self._stub_propose(goal, agent_pose, agent_yaw, max_candidates, trace)

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
            trace.tool_calls.append({"tool": "budget_exhausted_defer"})
            return []  # no usable answer after the tool budget → defer to frontier

        cand = self._ground_answer(goal, answer, agent_pose, agent_yaw, trace)
        return [cand] if cand is not None else []

    # ------------------------------------------------------------------
    # tool dispatch + LLM I/O helpers
    # ------------------------------------------------------------------

    def _dispatch_tool(
        self,
        name: str,
        arg: str,
        agent_pose: np.ndarray,
    ) -> List[Tuple[MemoryRecord, float]]:
        if name == "retrieve_from_text":
            return self.builder.retrieve_from_text(arg)
        if name == "retrieve_from_position":
            parts = [p.strip() for p in arg.split(",")]
            if len(parts) >= 3:
                try:
                    xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
                except ValueError:
                    return []
            else:
                return []
            return self.builder.retrieve_from_position(xyz)
        if name == "retrieve_from_time":
            try:
                t = float(arg.strip())
            except ValueError:
                return []
            return self.builder.retrieve_from_time(t)
        return []

    def _format_chat(self, sys_prompt: str, user_prompt: str, history: List[str]) -> str:
        """Build messages and apply the tokenizer's native chat template.

        ``history`` alternates [assistant_reply, tool_result, assistant_reply, ...].
        Tool results are surfaced to the LLM as user turns so the next reply
        attends to them as fresh input.

        Falls back to merging the system prompt into the first user turn for
        models whose template doesn't support a separate ``system`` role
        (e.g. Gemma, some Llama-2 templates).
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for i, turn in enumerate(history):
            role = "assistant" if i % 2 == 0 else "user"
            messages.append({"role": role, "content": turn})
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            merged = f"{sys_prompt}\n\n{user_prompt}"
            fallback: List[Dict[str, str]] = [{"role": "user", "content": merged}]
            for i, turn in enumerate(history):
                role = "assistant" if i % 2 == 0 else "user"
                fallback.append({"role": role, "content": turn})
            return self._tokenizer.apply_chat_template(
                fallback, tokenize=False, add_generation_prompt=True
            )

    def _llm_complete(self, prompt: str) -> str:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        n_input = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = self._llm.generate(
                **inputs,
                max_new_tokens=self.config.max_planner_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0, n_input:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------


def _rel_bearing(dx: float, dz: float, agent_yaw: float) -> float:
    """World-frame bearing to (dx, dz), then wrapped relative to ``agent_yaw``."""
    world_bearing = math.atan2(dx, dz)
    rel = world_bearing - float(agent_yaw)
    while rel > math.pi:
        rel -= 2.0 * math.pi
    while rel < -math.pi:
        rel += 2.0 * math.pi
    return rel


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
        payload = low[len("answer:"):].lstrip()
        if payload.startswith("explore"):
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


def _extract_float(s: str, key: str, default: Optional[float] = None) -> Optional[float]:
    idx = s.lower().find(key.lower())
    if idx < 0:
        return default
    rest = s[idx + len(key):]
    buf = []
    seen_digit = False
    for ch in rest.lstrip():
        if ch in "+-." or ch.isdigit() or (ch in "eE" and seen_digit):
            buf.append(ch)
            if ch.isdigit():
                seen_digit = True
        else:
            break
    try:
        return float("".join(buf))
    except ValueError:
        return default


def _summarize_hits(hits: List[Tuple[MemoryRecord, float]]) -> str:
    if not hits:
        return "TOOL_RESULT: (no hits)"
    lines = []
    for rec, score in hits[:3]:
        x, _, z = rec.position
        lines.append(f"  - t={rec.timestep} xz=({x:.1f},{z:.1f}) score={score:.2f} cap=\"{rec.caption[:60]}\"")
    return "TOOL_RESULT:\n" + "\n".join(lines)
