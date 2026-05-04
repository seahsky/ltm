"""
EmbodiedMemoryBridge — adapter between the Habitat loop and the existing
dialogue_memory LTM modules.

Owns one instance each of:
- ``HierarchicalLTM``  (fine / mid / coarse FAISS layers, primary index in
  text-embedding space — 768-d SBERT by default)
- ``DialogueConsolidation``  (importance score I = αR + βU + γN)
- ``PatternClusterer`` + ``MidLayerMemory``  (mid-layer pattern clusters)
- ``ResponseReranker``  (rerank frontier candidates by Score = w₁·S_succ +
  w₂·S_sim + w₃·S_phys, with a frontier-distance heuristic standing in for
  S_phys)

Visual CLIP embeddings are kept alongside each fine-layer record in metadata
for future use; the proof-of-life reranker only needs the text vector.

The ``dialogue_memory/`` modules are imported as-is — we never modify them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from dialogue_memory.consolidation import DialogueConsolidation, DialogueSegment
from dialogue_memory.ltm import HierarchicalLTM, MemoryEntry
from dialogue_memory.pattern_cluster import MidLayerMemory, PatternClusterer
from dialogue_memory.reranking import (
    CoherenceScorer,
    HistorySuccessScorer,
    MemorySimilarityScorer,
    RerankingResult,
    ResponseReranker,
    Scorer,
    ScoredResponse,
)

from .frontier_planner import FrontierCandidate
from .perception import Keyframe


# ----------------------------------------------------------------------
# Embodied record (one consolidated experience)
# ----------------------------------------------------------------------


@dataclass
class EmbodiedRecord:
    """One record handed to the consolidator at end-of-episode."""
    episode_id: str
    step_idx: int
    caption: str
    text_embedding: np.ndarray
    visual_embedding: Optional[np.ndarray] = None
    agent_position: Optional[np.ndarray] = None
    agent_yaw: float = 0.0
    action: Optional[int] = None
    reward: float = 0.0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Frontier physics scorer (S_phys)
# ----------------------------------------------------------------------


class FrontierPhysicsScorer(Scorer):
    """Stand-in for ReMEmbR's S_phys.

    Combines:
    - planner's intrinsic ``raw_score`` (cluster size + distance kernel)
    - bearing alignment (smaller |bearing| is better)
    - distance band (1–4 m preferred — not on top of the agent, not too far)
    """

    def score(self, candidate: str, candidate_embedding: np.ndarray, context: Dict[str, Any]) -> float:
        cand: Optional[FrontierCandidate] = context.get("frontier_candidate")
        if cand is None:
            return 0.5
        bearing_score = max(0.0, 1.0 - abs(float(cand.bearing_rad)) / np.pi)
        dist = float(cand.distance_m)
        if dist <= 0.0:
            dist_score = 0.0
        else:
            # Triangular preference centred at 2 m.
            dist_score = max(0.0, 1.0 - abs(dist - 2.0) / 4.0)
        score = 0.5 * float(cand.raw_score) + 0.3 * bearing_score + 0.2 * dist_score
        return float(np.clip(score, 0.0, 1.0))


# ----------------------------------------------------------------------
# The bridge
# ----------------------------------------------------------------------


class EmbodiedMemoryBridge:
    """Glue layer between the embodied loop and the dialogue_memory LTM stack."""

    def __init__(
        self,
        text_embed_dim: int = 768,
        visual_embed_dim: int = 512,
        text_encode_fn: Optional[Callable[[str], np.ndarray]] = None,
        cluster_every_n_episodes: int = 3,
        consolidation_top_k: int = 5,
        reranker_weights: Optional[Dict[str, float]] = None,
        coarse_seed_categories: Optional[List[str]] = None,
    ):
        if text_encode_fn is None:
            raise ValueError("text_encode_fn (str -> np.ndarray) is required")

        self.text_embed_dim = text_embed_dim
        self.visual_embed_dim = visual_embed_dim
        self.text_encode_fn = text_encode_fn
        self.cluster_every_n_episodes = cluster_every_n_episodes

        # 1. Hierarchical LTM (primary index = text embedding space).
        self.ltm = HierarchicalLTM(embed_dim=text_embed_dim)

        # 2. Consolidator (uses default α/β/γ; top_k tunable).
        self.consolidator = DialogueConsolidation(
            ltm=self.ltm,
            alpha=0.4,
            beta=0.3,
            gamma=0.3,
            top_k=consolidation_top_k,
        )

        # 3. Pattern clusterer + mid layer.
        self.clusterer = PatternClusterer(
            embed_dim=text_embed_dim,
            distance_threshold=1.5,
            max_clusters=64,
            min_cluster_size=2,
        )
        self.mid_memory = MidLayerMemory(
            clusterer=self.clusterer,
            ltm_layer=self.ltm.mid,
            encoder_func=text_encode_fn,
        )

        # 4. Reranker. Default weights bias S_succ for the POL run so memory
        # influence is more likely to flip the top-1 (criterion 3).
        weights = reranker_weights or {"history": 0.45, "memory": 0.35, "coherence": 0.20}
        self.reranker = ResponseReranker(
            mid_layer_memory=self.mid_memory,
            ltm=self.ltm,
            llm_client=None,
            weights=weights,
        )
        # Swap the dialogue CoherenceScorer out for the embodied physics scorer.
        # The reranker keys this slot as "coherence" — we keep the key, the
        # implementation is now frontier-distance based.
        self.reranker.scorers["coherence"] = FrontierPhysicsScorer()

        # Episode counter for periodic mid-layer clustering.
        self._episodes_seen = 0
        self._successful_episodes_seen = 0

        # Cache of pending records collected during the current episode.
        self._pending: List[EmbodiedRecord] = []

        # Tracking which modules have been invoked (for criterion 4 logging).
        self.modules_invoked: Dict[str, bool] = {
            "stm": False,
            "consolidation": False,
            "ltm_fine": False,
            "ltm_mid": False,
            "ltm_coarse": False,
            "rerank": False,
        }

        # Seed coarse layer once with HM3D-Semantics object categories.
        if coarse_seed_categories:
            self._seed_coarse(coarse_seed_categories)

    # ------------------------------------------------------------------
    # STM / per-episode buffering
    # ------------------------------------------------------------------

    def begin_episode(self, episode_id: str):
        self._pending = []
        self._current_episode_id = episode_id

    def observe_keyframe(
        self,
        keyframe: Keyframe,
        action: Optional[int],
        reward: float,
        success: bool = False,
    ):
        """Buffer a keyframe + action/reward as a candidate experience."""
        self.modules_invoked["stm"] = True
        record = EmbodiedRecord(
            episode_id=getattr(self, "_current_episode_id", "unknown"),
            step_idx=int(keyframe.step_idx),
            caption=keyframe.caption,
            text_embedding=keyframe.text_embedding.astype(np.float32),
            visual_embedding=keyframe.visual_embedding.astype(np.float32),
            agent_position=keyframe.agent_position,
            agent_yaw=float(keyframe.agent_yaw),
            action=action,
            reward=float(reward),
            success=bool(success),
        )
        self._pending.append(record)

    # ------------------------------------------------------------------
    # End-of-episode consolidation
    # ------------------------------------------------------------------

    def consolidate(self, episode_success: bool, episode_idx: int) -> Dict[str, List[str]]:
        """Run importance scoring, write key segments to LTM-fine, optionally
        cluster into LTM-mid every N successful episodes."""
        self.modules_invoked["consolidation"] = True
        segments = []
        for rec in self._pending:
            seg = DialogueSegment(
                session_id=episode_idx,
                dialogue_id=episode_idx,
                speaker="agent",
                utterance=rec.caption,
                response=None,
                embedding=rec.text_embedding,
                metadata={
                    "step_idx": rec.step_idx,
                    "action": rec.action,
                    "reward": rec.reward,
                    "agent_position": rec.agent_position.tolist() if rec.agent_position is not None else None,
                    "agent_yaw": rec.agent_yaw,
                    "visual_embedding": rec.visual_embedding.tolist() if rec.visual_embedding is not None else None,
                    "episode_id": rec.episode_id,
                    "episode_success": episode_success,
                },
            )
            segments.append(seg)

        inserted: Dict[str, List[str]] = {"fine": [], "mid": [], "coarse": []}
        if segments:
            inserted = self.consolidator.consolidate_session(
                segments=segments,
                encoder_func=self.text_encode_fn,
                dialogue_id=episode_idx,
            )
            if inserted.get("fine"):
                self.modules_invoked["ltm_fine"] = True

        self._episodes_seen += 1
        if episode_success:
            self._successful_episodes_seen += 1

        # Trigger mid-layer clustering periodically (over consolidated fine
        # segments, not raw stm — we want stable input).
        if (
            self._successful_episodes_seen > 0
            and self._successful_episodes_seen % self.cluster_every_n_episodes == 0
        ):
            self._refresh_mid_clusters()

        self._pending = []
        return inserted

    def _refresh_mid_clusters(self):
        """Re-cluster fine-layer entries into mid-layer pattern clusters."""
        fine_entries = self.ltm.fine.entries
        if not fine_entries:
            return
        for entry in fine_entries:
            success = bool(entry.metadata.get("episode_success", False))
            self.mid_memory.add_dialogue_pattern(
                embedding=entry.embedding,
                content=entry.content,
                pattern_topic=None,
                is_successful=success,
            )
        if len(self.mid_memory.ltm_layer):
            self.modules_invoked["ltm_mid"] = True

    def _seed_coarse(self, categories: List[str]):
        """Insert one coarse-layer record per HM3D-Semantics target category."""
        for cat in categories:
            text = f"category prior: agent searches for a {cat}"
            emb = self.text_encode_fn(text).astype(np.float32)
            self.ltm.insert(
                level="coarse",
                embedding=emb,
                content=text,
                metadata={"type": "category_prior", "category": cat},
            )
        if len(self.ltm.coarse):
            self.modules_invoked["ltm_coarse"] = True

    # ------------------------------------------------------------------
    # Retrieval + reranking
    # ------------------------------------------------------------------

    def retrieve(self, query_embedding: np.ndarray, top_k_per_layer: int = 3) -> Dict[str, List[Tuple[MemoryEntry, float]]]:
        results = self.ltm.multi_scale_search(query_embedding, top_k_per_layer=top_k_per_layer)
        if any(len(v) > 0 for v in results.values()):
            for layer in ("fine", "mid", "coarse"):
                if results.get(layer):
                    self.modules_invoked[f"ltm_{layer}"] = True
        return results

    def rerank(
        self,
        candidates: List[FrontierCandidate],
        query_text: str,
        stm_captions: Optional[List[str]] = None,
    ) -> Tuple[RerankingResult, Dict[str, List[Tuple[MemoryEntry, float]]]]:
        """Rerank frontier candidates by Score = w₁·S_succ + w₂·S_sim + w₃·S_phys.

        Returns the rerank result + the retrieval results that scored it
        (for logging / criterion checking).
        """
        if not candidates:
            raise ValueError("rerank requires at least one candidate")

        self.modules_invoked["rerank"] = True

        # Encode the query (current caption / target description) once.
        query_emb = self.text_encode_fn(query_text).astype(np.float32)
        retrieval = self.retrieve(query_emb, top_k_per_layer=3)

        # Build a textual stand-in per candidate so the SBERT-based scorers
        # have something to embed; subsequent scorers only need *embedding*
        # equality, not human readability.
        cand_texts: List[str] = []
        cand_embeddings: List[np.ndarray] = []
        for c in candidates:
            txt = (
                f"go to ({c.world_xy[0]:.1f}, {c.world_xy[1]:.1f}) "
                f"distance {c.distance_m:.1f}m bearing {c.bearing_rad:+.2f}rad "
                f"size {c.cluster_size}"
            )
            cand_texts.append(txt)
            cand_embeddings.append(self.text_encode_fn(txt).astype(np.float32))
        cand_emb_matrix = np.stack(cand_embeddings, axis=0)

        # The reranker's CoherenceScorer slot (now FrontierPhysicsScorer)
        # peeks at context["frontier_candidate"] for the current candidate.
        # We patch the rerank loop minimally by exposing per-candidate context
        # via a wrapper scorer that reads from a sliding ``context["__cands"]``
        # list keyed by index. The simplest way is to override .rerank by
        # iterating ourselves — but since ResponseReranker.rerank is the
        # public surface, we instead set context["frontier_candidates"] +
        # mutate context["frontier_candidate"] before each candidate via a
        # small subclass.

        result = self._rerank_with_per_candidate_context(
            candidates=candidates,
            cand_texts=cand_texts,
            cand_embeddings=cand_emb_matrix,
            retrieval=retrieval,
            stm_context=stm_captions or [],
        )
        return result, retrieval

    def _rerank_with_per_candidate_context(
        self,
        candidates: List[FrontierCandidate],
        cand_texts: List[str],
        cand_embeddings: np.ndarray,
        retrieval: Dict[str, List[Tuple[MemoryEntry, float]]],
        stm_context: List[str],
    ) -> RerankingResult:
        """Replicate ResponseReranker.rerank but plumb each candidate's
        ``FrontierCandidate`` through the per-call context. We can't reuse the
        upstream loop directly because it doesn't know about per-candidate
        side info; the math/weights are identical."""

        scored: List[ScoredResponse] = []
        weights = self.reranker.weights
        scorers = self.reranker.scorers

        for cand_obj, cand_text, cand_emb in zip(candidates, cand_texts, cand_embeddings):
            ctx = {
                "retrieval_results": retrieval,
                "stm_context": stm_context,
                "user_input": cand_text,
                "frontier_candidate": cand_obj,
            }
            scores: Dict[str, float] = {}
            weighted_sum = 0.0
            total_weight = 0.0

            if scorers.get("history") and weights.get("history", 0) > 0:
                s = scorers["history"].score(cand_text, cand_emb, ctx)
                scores["history"] = float(s)
                weighted_sum += weights["history"] * s
                total_weight += weights["history"]

            if scorers.get("memory") and weights.get("memory", 0) > 0:
                s = scorers["memory"].score(cand_text, cand_emb, ctx)
                scores["memory"] = float(s)
                weighted_sum += weights["memory"] * s
                total_weight += weights["memory"]

            if scorers.get("coherence") and weights.get("coherence", 0) > 0:
                s = scorers["coherence"].score(cand_text, cand_emb, ctx)
                scores["coherence"] = float(s)  # actually frontier physics
                weighted_sum += weights["coherence"] * s
                total_weight += weights["coherence"]

            final = weighted_sum / total_weight if total_weight > 0 else 0.5
            scored.append(
                ScoredResponse(
                    response=cand_text,
                    response_embedding=cand_emb,
                    scores=scores,
                    final_score=float(final),
                )
            )

        scored.sort(key=lambda x: x.final_score, reverse=True)
        for i, sc in enumerate(scored):
            sc.rank = i + 1

        return RerankingResult(
            candidates=scored,
            selected=scored[0] if scored else None,
            debug_info={
                "weights": weights,
                "num_candidates": len(scored),
                "top_scores": [
                    {"response": s.response[:80], "final_score": s.final_score, "scores": s.scores}
                    for s in scored[:3]
                ],
            },
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "ltm_counts": self.ltm.stats(),
            "n_clusters": len(self.clusterer.clusters),
            "episodes_seen": self._episodes_seen,
            "successful_episodes_seen": self._successful_episodes_seen,
            "modules_invoked": dict(self.modules_invoked),
        }
