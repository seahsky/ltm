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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity rescaled to [0, 1]."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.5
    cos = float(np.dot(a, b) / (na * nb))
    # Map [-1, 1] -> [0, 1]
    return 0.5 * (cos + 1.0)


class EmbodiedMemorySimilarityScorer(Scorer):
    """Candidate-aware S_sim for the embodied loop.

    The dialogue ``MemorySimilarityScorer`` ignores ``candidate_embedding``
    and returns the same value for every candidate in a decision — so it
    contributes a per-decision offset but no per-candidate ranking signal.
    Audit on the abl-s3 run showed `S_sim` saturating near 0.94 with stdev
    0.02 (4th-decimal noise), driving half the rerank disagreements as
    coin-flips on near-ties.

    This scorer instead computes the cosine similarity between
    ``candidate_embedding`` and each retrieved entry's embedding, weighted
    by layer (fine > mid > coarse). Output is in [0, 1] and varies *per
    candidate* against the same retrieval set.
    """

    def __init__(self, weight_fine: float = 0.5, weight_mid: float = 0.3, weight_coarse: float = 0.2):
        self.weight_fine = weight_fine
        self.weight_mid = weight_mid
        self.weight_coarse = weight_coarse

    def score(self, candidate: str, candidate_embedding: np.ndarray, context: Dict[str, Any]) -> float:
        retrieval = context.get("retrieval_results", {}) or {}
        if not retrieval:
            return 0.5
        total_score = 0.0
        total_weight = 0.0
        layer_weights = {"fine": self.weight_fine, "mid": self.weight_mid, "coarse": self.weight_coarse}
        for layer, w in layer_weights.items():
            hits = retrieval.get(layer) or []
            if not hits or w <= 0.0:
                continue
            sims = [_cosine(candidate_embedding, entry.embedding) for entry, _dist in hits]
            total_score += w * float(np.mean(sims))
            total_weight += w
        if total_weight <= 0.0:
            return 0.5
        return total_score / total_weight

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
    """Stand-in for ReMEmbR's S_phys. Source-aware.

    Planner candidates (frontier clusters from depth projection) are scored
    classically:
        ``score = 0.5·raw_score + 0.3·bearing_alignment + 0.2·distance_band``
    - ``raw_score``: planner intrinsic (cluster size + distance kernel)
    - ``bearing_alignment``: small |bearing| is better — cheap to head there
    - ``distance_band``: 1–4 m preferred (not on top, not too far)

    Memory candidates (LTM-injected by ``propose_memory_candidates``) carry
    raw_score = SBERT goal-vs-caption TEXT cosine. We score them goal-directed:
        ``score = sigmoid-ish boost in cosine + small distance preference``
    Bearing penalty is dropped — turning to face a memory waypoint is cheap
    compared to the value of going there. Non-match cosines (~0.2-0.3) keep
    memory from out-voting a strong planner choice; true caption matches
    (~0.5+) win comfortably.
    """

    # Calibration constants for memory-source physics. The memory raw_score is an
    # SBERT goal-vs-caption TEXT cosine (the LTM is indexed on caption text, not
    # CLIP images). Text-text separation is WIDE — a true caption match
    # ("a photo of a chair" vs "...featuring a wooden chair") ~0.5+, an unrelated
    # indoor caption ~0.2-0.3 — unlike the prior CLIP image-text signal which was
    # flat (~0.25 sighting vs ~0.228 baseline → memory picked wrong instances and
    # HURT, G4 + mini-ablation). Saturate at the match scale so a real match wins
    # while a non-match loses to a strong frontier. NOTE: provisional text-scale
    # values — verify/tune against the SBERT cosine distribution in the next smoke.
    #   cos=0.30 -> cos_norm 0    (non-match, loses to a strong frontier ~0.97)
    #   cos=0.50 -> cos_norm 1.0  (match saturates; wins when close + frontier weak)
    _MEM_COS_NULL = 0.30   # cos at-or-below this contributes nothing
    _MEM_COS_FULL = 0.50   # cos at-or-above this saturates the bonus
    _MEM_DIST_WEIGHT = 0.20

    def score(self, candidate: str, candidate_embedding: np.ndarray, context: Dict[str, Any]) -> float:
        cand: Optional[FrontierCandidate] = context.get("frontier_candidate")
        if cand is None:
            return 0.5

        dist = float(cand.distance_m)
        if dist <= 0.0:
            dist_score = 0.0
        else:
            dist_score = max(0.0, 1.0 - abs(dist - 2.0) / 4.0)

        if getattr(cand, "source", "planner") == "memory":
            cos = float(cand.raw_score)
            span = self._MEM_COS_FULL - self._MEM_COS_NULL
            cos_norm = (cos - self._MEM_COS_NULL) / span if span > 0 else 0.0
            cos_norm = float(np.clip(cos_norm, 0.0, 1.0))
            # 1 - _MEM_DIST_WEIGHT goes to the cosine, the rest to distance.
            score = (1.0 - self._MEM_DIST_WEIGHT) * cos_norm + self._MEM_DIST_WEIGHT * dist_score
        else:
            bearing_score = max(0.0, 1.0 - abs(float(cand.bearing_rad)) / np.pi)
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
        disable_stm: bool = False,
        disable_ltm: bool = False,
        disable_rerank: bool = False,
        clip_encoder: Optional[Any] = None,
        affordance_table: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        if text_encode_fn is None:
            raise ValueError("text_encode_fn (str -> np.ndarray) is required")

        self.text_embed_dim = text_embed_dim
        self.visual_embed_dim = visual_embed_dim
        self.text_encode_fn = text_encode_fn
        self.cluster_every_n_episodes = cluster_every_n_episodes

        # Ablation toggles. Setting 1 (paper-faithful baseline stand-in) sets
        # all three True; Setting 2 (STM only) keeps disable_stm=False; Setting 3
        # (full system) leaves all False (default).
        self.disable_stm = bool(disable_stm)
        self.disable_ltm = bool(disable_ltm)
        self.disable_rerank = bool(disable_rerank)

        # Path-1 fix: the LTM is now indexed in CLIP joint vision-language
        # space, not SBERT text space. Caption-based indexing was inert because
        # the HM3D-Semantics sensor wasn't loading, so every caption defaulted
        # to "agent at (x,y) sees: room interior | searching for ..." — i.e.
        # the LTM was indexing on positional perturbations of one identical
        # sentence. With CLIP, fine-layer entries are the agent's actual visual
        # observations and the coarse layer is seeded with CLIP text priors
        # ("a photo of a {category}") so goal-directed retrieval can surface
        # past sightings of the target.
        self.clip_encoder = clip_encoder
        # Index the LTM on the caption TEXT embedding (SBERT), not CLIP image
        # embeddings. The CLIP image-text cosine (ViT-B/32) was flat (~0.25
        # sighting vs ~0.228 baseline) and non-discriminative, so memory picked
        # wrong-but-similar instances and HURT (G4 + mini-ablation). The now-rich
        # Qwen-VL captions make SBERT goal-vs-caption similarity discriminative
        # again (~0.5 match vs ~0.2 non-match); the original reason for
        # image-indexing — degenerate all-"room interior" semantic-sensor captions
        # — no longer holds. clip_encoder is still kept for keyframe visual
        # encoding and the rerank's visual query.
        self._ltm_embed_dim = int(text_embed_dim)
        self._ltm_encode_text = text_encode_fn

        # 1. Hierarchical LTM (primary index = CLIP joint space when a
        # clip_encoder is supplied, else SBERT-text space as a fallback).
        self.ltm = HierarchicalLTM(embed_dim=self._ltm_embed_dim)

        # 2. Consolidator (uses default α/β/γ; top_k tunable).
        self.consolidator = DialogueConsolidation(
            ltm=self.ltm,
            alpha=0.4,
            beta=0.3,
            gamma=0.3,
            top_k=consolidation_top_k,
        )

        # 3. Pattern clusterer + mid layer. Dim matches the LTM (CLIP space
        # when clip_encoder is provided).
        self.clusterer = PatternClusterer(
            embed_dim=self._ltm_embed_dim,
            distance_threshold=1.5,
            max_clusters=64,
            min_cluster_size=2,
        )
        self.mid_memory = MidLayerMemory(
            clusterer=self.clusterer,
            ltm_layer=self.ltm.mid,
            encoder_func=self._ltm_encode_text,
        )

        # 4. Reranker. The POL weights were inherited from the dialogue
        # benchmark and an audit on `runs/abl-s3` showed they're miscalibrated
        # for embodied:
        #   - HistorySuccessScorer was a constant 0.5 across 11.8k scorings
        #     because the mid-layer never populates without successful eps
        #     (`_refresh_mid_clusters` gates on successful_episodes % N == 0).
        #   - MemorySimilarityScorer was 0.94±0.02 *and candidate-agnostic*,
        #     so it added a near-constant offset but no ranking signal.
        # Phase-1 fix: zero history until mid is populated, swap in a
        # candidate-aware memory scorer, and promote S_phys to dominant.
        weights = reranker_weights or {"history": 0.0, "memory": 0.30, "coherence": 0.70}
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
        # Swap in the candidate-aware memory scorer (cosine vs each retrieved
        # entry, not query-to-retrieval distance).
        self.reranker.scorers["memory"] = EmbodiedMemorySimilarityScorer()

        # Episode counter for periodic mid-layer clustering.
        self._episodes_seen = 0
        self._successful_episodes_seen = 0

        # Cache of pending records collected during the current episode.
        self._pending: List[EmbodiedRecord] = []

        # Cumulative count of keyframes the bridge actually ingested. Stays at 0
        # when disable_stm=True so the ablation can confirm the toggle worked.
        self._n_keyframes_observed = 0

        # Tracking which modules have been invoked (for criterion 4 logging).
        self.modules_invoked: Dict[str, bool] = {
            "stm": False,
            "consolidation": False,
            "ltm_fine": False,
            "ltm_mid": False,
            "ltm_coarse": False,
            "rerank": False,
        }

        # Stash the affordance table (may be None). Picked up by _seed_coarse
        # to condition prompts on the most-successful room for each category,
        # and exposed via stats() so analyzer scripts can confirm it loaded.
        self.affordance_table: Dict[str, Dict[str, float]] = dict(affordance_table or {})

        # Seed coarse layer once with HM3D-Semantics object categories. Skipped
        # when disable_ltm=True so Setting 1 starts with an empty LTM stack
        # (incl. coarse priors) — keeps `ltm_counts_final` cleanly zero.
        if coarse_seed_categories and not self.disable_ltm:
            self._seed_coarse(coarse_seed_categories)

    # ------------------------------------------------------------------
    # STM / per-episode buffering
    # ------------------------------------------------------------------

    def begin_episode(self, episode_id: str, scene_id: Optional[str] = None):
        self._pending = []
        self._current_episode_id = episode_id
        # Tracked so option-2 memory-injected candidates only surface entries
        # from the current scene (cross-scene world-xy positions would be
        # geometrically meaningless when habitat cycles to a new scene).
        self._current_scene_id = scene_id

    def observe_keyframe(
        self,
        keyframe: Keyframe,
        action: Optional[int],
        reward: float,
        success: bool = False,
    ):
        """Buffer a keyframe + action/reward as a candidate experience."""
        if self.disable_stm:
            return
        self.modules_invoked["stm"] = True
        self._n_keyframes_observed += 1
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
            metadata={"scene_id": getattr(self, "_current_scene_id", None)},
        )
        self._pending.append(record)

    # ------------------------------------------------------------------
    # End-of-episode consolidation
    # ------------------------------------------------------------------

    def consolidate(self, episode_success: bool, episode_idx: int) -> Dict[str, List[str]]:
        """Run importance scoring, write key segments to LTM-fine, optionally
        cluster into LTM-mid every N successful episodes."""
        if self.disable_ltm:
            self._pending = []
            self._episodes_seen += 1
            if episode_success:
                self._successful_episodes_seen += 1
            return {"fine": [], "mid": [], "coarse": []}

        self.modules_invoked["consolidation"] = True
        segments = []
        for rec in self._pending:
            # Index the fine layer on the caption TEXT embedding (SBERT) —
            # discriminative goal-vs-caption similarity now that captions are
            # rich VLM output. The image CLIP vector stays in metadata for
            # potential visual use but is no longer the retrieval index.
            primary_emb = rec.text_embedding
            seg = DialogueSegment(
                session_id=episode_idx,
                dialogue_id=episode_idx,
                speaker="agent",
                utterance=rec.caption,
                response=None,
                embedding=primary_emb,
                metadata={
                    "step_idx": rec.step_idx,
                    "action": rec.action,
                    "reward": rec.reward,
                    "agent_position": rec.agent_position.tolist() if rec.agent_position is not None else None,
                    "agent_yaw": rec.agent_yaw,
                    "visual_embedding": rec.visual_embedding.tolist() if rec.visual_embedding is not None else None,
                    "text_embedding": rec.text_embedding.tolist() if rec.text_embedding is not None else None,
                    "episode_id": rec.episode_id,
                    "episode_success": episode_success,
                    "scene_id": rec.metadata.get("scene_id"),
                },
            )
            segments.append(seg)

        inserted: Dict[str, List[str]] = {"fine": [], "mid": [], "coarse": []}
        if segments:
            inserted = self.consolidator.consolidate_session(
                segments=segments,
                encoder_func=self._ltm_encode_text,
                dialogue_id=episode_idx,
            )
            if inserted.get("fine"):
                self.modules_invoked["ltm_fine"] = True

            # The dialogue consolidator builds its own MemoryEntry.metadata
            # (dialogue_id / session_id / importance_score / breakdown) and
            # discards seg.metadata — so the embodied fields option-2 needs
            # (scene_id, agent_position, episode_id, step_idx) never reach
            # the LTM. Patch the freshly-inserted entries by matching their
            # embedding back to the originating EmbodiedRecord. Embeddings
            # come straight from rec.visual_embedding so np.array_equal is
            # exact.
            inserted_fine_ids = set(inserted.get("fine") or [])
            if inserted_fine_ids:
                new_entries = [e for e in self.ltm.fine.entries if e.id in inserted_fine_ids]
                for entry in new_entries:
                    for rec in self._pending:
                        # Match on the same primary embedding the fine layer was
                        # indexed with (caption text / SBERT).
                        rec_emb = rec.text_embedding
                        if rec_emb is None:
                            continue
                        if rec_emb.shape != entry.embedding.shape:
                            continue
                        if np.array_equal(rec_emb, entry.embedding):
                            entry.metadata.update({
                                "scene_id": rec.metadata.get("scene_id"),
                                "agent_position": (
                                    rec.agent_position.tolist()
                                    if rec.agent_position is not None else None
                                ),
                                "agent_yaw": float(rec.agent_yaw),
                                "episode_id": rec.episode_id,
                                "step_idx": int(rec.step_idx),
                                "episode_success": bool(episode_success),
                            })
                            break

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
        """Insert one coarse-layer record per HM3D-Semantics target category.

        Seeded in the same embedding space as the fine layer: when a CLIP
        encoder is configured we use the CLIP text prompt ``"a photo of a
        {category}"`` so a goal-directed retrieval query at decision time
        (also CLIP-text-encoded) can match these priors against past visual
        keyframes that captured the category.

        Phase 2 (affordance learning): if ``self.affordance_table`` carries
        per-(category, room-type) success rates from prior runs, we condition
        the prompt on the most-successful room — e.g.
        ``"a photo of a chair in a living_room"`` — and record the empirical
        success rate in metadata. Without a table we fall back to the
        category-only prompt, so the legacy behaviour is preserved.
        """
        for cat in categories:
            best_room: Optional[str] = None
            best_rate: Optional[float] = None
            cat_table = self.affordance_table.get(cat) or {}
            if cat_table:
                best_room, best_rate = max(cat_table.items(), key=lambda kv: kv[1])

            # Only condition the prompt on a room when we have *positive*
            # success evidence for that pairing — otherwise we'd be biasing
            # the prior toward places the agent already failed.
            has_evidence = bool(best_room) and (best_rate or 0.0) > 0.0
            if self.clip_encoder is not None:
                if has_evidence:
                    prompt = f"a photo of a {cat} in a {best_room.replace('_', ' ')}"
                else:
                    prompt = f"a photo of a {cat}"
            else:
                if has_evidence:
                    prompt = f"category prior: agent searches for a {cat} (typically in {best_room})"
                else:
                    prompt = f"category prior: agent searches for a {cat}"
            emb = self._ltm_encode_text(prompt).astype(np.float32)
            meta: Dict[str, Any] = {"type": "category_prior", "category": cat}
            if best_room is not None:
                meta["preferred_room"] = best_room
                meta["success_rate"] = float(best_rate or 0.0)
                meta["room_distribution"] = dict(cat_table)
            self.ltm.insert(
                level="coarse",
                embedding=emb,
                content=prompt,
                metadata=meta,
            )
        if len(self.ltm.coarse):
            self.modules_invoked["ltm_coarse"] = True

    @staticmethod
    def build_affordance_table(
        run_dirs: List[str],
        room_resolver: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        min_episodes_per_pair: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate ``runs/<dir>/episode_*.json`` into a per-(category,
        room-type) success-rate table.

        ``room_resolver`` maps an episode dict to a room-type string. When
        not supplied we default to ``episode["scene_id"]`` as a stand-in —
        HM3D-Semantics doesn't expose room types in the ObjectNav JSON, so
        the scene id is the next-best grouping until a proper resolver
        (e.g. nearest-region from the scene's region annotations) is wired.

        ``min_episodes_per_pair``: drop (cat, room) bins seen fewer than N
        times so the table doesn't memorize one-off lucky/unlucky runs.

        Note: Phase-1 runs have zero successes, so the table will be all
        zeros until a working backbone (i.e. ReMEmbR) closes the
        success-rate gap. The plumbing is still useful — the prompt
        conditioning generalizes once non-zero rates appear, and the
        resulting CLIP-text prompts (e.g. "a photo of a chair in a
        living room") are themselves better priors than the bare
        category form.
        """
        import glob
        import json
        import os as _os

        counts: Dict[str, Dict[str, List[int]]] = {}  # cat -> room -> [success, total]
        for run in run_dirs:
            for ep_path in sorted(glob.glob(_os.path.join(run, "episode_*.json"))):
                if ep_path.endswith("_error.json"):
                    continue
                try:
                    with open(ep_path, "r", encoding="utf-8") as f:
                        ep = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                cat = ep.get("target_category")
                if cat is None:
                    continue
                cat = str(cat).strip().lower()
                room = room_resolver(ep) if room_resolver else ep.get("scene_id")
                if room is None:
                    continue
                room = str(room)
                success = bool(ep.get("success", False))
                buf = counts.setdefault(cat, {}).setdefault(room, [0, 0])
                buf[0] += 1 if success else 0
                buf[1] += 1

        table: Dict[str, Dict[str, float]] = {}
        for cat, rooms in counts.items():
            for room, (succ, total) in rooms.items():
                if total < min_episodes_per_pair:
                    continue
                rate = succ / total if total else 0.0
                table.setdefault(cat, {})[room] = float(rate)
        return table

    # ------------------------------------------------------------------
    # Retrieval + reranking
    # ------------------------------------------------------------------

    def propose_memory_candidates(
        self,
        agent_pos: np.ndarray,
        agent_yaw: float,
        target_category: Optional[str],
        planner_world_xys: Optional[List[np.ndarray]] = None,
        top_k: int = 3,
        min_cosine: float = 0.25,
        dedup_radius_m: float = 1.5,
        max_distance_m: float = 30.0,
    ) -> List[FrontierCandidate]:
        """Option-2: turn LTM hits into extra frontier candidates.

        Query the fine layer with ``"a photo of a {target_category}"`` in SBERT
        text space; for each hit that (a) clears the cosine threshold,
        (b) belongs to the current scene, and (c) isn't a near-duplicate of an
        already-proposed planner candidate, materialize a ``FrontierCandidate``
        from the stored ``agent_position``. ``raw_score`` carries the cosine
        through so the downstream FrontierPhysicsScorer naturally favours
        semantically-strong memories. ``min_cosine`` (default 0.25) pre-filters
        clearly-unrelated captions on the SBERT scale.

        Returns ``[]`` whenever LTM is disabled, no text encoder is wired, the
        target is missing, or no hit clears the bar — so callers can always
        concatenate the result onto the planner's list without a guard.
        """
        if self.disable_ltm or self._ltm_encode_text is None or not target_category:
            return []
        if not len(self.ltm.fine):
            return []

        import math

        # Query the text-indexed fine layer with the goal phrase in SBERT space
        # (same encoder the layer was indexed with) → discriminative goal-vs-
        # caption cosine, not the flat CLIP image-text cosine.
        query = self._ltm_encode_text(
            f"a photo of a {target_category}"
        ).astype(np.float32)

        # Over-fetch: we'll discard scene-mismatched and threshold-failing hits.
        fetch_k = max(top_k * 4, 8)
        hits = self.ltm.fine.search(query, fetch_k)

        out: List[FrontierCandidate] = []
        seen_xys: List[np.ndarray] = list(planner_world_xys or [])
        ax = float(agent_pos[0])
        az = float(agent_pos[2])

        for entry, dist in hits:
            # FAISS returns L2² on unit-norm vectors → cos = 1 - d²/2.
            cos = max(-1.0, min(1.0, 1.0 - float(dist) / 2.0))
            if cos < min_cosine:
                continue
            if entry.metadata.get("scene_id") != self._current_scene_id:
                continue
            ap = entry.metadata.get("agent_position")
            if ap is None or len(ap) < 3:
                continue
            world_xy = np.asarray([ap[0], ap[2]], dtype=np.float32)

            # Skip "go back to where I'm standing" — and dedup against the
            # planner's own frontier proposals so we don't double-vote.
            if any(np.linalg.norm(world_xy - sx) < dedup_radius_m for sx in seen_xys):
                continue
            seen_xys.append(world_xy)

            dx = float(world_xy[0]) - ax
            dz = float(world_xy[1]) - az
            dist_m = math.hypot(dx, dz)
            if dist_m > max_distance_m:
                continue

            world_bearing = math.atan2(dx, dz)
            rel = world_bearing - float(agent_yaw)
            while rel > math.pi:
                rel -= 2.0 * math.pi
            while rel < -math.pi:
                rel += 2.0 * math.pi

            # raw_score = raw CLIP cosine. The source-aware physics scorer
            # (FrontierPhysicsScorer) interprets it differently for memory
            # vs planner candidates — see that class for the calibration.
            out.append(
                FrontierCandidate(
                    candidate_id=-1,  # caller assigns
                    world_xy=world_xy,
                    grid_rc=(-1, -1),
                    distance_m=dist_m,
                    bearing_rad=rel,
                    cluster_size=0,
                    raw_score=float(cos),
                    source="memory",
                    metadata={
                        "ltm_score": float(cos),
                        "ltm_step_idx": int(entry.metadata.get("step_idx", -1)),
                        "ltm_episode_id": entry.metadata.get("episode_id"),
                        "scene_id": entry.metadata.get("scene_id"),
                    },
                )
            )
            if len(out) >= top_k:
                break

        if out:
            # Counts as fine-layer activation for the criterion-4 module check.
            self.modules_invoked["ltm_fine"] = True

        return out

    def retrieve(self, query_embedding: np.ndarray, top_k_per_layer: int = 3) -> Dict[str, List[Tuple[MemoryEntry, float]]]:
        results = self.ltm.multi_scale_search(query_embedding, top_k_per_layer=top_k_per_layer)
        if any(len(v) > 0 for v in results.values()):
            for layer in ("fine", "mid", "coarse"):
                if results.get(layer):
                    self.modules_invoked[f"ltm_{layer}"] = True
        return results

    @staticmethod
    def _format_candidate_text(c: FrontierCandidate) -> str:
        return (
            f"go to ({c.world_xy[0]:.1f}, {c.world_xy[1]:.1f}) "
            f"distance {c.distance_m:.1f}m bearing {c.bearing_rad:+.2f}rad "
            f"size {c.cluster_size}"
        )

    def rerank(
        self,
        candidates: List[FrontierCandidate],
        query_text: str,
        stm_captions: Optional[List[str]] = None,
        target_category: Optional[str] = None,
        query_visual_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[RerankingResult, Dict[str, List[Tuple[MemoryEntry, float]]]]:
        """Rerank frontier candidates by Score = w₁·S_succ + w₂·S_sim + w₃·S_phys.

        Returns the rerank result + the retrieval results that scored it
        (for logging / criterion checking).
        """
        if not candidates:
            raise ValueError("rerank requires at least one candidate")

        # Setting 1/2 ablation: skip scoring entirely and pass through the
        # planner's raw top-1. We still package it as a RerankingResult so
        # the runner's downstream lookup keeps working unchanged.
        if self.disable_rerank:
            raw_top1 = candidates[0]
            cand_text = self._format_candidate_text(raw_top1)
            cand_emb = self.text_encode_fn(cand_text).astype(np.float32)
            scored = ScoredResponse(
                response=cand_text,
                response_embedding=cand_emb,
                scores={},
                final_score=float(raw_top1.raw_score),
            )
            scored.rank = 1
            empty_retrieval: Dict[str, List[Tuple[MemoryEntry, float]]] = {
                "fine": [], "mid": [], "coarse": [],
            }
            result = RerankingResult(
                candidates=[scored],
                selected=scored,
                debug_info={
                    "weights": {},
                    "num_candidates": 1,
                    "top_scores": [{
                        "response": cand_text[:80],
                        "final_score": float(raw_top1.raw_score),
                        "scores": {},
                    }],
                    "rerank_disabled": True,
                },
            )
            return result, empty_retrieval

        self.modules_invoked["rerank"] = True

        # Encode the query once. With disable_ltm=True we skip the multi-scale
        # search so retrieval is always empty — the score from
        # MemorySimilarityScorer collapses to its no-context default.
        # The LTM is indexed on SBERT caption text, so the query MUST live in the
        # same text space (a 512-d CLIP vector would mismatch the 384-d index).
        # Goal-directed when a target is given (surfaces past captions mentioning
        # the object), else the current caption. query_visual_embedding is no
        # longer used here — the index is text, not images.
        if self.disable_ltm:
            retrieval = {"fine": [], "mid": [], "coarse": []}
        else:
            if target_category:
                query_emb = self._ltm_encode_text(
                    f"a photo of a {target_category}"
                ).astype(np.float32)
            else:
                query_emb = self._ltm_encode_text(query_text).astype(np.float32)
            retrieval = self.retrieve(query_emb, top_k_per_layer=3)

        # Build a textual stand-in per candidate so the scorers have something
        # to embed; subsequent scorers only need *embedding* equality, not
        # human readability. We encode in whichever space the LTM lives in
        # (SBERT text) so the candidate-aware memory scorer can compute
        # well-defined cosines against retrieved entries.
        # NB: the memory scorer's per-candidate signal is intentionally weak
        # here — candidate texts are geometric "go to (x,y)" strings, so they
        # cluster tightly in text space. Path 1 is scaffolding; meaningful
        # candidate-level memory influence requires option 2 (planner-side
        # memory-injected candidates with their own visual embeddings).
        cand_texts: List[str] = []
        cand_embeddings: List[np.ndarray] = []
        for c in candidates:
            txt = self._format_candidate_text(c)
            cand_texts.append(txt)
            cand_embeddings.append(self._ltm_encode_text(txt).astype(np.float32))
        cand_emb_matrix = np.stack(cand_embeddings, axis=0)

        # If STM is disabled the in-episode caption stream isn't available, so
        # zero it out before it reaches the scorers' context.
        effective_stm = [] if self.disable_stm else (stm_captions or [])

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
            stm_context=effective_stm,
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
            "n_keyframes_observed": self._n_keyframes_observed,
            "modules_invoked": dict(self.modules_invoked),
            "ablation": {
                "disable_stm": self.disable_stm,
                "disable_ltm": self.disable_ltm,
                "disable_rerank": self.disable_rerank,
            },
            "affordance_table_size": sum(len(v) for v in self.affordance_table.values()),
            "affordance_categories": sorted(self.affordance_table.keys()),
        }
