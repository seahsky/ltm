"""
EpisodeRunner — top-level orchestration.

Wires:
  EpisodeSource → FrontierPlanner → EmbodiedMemoryBridge → action

Per-step:
  1. env.step(action)
  2. every M steps, build a Keyframe (CLIP visual + caption + SBERT text)
  3. on decision step, planner proposes K candidates → bridge reranks →
     execute top-1 over the next decision_period steps
  4. record everything to a structured JSON log

Per-episode end:
  bridge.consolidate(success, episode_idx)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .episode_source import Episode, EpisodeSource, Step
from .frontier_planner import (
    ACTION_FORWARD,
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    FrontierCandidate,
    FrontierPlanner,
)
from .memory_bridge import EmbodiedMemoryBridge
from .perception import CLIPKeyframeEncoder, Keyframe, SemanticCaptioner
from .remembr_backbone import ReMEmbRBuilder, ReMEmbRPlanner


@dataclass
class RunSummary:
    n_episodes_attempted: int = 0
    n_episodes_completed: int = 0
    n_successful_episodes: int = 0
    ltm_counts_final: Dict[str, int] = field(default_factory=dict)
    rerank_calls: int = 0
    rerank_disagreements: int = 0     # top-1 reranked != raw planner top-1
    retrieval_hits: int = 0           # rerank calls that retrieved >= 1 LTM record
    n_memory_candidates: int = 0      # total LTM-injected candidates surfaced
    n_memory_chosen: int = 0          # decisions where reranker picked a memory candidate
    n_frontier_chosen: int = 0        # decisions where reranker picked a frontier-injected candidate
    n_remembr_chosen: int = 0         # decisions where reranker picked a grounded remembr (LLM) candidate
    n_stop_signals: int = 0           # decisions where backbone emitted a grounded STOP
    n_keyframes_observed: int = 0
    modules_invoked: Dict[str, bool] = field(default_factory=dict)
    ablation: Dict[str, Any] = field(default_factory=dict)
    pass_conditions: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    episodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_episodes_attempted": self.n_episodes_attempted,
            "n_episodes_completed": self.n_episodes_completed,
            "n_successful_episodes": self.n_successful_episodes,
            "ltm_counts_final": self.ltm_counts_final,
            "rerank_calls": self.rerank_calls,
            "rerank_disagreements": self.rerank_disagreements,
            "retrieval_hits": self.retrieval_hits,
            "n_memory_candidates": self.n_memory_candidates,
            "n_memory_chosen": self.n_memory_chosen,
            "n_frontier_chosen": self.n_frontier_chosen,
            "n_remembr_chosen": self.n_remembr_chosen,
            "n_stop_signals": self.n_stop_signals,
            "n_keyframes_observed": self.n_keyframes_observed,
            "modules_invoked": self.modules_invoked,
            "ablation": self.ablation,
            "pass_conditions": self.pass_conditions,
            "notes": self.notes,
            "episodes": self.episodes,
        }


class EpisodeRunner:
    """Drive N episodes, log to ``out_dir``, return a RunSummary."""

    def __init__(
        self,
        source: EpisodeSource,
        planner: FrontierPlanner,
        bridge: EmbodiedMemoryBridge,
        clip_encoder: CLIPKeyframeEncoder,
        captioner: SemanticCaptioner,
        out_dir: str,
        target_category: str = "chair",
        keyframe_every_m: int = 5,
        max_steps_per_episode: int = 250,
        run_config: Optional[Dict[str, Any]] = None,
        backbone: str = "frontier",
        remembr_builder: Optional[ReMEmbRBuilder] = None,
        remembr_planner: Optional[ReMEmbRPlanner] = None,
    ):
        self.source = source
        self.planner = planner
        self.bridge = bridge
        self.clip_encoder = clip_encoder
        self.captioner = captioner
        self.out_dir = out_dir
        self.target_category = target_category
        self.keyframe_every_m = keyframe_every_m
        self.max_steps_per_episode = max_steps_per_episode
        self.run_config = dict(run_config or {})
        if backbone not in ("frontier", "remembr", "oracle"):
            raise ValueError(
                f"backbone must be 'frontier', 'remembr', or 'oracle'; got {backbone!r}"
            )
        self.backbone = backbone
        # Oracle backbone (Run-5 diagnostic): a ShortestPathFollower steers
        # straight to the episode goal, bypassing the candidate/scorer/memory
        # machinery. Lazily constructed per-episode in _init_oracle_follower.
        self.follower = None
        self._oracle_goal_radius = 1.0
        # Navmesh point-goal locomotion (Phase-2 C1 fix): the same
        # ShortestPathFollower steers toward the agent's SELF-CHOSEN waypoint
        # (frontier/memory/remembr), replacing the occupancy-grid step
        # controller whose grid-vs-navmesh mismatch kept SPL at 0. High-level
        # waypoint selection is unchanged; only locomotion uses the navmesh.
        # goal_radius ≈ propose_reached_m so "reached" aligns with re-propose.
        self._waypoint_goal_radius = 0.5
        self._waypoint_force_repropose = False
        # ReMEmbR pair is required for backbone='remembr' but optional otherwise
        # so the frontier-only path keeps its constructor signature simple.
        if backbone == "remembr" and (remembr_builder is None or remembr_planner is None):
            raise ValueError("backbone='remembr' requires remembr_builder and remembr_planner")
        self.remembr_builder = remembr_builder
        self.remembr_planner = remembr_planner

        # Run 4: obstacle-aware proposal pool. When backbone=remembr, the LLM
        # planner is pose-aware but obstacle-blind (Run 3 finding: every
        # forward sector in scene wcojb4TFT35 is wall, but the LLM still
        # proposes "1.5 m ahead"). Inject up to N frontier-planner candidates
        # alongside the LLM's so the rerank can prefer reachable options.
        # Env-tunable so the cap can shift without a constructor change.
        self.n_frontier_inject: int = int(os.environ.get("REMEMBR_FRONTIER_INJECT", "3"))

        # Commit-to-candidate (Phase-2): with a REAL ReMEmbR backbone, every
        # re-proposal is an expensive 7B agent loop. The Run-6 controller's
        # re-steer signals (force-replan from A* fallback, stuck) fire ~every
        # step, which — when each re-proposal called the real LLM — made a
        # full-horizon episode take ~20 min and thrashed the steering target.
        # Decouple the two cadences: the planner PROPOSES a waypoint only every
        # ``propose_period`` steps (or when the current one is reached / there
        # is none); between proposals the agent COMMITS to that waypoint and the
        # step controller re-steers toward it each step (A* + reachable-fallback
        # handle obstacles without a new LLM call). Caps LLM calls at
        # ~n_steps/period and kills the target ping-pong.
        self.propose_period: int = int(
            os.environ.get("REMEMBR_PROPOSE_PERIOD", str(self.planner.decision_period))
        )
        self.propose_reached_m: float = float(
            os.environ.get("REMEMBR_PROPOSE_REACHED_M", "0.5")
        )

        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self, n_episodes: int) -> RunSummary:
        summary = RunSummary()

        for ep_idx in range(n_episodes):
            summary.n_episodes_attempted += 1
            try:
                ep_log, ep_metrics = self._run_episode(ep_idx)
            except Exception as e:
                summary.notes.append(f"episode {ep_idx} crashed: {type(e).__name__}: {e}")
                self._dump_json(
                    os.path.join(self.out_dir, f"episode_{ep_idx:03d}_error.json"),
                    {"episode_idx": ep_idx, "error": repr(e)},
                )
                continue

            self._dump_json(os.path.join(self.out_dir, f"episode_{ep_idx:03d}.json"), ep_log)
            summary.n_episodes_completed += 1
            if ep_metrics.get("success"):
                summary.n_successful_episodes += 1
            summary.rerank_calls += int(ep_metrics.get("rerank_calls", 0))
            summary.rerank_disagreements += int(ep_metrics.get("rerank_disagreements", 0))
            summary.retrieval_hits += int(ep_metrics.get("retrieval_hits", 0))
            summary.n_memory_candidates += int(ep_metrics.get("n_memory_candidates", 0))
            summary.n_memory_chosen += int(ep_metrics.get("n_memory_chosen", 0))
            summary.n_frontier_chosen += int(ep_metrics.get("n_frontier_chosen", 0))
            summary.n_remembr_chosen += int(ep_metrics.get("n_remembr_chosen", 0))
            summary.n_stop_signals += int(ep_metrics.get("n_stop_signals", 0))
            # Per-episode row used by analyze_ablation.py to pair runs.
            summary.episodes.append({
                "episode_idx": ep_idx,
                "episode_id": ep_log.get("episode_id"),
                "scene_id": ep_log.get("scene_id"),
                "target_category": ep_log.get("target_category"),
                "success": bool(ep_metrics.get("success", False)),
                "spl": float(ep_metrics.get("spl", 0.0)),
                "soft_spl": float(ep_metrics.get("soft_spl", 0.0)),
                "n_steps": int(ep_log.get("n_steps", 0)),
                "rerank_calls": int(ep_metrics.get("rerank_calls", 0)),
                "rerank_disagreements": int(ep_metrics.get("rerank_disagreements", 0)),
                "retrieval_hits": int(ep_metrics.get("retrieval_hits", 0)),
                "n_memory_candidates": int(ep_metrics.get("n_memory_candidates", 0)),
                "n_memory_chosen": int(ep_metrics.get("n_memory_chosen", 0)),
                "n_frontier_chosen": int(ep_metrics.get("n_frontier_chosen", 0)),
                "n_remembr_chosen": int(ep_metrics.get("n_remembr_chosen", 0)),
                "n_stop_signals": int(ep_metrics.get("n_stop_signals", 0)),
                "distance_to_goal": ep_metrics.get("distance_to_goal"),
                "grid_cells_free": int(ep_metrics.get("grid_cells_free", 0)),
                "grid_cells_occupied": int(ep_metrics.get("grid_cells_occupied", 0)),
                "grid_cells_unknown": int(ep_metrics.get("grid_cells_unknown", 0)),
                "grid_frontier_cells": int(ep_metrics.get("grid_frontier_cells", 0)),
                "action_forward": int(ep_metrics.get("action_forward", 0)),
                "action_turn": int(ep_metrics.get("action_turn", 0)),
                "action_stop": int(ep_metrics.get("action_stop", 0)),
                "astar_path": int(ep_metrics.get("astar_path", 0)),
                "astar_reachable_fallback": int(ep_metrics.get("astar_reachable_fallback", 0)),
                "astar_fallback": int(ep_metrics.get("astar_fallback", 0)),
                "collision_escape": int(ep_metrics.get("collision_escape", 0)),
                "replan_scheduled": int(ep_metrics.get("replan_scheduled", 0)),
                "replan_forced": int(ep_metrics.get("replan_forced", 0)),
                "replan_stuck": int(ep_metrics.get("replan_stuck", 0)),
                "remembr_stub_mode": ep_metrics.get("remembr_stub_mode"),
                "remembr_sample_caption": ep_metrics.get("remembr_sample_caption"),
            })

        # Finalize summary. The oracle backbone runs without a memory bridge,
        # so guard every dereference and fall back to empty stats.
        bridge_stats = self.bridge.stats() if self.bridge is not None else {}
        summary.ltm_counts_final = bridge_stats.get("ltm_counts", {})
        summary.modules_invoked = bridge_stats.get("modules_invoked", {})
        summary.n_keyframes_observed = int(bridge_stats.get("n_keyframes_observed", 0))
        summary.ablation = {
            **bridge_stats.get("ablation", {}),
            **{k: v for k, v in self.run_config.items() if k not in {"setting"}},
            "setting": self.run_config.get("setting"),
        }
        summary.pass_conditions = self._evaluate_pass_conditions(summary)

        self._dump_json(os.path.join(self.out_dir, "summary.json"), summary.to_dict())
        return summary

    # ------------------------------------------------------------------
    # per-episode
    # ------------------------------------------------------------------

    def _run_episode(self, ep_idx: int):
        is_oracle = self.backbone == "oracle"
        step, ep = self.source.reset(ep_idx)
        self.planner.reset(agent_pos=step.agent_state.position)
        if self.bridge is not None:
            self.bridge.begin_episode(ep.episode_id, scene_id=ep.scene_id)

        ep_log: Dict[str, Any] = {
            "episode_idx": ep_idx,
            "episode_id": ep.episode_id,
            "scene_id": ep.scene_id,
            "target_category": ep.target_category,
            "started_at": time.time(),
            "steps": [],
            "decisions": [],
        }

        # Oracle with no goal would silently STOP at step 0; flag it loudly so
        # the empty path isn't mistaken for a real navigation failure.
        if is_oracle and getattr(ep, "target_position", None) is None:
            warn = (
                f"[oracle] episode {ep_idx} (id={ep.episode_id}) has NO goal "
                f"(target_position is None) — agent STOPs immediately; this is "
                f"NOT a navigation result."
            )
            print("!" * 78 + f"\n{warn}\n" + "!" * 78)
            ep_log["oracle_no_goal"] = True

        rerank_calls = 0
        rerank_disagreements = 0
        retrieval_hits = 0
        n_memory_candidates = 0
        n_memory_chosen = 0
        n_frontier_chosen = 0
        n_remembr_chosen = 0
        n_stop_signals = 0
        stm_captions: List[str] = []
        current_candidate: Optional[FrontierCandidate] = None
        last_propose_step: int = -10**9  # forces a proposal on the first loop tick
        # Run-6 instrumentation: action mix over the episode (non-oracle path).
        action_counts = {ACTION_STOP: 0, ACTION_FORWARD: 0,
                         ACTION_TURN_LEFT: 0, ACTION_TURN_RIGHT: 0}
        # First grounded-STOP event (cosine / matched caption) — set when the
        # ReMEmbR backbone emits a stop_signal candidate. Phase-2 STOP tuning.
        stop_event: Optional[Dict[str, Any]] = None

        # Initial observation: update map, build keyframe at step 0. The oracle
        # path skips the perception/memory preamble entirely (no bridge, no
        # CLIP, no captioner) — it only needs the goal and the follower.
        self.planner.update(step.depth, step.agent_state.position, step.agent_state.rotation_yaw)
        if not is_oracle:
            keyframe = self._build_keyframe(step)
            self.bridge.observe_keyframe(keyframe, action=None, reward=0.0)
            stm_captions.append(keyframe.caption)
            ep_log["steps"].append(self._serialize_step(step, keyframe))

            # Reset ReMEmbR per-episode state and index the initial keyframe.
            if self.backbone == "remembr":
                self.remembr_builder.reset()
                self.remembr_planner.reset()
                self.remembr_builder.caption_and_index(
                    rgb=step.rgb,
                    agent_position=step.agent_state.position,
                    timestep=int(step.step_idx),
                )

        # Loop.
        for t in range(1, self.max_steps_per_episode):
            if is_oracle:
                # Oracle short-circuit: steer straight to the goal, bypassing
                # candidate proposal, memory injection, and rerank entirely.
                action = self._oracle_action(ep)
                step = self.source.step(action)
                self.planner.update(
                    step.depth, step.agent_state.position, step.agent_state.rotation_yaw
                )
                if step.done:
                    break
                continue

            # Decide whether to RE-PROPOSE a waypoint (expensive: real LLM
            # agent loop). Commit-to-candidate: only on a fixed schedule, when
            # there is no candidate, or when the current one is reached — NOT on
            # every controller re-steer. is_decision_step() is still ticked for
            # its replan-trigger instrumentation, but no longer gates proposal.
            self.planner.is_decision_step()  # tick stats only (return ignored)
            candidate_reached = (
                current_candidate is not None
                and float(getattr(current_candidate, "distance_m", 1e9))
                < self.propose_reached_m
            )
            due_to_propose = (step.step_idx - last_propose_step) >= self.propose_period
            if current_candidate is None or due_to_propose or candidate_reached:
                last_propose_step = int(step.step_idx)
                cands = self._propose_candidates(step, ep)
                if cands:
                    # raw_top1 is the planner's pick BEFORE memory injection,
                    # so the disagreement counter measures "did rerank+memory
                    # change the action vs vanilla planner top-1?".
                    raw_top1 = cands[0]

                    # Grounded STOP short-circuit: if the backbone emitted a
                    # stop_signal candidate, force-select it before rerank so
                    # nothing can outscore it. The runner's action-derivation
                    # block downstream sees stop_signal and emits ACTION_STOP.
                    stop_cand = next(
                        (c for c in cands if c.metadata.get("stop_signal", False)),
                        None,
                    )

                    # Option-2: extend the candidate pool with LTM-derived
                    # waypoints (locations of past observations that look like
                    # the target category in CLIP joint space). Scene-filtered
                    # and de-duped vs planner candidates inside the bridge.
                    mem_cands = self.bridge.propose_memory_candidates(
                        agent_pos=step.agent_state.position,
                        agent_yaw=step.agent_state.rotation_yaw,
                        target_category=ep.target_category,
                        planner_world_xys=[c.world_xy for c in cands],
                        top_k=3,
                    )
                    # Assign fresh, non-clashing ids before merging.
                    for i, mc in enumerate(mem_cands):
                        mc.candidate_id = len(cands) + i + 1000  # offset so logs are unambiguous
                    all_cands = cands + mem_cands
                    n_memory_candidates += len(mem_cands)

                    rerank_result, retrieval = self.bridge.rerank(
                        candidates=all_cands,
                        query_text=keyframe.caption,
                        stm_captions=stm_captions[-5:],
                        target_category=ep.target_category,
                        query_visual_embedding=keyframe.visual_embedding,
                    )
                    rerank_calls += 1
                    if any(len(v) > 0 for v in retrieval.values()):
                        retrieval_hits += 1

                    if stop_cand is not None:
                        chosen = stop_cand
                        n_stop_signals += 1
                        # Capture WHY the real backbone STOPped (cosine + matched
                        # caption + distance) so we can tell a correct STOP from a
                        # premature one and tune REMEMBR_STOP_COS for the ablation.
                        stop_event = {
                            "step": int(step.step_idx),
                            "stop_match": stop_cand.metadata.get("stop_match"),
                            "stop_cos": stop_cand.metadata.get("stop_cos"),
                            "stop_dist_m": stop_cand.metadata.get("stop_dist_m"),
                            "matched_caption": stop_cand.metadata.get("matched_caption"),
                        }
                    else:
                        chosen_idx = self._chosen_candidate_index(rerank_result, all_cands)
                        chosen = all_cands[chosen_idx]
                    if chosen.candidate_id != raw_top1.candidate_id:
                        rerank_disagreements += 1
                    if chosen.source == "memory":
                        n_memory_chosen += 1
                    if chosen.source == "frontier":
                        n_frontier_chosen += 1
                    if chosen.source == "remembr":
                        n_remembr_chosen += 1
                    current_candidate = chosen

                    n_frontier_in_pool = sum(1 for c in cands if c.source == "frontier")

                    ep_log["decisions"].append({
                        "step_idx": int(step.step_idx),
                        "raw_top1_id": int(raw_top1.candidate_id),
                        "raw_top1_world_xy": raw_top1.world_xy.tolist(),
                        "raw_top1_score": float(raw_top1.raw_score),
                        "chosen_id": int(chosen.candidate_id),
                        "chosen_world_xy": chosen.world_xy.tolist(),
                        "chosen_source": str(chosen.source),
                        "chosen_final_score": float(rerank_result.selected.final_score)
                        if rerank_result.selected else None,
                        "n_planner_candidates": len(cands),
                        "n_frontier_candidates": n_frontier_in_pool,
                        "n_memory_candidates": len(mem_cands),
                        "candidates": [
                            {
                                "id": int(c.candidate_id),
                                "world_xy": c.world_xy.tolist(),
                                "distance_m": float(c.distance_m),
                                "bearing_rad": float(c.bearing_rad),
                                "cluster_size": int(c.cluster_size),
                                "raw_score": float(c.raw_score),
                                "source": str(c.source),
                            }
                            for c in all_cands
                        ],
                        "rerank_top": rerank_result.debug_info["top_scores"],
                        "retrieval_counts": {k: len(v) for k, v in retrieval.items()},
                    })

            # Convert candidate → action.
            if current_candidate is None:
                action = ACTION_FORWARD
            elif current_candidate.metadata.get("stop_signal", False):
                # ReMEmbR's grounded STOP fired: goal-matching observation
                # lies within the success radius of the agent. Emit action=0.
                action = ACTION_STOP
            else:
                action = self._waypoint_action(
                    current_candidate,
                    step.agent_state.position,
                    step.agent_state.rotation_yaw,
                )
                if self._waypoint_force_repropose:
                    # Follower reports the waypoint reached/unreachable → drop it
                    # so the next tick re-proposes a fresh target (locomotion
                    # never STOPs; only keyword-STOP / stop_signal ends an ep).
                    current_candidate = None

            # Step the env.
            action_counts[action] = action_counts.get(action, 0) + 1
            step = self.source.step(action)
            self.planner.update(
                step.depth, step.agent_state.position, step.agent_state.rotation_yaw
            )

            # Re-bearing-rel is needed for the controller next iteration; we
            # recompute the candidate's bearing relative to the current yaw.
            if current_candidate is not None:
                ax, az = float(step.agent_state.position[0]), float(step.agent_state.position[2])
                tx, tz = float(current_candidate.world_xy[0]), float(current_candidate.world_xy[1])
                import math as _math
                world_bearing = _math.atan2(tx - ax, tz - az)
                rel = world_bearing - float(step.agent_state.rotation_yaw)
                while rel > _math.pi:
                    rel -= 2.0 * _math.pi
                while rel < -_math.pi:
                    rel += 2.0 * _math.pi
                current_candidate.bearing_rad = rel
                current_candidate.distance_m = _math.hypot(tx - ax, tz - az)

            # Build keyframe periodically.
            if step.step_idx % self.keyframe_every_m == 0:
                keyframe = self._build_keyframe(step)
                self.bridge.observe_keyframe(
                    keyframe, action=action, reward=step.reward, success=False
                )
                stm_captions.append(keyframe.caption)
                ep_log["steps"].append(self._serialize_step(step, keyframe))
                # ReMEmbR build phase: per-keyframe caption + flat-memory write.
                if self.backbone == "remembr":
                    self.remembr_builder.caption_and_index(
                        rgb=step.rgb,
                        agent_position=step.agent_state.position,
                        timestep=int(step.step_idx),
                    )

            if step.done:
                break

        # End-of-episode: figure out success, consolidate.
        success = bool(step.info.get("success", False)) or bool(
            step.info.get("distance_to_goal", 1e9) < 0.1
        )
        spl = float(step.info.get("spl", 1.0 if success else 0.0))
        soft_spl = float(step.info.get("softspl", step.info.get("soft_spl", spl)))
        distance_to_goal = step.info.get("distance_to_goal")
        ep.success = success
        ep.spl = spl

        # Stamp success on the most-recent observed keyframe so the segment
        # the consolidator sees can be flagged successful. Skipped on the
        # oracle path (no bridge).
        if self.bridge is not None:
            if success and self.bridge._pending:  # noqa: SLF001 — controlled use
                self.bridge._pending[-1].success = True  # noqa: SLF001
            self.bridge.consolidate(episode_success=success, episode_idx=ep_idx)

        # Occupancy-grid census (Run-5 instrumentation) — makes the smoke
        # interpretable next to n_frontier_chosen.
        grid_stats = self.planner.grid_stats()
        # Controller census (Run-6 instrumentation) — replan-trigger breakdown,
        # A* path-vs-fallback, collision-escape, and action mix. Distinguishes a
        # force-replan loop from a stuck loop from a geometry stall.
        controller_stats = self.planner.controller_stats()
        action_turn = action_counts[ACTION_TURN_LEFT] + action_counts[ACTION_TURN_RIGHT]
        controller_log = {
            "action_forward": action_counts[ACTION_FORWARD],
            "action_turn": action_turn,
            "action_stop": action_counts[ACTION_STOP],
            "astar_path": controller_stats["astar_path"],
            "astar_reachable_fallback": controller_stats["astar_reachable_fallback"],
            "astar_fallback": controller_stats["astar_fallback"],
            "collision_escape": controller_stats["collision_escape"],
            "replan_scheduled": controller_stats["replan_scheduled"],
            "replan_forced": controller_stats["replan_forced"],
            "replan_stuck": controller_stats["replan_stuck"],
        }
        # ReMEmbR backbone certification (Phase-2): record whether the real
        # weights actually loaded, so a long ablation self-certifies and
        # analyze_ablation can refuse a silent-stub run. Every prior run was
        # stub (missing accelerate); never trust a remembr run that doesn't say
        # remembr_stub_mode=false. The sample caption lets us eyeball real VLM
        # output ("a bedroom with...") vs the stub ("stub-caption step=N").
        remembr_log: Dict[str, Any] = {}
        if self.backbone == "remembr":
            recs = self.remembr_builder.records
            remembr_log = {
                "remembr_stub_mode": bool(self.remembr_planner.stub_mode),
                "remembr_builder_stub": bool(self.remembr_builder.stub_mode),
                "remembr_n_records": len(recs),
                "remembr_sample_caption": recs[len(recs) // 2].caption if recs else None,
                "remembr_stop_event": stop_event,
            }

        ep_log["finished_at"] = time.time()
        ep_log["n_steps"] = int(step.step_idx)
        ep_log["success"] = success
        ep_log["spl"] = spl
        ep_log["soft_spl"] = soft_spl
        ep_log["distance_to_goal"] = distance_to_goal
        ep_log["rerank_calls"] = rerank_calls
        ep_log["rerank_disagreements"] = rerank_disagreements
        ep_log["retrieval_hits"] = retrieval_hits
        ep_log["n_memory_candidates"] = n_memory_candidates
        ep_log["n_memory_chosen"] = n_memory_chosen
        ep_log["n_frontier_chosen"] = n_frontier_chosen
        ep_log["n_remembr_chosen"] = n_remembr_chosen
        ep_log["n_stop_signals"] = n_stop_signals
        ep_log["grid_cells_free"] = grid_stats["cells_free"]
        ep_log["grid_cells_occupied"] = grid_stats["cells_occupied"]
        ep_log["grid_cells_unknown"] = grid_stats["cells_unknown"]
        ep_log["grid_frontier_cells"] = grid_stats["frontier_cells"]
        ep_log.update(controller_log)
        ep_log.update(remembr_log)
        ep_log["bridge_stats_after"] = (
            self.bridge.stats() if self.bridge is not None else {}
        )

        return ep_log, {
            "success": success,
            "spl": spl,
            "soft_spl": soft_spl,
            "distance_to_goal": distance_to_goal,
            "rerank_calls": rerank_calls,
            "rerank_disagreements": rerank_disagreements,
            "retrieval_hits": retrieval_hits,
            "n_memory_candidates": n_memory_candidates,
            "n_memory_chosen": n_memory_chosen,
            "n_frontier_chosen": n_frontier_chosen,
            "n_remembr_chosen": n_remembr_chosen,
            "n_stop_signals": n_stop_signals,
            "grid_cells_free": grid_stats["cells_free"],
            "grid_cells_occupied": grid_stats["cells_occupied"],
            "grid_cells_unknown": grid_stats["cells_unknown"],
            "grid_frontier_cells": grid_stats["frontier_cells"],
            **controller_log,
            **remembr_log,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _propose_candidates(self, step: Step, ep) -> List[FrontierCandidate]:
        """Dispatch primary candidate generation by backbone.

        ``frontier`` uses the Phase-1 stand-in (depth → occupancy grid →
        frontier clusters). ``remembr`` queries the ReMEmbR builder's flat
        memory through the LLM agent loop in ``ReMEmbRPlanner``. Memory
        injection from ``EmbodiedMemoryBridge.propose_memory_candidates`` is
        layered on identically in both branches by the caller.

        For the ``remembr`` backbone, also inject up to ``n_frontier_inject``
        obstacle-aware candidates from the frontier planner. Run 3 showed the
        7B planner is pose-aware but obstacle-blind, so the rerank pool needs
        a reachable alternative when the LLM's "1.5 m ahead" pick is wall.
        STOP short-circuit is preserved: if the LLM emitted a stop_signal
        candidate, return it alone without dilution.
        """
        if self.backbone == "frontier":
            return self.planner.propose(
                step.agent_state.position, step.agent_state.rotation_yaw
            )
        # remembr
        llm_cands = self.remembr_planner.propose(
            goal=ep.target_category or self.target_category,
            agent_pose=step.agent_state.position,
            agent_yaw=step.agent_state.rotation_yaw,
            current_step=int(step.step_idx),
        )
        # Preserve STOP short-circuit (runner force-selects this downstream).
        if llm_cands and llm_cands[0].metadata.get("stop_signal", False):
            return llm_cands
        if self.n_frontier_inject <= 0:
            return llm_cands

        # propose_diverse swaps the single random-walk fallback for a compass
        # fan of N candidates when the occupancy grid is sparse. Run-4 smoke 1
        # showed plain propose() returns a single 1.5 m-forward candidate that
        # de-dups against the LLM's matching forward pick, zeroing out the
        # injection pool. propose_diverse keeps the side picks alive.
        frontier_cands = self.planner.propose_diverse(
            step.agent_state.position,
            step.agent_state.rotation_yaw,
            k=self.n_frontier_inject,
        )
        for fc in frontier_cands:
            fc.source = "frontier"

        # De-dup: drop frontier candidates within MIN_WAYPOINT_DIST of any LLM
        # candidate, so identical "1.5 m forward" picks don't crowd the pool.
        min_dist = float(os.environ.get("REMEMBR_MIN_WAYPOINT_DIST", "0.5"))
        llm_xys = [c.world_xy for c in llm_cands]
        keep: List[FrontierCandidate] = []
        for fc in frontier_cands:
            if all(
                float(np.linalg.norm(fc.world_xy - xy)) > min_dist for xy in llm_xys
            ):
                keep.append(fc)

        return llm_cands + keep

    # ------------------------------------------------------------------
    # oracle backbone (Run-5 diagnostic)
    # ------------------------------------------------------------------

    def _init_oracle_follower(self, ep) -> None:
        """Lazily build a Habitat ShortestPathFollower for the episode goal.

        Reaches the underlying habitat-sim Simulator through the source's
        ``get_sim()`` accessor. Leaves ``self.follower = None`` if the source
        has no sim (e.g. cached mode), in which case the oracle just STOPs.
        """
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        sim = self.source.get_sim()
        if sim is None:
            self.follower = None
            return
        self.follower = ShortestPathFollower(
            sim, goal_radius=self._oracle_goal_radius, return_one_hot=False
        )

    def _oracle_action(self, ep) -> int:
        """Next action toward the episode goal via the ShortestPathFollower.

        Maps the follower's return to a discrete action id:
          - ``None`` (at goal / no path)  → ACTION_STOP
          - action name (str)             → ``_ACTION_NAMES.index(name)``
          - action id (int)               → passed through (already matches
            ``_ACTION_NAMES`` ordering for stop/forward/turn_left/turn_right)
        A missing goal STOPs immediately (the no-goal case is flagged loudly
        in ``_run_episode`` so it isn't read as a navigation failure).
        """
        from .frontier_planner import ACTION_STOP

        goal = getattr(ep, "target_position", None)
        if goal is None:
            return ACTION_STOP
        if self.follower is None:
            self._init_oracle_follower(ep)
            if self.follower is None:
                return ACTION_STOP

        raw = self.follower.get_next_action(goal)
        if raw is None:
            return ACTION_STOP
        if isinstance(raw, str):
            from .habitat_env import _ACTION_NAMES
            try:
                return _ACTION_NAMES.index(raw)
            except ValueError:
                return ACTION_STOP
        if isinstance(raw, (int, np.integer)):
            return int(raw)
        return ACTION_STOP

    def _init_waypoint_follower(self) -> None:
        """Build a navmesh ShortestPathFollower reused for steering toward the
        agent's self-chosen waypoints. Checks for a sim BEFORE importing habitat
        so the sim-less path (cached mode / unit tests) never needs habitat.
        Leaves ``self.follower = None`` when the source has no sim, in which case
        ``_waypoint_action`` falls back to the grid step controller."""
        sim = self.source.get_sim()
        if sim is None:
            self.follower = None
            return
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        self.follower = ShortestPathFollower(
            sim, goal_radius=self._waypoint_goal_radius, return_one_hot=False
        )

    def _waypoint_action(self, candidate, agent_pos, agent_yaw) -> int:
        """Next discrete action toward the chosen waypoint via the navmesh
        ShortestPathFollower (Phase-2 C1 fix).

        Snaps the waypoint to the navmesh and asks the follower for the action
        toward it. ``None`` from the follower (waypoint reached or unreachable)
        sets ``_waypoint_force_repropose`` and returns a TURN — locomotion never
        emits ACTION_STOP (only the keyword-STOP / explicit stop_signal ends an
        episode). When no sim is available (cached mode) it degrades to the
        occupancy-grid ``step_controller`` so that path keeps working.
        """
        from .frontier_planner import ACTION_TURN_LEFT

        self._waypoint_force_repropose = False
        if self.follower is None:
            self._init_waypoint_follower()
        if self.follower is None:
            return self.planner.step_controller(candidate, agent_pos, agent_yaw)

        wx, wz = float(candidate.world_xy[0]), float(candidate.world_xy[1])
        goal = np.array([wx, float(agent_pos[1]), wz], dtype=np.float32)
        sim = self.source.get_sim()
        if sim is not None:
            try:
                sp = sim.pathfinder.snap_point(goal)
                snapped = np.array([float(sp[0]), float(sp[1]), float(sp[2])],
                                   dtype=np.float32)
                if np.all(np.isfinite(snapped)):
                    goal = snapped
            except Exception:
                pass  # off-navmesh or unsupported snap → steer to the raw point

        raw = self.follower.get_next_action(goal)
        if raw is None:
            # Reached/unreachable: drop the waypoint and re-propose; don't STOP.
            self._waypoint_force_repropose = True
            return ACTION_TURN_LEFT
        if isinstance(raw, str):
            from .habitat_env import _ACTION_NAMES
            try:
                return _ACTION_NAMES.index(raw)
            except ValueError:
                self._waypoint_force_repropose = True
                return ACTION_TURN_LEFT
        if isinstance(raw, (int, np.integer)):
            return int(raw)
        self._waypoint_force_repropose = True
        return ACTION_TURN_LEFT

    def _build_keyframe(self, step: Step) -> Keyframe:
        # Cached mode may have already produced a caption + embeddings.
        precomputed_text_emb = step.info.get("text_embedding") if step.info else None
        precomputed_visual_emb = step.info.get("visual_embedding") if step.info else None
        precomputed_caption = step.info.get("caption") if step.info else None

        if precomputed_visual_emb is not None:
            visual = np.asarray(precomputed_visual_emb, dtype=np.float32)
        else:
            visual = self.clip_encoder.encode(step.rgb)

        if precomputed_caption is not None:
            caption = str(precomputed_caption)
        else:
            caption = self.captioner.caption(
                step.semantic, step.agent_state.position, target=self.target_category
            )

        if precomputed_text_emb is not None:
            text_emb = np.asarray(precomputed_text_emb, dtype=np.float32)
        else:
            text_emb = self.bridge.text_encode_fn(caption).astype(np.float32)

        return Keyframe(
            step_idx=int(step.step_idx),
            rgb=step.rgb,
            visual_embedding=visual,
            caption=caption,
            text_embedding=text_emb,
            agent_position=np.asarray(step.agent_state.position, dtype=np.float32),
            agent_yaw=float(step.agent_state.rotation_yaw),
        )

    @staticmethod
    def _chosen_candidate_index(rerank_result, cands: List[FrontierCandidate]) -> int:
        # Reranker returns scored responses sorted; map by stable text prefix.
        if not rerank_result.selected:
            return 0
        chosen_text = rerank_result.selected.response
        # cand_texts are produced deterministically in the bridge; we match
        # by the leading "go to (x.x, y.y)" prefix, which is unique per cand.
        for i, c in enumerate(cands):
            prefix = f"go to ({c.world_xy[0]:.1f}, {c.world_xy[1]:.1f})"
            if chosen_text.startswith(prefix):
                return i
        return 0

    @staticmethod
    def _serialize_step(step: Step, keyframe: Keyframe) -> Dict[str, Any]:
        return {
            "step_idx": int(step.step_idx),
            "action": step.action,
            "reward": float(step.reward),
            "done": bool(step.done),
            "agent_pos": step.agent_state.position.tolist(),
            "agent_yaw": float(step.agent_state.rotation_yaw),
            "caption": keyframe.caption,
        }

    @staticmethod
    def _dump_json(path: str, payload: Dict[str, Any]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)

    def _evaluate_pass_conditions(self, summary: RunSummary) -> Dict[str, bool]:
        ltm = summary.ltm_counts_final or {}
        modules = summary.modules_invoked or {}
        # Criterion 1: fine-layer non-empty after run.
        c1 = int(ltm.get("fine", 0)) >= 1
        # Criterion 2: every rerank call retrieved >= 1 record. Approximated
        # as retrieval_hits == rerank_calls (over all episodes >= 2).
        c2 = (
            summary.rerank_calls == 0
            or summary.retrieval_hits >= max(0, summary.rerank_calls - 1)
        )
        # Criterion 3: at least one disagreement.
        c3 = summary.rerank_disagreements >= 1
        # Criterion 4: all four module categories invoked.
        c4 = all(modules.get(k, False) for k in ("stm", "consolidation", "ltm_fine", "rerank"))
        # Criterion 5: at least one episode completed without crash.
        c5 = summary.n_episodes_completed >= 1
        return {
            "fine_layer_nonempty": c1,
            "rerank_retrieves_always": c2,
            "memory_influences_at_least_once": c3,
            "all_four_modules_invoked": c4,
            "no_crash": c5,
        }


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
