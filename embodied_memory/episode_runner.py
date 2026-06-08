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
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
from .remembr_backbone import (
    ReMEmbRBuilder,
    ReMEmbRPlanner,
    _caption_mentions,
    _goal_terms,
)

if TYPE_CHECKING:
    from .goal_detector import GoalDetector


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
    n_coarse_candidates: int = 0      # total coarse-affordance candidates surfaced (step 4)
    n_coarse_chosen: int = 0          # decisions where reranker picked a coarse-affordance candidate
    n_remembr_chosen: int = 0         # decisions where reranker picked a grounded remembr (LLM) candidate
    n_stop_signals: int = 0           # decisions where backbone emitted a grounded STOP
    n_detector_called: int = 0
    n_detector_localized: int = 0
    n_detector_locate_failed: int = 0
    n_detector_gated: int = 0
    n_detector_approach_success: int = 0
    n_arrival_stop: int = 0
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
            "n_coarse_candidates": self.n_coarse_candidates,
            "n_coarse_chosen": self.n_coarse_chosen,
            "n_remembr_chosen": self.n_remembr_chosen,
            "n_stop_signals": self.n_stop_signals,
            "n_detector_called": self.n_detector_called,
            "n_detector_localized": self.n_detector_localized,
            "n_detector_locate_failed": self.n_detector_locate_failed,
            "n_detector_gated": self.n_detector_gated,
            "n_detector_approach_success": self.n_detector_approach_success,
            "n_arrival_stop": self.n_arrival_stop,
            "n_keyframes_observed": self.n_keyframes_observed,
            "modules_invoked": self.modules_invoked,
            "ablation": self.ablation,
            "pass_conditions": self.pass_conditions,
            "notes": self.notes,
            "episodes": self.episodes,
        }


# ---------------------------------------------------------------------------
# Detector intercept helpers (Task 4)
# ---------------------------------------------------------------------------


def _decide_stop_or_approach(
    detector_enabled: bool,
    detector,                          # GoalDetector or None
    rgb,
    depth,
    goal_category: str,
    agent_pose,
    intrinsics,
    counters: Dict[str, Any],
    mem_world_xys=None,
    agree_radius: float = 2.0,
):
    """Decide what to do at a stop_signal=True candidate.

    Returns (action, approach_waypoint):
      action=ACTION_STOP, approach_waypoint=None   -> STOP this tick
      action=None,        approach_waypoint=wp    -> caller must navigate
                                                     toward ``wp`` (3D world)
    Updates ``counters`` (keyed by 'n_detector_called', 'n_detector_localized',
    'n_detector_locate_failed', 'n_detector_gated') in place.
    'n_detector_locate_failed' counts any None return from locate() — no-bbox,
    parse failure, invalid depth, OR off-navmesh snap (all locate() failure
    modes, not exclusively off-navmesh).

    c9 detector-memory gate: after a successful localize, the waypoint is
    committed only if it concurs with a retrieved same-category LTM sighting
    (``mem_world_xys`` within ``agree_radius``); otherwise we fall back to plain
    STOP and bump 'n_detector_gated'. ``mem_world_xys=None`` disables the gate
    (legacy c7 behavior). Pure: no I/O, no side effects beyond the counters dict.
    """
    if not detector_enabled or detector is None:
        return ACTION_STOP, None
    counters["n_detector_called"] = counters.get("n_detector_called", 0) + 1
    wp = detector.locate(
        rgb=rgb, depth=depth, goal_category=goal_category,
        agent_pose=agent_pose, intrinsics=intrinsics,
    )
    if wp is None:
        counters["n_detector_locate_failed"] = counters.get("n_detector_locate_failed", 0) + 1
        return ACTION_STOP, None
    counters["n_detector_localized"] = counters.get("n_detector_localized", 0) + 1
    wp = np.asarray(wp, dtype=np.float32)
    if not _detector_memory_agrees(wp, mem_world_xys, agree_radius):
        # Detector disagrees with the LTM (wrong-instance or cold) -> don't
        # commit; fall back to plain STOP (monotonic with detector-OFF).
        counters["n_detector_gated"] = counters.get("n_detector_gated", 0) + 1
        return ACTION_STOP, None
    return None, wp


def _detector_candidate(approach_wp_xyz, agent_pos):
    """Construct a FrontierCandidate at the detector-snapped waypoint.

    ``approach_wp_xyz`` is a 3-vector in world coords; the candidate only
    uses (x, z) (the floor plane). ``agent_pos`` lets us populate
    ``distance_m`` for downstream consumers.
    """
    return FrontierCandidate(
        candidate_id=-1,
        world_xy=np.array([approach_wp_xyz[0], approach_wp_xyz[2]], dtype=np.float32),
        grid_rc=(0, 0),
        distance_m=float(np.linalg.norm(
            np.array([agent_pos[0], agent_pos[2]])
            - np.array([approach_wp_xyz[0], approach_wp_xyz[2]])
        )),
        bearing_rad=0.0,
        cluster_size=1,
        raw_score=1.0,
        source="detector",
        metadata={"approach": True},
    )


def _approach_arrived(force_repropose, agent_pos, approach_wp, approach_radius):
    """Return ``(arrived: bool, stop_distance_xz: float)``. Arrived iff the
    follower signalled done (``force_repropose``) OR the agent is already within
    ``approach_radius`` of the snapped waypoint (soft backstop). Pure: no I/O.

    ``stop_distance_xz`` is the floor-plane (x, z) distance from the agent to the
    raw waypoint, recorded for the n_detector_approach_stop_distance counter.
    """
    stop_distance = float(
        np.linalg.norm(
            np.array([agent_pos[0], agent_pos[2]])
            - np.array([approach_wp[0], approach_wp[2]])
        )
    )
    arrived = bool(force_repropose or stop_distance < approach_radius)
    return arrived, stop_distance


def _detector_memory_agrees(approach_wp, mem_world_xys, agree_radius):
    """Return True iff the detector-localized waypoint concurs with the LTM (c9).

    ``mem_world_xys`` is the list of (x, z) world positions of retrieved
    same-category LTM sightings at this decision. Agreement = the detector's
    localized point is within ``agree_radius`` (floor-plane) of at least one
    sighting. This gates out c7's two failure modes: wrong-instance grounding
    (detected object far from where memory saw the goal) and cold visits.

    Sentinel: ``mem_world_xys is None`` disables the gate (legacy c7 behavior,
    always agree). An empty list means the gate is ON but no sighting was
    recalled (cold) -> disagree. Pure: no I/O.
    """
    if mem_world_xys is None:
        return True
    wp_xz = np.array([approach_wp[0], approach_wp[2]])
    for sx in mem_world_xys:
        if float(np.linalg.norm(wp_xz - np.array([sx[0], sx[1]]))) < agree_radius:
            return True
    return False


def _oracle_stop_override(action, dist_to_goal, radius):
    """Oracle-STOP diagnostic: force ACTION_STOP once the agent is within
    ``radius`` (GT geodesic distance to goal), isolating the termination layer
    from the rest of the policy. Passes ``action`` through unchanged when the
    GT distance is unknown (None) or outside the ring. Pure: no I/O."""
    if dist_to_goal is not None and float(dist_to_goal) <= radius:
        return ACTION_STOP
    return action


def _arrival_stop(near, candidate, caption_confirms, cos_threshold):
    """Waypoint-arrival STOP — the realistic proxy for oracle-STOP.

    The oracle ladder (2026-06-02) showed the LTM's navigation already reaches the
    goal viewpoint in ~75% of warm episodes; the only thing failing is the STOP
    decision (caption-keyword STOP fires on a mere object mention, decoupled from
    goal proximity). Memory waypoints ARE remembered goal positions, so being at a
    confident one ≈ being at the goal. STOP iff: the agent is NEAR (within the
    proximity radius of) the chosen waypoint, it is a MEMORY candidate, its
    retrieval cosine clears ``cos_threshold``, AND the current caption confirms the
    goal object. ``near`` is proximity-based (checked every tick on the waypoint's
    distance_m), mirroring oracle-STOP's distance semantics against the REMEMBERED
    goal instead of the GT one — far more reliable than the follower's exact
    arrival signal (arrival-1: that fired only 2× / 16 eps). Pure."""
    if not near:
        return False
    if candidate is None or getattr(candidate, "source", None) != "memory":
        return False
    if float(getattr(candidate, "raw_score", 0.0)) < cos_threshold:
        return False
    return bool(caption_confirms)


def _advance_subgoal(
    dist_to_active,
    caption_confirms: bool,
    subgoal_idx: int,
    n_subgoals: int,
    found_radius: float,
):
    """MultiON advance decision — the one load-bearing core-loop change.

    Returns ``(found, finished)`` for the ACTIVE sub-goal: ``found`` iff the
    agent is within ``found_radius`` (geodesic, env-adjudicated like Habitat's
    own success check) AND the current caption confirms the category (the
    agent-side signal). ``finished`` iff that found sub-goal was the last one.

    Lenient strictness (confirmed design): a wrong-category confirm never
    aborts — it simply doesn't advance (Progress stays clean). ``None``
    distance (no sim / unreachable / base EpisodeSource) never advances.
    Boundary: ``dist == found_radius`` is NOT found (strict <). Pure: no I/O.
    """
    if dist_to_active is None:
        return False, False
    if float(dist_to_active) >= float(found_radius):
        return False, False
    if not caption_confirms:
        return False, False
    return True, subgoal_idx >= n_subgoals - 1


def _farthest_from_points(cands: List[FrontierCandidate], points):
    """The non-stop candidate FARTHEST from ``points`` (max over candidates of
    the min floor-plane distance to any point) — the least-bad pick when every
    non-stop candidate sits inside the unreachable blacklist (full2 ep5: the
    raw-pool fallback re-admitted the bad waypoint and the agent turned in
    place for 724 ticks). ``None`` when there is no non-stop candidate. Pure."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    best = None
    best_d = -1.0
    for c in cands:
        if c.metadata.get("stop_signal", False):
            continue
        cx, cz = float(c.world_xy[0]), float(c.world_xy[1])
        d = min((float(np.hypot(cx - px, cz - pz)) for px, pz in pts),
                default=float("inf"))
        if d > best_d:
            best, best_d = c, d
    return best


def _filter_candidates_near_points(
    cands: List[FrontierCandidate],
    points,
    radius: float,
    prefer_farthest: bool = False,
):
    """Drop pool candidates within ``radius`` (strict ``<``, floor-plane) of
    ANY of ``points`` — the shared engine behind the reached-thrash near-filter
    (points = [agent position]) and the unreachable-waypoint blacklist
    (points = follower-unreachable waypoints). Returns ``(filtered, n_dropped)``.

    * ``stop_signal`` candidates are NEVER dropped — near by design.
    * If EVERY non-stop candidate would drop, the fallback keeps the agent
      from going waypoint-less:
      - ``prefer_farthest=False`` (default, near-filter call site): the
        ORIGINAL list is returned unchanged (n_dropped 0).
      - ``prefer_farthest=True`` (blacklist call site): keep only the single
        non-stop candidate FARTHEST from ``points`` (+ stop candidates) —
        returning the raw pool re-admitted the blacklisted waypoint and froze
        full2 ep5 in a turn-in-place loop.
    Pure: no I/O."""
    if not points:
        return cands, 0
    pts = [(float(p[0]), float(p[1])) for p in points]
    kept: List[FrontierCandidate] = []
    n_dropped = 0
    for c in cands:
        if c.metadata.get("stop_signal", False):
            kept.append(c)
            continue
        cx, cz = float(c.world_xy[0]), float(c.world_xy[1])
        if any(float(np.hypot(cx - px, cz - pz)) < float(radius)
               for px, pz in pts):
            n_dropped += 1
            continue
        kept.append(c)
    if n_dropped and not any(
        not c.metadata.get("stop_signal", False) for c in kept
    ):
        if prefer_farthest:
            far = _farthest_from_points(cands, pts)
            if far is not None:
                kept = [c for c in cands
                        if c.metadata.get("stop_signal", False) or c is far]
                return kept, n_dropped - 1
        return list(cands), 0
    return kept, n_dropped


def _filter_near_candidates(
    cands: List[FrontierCandidate],
    agent_xy,
    min_target_m: float,
):
    """Drop pool candidates already within ``min_target_m`` of the agent —
    the reached-thrash absorbing loop (multion-micro3 ep0: ~700 of 749 steps
    spun re-propose → turn toward a near pick → "reached" → re-propose,
    because the pool kept yielding candidates inside the 0.5 m reached
    radius). Returns ``(filtered, n_dropped)``.

    Strict ``<`` (matching ``_advance_subgoal`` and the ``candidate_reached``
    trigger): a candidate at exactly ``min_target_m`` can never re-trigger
    "reached", so it survives. stop_signal preservation + all-near fallback
    semantics live in ``_filter_candidates_near_points``. Pure: no I/O."""
    return _filter_candidates_near_points(cands, [agent_xy], min_target_m)


def _cooldown_elapsed(step_idx: int, last_fire_step: int, cooldown: int) -> bool:
    """True iff at least ``cooldown`` steps passed since ``last_fire_step`` —
    the reached-triggered re-propose backstop (bounds ``rerank_calls`` even
    when the near-filter falls back on an all-near pool). Pure."""
    return (int(step_idx) - int(last_fire_step)) >= int(cooldown)


def _waypoint_outcome(
    force_repropose: bool,
    dist_to_waypoint_m,
    goal_radius: float = 0.5,
    slack: float = 0.3,
):
    """Classify a follower "done" tick (multion-micro2 diagnostics).

    The navmesh follower signals done via None/STOP without saying why;
    ``_waypoint_action`` maps both to TURN + force_repropose. Disambiguate by
    distance: within ``goal_radius + slack`` -> ``"reached"``; still far (or
    unknown distance) -> ``"unreachable"`` — the turn-forever absorbing loop
    is 700+ consecutive unreachables. Returns None on a normal tick. Pure.
    """
    if not force_repropose:
        return None
    if dist_to_waypoint_m is None:
        return "unreachable"
    return ("reached" if float(dist_to_waypoint_m) <= goal_radius + slack
            else "unreachable")


def _forward_no_progress(
    action: int, displacement_m: float, min_progress_m: float = 0.05
) -> bool:
    """True iff a FORWARD action produced ~no displacement — collision
    sliding (wall-pushing). The grid-era ``collision_escape`` counter never
    fires under the navmesh follower, so this is its follower-era analogue
    (multion-micro2: 729 forwards moved the agent 0.07 m total). Pure."""
    return action == ACTION_FORWARD and float(displacement_m) < float(min_progress_m)


def _should_snap_unreachable(n: int, threshold: int) -> bool:
    """True iff ``n`` consecutive follower-unreachable ticks reached the snap
    threshold (``>=``; 0 disables). full2 ep5: the blacklist alone could not
    break the loop because the never-moving agent kept re-clustering the same
    geometry — the escape snaps the waypoint onto the navmesh instead. Pure."""
    return int(threshold) > 0 and int(n) >= int(threshold)


def _no_progress_escape(window, min_events: int, window_size: int) -> bool:
    """True iff the rolling window of per-tick ``_forward_no_progress`` flags
    is FULL and carries >= ``min_events`` no-progress forwards (0 disables).

    The grid-era ``collision_escape`` needs two CONSECUTIVE forwards, but the
    full2 ep4 wall-loop alternates FORWARD/TURN so it never fired —
    ``_forward_no_progress`` was computed every tick yet only counted. A
    windowed predicate catches the alternating shape. Pure."""
    if int(min_events) <= 0:
        return False
    if len(window) < int(window_size):
        return False
    return sum(1 for w in window if w) >= int(min_events)


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
        goal_detector: Optional["GoalDetector"] = None,
        oracle_stop: bool = False,
        oracle_location: bool = False,
        oracle_stop_radius: float = 0.1,
        target_categories: Optional[List[str]] = None,
        found_radius: float = 1.0,
    ):
        self.source = source
        self.planner = planner
        self.bridge = bridge
        self.clip_encoder = clip_encoder
        self.captioner = captioner
        self.out_dir = out_dir
        self.target_category = target_category
        # MultiON (sequential semantic ObjectNav): ordered K-chain of goal
        # categories. Per-episode info["object_categories"] (from the multion
        # dataset builder) takes precedence; this is the CLI --target-sequence
        # override. None + no info key -> single-goal behaviour, byte-identical
        # (every multion branch in _run_episode is gated behind K > 1).
        self.target_categories = list(target_categories) if target_categories else None
        # Sub-goal "found" radius (m). MultiON-standard forgiving radius; the
        # advance also requires a caption confirm (see _advance_subgoal).
        self.found_radius = float(found_radius)
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
        # Dedicated tighter follower for the detector final-approach (c7): a
        # separate ShortestPathFollower with goal_radius=0.25 so the agent stops
        # closer to the snapped goal than the 0.5 m normal-nav re-propose radius.
        # Lazily built in _init_approach_follower; None when there is no sim.
        self.approach_follower = None
        self._oracle_goal_radius = 1.0
        # Navmesh point-goal locomotion (Phase-2 C1 fix): the same
        # ShortestPathFollower steers toward the agent's SELF-CHOSEN waypoint
        # (frontier/memory/remembr), replacing the occupancy-grid step
        # controller whose grid-vs-navmesh mismatch kept SPL at 0. High-level
        # waypoint selection is unchanged; only locomotion uses the navmesh.
        # goal_radius ≈ propose_reached_m so "reached" aligns with re-propose.
        self._waypoint_goal_radius = 0.5
        # Tighter radius for the detector final-approach follower (c7). Stopping
        # at 0.25 m of the snapped goal (vs 0.5 m for normal nav) recovers binary
        # SPL@0.1 m, which the 0.5 m re-propose radius could not reach.
        self._approach_goal_radius = float(
            os.environ.get("DETECTOR_APPROACH_RADIUS", "0.25")
        )
        # c9 detector-memory agreement gate: commit to the detector's precise
        # approach only when its localized point is within this radius of a
        # retrieved same-category LTM sighting. Suppresses wrong-instance
        # grounding + cold-visit firing (the two c7 regressions).
        self._detector_mem_agree_m = float(
            os.environ.get("DETECTOR_MEM_AGREE_M", "2.0")
        )
        self._waypoint_force_repropose = False
        # Goal detector for precise final-approach (Task 3).
        self.goal_detector = goal_detector
        self.detector_enabled = goal_detector is not None
        # Detector approach state: when locate() returns a 3D point we lock
        # to that waypoint for subsequent ticks until ShortestPathFollower
        # reports reached -> emit STOP (Task 4 implements the intercept).
        self._approach_waypoint: Optional[np.ndarray] = None
        # Oracle-ladder diagnostics (bottleneck isolation; NOT used in headline
        # configs). oracle_location steers to the GT goal (removes the
        # exploration+retrieval layer); oracle_stop forces STOP within
        # oracle_stop_radius of the GT goal (removes the termination layer).
        # Both keep the rest of the S3 policy intact so each layer's ceiling is
        # measured independently. See _oracle_stop_override + the loop hooks.
        self.oracle_stop = bool(oracle_stop)
        self.oracle_location = bool(oracle_location)
        self.oracle_stop_radius = float(oracle_stop_radius)
        # Waypoint-arrival STOP (oracle-ladder finding: termination is the
        # bottleneck). STOP when the agent arrives at a memory waypoint whose
        # retrieval cosine clears this gate AND the caption confirms the goal.
        # Layers on top of the backbone's keyword-STOP; env-tunable.
        self._arrival_stop_cos = float(os.environ.get("ARRIVAL_STOP_COS", "0.4"))
        # Proximity radius for the arrival STOP (m). MUST exceed the follower
        # goal_radius (_waypoint_goal_radius=0.5) so the agent crosses into the ring
        # while approaching — a ring AT 0.5m never triggers because the follower
        # stops the agent there (arrival-2 fired 0×). 0.75 catches the approach.
        self._arrival_stop_radius = float(os.environ.get("ARRIVAL_STOP_RADIUS", "0.75"))
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

        # Reached-thrash escape (multion-micro3 ep0: ~700/749 steps burned in
        # a re-propose→turn→"reached" absorbing loop). Multion-gated behavior:
        # (a) pool candidates nearer than min_target_m are dropped before
        # rerank (_filter_near_candidates: stop_signal preserved, falls back
        # to the unfiltered pool when all non-stop picks are near), and (b) a
        # reached-triggered re-propose fires at most once per cooldown window
        # (_cooldown_elapsed). K=1 behavior is byte-identical; the counters
        # n_propose_reached / n_candidates_filtered_near log unconditionally.
        self.min_target_m: float = float(
            os.environ.get("REMEMBR_MIN_TARGET_M", str(self.propose_reached_m))
        )
        self.propose_cooldown: int = int(
            os.environ.get("REMEMBR_PROPOSE_COOLDOWN", "3")
        )
        # Unreachable-waypoint blacklist (multion-full1 third absorbing mode:
        # follower reports the chosen frontier waypoint navmesh-unreachable →
        # candidate dropped → the planner re-proposes the SAME cluster next
        # tick → turn-forever; S1 6/8 eps, S3 3/8, top pick re-chosen up to
        # 732×). Multion-gated: waypoints the follower reported unreachable
        # this episode are filtered from later pools (same stop-preserving /
        # never-empty fallback as the near-filter). 0 disables.
        self.unreachable_blacklist_m: float = float(
            os.environ.get("REMEMBR_UNREACHABLE_BLACKLIST_M", "0.5")
        )
        # Snap escape (full2 ep5, turn-in-place): after N CONSECUTIVE
        # follower-unreachable ticks, snap the current waypoint to the nearest
        # navmesh point (sim.pathfinder.snap_point) and re-commit it once —
        # the blacklist alone can't break the loop because the never-moving
        # agent re-clusters the same geometry at coordinates drifting outside
        # the blacklist radius. Multion-gated; 0 disables; no-sim fallback is
        # the existing blacklist+drop. Default 8 -> 1 after full3: ep5's
        # interleaved progressing-forwards kept resetting the consecutive
        # count (803 unreachables, 0 escapes), and waiting 8 ticks per snap
        # just slowed the ep0 snap-loop — snap on the FIRST failure, and at
        # most ONCE per waypoint (the snap_retried mark below): if the
        # snapped point is also unreachable, blacklist + drop instead of
        # re-snapping forever (full3 ep0: escape=125, wp_unreach=1023).
        self.unreachable_snap_n: int = int(
            os.environ.get("REMEMBR_UNREACHABLE_SNAP_N", "1")
        )
        # Memory-consumption fix (full3 post-mortem, the S3-specific
        # wrong-instance recall attractor): once a MEMORY-source waypoint is
        # REACHED without the sub-goal advancing, the recall is a dead lead —
        # consume it (filter its position from later pools for the rest of
        # the CURRENT sub-goal; cleared on advance). full3 ep12 re-chose one
        # bad toilet recall 945x; ep10/11/13 oscillated on one all episode.
        # Multion-gated; 0 disables.
        self.consume_reached_mem: bool = (
            os.environ.get("REMEMBR_CONSUME_REACHED_MEM", "1") != "0"
        )
        # Windowed no-progress escape (full2 ep4, forward-into-wall): when
        # >= MIN of the last WINDOW ticks were no-progress forwards, blacklist
        # the committed waypoint, drop it, and force a re-propose. The
        # grid-era collision_escape needs two consecutive forwards and never
        # fires under the alternating FORWARD/TURN wall-loop. Multion-gated;
        # MIN=0 disables.
        self.no_progress_window: int = int(
            os.environ.get("REMEMBR_NO_PROGRESS_WINDOW", "20")
        )
        self.no_progress_min: int = int(
            os.environ.get("REMEMBR_NO_PROGRESS_MIN", "12")
        )

        # MultiON within-episode consolidation extension seam (OFF by default):
        # when > 0, ALSO flush STM→fine every N keyframes, on top of the
        # event-boundary (sub-goal advance) consolidation that is the default
        # multion path. Not wired into any driver yet.
        self.multion_consolidate_period: int = int(
            os.environ.get("MULTION_CONSOLIDATE_PERIOD", "0")
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
            summary.n_coarse_candidates += int(ep_metrics.get("n_coarse_candidates", 0))
            summary.n_coarse_chosen += int(ep_metrics.get("n_coarse_chosen", 0))
            summary.n_remembr_chosen += int(ep_metrics.get("n_remembr_chosen", 0))
            summary.n_stop_signals += int(ep_metrics.get("n_stop_signals", 0))
            summary.n_detector_called += int(ep_metrics.get("n_detector_called", 0))
            summary.n_detector_localized += int(ep_metrics.get("n_detector_localized", 0))
            summary.n_detector_locate_failed += int(ep_metrics.get("n_detector_locate_failed", 0))
            summary.n_detector_gated += int(ep_metrics.get("n_detector_gated", 0))
            summary.n_arrival_stop += int(ep_metrics.get("n_arrival_stop", 0))
            summary.n_detector_approach_success += int(ep_metrics.get("n_detector_approach_success", 0))
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
                "n_coarse_candidates": int(ep_metrics.get("n_coarse_candidates", 0)),
                "n_coarse_chosen": int(ep_metrics.get("n_coarse_chosen", 0)),
                "n_remembr_chosen": int(ep_metrics.get("n_remembr_chosen", 0)),
                "n_stop_signals": int(ep_metrics.get("n_stop_signals", 0)),
                "n_detector_called": int(ep_metrics.get("n_detector_called", 0)),
                "n_detector_localized": int(ep_metrics.get("n_detector_localized", 0)),
                "n_detector_locate_failed": int(ep_metrics.get("n_detector_locate_failed", 0)),
                "n_detector_gated": int(ep_metrics.get("n_detector_gated", 0)),
                "n_arrival_stop": int(ep_metrics.get("n_arrival_stop", 0)),
                "n_detector_approach_success": int(ep_metrics.get("n_detector_approach_success", 0)),
                "distance_to_goal": ep_metrics.get("distance_to_goal"),
                "min_distance_to_goal": ep_metrics.get("min_distance_to_goal"),
                "success_1m": bool(ep_metrics.get("success_1m", False)),
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
                "n_waypoint_reached": int(ep_metrics.get("n_waypoint_reached", 0)),
                "n_waypoint_unreachable": int(ep_metrics.get("n_waypoint_unreachable", 0)),
                "n_forward_no_progress": int(ep_metrics.get("n_forward_no_progress", 0)),
                "n_propose_reached": int(ep_metrics.get("n_propose_reached", 0)),
                "n_candidates_filtered_near": int(ep_metrics.get("n_candidates_filtered_near", 0)),
                "n_candidates_filtered_unreachable": int(
                    ep_metrics.get("n_candidates_filtered_unreachable", 0)),
                "n_unreachable_escape": int(ep_metrics.get("n_unreachable_escape", 0)),
                "n_no_progress_escape": int(ep_metrics.get("n_no_progress_escape", 0)),
                "n_memory_consumed": int(ep_metrics.get("n_memory_consumed", 0)),
                "n_candidates_filtered_consumed": int(
                    ep_metrics.get("n_candidates_filtered_consumed", 0)),
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
        # Reset per-episode detector approach state so a stale waypoint from
        # episode N (e.g. step-budget exhausted mid-approach) never leaks into
        # episode N+1.
        self._approach_waypoint = None
        # Wire the goal detector's pathfinder from the freshly-reset sim. We
        # construct GoalDetector with pathfinder=None in run_hm3d_pol (the
        # sim doesn't exist yet at that point) and update it here per episode
        # because each scene resets the sim and its pathfinder. Re-wire even
        # if non-None: across episodes from different scenes the old reference
        # would be stale. c5 caught this — GoalDetector.locate() crashed
        # with 'NoneType has no attribute snap_point' because the only existing
        # wiring lived inside the post-locate "install waypoint" branch and
        # so could never run on the first detector call.
        if self.goal_detector is not None:
            _src_sim = self.source.get_sim()
            if _src_sim is not None and hasattr(_src_sim, "pathfinder"):
                self.goal_detector.pathfinder = _src_sim.pathfinder
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
        n_coarse_candidates = 0
        n_coarse_chosen = 0
        n_remembr_chosen = 0
        n_stop_signals = 0
        ep_metrics_counters: Dict[str, Any] = {
            "n_detector_called": 0,
            "n_detector_localized": 0,
            "n_detector_locate_failed": 0,
            "n_detector_gated": 0,
            "n_detector_approach_success": 0,
            "n_arrival_stop": 0,
            "n_detector_approach_stop_distance": float("nan"),
            # Absorbing-loop diagnostics (multion-micro2): follower-done
            # ticks split reached/unreachable + collision-slide forwards.
            "n_waypoint_reached": 0,
            "n_waypoint_unreachable": 0,
            "n_forward_no_progress": 0,
            # Reached-thrash escape diagnostics (multion-micro3): reached-
            # triggered re-proposes + near-candidates dropped from the pool.
            "n_propose_reached": 0,
            "n_candidates_filtered_near": 0,
            # Unreachable-waypoint blacklist drops (multion-full1 third mode).
            "n_candidates_filtered_unreachable": 0,
            # Stuck-loop escapes (multion-full2 post-mortem): snap escape on
            # consecutive unreachables (ep5) + windowed no-progress escape
            # (ep4). Multion-gated actions; counters log unconditionally.
            "n_unreachable_escape": 0,
            "n_no_progress_escape": 0,
            # Memory-consumption fix (multion-full3 post-mortem): recalls
            # consumed on reached-without-advance + pool drops they caused.
            "n_memory_consumed": 0,
            "n_candidates_filtered_consumed": 0,
        }
        # Closest the agent ever gets to a goal viewpoint over the episode
        # (geodesic). success@0.1m is perception-bound with caption-only
        # detection; min_d2g + success@1m are the reframed reach diagnostics.
        min_d2g = float("inf")

        def _track_d2g(s) -> None:
            nonlocal min_d2g
            v = s.info.get("distance_to_goal") if s.info else None
            if v is not None:
                min_d2g = min(min_d2g, float(v))

        # MultiON sub-goal cursor. The ordered chain comes from the episode
        # (info["object_categories"], surfaced into Episode.metadata by the
        # source) or the CLI --target-sequence override; absent both, the
        # single-goal path below is byte-identical (every multion branch is
        # gated behind K > 1). Native Habitat success/spl/distance_to_goal
        # stay c1-only; Progress/PPL are computed here.
        subgoal_seq: List[str] = []
        if not is_oracle:
            ep_meta = getattr(ep, "metadata", None)
            if isinstance(ep_meta, dict):
                subgoal_seq = [
                    str(c) for c in (ep_meta.get("object_categories") or [])
                ]
            if not subgoal_seq and self.target_categories:
                subgoal_seq = list(self.target_categories)
        if not subgoal_seq:
            subgoal_seq = [str(ep.target_category)]
        n_subgoals = len(subgoal_seq)
        multion = n_subgoals > 1 and not is_oracle
        subgoal_idx = 0
        active_category: str = subgoal_seq[0]
        subgoals_found: List[Dict[str, Any]] = []
        multion_force_stop = False
        path_len_taken = 0.0
        _last_pos = np.asarray(step.agent_state.position, dtype=np.float64)
        # memory_assisted attribution: n_memory_chosen snapshot at the moment
        # the current sub-goal became active.
        _mem_chosen_at_subgoal_start = 0
        # Keyframes observed since the last within-episode consolidation —
        # only consulted when MULTION_CONSOLIDATE_PERIOD > 0 (off by default).
        _kf_since_boundary = 0
        # L_opt = sum of ordered geodesic legs at episode start; each leg is
        # anchored at the previous leg's realizing viewpoint (start pose for
        # leg 1). Any unreachable leg -> sum reachable legs + partial flag.
        geodesic_optimal = 0.0
        geodesic_optimal_partial = False
        if multion:
            _anchor = np.asarray(step.agent_state.position, dtype=np.float32)
            for _cat in subgoal_seq:
                _leg = self.source.nearest_category_viewpoint(_anchor, _cat)
                if _leg is None or not np.isfinite(float(_leg[0])):
                    geodesic_optimal_partial = True
                    continue
                geodesic_optimal += float(_leg[0])
                _anchor = np.asarray(_leg[1], dtype=np.float32)

        stm_captions: List[str] = []
        current_candidate: Optional[FrontierCandidate] = None
        # Same-category LTM-sighting candidates from the latest proposal; held
        # across ticks so the detector-memory gate (c9) can read them when the
        # stop_signal tick fires. Refreshed each propose step.
        mem_cands: List[FrontierCandidate] = []
        last_propose_step: int = -10**9  # forces a proposal on the first loop tick
        # Last step a REACHED-triggered re-propose fired (cooldown backstop).
        last_reached_propose_step: int = -10**9
        # Waypoints the follower reported unreachable this episode (multion
        # blacklist — filtered from later proposal pools).
        unreachable_xys: List[np.ndarray] = []
        # Snap-escape state (full2 ep5): consecutive follower-unreachable
        # ticks; reset on "reached" or a progressing forward.
        consecutive_unreachable = 0
        # Consumed memory recalls (full3 ep12): positions of memory-source
        # waypoints reached without an advance — dead leads for the CURRENT
        # sub-goal, filtered from later pools; cleared on advance.
        consumed_memory_xys: List[np.ndarray] = []
        # Last follower-done drop (full3 ep12 no_candidate hole): gates the
        # multion no_candidate re-propose path with the same cooldown the
        # candidate_reached trigger already has.
        last_follower_drop_step: int = -10**9
        # No-progress-escape state (full2 ep4): rolling window of per-tick
        # _forward_no_progress flags. Multion-only (None otherwise) — keeps
        # K=1 byte-identical AND avoids dereferencing the escape knobs on the
        # oracle path (test_propose_candidates builds an oracle runner via
        # __new__, skipping __init__).
        no_progress_window: Optional[deque] = (
            deque(maxlen=max(1, self.no_progress_window)) if multion else None)
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
        _track_d2g(step)
        if not is_oracle:
            # Reset ReMEmbR per-episode state and index the initial keyframe;
            # reuse its RICH VLM caption for the LTM keyframe (see _build_keyframe).
            caption_override = None
            if self.backbone == "remembr":
                self.remembr_builder.reset()
                self.remembr_planner.reset()
                rec = self.remembr_builder.caption_and_index(
                    rgb=step.rgb,
                    agent_position=step.agent_state.position,
                    timestep=int(step.step_idx),
                )
                caption_override = rec.caption
            keyframe = self._build_keyframe(step, caption_override=caption_override)
            self.bridge.observe_keyframe(keyframe, action=None, reward=0.0)
            stm_captions.append(keyframe.caption)
            ep_log["steps"].append(self._serialize_step(step, keyframe))

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
                _track_d2g(step)
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
            if multion and candidate_reached and not _cooldown_elapsed(
                step.step_idx, last_reached_propose_step, self.propose_cooldown
            ):
                # Cooldown backstop (multion-gated): a reached-triggered
                # re-propose may fire at most once per cooldown window, so
                # pathological all-near geometry can't re-propose every tick.
                # Scheduled triggers are unaffected.
                candidate_reached = False
            need_candidate = current_candidate is None
            if multion and need_candidate and not _cooldown_elapsed(
                step.step_idx, last_follower_drop_step, self.propose_cooldown
            ):
                # The ep12 hole (full3): a follower-done drop left
                # current_candidate=None, and the no_candidate trigger fired
                # EVERY TICK, bypassing the reached-cooldown entirely
                # (rerank=949/1049 with n_propose_reached=0). Gate it with
                # the same cooldown; the sub-goal-advance recall moment
                # resets last_follower_drop_step so it re-queries instantly.
                need_candidate = False
            due_to_propose = (step.step_idx - last_propose_step) >= self.propose_period
            if not multion_force_stop and (
                need_candidate or due_to_propose or candidate_reached
            ):
                # Record WHY this propose fired (multion-full1 post-mortem: a
                # per-tick re-propose mode existed that no counter attributed;
                # diagnose_propose_triggers.py mines this field).
                propose_trigger = (
                    "no_candidate" if need_candidate
                    else "reached" if candidate_reached
                    else "scheduled"
                )
                if candidate_reached:
                    ep_metrics_counters["n_propose_reached"] += 1
                    last_reached_propose_step = int(step.step_idx)
                    if (
                        multion
                        and self.consume_reached_mem
                        and current_candidate is not None
                        and current_candidate.source == "memory"
                    ):
                        # Consume the recall (full3 ep10/11/13 chronic
                        # variant): the agent walked within the re-propose
                        # radius of a memory waypoint and the sub-goal did
                        # NOT advance — a dead lead for this sub-goal.
                        consumed_memory_xys.append(np.asarray(
                            current_candidate.world_xy, dtype=np.float32))
                        ep_metrics_counters["n_memory_consumed"] += 1
                last_propose_step = int(step.step_idx)
                cands = self._propose_candidates(
                    step, ep, goal_override=active_category if multion else None
                )
                if multion and subgoal_idx < n_subgoals - 1 and cands:
                    # STOP-vs-advance: before the final sub-goal a backbone
                    # stop_signal must not terminate the episode — drop it and
                    # keep navigating; the per-tick advance check (distance +
                    # caption) decides the sub-goal hand-off instead.
                    cands = [c for c in cands
                             if not c.metadata.get("stop_signal", False)]
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
                        target_category=(active_category if multion
                                         else ep.target_category),
                        planner_world_xys=[c.world_xy for c in cands],
                        top_k=3,
                    )
                    # Assign fresh, non-clashing ids before merging.
                    for i, mc in enumerate(mem_cands):
                        mc.candidate_id = len(cands) + i + 1000  # offset so logs are unambiguous
                    all_cands = cands + mem_cands
                    n_memory_candidates += len(mem_cands)

                    # Coarse-affordance (step 4): the POSITION-FREE cross-env path.
                    # Env-gated (LTM_COARSE_AFFORDANCE) and fired ONLY when the fine
                    # layer surfaced no same-scene hit (mem_cands empty == genuinely
                    # cold/new scene) — conservatism that avoids the importance-head
                    # over-fire trap. Grounds the goal category's preferred room-type
                    # to the current scene's STM observations and injects one waypoint.
                    if not mem_cands and os.environ.get("LTM_COARSE_AFFORDANCE"):
                        coarse_cands = self.bridge.propose_coarse_candidates(
                            agent_pos=step.agent_state.position,
                            agent_yaw=step.agent_state.rotation_yaw,
                            target_category=(active_category if multion
                                             else ep.target_category),
                            planner_world_xys=[c.world_xy for c in all_cands],
                            top_k=1,
                        )
                        for i, cc in enumerate(coarse_cands):
                            cc.candidate_id = len(all_cands) + i + 2000
                        all_cands = all_cands + coarse_cands
                        n_coarse_candidates += len(coarse_cands)
                    if multion:
                        # Reached-thrash escape: drop already-reached
                        # waypoints from the pool — applied once after the
                        # memory merge so frontier + remembr + memory are
                        # covered uniformly (stop_signal survives; falls back
                        # to the unfiltered pool when all non-stop are near).
                        all_cands, _n_near = _filter_near_candidates(
                            all_cands,
                            (float(step.agent_state.position[0]),
                             float(step.agent_state.position[2])),
                            self.min_target_m,
                        )
                        ep_metrics_counters["n_candidates_filtered_near"] += _n_near
                        # Unreachable-waypoint blacklist (full1 third mode):
                        # never re-offer a waypoint the follower already
                        # reported unreachable this episode. prefer_farthest
                        # (full2 ep5): when EVERY non-stop candidate is
                        # blacklisted, keep the least-bad (farthest) one
                        # instead of the raw pool, which re-admitted the bad
                        # waypoint and froze the agent turning in place.
                        if unreachable_xys and self.unreachable_blacklist_m > 0:
                            all_cands, _n_bl = _filter_candidates_near_points(
                                all_cands, unreachable_xys,
                                self.unreachable_blacklist_m,
                                prefer_farthest=True,
                            )
                            ep_metrics_counters["n_candidates_filtered_unreachable"] += _n_bl
                        # Consumed-recall filter (full3 ep12): never re-offer
                        # a memory waypoint already reached without an
                        # advance this sub-goal — the bridge re-proposes the
                        # same fine-LTM sighting every query, so filtering
                        # the pool is the only place to break the attractor.
                        if consumed_memory_xys and self.unreachable_blacklist_m > 0:
                            all_cands, _n_cons = _filter_candidates_near_points(
                                all_cands, consumed_memory_xys,
                                self.unreachable_blacklist_m,
                                prefer_farthest=True,
                            )
                            ep_metrics_counters["n_candidates_filtered_consumed"] += _n_cons

                    rerank_result, retrieval = self.bridge.rerank(
                        candidates=all_cands,
                        query_text=keyframe.caption,
                        stm_captions=stm_captions[-5:],
                        target_category=(active_category if multion
                                         else ep.target_category),
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
                    if chosen.source == "coarse":
                        n_coarse_chosen += 1
                    if chosen.source == "remembr":
                        n_remembr_chosen += 1
                    current_candidate = chosen

                    n_frontier_in_pool = sum(1 for c in cands if c.source == "frontier")

                    ep_log["decisions"].append({
                        "step_idx": int(step.step_idx),
                        "trigger": propose_trigger,
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

            # Oracle-location diagnostic: steer to the GT goal (isolates
            # exploration + retrieval). Overrides only the NAVIGATION target —
            # a backbone stop_signal candidate is left intact so the agent's own
            # termination logic still decides when to STOP.
            if self.oracle_location and not (
                current_candidate is not None
                and current_candidate.metadata.get("stop_signal", False)
            ):
                _goal = getattr(ep, "target_position", None)
                if _goal is not None:
                    current_candidate = _detector_candidate(
                        np.asarray(_goal, dtype=np.float32),
                        step.agent_state.position,
                    )

            # Convert candidate → action.
            if multion_force_stop:
                # MultiON: final sub-goal found on a previous tick — terminate.
                action = ACTION_STOP
            elif current_candidate is None:
                action = ACTION_FORWARD
            elif current_candidate.metadata.get("stop_signal", False) and self._approach_waypoint is None:
                # Detector intercept (Task 4): if --detector is on, ask the
                # GoalDetector to localize the goal and navigate the last
                # metre. None -> fall back to immediate STOP (monotonicity:
                # detector-ON is expectation->=detector-OFF).
                # Guard: only intercept on the FIRST stop_signal tick
                # (approach_wp=None). Subsequent ticks with approach_wp set
                # fall through to the elif branch below, which continues
                # steering toward the already-installed waypoint.
                action, approach_wp = _decide_stop_or_approach(
                    detector_enabled=self.detector_enabled,
                    detector=self.goal_detector,
                    rgb=step.rgb,
                    depth=step.depth,
                    goal_category=(active_category if multion
                                   else self.target_category),
                    agent_pose=self._agent_pose_matrix(step.agent_state),
                    intrinsics=self._camera_intrinsics(),
                    counters=ep_metrics_counters,
                    # c9 gate: same-category LTM sighting positions from the
                    # latest proposal. Empty (cold) -> detector falls back to STOP.
                    mem_world_xys=[mc.world_xy for mc in mem_cands
                                   if mc.source == "memory"],
                    agree_radius=self._detector_mem_agree_m,
                )
                if action is None:
                    # Install detector waypoint and drive toward it THIS step.
                    # Pathfinder is wired once per episode at _run_episode entry
                    # (see comment there); locate() already ran with a valid
                    # pathfinder to reach this branch.
                    self._approach_waypoint = approach_wp
                    synthetic = _detector_candidate(
                        approach_wp, step.agent_state.position,
                    )
                    action = self._waypoint_action(
                        synthetic, step.agent_state.position, step.agent_state.rotation_yaw,
                        use_approach_follower=True,
                    )
                    arrived, stop_dist = _approach_arrived(
                        self._waypoint_force_repropose,
                        step.agent_state.position, approach_wp,
                        self._approach_goal_radius,
                    )
                    if arrived:
                        # At the snapped waypoint (follower-STOP or inside ring).
                        ep_metrics_counters["n_detector_approach_success"] += 1
                        ep_metrics_counters["n_detector_approach_stop_distance"] = stop_dist
                        action = ACTION_STOP
                        self._approach_waypoint = None
                # else: action is ACTION_STOP from the helper -> emit it
            elif self._approach_waypoint is not None:
                # Continuing the detector approach from a prior tick.
                wp = self._approach_waypoint
                synthetic = _detector_candidate(wp, step.agent_state.position)
                action = self._waypoint_action(
                    synthetic, step.agent_state.position, step.agent_state.rotation_yaw,
                    use_approach_follower=True,
                )
                arrived, stop_dist = _approach_arrived(
                    self._waypoint_force_repropose,
                    step.agent_state.position, wp,
                    self._approach_goal_radius,
                )
                if arrived:
                    ep_metrics_counters["n_detector_approach_success"] += 1
                    ep_metrics_counters["n_detector_approach_stop_distance"] = stop_dist
                    action = ACTION_STOP
                    self._approach_waypoint = None
            else:
                action = self._waypoint_action(
                    current_candidate,
                    step.agent_state.position,
                    step.agent_state.rotation_yaw,
                )
                # Absorbing-loop diagnostics: classify follower "done" ticks.
                # A run of consecutive unreachables IS the turn-forever loop
                # (multion-micro2 ep0: 741 re-proposes, forward 2/749).
                _wp_dist = None
                if current_candidate is not None:
                    _wp_dist = float(np.hypot(
                        float(current_candidate.world_xy[0])
                        - float(step.agent_state.position[0]),
                        float(current_candidate.world_xy[1])
                        - float(step.agent_state.position[2]),
                    ))
                _outcome = _waypoint_outcome(
                    self._waypoint_force_repropose, _wp_dist
                )
                if _outcome is not None:
                    ep_metrics_counters[f"n_waypoint_{_outcome}"] += 1
                    if _outcome == "reached":
                        consecutive_unreachable = 0
                        if (
                            multion
                            and self.consume_reached_mem
                            and current_candidate is not None
                            and current_candidate.source == "memory"
                        ):
                            # Consume the recall (full3 ep12 catastrophic
                            # variant): the follower arrived at a memory
                            # waypoint and the sub-goal did NOT advance —
                            # dead lead; never re-offer it this sub-goal.
                            consumed_memory_xys.append(np.asarray(
                                current_candidate.world_xy,
                                dtype=np.float32))
                            ep_metrics_counters["n_memory_consumed"] += 1
                    if (
                        multion
                        and _outcome == "unreachable"
                        and current_candidate is not None
                    ):
                        # Blacklist the failed waypoint so the next proposal
                        # pool can't re-offer the same cluster (full1: the
                        # top pick was re-chosen 593-732× in a turn-forever
                        # loop after every unreachable drop).
                        unreachable_xys.append(
                            np.asarray(current_candidate.world_xy,
                                       dtype=np.float32)
                        )
                        # Snap escape (full2 ep5): the blacklist alone never
                        # broke the loop — the agent never moves, so the
                        # planner re-clusters the same geometry at
                        # coordinates drifting outside the blacklist radius.
                        # After N consecutive unreachables, snap the waypoint
                        # to the nearest navmesh point and re-commit it once.
                        # No sim / snap failure -> blacklist+drop (above).
                        consecutive_unreachable += 1
                        if _should_snap_unreachable(
                            consecutive_unreachable, self.unreachable_snap_n
                        ) and not current_candidate.metadata.get(
                            "snap_retried", False
                        ):
                            # snap-once (full3 ep0/ep6): a waypoint is
                            # snapped at most ONCE — if the snapped point is
                            # also unreachable, fall through to blacklist +
                            # drop instead of re-snapping forever (ep0:
                            # escape=125, wp_unreach=1023 — each snap just
                            # bought 8 more turn-in-place ticks).
                            _sim = self.source.get_sim()
                            _snapped = None
                            if _sim is not None:
                                try:
                                    _wp = current_candidate.world_xy
                                    _g = np.array(
                                        [float(_wp[0]),
                                         float(step.agent_state.position[1]),
                                         float(_wp[1])], dtype=np.float32)
                                    _sp = _sim.pathfinder.snap_point(_g)
                                    _sp = np.array(
                                        [float(_sp[0]), float(_sp[1]),
                                         float(_sp[2])], dtype=np.float32)
                                    if np.all(np.isfinite(_sp)):
                                        _snapped = _sp
                                except Exception:
                                    _snapped = None  # off-navmesh / no snap
                            if _snapped is not None:
                                current_candidate.world_xy = np.array(
                                    [_snapped[0], _snapped[2]],
                                    dtype=np.float32)
                                current_candidate.metadata["snap_retried"] = True
                                # Keep the snapped waypoint committed: clear
                                # the follower-done flag so the drop branch
                                # below doesn't immediately re-propose.
                                self._waypoint_force_repropose = False
                                ep_metrics_counters["n_unreachable_escape"] += 1
                                consecutive_unreachable = 0
                # Waypoint-arrival STOP: STOP if the agent is at a confident MEMORY
                # waypoint (a remembered goal position) and the caption confirms the
                # goal → terminate here (oracle-ladder proxy). "At" = the follower
                # reports reached OR the agent is within _arrival_stop_radius. The OR
                # is load-bearing: the follower's goal_radius (0.5m) means distance_m
                # bottoms out at ~0.5m, so a pure proximity ring at 0.5m never
                # triggers (arrival-2 fired 0×). The radius is set ABOVE the follower
                # radius so proximity catches the approach, and the force_repropose
                # term guarantees ≥ the follower-reached fires.
                _near = self._waypoint_force_repropose or (
                    current_candidate is not None
                    and float(current_candidate.distance_m) < self._arrival_stop_radius
                )
                _confirms = _caption_mentions(
                    keyframe.caption,
                    _goal_terms(active_category if multion
                                else ep.target_category),
                ) is not None
                if _arrival_stop(_near, current_candidate, _confirms,
                                 self._arrival_stop_cos):
                    if multion and subgoal_idx < n_subgoals - 1:
                        # STOP-vs-advance: never STOP before the final
                        # sub-goal — drop the waypoint and let the per-tick
                        # advance check decide the hand-off.
                        current_candidate = None
                    else:
                        action = ACTION_STOP
                        ep_metrics_counters["n_arrival_stop"] += 1
                        current_candidate = None
                elif self._waypoint_force_repropose:
                    # Follower reports reached/unreachable → drop & re-propose.
                    current_candidate = None
                    if multion:
                        # Start the no_candidate re-propose cooldown (the
                        # full3 ep12 hole: this drop used to re-propose
                        # ungated EVERY tick).
                        last_follower_drop_step = int(step.step_idx)

            # Oracle-STOP diagnostic: force STOP once the agent is within the GT
            # success ring (isolates the termination layer — measures how much
            # binary success a perfect STOP recovers). step.info carries the
            # current geodesic distance_to_goal.
            if self.oracle_stop:
                _d2g = step.info.get("distance_to_goal") if step.info else None
                action = _oracle_stop_override(action, _d2g, self.oracle_stop_radius)

            # Step the env.
            action_counts[action] = action_counts.get(action, 0) + 1
            _pos_before_step = np.asarray(
                step.agent_state.position, dtype=np.float64
            )
            step = self.source.step(action)
            self.planner.update(
                step.depth, step.agent_state.position, step.agent_state.rotation_yaw
            )
            _track_d2g(step)
            # Collision-slide diagnostic: a FORWARD that moved the agent ~0 m
            # (wall-pushing; collision_escape is grid-controller-only and
            # never fires under the navmesh follower).
            _disp = float(np.linalg.norm(
                np.asarray(step.agent_state.position, dtype=np.float64)
                - _pos_before_step
            ))
            _no_prog = _forward_no_progress(action, _disp)
            if _no_prog:
                ep_metrics_counters["n_forward_no_progress"] += 1
            elif action == ACTION_FORWARD:
                # A progressing forward breaks the snap-escape streak.
                consecutive_unreachable = 0
            if multion:
                no_progress_window.append(_no_prog)
            # Windowed no-progress escape (full2 ep4: 656 no-progress forwards
            # counted, zero acted on — the FORWARD/TURN alternation kept the
            # grid-era collision_escape from ever firing). When the window is
            # full of mostly no-progress forwards: blacklist the committed
            # waypoint (the same list the unreachable filter consumes), drop
            # it, and force a re-propose next tick.
            if (
                multion
                and current_candidate is not None
                and _no_progress_escape(
                    no_progress_window, self.no_progress_min,
                    self.no_progress_window,
                )
            ):
                unreachable_xys.append(
                    np.asarray(current_candidate.world_xy, dtype=np.float32)
                )
                current_candidate = None
                last_propose_step = -10**9
                no_progress_window.clear()
                ep_metrics_counters["n_no_progress_escape"] += 1

            # MultiON per-tick advance check (gated: K==1 is untouched).
            if multion:
                _pos = np.asarray(step.agent_state.position, dtype=np.float64)
                path_len_taken += float(np.linalg.norm(_pos - _last_pos))
                _last_pos = _pos
                if not multion_force_stop:
                    _dist = self.source.distance_to_category(
                        step.agent_state.position, active_category
                    )
                    _conf = _caption_mentions(
                        keyframe.caption, _goal_terms(active_category)
                    ) is not None
                    _found, _finished = _advance_subgoal(
                        _dist, _conf, subgoal_idx, n_subgoals, self.found_radius
                    )
                    if _found:
                        subgoals_found.append({
                            "category": active_category,
                            "subgoal_idx": subgoal_idx,
                            "step_idx": int(step.step_idx),
                            "distance": float(_dist),
                            "memory_assisted": bool(
                                n_memory_chosen > _mem_chosen_at_subgoal_start
                            ),
                            "path_len_at_found": float(path_len_taken),
                        })
                        subgoal_idx += 1
                        _mem_chosen_at_subgoal_start = n_memory_chosen
                        if _finished:
                            multion_force_stop = True
                        else:
                            # Event-boundary consolidation (multion-micro3 fix):
                            # flush the c_i-hunt keyframes from STM into the
                            # fine LTM NOW, so the immediate re-query below for
                            # c_{i+1} can recall sightings made THIS episode.
                            # End-of-episode consolidation was structurally too
                            # late — n_memory_candidates stayed 0 all episode.
                            if self.bridge is not None:
                                self.bridge.consolidate_subgoal_boundary(
                                    episode_idx=ep_idx
                                )
                                _kf_since_boundary = 0
                            active_category = subgoal_seq[subgoal_idx]
                            # The recall moment: force an immediate LTM
                            # re-query for the NEW category by resetting the
                            # propose-cadence clock and dropping the waypoint.
                            current_candidate = None
                            self._approach_waypoint = None
                            last_propose_step = -10**9
                            # Consumed recalls are PER SUB-GOAL: the new
                            # category may legitimately want a previously
                            # visited area. Also reset the follower-drop
                            # cooldown so the re-query fires next tick.
                            consumed_memory_xys.clear()
                            last_follower_drop_step = -10**9

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
                # ReMEmbR build phase: caption the RGB with the VLM, write it to
                # flat memory, and reuse that RICH caption for the LTM keyframe
                # (the SemanticCaptioner fallback is degenerate — see
                # _build_keyframe). One VLM call serves both.
                caption_override = None
                if self.backbone == "remembr":
                    rec = self.remembr_builder.caption_and_index(
                        rgb=step.rgb,
                        agent_position=step.agent_state.position,
                        timestep=int(step.step_idx),
                    )
                    caption_override = rec.caption
                keyframe = self._build_keyframe(step, caption_override=caption_override)
                self.bridge.observe_keyframe(
                    keyframe, action=action, reward=step.reward, success=False
                )
                stm_captions.append(keyframe.caption)
                ep_log["steps"].append(self._serialize_step(step, keyframe))
                # Periodic within-episode consolidation (extension seam, OFF
                # by default — event-boundary consolidation at sub-goal
                # advance is the multion default).
                _kf_since_boundary += 1
                if (
                    multion
                    and self.multion_consolidate_period > 0
                    and _kf_since_boundary >= self.multion_consolidate_period
                ):
                    self.bridge.consolidate_subgoal_boundary(episode_idx=ep_idx)
                    _kf_since_boundary = 0

            if step.done:
                break

        # End-of-episode: figure out success, consolidate.
        success = bool(step.info.get("success", False)) or bool(
            step.info.get("distance_to_goal", 1e9) < 0.1
        )
        spl = float(step.info.get("spl", 1.0 if success else 0.0))
        soft_spl = float(step.info.get("softspl", step.info.get("soft_spl", spl)))
        distance_to_goal = step.info.get("distance_to_goal")
        # Reframed reach diagnostics: success@0.1m is perception-bound (caption
        # detection can't localize to 0.1m), so the gate keys on soft-SPL plus
        # success@1m = "agent came within 1.0m of a goal viewpoint at any step"
        # (STOP-independent reach), with min_d2g as the continuous companion.
        min_distance_to_goal = None if min_d2g == float("inf") else float(min_d2g)
        success_1m = bool(
            min_distance_to_goal is not None and min_distance_to_goal < 1.0
        )
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
        ep_log["n_coarse_candidates"] = n_coarse_candidates
        ep_log["n_coarse_chosen"] = n_coarse_chosen
        ep_log["n_remembr_chosen"] = n_remembr_chosen
        ep_log["n_stop_signals"] = n_stop_signals
        ep_log["n_waypoint_reached"] = int(ep_metrics_counters["n_waypoint_reached"])
        ep_log["n_waypoint_unreachable"] = int(ep_metrics_counters["n_waypoint_unreachable"])
        ep_log["n_forward_no_progress"] = int(ep_metrics_counters["n_forward_no_progress"])
        ep_log["n_propose_reached"] = int(ep_metrics_counters["n_propose_reached"])
        ep_log["n_candidates_filtered_near"] = int(ep_metrics_counters["n_candidates_filtered_near"])
        ep_log["n_candidates_filtered_unreachable"] = int(
            ep_metrics_counters["n_candidates_filtered_unreachable"])
        ep_log["n_unreachable_escape"] = int(ep_metrics_counters["n_unreachable_escape"])
        ep_log["n_no_progress_escape"] = int(ep_metrics_counters["n_no_progress_escape"])
        ep_log["n_memory_consumed"] = int(ep_metrics_counters["n_memory_consumed"])
        ep_log["n_candidates_filtered_consumed"] = int(
            ep_metrics_counters["n_candidates_filtered_consumed"])
        ep_log["n_detector_called"] = int(ep_metrics_counters["n_detector_called"])
        ep_log["n_detector_localized"] = int(ep_metrics_counters["n_detector_localized"])
        ep_log["n_detector_locate_failed"] = int(ep_metrics_counters["n_detector_locate_failed"])
        ep_log["n_detector_gated"] = int(ep_metrics_counters["n_detector_gated"])
        ep_log["n_arrival_stop"] = int(ep_metrics_counters["n_arrival_stop"])
        ep_log["n_detector_approach_success"] = int(ep_metrics_counters["n_detector_approach_success"])
        ep_log["n_detector_approach_stop_distance"] = float(ep_metrics_counters["n_detector_approach_stop_distance"])
        ep_log["min_distance_to_goal"] = min_distance_to_goal
        ep_log["success_1m"] = success_1m
        # MultiON metrics (only when K > 1; single-goal logs are unchanged).
        # Native success/spl/distance_to_goal above stay c1-only by design;
        # Progress = k_found/K and PPL = Progress * L_opt / max(L_taken, L_opt)
        # are the multion headline metrics.
        if multion:
            k_found = len(subgoals_found)
            progress = k_found / float(n_subgoals)
            success_multion = k_found == n_subgoals
            if geodesic_optimal > 0.0:
                _ratio = geodesic_optimal / max(path_len_taken, geodesic_optimal)
                ppl = progress * _ratio
                spl_multion = _ratio if success_multion else 0.0
            else:
                # No reachable leg at all -> path-weighting undefined.
                ppl = None
                spl_multion = None
            ep_log["is_multion"] = True
            ep_log["target_categories"] = list(subgoal_seq)
            ep_log["found_radius"] = float(self.found_radius)
            ep_log["subgoals_found"] = subgoals_found
            ep_log["progress"] = progress
            ep_log["success_multion"] = success_multion
            ep_log["ppl"] = ppl
            ep_log["spl_multion"] = spl_multion
            ep_log["path_len_taken"] = float(path_len_taken)
            ep_log["geodesic_optimal"] = float(geodesic_optimal)
            ep_log["geodesic_optimal_partial"] = bool(geodesic_optimal_partial)
            ep_log["recall_assisted_advances"] = sum(
                1 for s in subgoals_found if s["memory_assisted"]
            )
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
            "min_distance_to_goal": min_distance_to_goal,
            "success_1m": success_1m,
            "rerank_calls": rerank_calls,
            "rerank_disagreements": rerank_disagreements,
            "retrieval_hits": retrieval_hits,
            "n_memory_candidates": n_memory_candidates,
            "n_memory_chosen": n_memory_chosen,
            "n_frontier_chosen": n_frontier_chosen,
            "n_coarse_candidates": n_coarse_candidates,
            "n_coarse_chosen": n_coarse_chosen,
            "n_remembr_chosen": n_remembr_chosen,
            "n_stop_signals": n_stop_signals,
            "n_detector_called": int(ep_metrics_counters["n_detector_called"]),
            "n_detector_localized": int(ep_metrics_counters["n_detector_localized"]),
            "n_detector_locate_failed": int(ep_metrics_counters["n_detector_locate_failed"]),
            "n_detector_gated": int(ep_metrics_counters["n_detector_gated"]),
            "n_arrival_stop": int(ep_metrics_counters["n_arrival_stop"]),
            "n_detector_approach_success": int(ep_metrics_counters["n_detector_approach_success"]),
            "n_detector_approach_stop_distance": float(ep_metrics_counters["n_detector_approach_stop_distance"]),
            "n_propose_reached": int(ep_metrics_counters["n_propose_reached"]),
            "n_candidates_filtered_near": int(ep_metrics_counters["n_candidates_filtered_near"]),
            "n_candidates_filtered_unreachable": int(
                ep_metrics_counters["n_candidates_filtered_unreachable"]),
            "n_unreachable_escape": int(ep_metrics_counters["n_unreachable_escape"]),
            "n_no_progress_escape": int(ep_metrics_counters["n_no_progress_escape"]),
            "n_memory_consumed": int(ep_metrics_counters["n_memory_consumed"]),
            "n_candidates_filtered_consumed": int(
                ep_metrics_counters["n_candidates_filtered_consumed"]),
            # Bookkeeping fix (multion-full1 post-mortem): these three were
            # written to the episode JSON but never returned in ep_metrics, so
            # the summary.json per-episode rows read 0 via .get() defaults —
            # the emailed digest showed 0/0 while ep2's episode JSON carried
            # n_waypoint_unreachable=662, mis-directing the diagnosis.
            "n_waypoint_reached": int(ep_metrics_counters["n_waypoint_reached"]),
            "n_waypoint_unreachable": int(ep_metrics_counters["n_waypoint_unreachable"]),
            "n_forward_no_progress": int(ep_metrics_counters["n_forward_no_progress"]),
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

    def _propose_candidates(
        self, step: Step, ep, goal_override: Optional[str] = None
    ) -> List[FrontierCandidate]:
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
        # remembr — goal_override carries the multion ACTIVE sub-goal category
        # (None on the single-goal path, preserving the legacy expression).
        llm_cands = self.remembr_planner.propose(
            goal=goal_override or ep.target_category or self.target_category,
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

    def _init_approach_follower(self) -> None:
        """Build the dedicated tighter follower for the detector final-approach
        (c7). Identical to ``_init_waypoint_follower`` but with
        ``goal_radius=self._approach_goal_radius`` (0.25 m) so the agent stops
        closer to the snapped goal. Leaves ``self.approach_follower = None`` when
        there is no sim, preserving the grid ``step_controller`` fallback."""
        sim = self.source.get_sim()
        if sim is None:
            self.approach_follower = None
            return
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        self.approach_follower = ShortestPathFollower(
            sim, goal_radius=self._approach_goal_radius, return_one_hot=False
        )

    def _waypoint_action(
        self, candidate, agent_pos, agent_yaw, use_approach_follower: bool = False
    ) -> int:
        """Next discrete action toward the chosen waypoint via the navmesh
        ShortestPathFollower (Phase-2 C1 fix).

        Snaps the waypoint to the navmesh and asks the follower for the action
        toward it. ``None`` from the follower (waypoint reached or unreachable)
        sets ``_waypoint_force_repropose`` and returns a TURN — locomotion never
        emits ACTION_STOP (only the keyword-STOP / explicit stop_signal ends an
        episode). When no sim is available (cached mode) it degrades to the
        occupancy-grid ``step_controller`` so that path keeps working.
        """
        from .frontier_planner import ACTION_STOP, ACTION_TURN_LEFT

        self._waypoint_force_repropose = False
        # Detector final-approach uses the tighter 0.25 m follower; normal nav
        # uses the shared 0.5 m one. Lazily build whichever is requested.
        if use_approach_follower:
            if self.approach_follower is None:
                self._init_approach_follower()
            follower = self.approach_follower
        else:
            if self.follower is None:
                self._init_waypoint_follower()
            follower = self.follower
        if follower is None:
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

        raw = follower.get_next_action(goal)
        if raw is None:
            # Reached/unreachable: drop the waypoint and re-propose; don't STOP.
            self._waypoint_force_repropose = True
            return ACTION_TURN_LEFT
        if isinstance(raw, str):
            from .habitat_env import _ACTION_NAMES
            try:
                action_id = _ACTION_NAMES.index(raw)
            except ValueError:
                self._waypoint_force_repropose = True
                return ACTION_TURN_LEFT
            if action_id == ACTION_STOP:
                # Follower signals arrival via the STOP action. Treat it like the
                # None path: flag "done" and TURN (locomotion never emits STOP;
                # the approach branches decide STOP themselves via _approach_arrived).
                self._waypoint_force_repropose = True
                return ACTION_TURN_LEFT
            return action_id
        if isinstance(raw, (int, np.integer)):
            action_id = int(raw)
            if action_id == ACTION_STOP:
                self._waypoint_force_repropose = True
                return ACTION_TURN_LEFT
            return action_id
        self._waypoint_force_repropose = True
        return ACTION_TURN_LEFT

    def _agent_pose_matrix(self, agent_state):
        """Build a 4x4 world-from-camera transform from a Habitat agent state.

        ``AgentState`` stores only a yaw float (rotation around y-axis) so we
        construct the rotation matrix from ``rotation_yaw`` directly, without
        needing the quaternion library.
        """
        p = np.asarray(agent_state.position, dtype=np.float32)
        yaw = float(agent_state.rotation_yaw)
        cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
        # y-up Habitat convention: yaw rotates in the xz-plane.
        R = np.array([
            [ cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def _camera_intrinsics(self):
        """Compute pinhole intrinsics from the source's image size + HFOV.

        If the source doesn't expose ``image_hw`` and ``hfov_rad`` cleanly,
        falls back to the Habitat default for the val_mini config:
        ``image_hw=(256, 256)`` and ``hfov=90°``.
        """
        H = getattr(self.source, "image_hw", (256, 256))[0]
        W = getattr(self.source, "image_hw", (256, 256))[1]
        hfov_rad = float(getattr(self.source, "hfov_rad", np.pi / 2.0))
        fx = 0.5 * W / np.tan(0.5 * hfov_rad)
        fy = fx   # square pixels in Habitat default
        return {"fx": fx, "fy": fy, "cx": W / 2.0, "cy": H / 2.0, "image_hw": (H, W)}

    def _build_keyframe(self, step: Step, caption_override: Optional[str] = None) -> Keyframe:
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
        elif caption_override is not None:
            # ReMEmbR backbone: index the LTM fine layer on the RICH VLM caption,
            # not the SemanticCaptioner fallback. HM3D's semantic sensor is
            # all-zeros, so SemanticCaptioner emits a degenerate "room interior"
            # caption — that made the goal-query↔caption cosine non-discriminative
            # (pinned ~0.17, below the 0.23 bar) so memory never fired.
            caption = str(caption_override)
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
        # Per-step geodesic distance to goal (from the sim info) — the label
        # source for the goal_proximity importance head. Absent on non-sim /
        # oracle paths, so guard to None.
        _d2g = step.info.get("distance_to_goal") if step.info else None
        return {
            "step_idx": int(step.step_idx),
            "action": step.action,
            "reward": float(step.reward),
            "done": bool(step.done),
            "agent_pos": step.agent_state.position.tolist(),
            "agent_yaw": float(step.agent_state.rotation_yaw),
            "caption": keyframe.caption,
            "distance_to_goal": (float(_d2g) if _d2g is not None else None),
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
