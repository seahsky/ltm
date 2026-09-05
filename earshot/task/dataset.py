"""The episode builder: where the one anomaly source goes, and what it is called.

Task spec §2.1-§2.2 and ADR-0010. One positioned source per episode, within ``|Δy| < 1.0
m`` of **both the primary goal and the episode start**, and far enough from the primary
goal in ``xz`` that investigating it is a real detour rather than a second route to it.

**The floor rule is checked here and nowhere else.** ADR-0010 moved it from invariant to
builder policy: with no grid and no ``nearest`` there is nothing to snap silently, so
there is no runtime guard and none is needed. What survives is the *controller* argument
— a greedy energy climb over ``move_forward`` / ``turn_left`` / ``turn_right`` cannot
fund a stair-climb.

**Measured, not predicted.** ADR-0010 guessed the failure would look like walking into a
wall while the energy rises through the ceiling. Ticket 26's first full box episode shows
otherwise, and the real shape is worse to diagnose: the agent walked **cleanly** — 62
forwards, 0 collisions, 15.5 m — for its whole 120-step budget on the floor above the
source, while the measured RMS *fell* 0.0407 -> 0.0121 as it chased sound leaking through
a stairwell, and ``source_is_visible`` stayed false at all 153 steps. Nothing looked
broken. That is why the rule now covers the start as well as the goal.

**Audibility is not screened (§2.5).** Nothing here renders, computes an RIR, or asks
how loud the source will be from anywhere — pre-screening would reintroduce offline
rendering by the back door, and the attrition it would hide belongs in the funnel's
stage 3 instead. ``tests/mac/test_task_dataset.py`` holds that structurally: this module
may not import ``audio``.

Carried from ``make_anomaly_response_smoke.pick_anomaly_source``, with its preference
order intact — a **different category** first, so ``anomaly_object`` differs from the
find-target and the regime is genuinely decoupled; then a different instance of the
primary category; and among qualifiers the **nearest** that still clears the bar, because
proximity correlates with being on the same navmesh component and the farthest-first pick
repeatedly landed on disconnected islands.

**One strengthening, disclosed.** The old builder measured the separation against the
*one* goal view point it had chosen as the cold start. These episodes come from the
published dataset and the agent succeeds at **any** instance of the category
(``Episode.view_points()`` flattens all of them), so separation from one instance is not
separation from the goal: a source 3 m from instance A can be 0.5 m from instance B, and
the detour is degenerate again. The bar is therefore against every primary view point.
The cost is that a scene can fail to place a source, which is reported per episode rather
than hidden.

Pure and stdlib-only, like ``episodes.py``: no numpy, no simulator, no I/O. The navmesh
never sees a candidate here — the source is a real goal view point, which is where its
navigability comes from, and it is the same thing that gives the source a **named
object** for the detector to visually confirm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from earshot.task.episodes import Episode, EpisodeDataset, ObjectGoal
from earshot.types import Xyz

__all__ = [
    "PlacementError",
    "EmptyDatasetError",
    "SourcePlacement",
    "AnomalyEpisode",
    "DatasetBuild",
    "goal_table",
    "primary_anchor",
    "derive_t_anom",
    "place_anomaly_source",
    "build_anomaly_episodes",
    "MIN_SOURCE_START_SEP_M",
]

# provenance: source — `sim.world`'s action spec (`step_size_m=0.25`). One `move_forward`
# advances the agent at most this far, and with `allow_sliding=False` a collided one
# advances less. It bounds travel-per-step from **above**, which is the direction that
# makes a step count derived from a distance a genuine lower bound.
FORWARD_STEP_M = 0.25

# provenance: source — `agent.config.DetectorConfig.oracle_radius_m`. The find ends when
# the agent is within this of a primary goal view point, so this much of the route is
# never walked and comes off the distance before it is turned into steps.
ARRIVAL_RADIUS_M = 1.0

# Both are duplicated rather than imported: this module may not reach into `sim` (the
# navmesh never sees a candidate here) and `sim/world.py` imports torch at module scope,
# so a Mac cannot load it at all. `test_task_dataset.py` reads both definitions out of
# their own source with `ast` and fails if these drift.

# provenance: fake — how far into the find the agent gets before the anomaly starts. Any
# value below 1 makes the derived `t_anom` strictly earlier than the earliest step the
# find can end on (`derive_t_anom` carries the argument); a half puts the interrupt near
# the middle of the search rather than at either end of it.
T_ANOM_FRACTION = 0.5

# provenance: fake — §3.1's first invariant is checked on each pre-onset step, so a
# `t_anom` of 0 leaves it unexercised and `assert_provenance` says so rather than
# passing. Three readings is the smallest number that is not one.
T_ANOM_FLOOR_STEPS = 3

# provenance: MEASURED — `detour-1`, 2026-08-06, and the one constant on this map that is
# not a guess. That run's `d_min` (closest approach during the detour) came out bimodal
# with no overlap: the eight detours that reached the source ended at 0.31-0.78 m, the
# twelve that did not plateaued at 2.06-9.26 m. 2.0 m sits in the empty gap between them.
#
# So a source closer than this to the agent's start is one the agent is *already inside
# the outcome of*: episode 18 of that run placed one 0.75 m away and the loop closed in
# two steps, INVESTIGATE at 5 and RESUME at 7, counting as one of the eight successes.
# Below the gap there is no detour to measure, which makes the episode a null dressed as
# a pass — the failure mode §2.5 and ADR-0014 are both about.
#
# It costs yield, and the cost is not yet known: every yield measured before this rule
# existed is an OVERESTIMATE and has to be re-measured, not adjusted.
MIN_SOURCE_START_SEP_M = 2.0


class PlacementError(ValueError):
    """No object in this scene can carry the anomaly source for this episode.

    A property of the scene and the rules, not a program error: a single-object scene,
    or one whose only other objects are upstairs, cannot express a decoupled anomaly
    response. ``build_anomaly_episodes`` catches it, skips the episode and reports it, so
    the attrition is counted rather than crashing a run — but ``place_anomaly_source``
    raises rather than returning a degenerate placement, because a source at the goal is
    the exact degeneracy this module exists to prevent.
    """


class EmptyDatasetError(PlacementError):
    """No episode in this scene could be built at all — a 0% yield, which is DATA.

    Separate from ``PlacementError`` because the two are answered differently. One episode
    that cannot be placed is attrition and gets skipped; a whole scene that cannot is the
    most informative point a yield denominator has, and it must reach disk before the run
    stops. It therefore carries the whole ``DatasetBuild`` — every candidate and its
    reason — rather than the first five formatted into a message.

    yield-1 is why. ``mL8ThkuaVTM`` offered 99 candidates and placed none, in both of that
    tag's invocations. The raise happened before ``write_run_summary``, so the scene left
    no record, so ``yield_report`` aggregated 19 scenes and called it the yield of 20 —
    excluding the only one that yielded nothing.

    It still raises. A run asked for episodes and produced none, and that is a failure for
    whoever asked; what changes is that the failure is now written down first.
    """

    def __init__(self, message: str, *, scene_label: str, build: "DatasetBuild") -> None:
        super().__init__(message)
        self.scene_label = scene_label
        self.build = build


@dataclass(frozen=True)
class SourcePlacement:
    """Where the anomaly sounds from, what stands there, and the evidence for both.

    ``anomaly_object`` is the load-bearing one. The controller makes it the active goal
    during INVESTIGATE, and the realizable arm's arrival is peak-or-plateau **plus visual
    confirm** — so an unnamed source could only ever leave INVESTIGATE through the
    step-budget abort. Every placement names it, because every candidate is a real goal
    object with a category.

    The three measurements are recorded rather than merely checked: ``separation_m`` and
    ``height_difference_m`` are what ADR-0010's rule is about, and
    ``height_difference_to_start_m`` is the start-to-source drop the controller pays for.

    **That last one used to be recorded and unconstrained, and the box closed the gap.**
    The reasoning was that the number would say so afterwards — but it never reached a
    run's metrics (the audit surfaced ``source_dy_m``, the *anchor* difference, which read
    0.000), so ticket 26's first full episode ran as a silent null with its source 2.6 m
    below the agent's start. The floor rule now covers both anchors and this field is
    bounded by it; ``runner`` publishes it as ``source_dy_start_m`` so a future violation
    is visible in the record rather than inferred from raw coordinates.
    """

    position: Xyz
    anomaly_object: str
    object_id: Optional[str]
    separation_m: float
    height_difference_m: float
    height_difference_to_start_m: float
    same_category: bool
    # Whether the source landed on the class's OWN anchor category, or on the geometric
    # fallback because no instance of that anchor qualified. Recorded rather than merely
    # preferred: the memory arm's prior recalls a category, and an episode whose source is
    # NOT at that category is one the prior could not have got right. Splitting the readout
    # on this is the difference between "the memory was wrong" and "the memory was right
    # and this episode did not follow the rule" -- and pooling them would charge the first
    # for the second.
    at_class_anchor: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "position": list(self.position.as_tuple()),
            "anomaly_object": self.anomaly_object,
            "object_id": self.object_id,
            "separation_m": float(self.separation_m),
            "height_difference_m": float(self.height_difference_m),
            "height_difference_to_start_m": float(self.height_difference_to_start_m),
            "same_category": bool(self.same_category),
            "at_class_anchor": bool(self.at_class_anchor),
        }


@dataclass(frozen=True)
class AnomalyEpisode:
    """One ObjectNav episode plus the anomaly the agent will be interrupted by."""

    episode: Episode
    source: SourcePlacement
    anomaly_class: str
    t_anom: int

    @property
    def primary_category(self) -> str:
        return self.episode.object_category


@dataclass(frozen=True)
class DatasetBuild:
    """The episodes that could be built, and the ones that could not, with reasons.

    Skips are carried rather than swallowed for the reason §6 gives about the funnel: an
    aggregate that mixes "the scene could not express this episode" with "the agent never
    heard the sound" hides the mechanism, and this project's record has more than one case
    where it did.
    """

    episodes: Tuple[AnomalyEpisode, ...]
    skipped: Tuple[Tuple[str, str], ...] = ()

    def summary(self) -> str:
        lines = ["dataset: {} episode(s) built, {} skipped".format(
            len(self.episodes), len(self.skipped)
        )]
        for episode_id, reason in self.skipped:
            lines.append("  skipped {:<12} {}".format(episode_id, reason))
        return "\n".join(lines)


def goal_table(dataset: EpisodeDataset) -> Dict[str, Tuple[ObjectGoal, ...]]:
    """``{category: goals}`` across the whole scene, from the episodes themselves.

    The published content file hoists one goal list per category
    (``goals_by_category``), and ``episodes.parse_content`` resolves it onto each episode
    — so every category present in the scene appears on some episode, and gathering them
    needs no second read of the file and no loader change.

    Instances are de-duplicated by ``(category, position)``, because every episode of a
    category carries the same goal list and a naive concatenation would offer the same
    object as a candidate once per episode.
    """
    table: Dict[str, List[ObjectGoal]] = {}
    seen: Dict[str, set] = {}
    for episode in dataset.episodes:
        category = episode.object_category
        bucket = table.setdefault(category, [])
        keys = seen.setdefault(category, set())
        for goal in episode.goals:
            key = goal.position.as_tuple()
            if key in keys:
                continue
            keys.add(key)
            bucket.append(goal)
    return {category: tuple(goals) for category, goals in table.items()}


def _first_view_point(goal: ObjectGoal) -> Optional[Xyz]:
    """A navigable pose at this object, or ``None`` if it publishes none.

    The first view point rather than the highest-IoU one, deliberately. The old builder
    took ``vps[0]`` for a stated reason — the highest-IoU view point can be a malformed
    entry with no position, which would raise and skip the whole category — and
    ``episodes._parse_view_point`` requires a position, so entry zero is always real.
    """
    return goal.view_points[0].position if goal.view_points else None


def primary_anchor(episode: Episode) -> Xyz:
    """The primary goal view point the separation and floor rules are measured from.

    The one nearest the episode's start in three dimensions. "The primary goal" is not a
    single point when a category has several instances, so the rule needs a defined
    anchor, and the nearest-to-start instance is the one the episode is realistically
    about: it is the goal a working agent reaches, so it is the goal the detour has to be
    decoupled *from*.

    Falls back to the goal object's own position when an instance publishes no view
    points, which keeps a malformed instance from removing the anchor entirely.
    """
    if not episode.goals:
        raise PlacementError(
            "episode {} has no goals, so there is nothing to place a source relative "
            "to".format(episode.episode_id)
        )
    start = episode.start_position
    candidates: List[Xyz] = []
    for goal in episode.goals:
        view_point = _first_view_point(goal)
        candidates.append(view_point if view_point is not None else goal.position)
    return min(
        candidates,
        key=lambda p: (
            (p.x - start.x) ** 2 + (p.y - start.y) ** 2 + (p.z - start.z) ** 2
        ),
    )


def derive_t_anom(
    episode: Episode,
    *,
    fraction: float = T_ANOM_FRACTION,
    floor_steps: int = T_ANOM_FLOOR_STEPS,
) -> int:
    """The step this episode's anomaly starts playing, derived from its own geometry.

    **Why this is not a constant.** It was one — ``t_anom = 30``, tagged ``fake``, chosen
    "low enough that a 500-step episode has room for the detour and the resume". That
    reasoning measures against the step *budget*, and under an oracle STOP the binding
    constraint is the *find*: the episode ends when the agent reaches its primary goal,
    not when the budget runs out. The smoke's second box episode found its bed at step 30
    after 3.75 m and 15 forwards — the same step the source started sounding — so the
    anomaly arrived on the last step of the run and the loop under test never ran. A
    number chosen against 500 was spent on an episode that lasted 31.

    So it is derived per episode, from the one thing that decides how long the find is:

    - ``reach`` — the straight-line ``xz`` distance from the start to the nearest primary
      view point. A straight line is never longer than the navmesh route, so this is a
      lower bound on the distance the agent must actually cover.
    - minus ``ARRIVAL_RADIUS_M``, the part of that route the oracle STOP means is never
      walked.
    - divided by ``FORWARD_STEP_M``, an upper bound on travel per step, which turns a
      lower-bound distance into a lower bound on the number of steps.

    Every approximation therefore leans the same way — earlier — and the result is that
    the find **cannot** end before ``floor((reach - radius) / step)``. Taking ``fraction``
    of that lands the onset strictly inside the search, by an argument rather than by a
    guess about any particular scene.

    The one exception is stated rather than hidden: ``floor_steps`` wins when the goal is
    within about two metres of the start, and in that episode the find can end before the
    source ever sounds. That is a degenerate episode — there is no search to interrupt —
    and §2.5's rule applies, so it shows up as a funnel stage rather than being screened.

    Pin ``RunConfig.t_anom`` to an integer to override this; ``None`` means derive.
    """
    view_points = episode.view_points()
    if not view_points:
        raise PlacementError(
            "episode {} publishes no goal view points, so there is no distance to derive "
            "t_anom from".format(episode.episode_id)
        )
    start = episode.start_position
    reach = min(start.horizontal_distance_to(point.position) for point in view_points)
    walk_m = max(0.0, reach - ARRIVAL_RADIUS_M)
    earliest_end = int(math.floor(walk_m / FORWARD_STEP_M))
    return max(int(floor_steps), int(math.floor(float(fraction) * earliest_end)))


def _primary_keep_out(episode: Episode) -> Tuple[Xyz, ...]:
    """Every point the source must stay clear of: all primary view points and objects.

    The strengthening this module's docstring discloses. Object positions are included
    alongside view points because a source placed *at* an instance is as degenerate as
    one placed at its viewing pose, and an object whose view points are all malformed
    would otherwise vanish from the bar.
    """
    points: List[Xyz] = []
    for goal in episode.goals:
        points.append(goal.position)
        points.extend(view_point.position for view_point in goal.view_points)
    return tuple(points)


def place_anomaly_source(
    episode: Episode,
    table: Dict[str, Tuple[ObjectGoal, ...]],
    *,
    anchor_category: Optional[str] = None,
    min_sep_m: float = 3.0,
    max_dy_m: float = 1.0,
    min_start_sep_m: float = MIN_SOURCE_START_SEP_M,
) -> SourcePlacement:
    """Choose the one positioned source for this episode, or raise.

    The rules, in the order they are applied:

    1. The candidate is a real goal view point, so it is navigable and has a named
       object standing at it.
    2. It clears ``min_sep_m`` in ``xz`` from **every** primary goal view point and
       object position — the decoupling.
    2b. It clears ``min_start_sep_m`` in ``xz`` from the episode START, so there is an
       investigation to run at all.

       **The mirror of rule 2, and it was missing.** Rule 2 keeps the source away from
       the *goal*; nothing kept it away from the *agent*. `detour-1`'s episode 18 placed
       one 0.75 m from the start — inside the arrival radius before the anomaly had
       sounded — and it counted as a completed anomaly-response loop: INVESTIGATE at step
       5, RESUME at step 7, two steps of detour. One of that run's eight successes was a
       source already at the agent's feet, which makes 8/20 read as 7/20 honestly.

       It is measured against the start rather than against the pose at ``t_anom``,
       because the builder runs before anything is simulated and the agent's position
       when the anomaly fires is not knowable here. The start is a lower bound on it in
       the only direction that matters: an agent that has walked away from a source
       placed ``min_start_sep_m`` off has a real detour either way, and one that has
       walked *toward* it was heading there anyway.
    3. It is within ``max_dy_m`` in ``y`` of **both** the primary anchor and the episode
       start — ADR-0010's floor rule, checked **before** the nearest-first tie-break,
       which would otherwise actively prefer a cross-floor candidate for being ``xz``-near.

       **Both anchors, and the start was the one that was missing.** HM3D ObjectNav
       episodes routinely begin a storey from their goal: the smoke's episode 0 starts at
       y +2.064 with its nearest bed view point at y -0.536, 2.6 m apart with an authored
       geodesic of 5.98 m via stairs. Measuring only against the anchor let a source sit
       at the goal's level and pass — ``|anchor - source|`` 0.000 — while a full storey
       below where the agent begins. The episode was then legal by this function's own
       test and unwinnable in practice: the onset fired at step 30 with the agent still
       upstairs, and a greedy energy climb cannot take stairs. It spent its whole 120-step
       budget on the wrong floor, 62 forwards and 0 collisions, while the measured RMS
       *fell* 0.0407 -> 0.0121 chasing sound through a stairwell. Smoke criterion 5 was
       unreachable by construction.

       ``t_anom`` is why it must be both rather than either: the anomaly fires mid-episode,
       so the agent may be on the start's floor or the goal's, and only a source within
       reach of both is climbable either way. The side effect is the right one — in a
       cross-floor episode the two anchors are further apart than ``max_dy_m``, so nothing
       qualifies and the episode is skipped with a reason instead of running as a silent
       null.
    4. Among survivors: **the class's own anchor category first**, then a different
       category from the primary goal, then the nearest.

       **Rule 4's first key is new and it changes what the task IS.** Before it, the
       source was placed by geometry alone -- nothing in this module read
       `vocabulary.anchor_object`, so an alarm sat at whatever object cleared the
       separation rules. Every episode this repo ran before 2026-09-02, `abl-1` included,
       was built that way. A semantic memory that learns "an alarm is heard at a bed" has
       nothing to predict in a world where the alarm is wherever the geometry put it, so
       ADR-0018's heard axis could not have measured anything. This key is what gives the
       world the structure the memory is supposed to learn.

       **It is a PREFERENCE, not a filter.** When no instance of the anchor qualifies --
       the scene has no bed, or every bed is too near the goal or on another floor -- the
       ranking falls through to exactly the pre-2026-09-02 behaviour and the placement
       records `at_class_anchor=False`. So yield cannot drop, and an episode that could
       not follow the rule says so on its own record rather than being counted against
       the memory that recalled correctly.

       `anchor_category=None` reproduces the old ordering exactly, which is what every
       caller that does not know its sound class still gets.

    Nothing about audibility is consulted, and nothing renders (§2.5).
    """
    anchor = primary_anchor(episode)
    start = episode.start_position
    keep_out = _primary_keep_out(episode)
    primary_category = episode.object_category

    # (separation, same_category, category, position, object_id, at_class_anchor)
    qualifying: List[Tuple[float, bool, str, Xyz, Optional[str], bool]] = []
    n_too_near = n_wrong_floor = n_no_view_point = n_at_the_start = 0
    for category in sorted(table):
        for goal in table[category]:
            position = _first_view_point(goal)
            if position is None:
                n_no_view_point += 1
                continue
            separation = min(
                position.horizontal_distance_to(point) for point in keep_out
            )
            if separation < float(min_sep_m):
                n_too_near += 1
                continue
            # Counted apart from `too_near`: "the source would be on top of the goal" and
            # "the source would be on top of the agent" are different degeneracies, and a
            # yield report that pooled them would name the wrong rule to revisit.
            if position.horizontal_distance_to(start) < float(min_start_sep_m):
                n_at_the_start += 1
                continue
            if abs(position.height_difference_to(anchor)) > float(max_dy_m) or abs(
                position.height_difference_to(start)
            ) > float(max_dy_m):
                n_wrong_floor += 1
                continue
            qualifying.append(
                (
                    separation,
                    category == primary_category,
                    category,
                    position,
                    goal.object_id,
                    anchor_category is not None and category == anchor_category,
                )
            )

    if not qualifying:
        # When the start and the anchor are themselves more than `max_dy_m` apart, NO
        # candidate can satisfy both and the count above says "another floor" for a
        # reason that is about the episode rather than the scene. Said plainly, because
        # a skip reason that only blamed the scene is what let this run as a silent null.
        start_to_anchor = abs(start.height_difference_to(anchor))
        cross_floor = (
            " The episode's own start is {:.2f} m in y from its primary anchor, which "
            "already exceeds the {:.2f} m rule, so no placement could have qualified: "
            "this episode spans floors and the greedy climb cannot take stairs.".format(
                start_to_anchor, float(max_dy_m)
            )
            if start_to_anchor > float(max_dy_m)
            else ""
        )
        raise PlacementError(
            "no object in {} is >= {:.2f} m (xz) from every {!r} goal AND >= {:.2f} m "
            "(xz) from the episode start AND within {:.2f} m in y of BOTH the primary "
            "anchor and the episode start (rejected: {} too near, {} at the start, "
            "{} on another floor, {} with no view point). The scene cannot express a "
            "decoupled anomaly response for this episode.{}".format(
                episode.scene_label,
                float(min_sep_m),
                primary_category,
                float(min_start_sep_m),
                float(max_dy_m),
                n_too_near,
                n_at_the_start,
                n_wrong_floor,
                n_no_view_point,
                cross_floor,
            )
        )

    # A different category (False) before the same one (True), then nearest first, then
    # the category name so the pick is reproducible when two candidates tie exactly.
    # `not row[5]` first: an anchor candidate sorts False and therefore ahead. The
    # decoupling preference (`row[1]`) stays, one rank down, so it still breaks ties among
    # anchors and still orders the fallback exactly as it always did.
    qualifying.sort(key=lambda row: (not row[5], row[1], row[0], row[2]))
    separation, same_category, category, position, object_id, at_anchor = qualifying[0]
    return SourcePlacement(
        position=position,
        anomaly_object=category,
        object_id=object_id,
        separation_m=separation,
        at_class_anchor=bool(at_anchor),
        height_difference_m=position.height_difference_to(anchor),
        height_difference_to_start_m=position.height_difference_to(
            episode.start_position
        ),
        same_category=same_category,
    )


def build_anomaly_episodes(
    dataset: EpisodeDataset,
    *,
    anomaly_class: str,
    # WHICH OBJECT CATEGORY THIS CLASS BELONGS AT, passed in rather than looked up.
    # The lookup lives in `task/prior_build.py` because it reads `audio.vocabulary`, and
    # `test_task_dataset.TestAudibilityIsNotScreened` holds that this module reaches into
    # `audio/` for nothing at all -- §2.5's rule is that audibility is not screened at
    # build time, and the cheapest way to keep that true is to import none of it. So the
    # DECISION stays here, in `place_anomaly_source`'s ranking; only the table is elsewhere,
    # and a caller that does not know its class still gets the pre-2026-09-02 ordering.
    anchor_category: Optional[str] = None,
    t_anom: Optional[int] = None,
    category: Optional[str] = None,
    n_episodes: Optional[int] = None,
    min_sep_m: float = 3.0,
    max_dy_m: float = 1.0,
    min_start_sep_m: float = MIN_SOURCE_START_SEP_M,
) -> DatasetBuild:
    """Build up to ``n_episodes`` anomaly episodes from one scene's ObjectNav episodes.

    ``category`` filters the **primary** goal only; the source is still drawn from every
    category in the scene, which is what makes a different-category source available at
    all.

    ``t_anom`` is a pin. ``None`` — the default — derives one per episode from that
    episode's own start-to-goal distance (``derive_t_anom``), so the anomaly lands inside
    the find it is supposed to interrupt rather than at a step index chosen once for every
    scene. An integer forces that value on every episode, which is what an experiment
    holding the onset fixed wants and what the ``--t-anom`` flag is for.

    Episodes whose placement fails are skipped with the reason recorded rather than
    aborting the build — a scene that can express three of its episodes should run three.
    An empty result raises, because a build that produced nothing and said so only in a
    list is a run that would otherwise start and immediately do nothing.

    **The empty result is still a measurement**, and it raises as ``EmptyDatasetError``
    carrying the whole build so the caller can write it down before the run stops. That
    distinction cost the yield-1 sweep its most informative scene: ``mL8ThkuaVTM`` offered
    99 candidates and could place none of them — a true 0% yield, the single number a
    denominator most wants — and because the raise carried nothing but a message, the
    scene left no ``summary.json`` and ``yield_report`` never saw it. The tool that
    measures attrition was blind to total attrition, in the direction that flatters.
    """
    table = goal_table(dataset)
    candidates = [
        episode
        for episode in dataset.episodes
        if category is None or episode.object_category == category
    ]
    if not candidates:
        raise PlacementError(
            "no episode in {} has object_category={!r}; present: {}".format(
                dataset.scene_label, category, ", ".join(dataset.categories())
            )
        )

    wanted = len(candidates) if n_episodes is None else max(0, int(n_episodes))
    built: List[AnomalyEpisode] = []
    skipped: List[Tuple[str, str]] = []
    for position, episode in enumerate(candidates):
        if len(built) >= wanted:
            break
        try:
            placement = place_anomaly_source(
                episode, table,
                # THE RUN'S OWN CLASS DECIDES WHERE ITS SOURCE GOES. Passing it here is
                # what makes the class-to-category association a fact about the world
                # rather than a fiction the prior pass teaches. Every episode of a build
                # shares one class, so this is constant across the loop and is computed
                # once above it.
                anchor_category=anchor_category,
                min_sep_m=min_sep_m, max_dy_m=max_dy_m,
                min_start_sep_m=min_start_sep_m,
            )
        except PlacementError as exc:
            # The candidate's POSITION leads the label: HM3D authors `episode_id` as
            # "0" on every episode of a scene, and matrix-1 wrote 65 skip rows for one
            # scene that were indistinguishable for exactly that reason.
            skipped.append((
                "cand{:04d} id={}".format(position, episode.episode_id), str(exc),
            ))
            continue
        built.append(
            AnomalyEpisode(
                episode=episode,
                source=placement,
                anomaly_class=str(anomaly_class),
                t_anom=derive_t_anom(episode) if t_anom is None else int(t_anom),
            )
        )

    if not built:
        raise EmptyDatasetError(
            "no episode in {} could be built ({} candidate(s), all skipped):\n  {}".format(
                dataset.scene_label,
                len(candidates),
                "\n  ".join("{}: {}".format(eid, why) for eid, why in skipped[:5]),
            ),
            scene_label=dataset.scene_label,
            build=DatasetBuild(episodes=(), skipped=tuple(skipped)),
        )
    return DatasetBuild(episodes=tuple(built), skipped=tuple(skipped))
