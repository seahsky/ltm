"""The episode builder: where the one anomaly source goes, and what it is called.

Task spec §2.1-§2.2 and ADR-0010. One positioned source per episode, on the primary
goal's floor at ``|Δy| < 1.0 m``, and far enough from the primary goal in ``xz`` that
investigating it is a real detour rather than a second route to the goal.

**The floor rule is checked here and nowhere else.** ADR-0010 moved it from invariant to
builder policy: with no grid and no ``nearest`` there is nothing to snap silently, so
there is no runtime guard and none is needed. What survives is the *controller* argument
— a greedy energy climb over ``move_forward`` / ``turn_left`` / ``turn_right`` cannot
fund a stair-climb, and across floors it fails in a specific ugly way, walking into a
wall while the energy rises through the ceiling.

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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from earshot.task.episodes import Episode, EpisodeDataset, ObjectGoal
from earshot.types import Xyz

__all__ = [
    "PlacementError",
    "SourcePlacement",
    "AnomalyEpisode",
    "DatasetBuild",
    "goal_table",
    "primary_anchor",
    "place_anomaly_source",
    "build_anomaly_episodes",
]


class PlacementError(ValueError):
    """No object in this scene can carry the anomaly source for this episode.

    A property of the scene and the rules, not a program error: a single-object scene,
    or one whose only other objects are upstairs, cannot express a decoupled anomaly
    response. ``build_anomaly_episodes`` catches it, skips the episode and reports it, so
    the attrition is counted rather than crashing a run — but ``place_anomaly_source``
    raises rather than returning a degenerate placement, because a source at the goal is
    the exact degeneracy this module exists to prevent.
    """


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
    ``height_difference_to_start_m`` is the number the ADR does *not* constrain but the
    controller pays for — if the agent starts a storey below the goal, the detour is a
    stair-climb no matter how faithful the source placement is.
    """

    position: Xyz
    anomaly_object: str
    object_id: Optional[str]
    separation_m: float
    height_difference_m: float
    height_difference_to_start_m: float
    same_category: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "position": list(self.position.as_tuple()),
            "anomaly_object": self.anomaly_object,
            "object_id": self.object_id,
            "separation_m": float(self.separation_m),
            "height_difference_m": float(self.height_difference_m),
            "height_difference_to_start_m": float(self.height_difference_to_start_m),
            "same_category": bool(self.same_category),
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
    min_sep_m: float = 3.0,
    max_dy_m: float = 1.0,
) -> SourcePlacement:
    """Choose the one positioned source for this episode, or raise.

    The rules, in the order they are applied:

    1. The candidate is a real goal view point, so it is navigable and has a named
       object standing at it.
    2. It clears ``min_sep_m`` in ``xz`` from **every** primary goal view point and
       object position — the decoupling.
    3. It is within ``max_dy_m`` of the primary anchor in ``y`` — ADR-0010's floor rule,
       checked **before** the nearest-first tie-break, which would otherwise actively
       prefer a cross-floor candidate for being ``xz``-near.
    4. Among survivors: a different category first, then the nearest.

    Nothing about audibility is consulted, and nothing renders (§2.5).
    """
    anchor = primary_anchor(episode)
    keep_out = _primary_keep_out(episode)
    primary_category = episode.object_category

    # (separation, same_category, category, position, object_id)
    qualifying: List[Tuple[float, bool, str, Xyz, Optional[str]]] = []
    n_too_near = n_wrong_floor = n_no_view_point = 0
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
            if abs(position.height_difference_to(anchor)) > float(max_dy_m):
                n_wrong_floor += 1
                continue
            qualifying.append(
                (
                    separation,
                    category == primary_category,
                    category,
                    position,
                    goal.object_id,
                )
            )

    if not qualifying:
        raise PlacementError(
            "no object in {} is >= {:.2f} m (xz) from every {!r} goal AND within "
            "{:.2f} m in y of the primary anchor (rejected: {} too near, {} on another "
            "floor, {} with no view point). The scene cannot express a decoupled anomaly "
            "response for this episode.".format(
                episode.scene_label,
                float(min_sep_m),
                primary_category,
                float(max_dy_m),
                n_too_near,
                n_wrong_floor,
                n_no_view_point,
            )
        )

    # A different category (False) before the same one (True), then nearest first, then
    # the category name so the pick is reproducible when two candidates tie exactly.
    qualifying.sort(key=lambda row: (row[1], row[0], row[2]))
    separation, same_category, category, position, object_id = qualifying[0]
    return SourcePlacement(
        position=position,
        anomaly_object=category,
        object_id=object_id,
        separation_m=separation,
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
    t_anom: int,
    category: Optional[str] = None,
    n_episodes: Optional[int] = None,
    min_sep_m: float = 3.0,
    max_dy_m: float = 1.0,
) -> DatasetBuild:
    """Build up to ``n_episodes`` anomaly episodes from one scene's ObjectNav episodes.

    ``category`` filters the **primary** goal only; the source is still drawn from every
    category in the scene, which is what makes a different-category source available at
    all.

    Episodes whose placement fails are skipped with the reason recorded rather than
    aborting the build — a scene that can express three of its episodes should run three.
    An empty result raises, because a build that produced nothing and said so only in a
    list is a run that would otherwise start and immediately do nothing.
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
    for episode in candidates:
        if len(built) >= wanted:
            break
        try:
            placement = place_anomaly_source(
                episode, table, min_sep_m=min_sep_m, max_dy_m=max_dy_m
            )
        except PlacementError as exc:
            skipped.append((episode.episode_id, str(exc)))
            continue
        built.append(
            AnomalyEpisode(
                episode=episode,
                source=placement,
                anomaly_class=str(anomaly_class),
                t_anom=int(t_anom),
            )
        )

    if not built:
        raise PlacementError(
            "no episode in {} could be built ({} candidate(s), all skipped):\n  {}".format(
                dataset.scene_label,
                len(candidates),
                "\n  ".join("{}: {}".format(eid, why) for eid, why in skipped[:5]),
            )
        )
    return DatasetBuild(episodes=tuple(built), skipped=tuple(skipped))
