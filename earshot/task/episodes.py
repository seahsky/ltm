"""The ObjectNav ``.json.gz`` loader — habitat-lab's job, done in ~300 lines of stdlib.

Extracted out of ``embodied_memory/habitat_env.py`` (623 LOC), which does not carry:
that module is a wrapper around ``habitat.Env``, and habitat-lab is deliberately not a
dependency of the clean room (``tests/mac/test_layering.py`` asserts nothing imports
it). What habitat-lab supplied and this replaces is exactly three things — the
dataset-path search, scene-label resolution, and the lazy ``content/<scene>.json.gz``
load — plus the goal/view-point schema the runner and the dataset builder share.

**Pure and stdlib-only.** No numpy, no habitat-sim, no I/O beyond ``gzip`` + ``json``,
so the whole module is Mac-testable against real dataset bytes rather than against a
fake (ADR-0014's fake layer is for bindings we did not write; this is our own logic).

The schema is habitat-lab's, read from its source rather than inferred. Citations are
against ``habitat_lab-0.3.320250127``, the version the old tree ran:

- ``datasets/pointnav/pointnav_dataset.py:26-27`` — ``content_scenes_path`` is
  ``"{data_path}/content/{scene}.json.gz"`` and ``DEFAULT_SCENE_PATH_PREFIX`` is
  ``"data/scene_datasets/"``.
- ``datasets/object_nav/object_nav_dataset.py:163-168`` — the prefix is stripped from
  each episode's ``scene_id`` and the remainder is joined onto ``scenes_dir``.
- ``tasks/nav/object_nav_task.py:42-44`` — ``goals_key`` is
  ``f"{basename(scene_id)}_{object_category}"``, e.g. ``TEEsavR23oF.basis.glb_chair``.
- ``utils/geometry_utils.py:55-60`` — ``start_rotation`` and every view point's
  rotation are quaternion coefficients in **[x, y, z, w]** order, not [w, x, y, z].
  Getting that backwards points the agent somewhere plausible and wrong, which is why
  it is read from source and pinned by a test rather than assumed.

**One deliberate divergence, and it is a fix.** habitat-lab overwrites every authored
``episode_id`` with the load index (``object_nav_dataset.py:141``,
``episode.episode_id = str(i)``). That renumbering is why the old analysis pipeline had
to re-key onto ``(scene_id, target_category, visit_order)`` after silently dropping
pairs, and why ``seed_only`` had to ride in ``episode.info`` because the id could not
carry it. This loader keeps the authored id and exposes the load index separately, so
both are available and neither is a lie.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from earshot.types import Xyz

__all__ = [
    "ViewPoint",
    "ObjectGoal",
    "Episode",
    "EpisodeDataset",
    "DEFAULT_SCENE_PATH_PREFIX",
    "SPLIT_DIR_CANDIDATES",
    "SCENES_DIR_CANDIDATES",
    "find_split_dir",
    "find_scenes_dir",
    "available_scenes",
    "scene_label",
    "resolve_scene_path",
    "goals_key",
    "load_scene",
]

# habitat-lab writes every ObjectNav episode's `scene_id` relative to this prefix, then
# strips it and re-joins against `scenes_dir` (pointnav_dataset.py:26). The prefix is a
# property of the published dataset files, so it stays even though we never use
# habitat-lab's default `scenes_dir`.
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"

# Where the ObjectNav episode splits live, most canonical first. The first entry is the
# layout on the box (runbook section 6); the second is the older flat layout that
# `download_hm3d.sh` produced and that some machines still carry.
SPLIT_DIR_CANDIDATES: Tuple[str, ...] = (
    "data/hm3d/datasets/objectnav/hm3d/v1",
    "data/datasets/objectnav/hm3d/v1",
)

# Where the meshes live. Same ordering rule, and the same reason.
SCENES_DIR_CANDIDATES: Tuple[str, ...] = (
    "data/hm3d/scene_datasets",
    "data/scene_datasets",
)

_CONTENT_SUFFIX = ".json.gz"


class EpisodeDataError(ValueError):
    """The dataset on disk is not the shape this loader was written against.

    Raised rather than defaulted, because every field this module reads is load-bearing
    for navigation: a missing ``start_position`` silently becomes the origin, and a
    missing goal set silently becomes an unreachable episode that still produces a
    number.
    """


@dataclass(frozen=True)
class ViewPoint:
    """A navigable pose from which the goal object is visible.

    The success ring is measured to a view point, not to the object (audit caveat 5),
    so this is the geometry the metrics actually key on.
    """

    position: Xyz
    rotation: Tuple[float, float, float, float]  # [x, y, z, w]
    iou: Optional[float] = None


@dataclass(frozen=True)
class ObjectGoal:
    """One instance of the goal category, with the poses that can see it."""

    position: Xyz
    view_points: Tuple[ViewPoint, ...]
    object_id: Optional[str] = None
    object_category: Optional[str] = None


@dataclass(frozen=True)
class Episode:
    """One ObjectNav episode, resolved against the meshes on this machine.

    ``scene_path`` is a real filesystem path to the ``.basis.glb``, which is what
    ``sim.World`` takes. There is no scene-dataset-config indirection — see
    ``resolve_scene_path``.
    """

    episode_id: str
    index: int
    scene_id: str  # as authored, e.g. "hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    scene_label: str  # "TEEsavR23oF"
    scene_path: str  # resolved: "<scenes_dir>/hm3d/val/.../TEEsavR23oF.basis.glb"
    object_category: str
    start_position: Xyz
    start_rotation: Tuple[float, float, float, float]  # [x, y, z, w]
    goals: Tuple[ObjectGoal, ...]
    info: Mapping[str, Any]

    def view_points(self) -> Tuple[ViewPoint, ...]:
        """Every view point across every instance of the goal category, flattened.

        The runner's arrival test and the SPL denominator both want the whole set — the
        agent succeeds by reaching any of them, not a designated one.
        """
        return tuple(vp for goal in self.goals for vp in goal.view_points)


@dataclass(frozen=True)
class EpisodeDataset:
    """One scene's episodes plus the goal table they were resolved against."""

    scene_label: str
    scene_path: str
    episodes: Tuple[Episode, ...]

    def categories(self) -> Tuple[str, ...]:
        """The goal categories present, sorted. What the scale-up planner enumerated."""
        return tuple(sorted({ep.object_category for ep in self.episodes}))

    def filter_category(self, category: str) -> Tuple[Episode, ...]:
        return tuple(ep for ep in self.episodes if ep.object_category == category)


# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------


def find_split_dir(split: str = "val", root: str = ".") -> str:
    """Locate ``<root>/<candidate>/<split>``, most canonical layout first.

    Raises rather than returning ``None`` — the old tree's ``_default_episodes_path``
    returned ``None`` on failure and handed it to habitat-lab, which then failed
    somewhere else with a message about a different path.
    """
    tried = []
    for candidate in SPLIT_DIR_CANDIDATES:
        path = os.path.join(root, candidate, split)
        tried.append(path)
        if os.path.isdir(path):
            return path
    raise EpisodeDataError(
        "no ObjectNav split directory for split={!r}; tried:\n  {}".format(
            split, "\n  ".join(tried)
        )
    )


def find_scenes_dir(root: str = ".") -> str:
    """Locate the directory the episodes' ``scene_id`` paths are relative to."""
    tried = []
    for candidate in SCENES_DIR_CANDIDATES:
        path = os.path.join(root, candidate)
        tried.append(path)
        if os.path.isdir(path):
            return path
    raise EpisodeDataError(
        "no HM3D scenes directory; tried:\n  {}".format("\n  ".join(tried))
    )


def available_scenes(split_dir: str) -> Tuple[str, ...]:
    """Scene labels with their own ``content/<scene>.json.gz``, sorted.

    Replaces habitat-lab's ``_get_scenes_from_folder``
    (``pointnav_dataset.py:75-91``). Reading the directory is the whole discovery
    mechanism — the split's top-level ``<split>.json.gz`` holds no episodes for HM3D.
    """
    content_dir = os.path.join(split_dir, "content")
    if not os.path.isdir(content_dir):
        raise EpisodeDataError("no content/ directory under {}".format(split_dir))
    return tuple(
        sorted(
            name[: -len(_CONTENT_SUFFIX)]
            for name in os.listdir(content_dir)
            if name.endswith(_CONTENT_SUFFIX)
        )
    )


def scene_label(scene_id: str) -> str:
    """``.../00800-TEEsavR23oF/TEEsavR23oF.basis.glb`` -> ``TEEsavR23oF``.

    The label keys the content file, the RIR-free audio config, and every per-scene
    analysis join. It is the basename up to the first dot, so ``.basis.glb`` and
    ``.glb`` both reduce to the same label — which is what makes a run comparable
    across the two mesh variants.
    """
    base = os.path.basename(scene_id)
    return base.split(".", 1)[0]


def resolve_scene_path(scene_id: str, scenes_dir: str) -> str:
    """Turn an authored ``scene_id`` into a filesystem path to the mesh.

    **This is where ticket 08's outstanding box fact is settled, and the question had a
    false premise.** It asked whether ObjectNav HM3D v1 loads against
    ``hm3d_basis.scene_dataset_config.json`` or requires
    ``hm3d_annotated_basis.scene_dataset_config.json``. Stock habitat-lab uses
    **neither**: ``benchmark/nav/objectnav/objectnav_hm3d.yaml`` never sets
    ``scene_dataset``, so it keeps the default ``"default"``
    (``config/default_structured_configs.py:1744``), which
    ``habitat_simulator.py:326`` assigns straight to
    ``SimulatorConfiguration.scene_dataset_config_file`` — and ``"default"`` is also
    habitat-sim's own constructor default. The episode's ``scene_id`` is resolved as a
    plain filesystem path (``object_nav_dataset.py:163-168``), which is exactly the
    form ticket 04 and ticket 16 already rendered against on the box.

    The old tree pointed at the *annotated* config
    (``habitat_env.py:132``) for one reason: it added a semantic sensor. That sensor is
    gone — ADR-0007 turns materials off permanently, ticket 03 found HM3D's v0.2
    texture-based semantics hand the audio context an empty mesh, and CLAUDE.md records
    that the sensor returned all-zeros for the whole of the earlier work. So the clean
    room needs no scene-dataset config of any kind, and the semantic annotations
    ticket 10 kept only against this question are no longer load-bearing.

    Recorded as source-derived. ``tests/box/test_world_box.py`` measures it.
    """
    relative = scene_id
    if relative.startswith(DEFAULT_SCENE_PATH_PREFIX):
        relative = relative[len(DEFAULT_SCENE_PATH_PREFIX) :]
    return os.path.join(scenes_dir, relative)


def goals_key(scene_id: str, object_category: str) -> str:
    """habitat-lab's ``ObjectGoalNavEpisode.goals_key`` (``object_nav_task.py:42-44``)."""
    return "{}_{}".format(os.path.basename(scene_id), object_category)


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise EpisodeDataError("{} is missing required key {!r}".format(where, key))
    return mapping[key]


def _coeffs(values: Sequence[float], where: str) -> Tuple[float, float, float, float]:
    """Quaternion coefficients in habitat-lab's [x, y, z, w] order."""
    coeffs = tuple(float(v) for v in values)
    if len(coeffs) != 4:
        raise EpisodeDataError(
            "{}: expected 4 quaternion coefficients [x, y, z, w], got {}".format(
                where, len(coeffs)
            )
        )
    return coeffs  # type: ignore[return-value]


def _parse_view_point(raw: Mapping[str, Any], where: str) -> ViewPoint:
    state = raw.get("agent_state") or {}
    return ViewPoint(
        position=Xyz.from_sequence(_require(state, "position", where)),
        rotation=_coeffs(_require(state, "rotation", where), where),
        iou=None if raw.get("iou") is None else float(raw["iou"]),
    )


def _parse_goal(raw: Mapping[str, Any], where: str) -> ObjectGoal:
    view_points = tuple(
        _parse_view_point(vp, "{} view_point[{}]".format(where, i))
        for i, vp in enumerate(raw.get("view_points") or ())
    )
    return ObjectGoal(
        position=Xyz.from_sequence(_require(raw, "position", where)),
        view_points=view_points,
        object_id=None if raw.get("object_id") is None else str(raw["object_id"]),
        object_category=(
            None if raw.get("object_category") is None else str(raw["object_category"])
        ),
    )


def parse_content(
    content: Mapping[str, Any],
    *,
    scenes_dir: str,
    source: str = "<content>",
) -> Tuple[Episode, ...]:
    """Turn a decoded ObjectNav content dict into episodes. Pure — no I/O.

    Split out from ``load_scene`` so the whole parse is testable against a literal dict
    as well as against real bytes.
    """
    raw_episodes = content.get("episodes")
    if not raw_episodes:
        raise EpisodeDataError("{} holds no episodes".format(source))
    by_category: Dict[str, Any] = content.get("goals_by_category") or {}

    episodes: List[Episode] = []
    for index, raw in enumerate(raw_episodes):
        where = "{} episode[{}]".format(source, index)
        scene_id = str(_require(raw, "scene_id", where))
        category = str(_require(raw, "object_category", where))

        # `goals_by_category` is the published form: `dedup_goals` empties every
        # episode's own `goals` list and hoists one copy per category
        # (`object_nav_dataset.py:38-58`). Inline goals are the pre-dedup form, which
        # the smoke builders in the old tree also emitted, so both are accepted.
        raw_goals = by_category.get(goals_key(scene_id, category)) or raw.get("goals")
        if not raw_goals:
            raise EpisodeDataError(
                "{}: no goals for category {!r} — looked for goals_by_category[{!r}] "
                "and an inline 'goals' list".format(
                    where, category, goals_key(scene_id, category)
                )
            )

        episodes.append(
            Episode(
                # Authored, NOT the load index — see the module docstring.
                episode_id=str(raw.get("episode_id", index)),
                index=index,
                scene_id=scene_id,
                scene_label=scene_label(scene_id),
                scene_path=resolve_scene_path(scene_id, scenes_dir),
                object_category=category,
                start_position=Xyz.from_sequence(
                    _require(raw, "start_position", where)
                ),
                start_rotation=_coeffs(_require(raw, "start_rotation", where), where),
                goals=tuple(
                    _parse_goal(goal, "{} goal[{}]".format(where, i))
                    for i, goal in enumerate(raw_goals)
                ),
                info=dict(raw.get("info") or {}),
            )
        )
    return tuple(episodes)


def load_scene(
    split_dir: str,
    scene: str,
    *,
    scenes_dir: str,
    category: Optional[str] = None,
) -> EpisodeDataset:
    """Load one scene's ``content/<scene>.json.gz``, resolved and validated.

    Lazy by scene, which is the whole reason HM3D ships per-scene content files: the
    ``val`` split is 36 scenes and the smoke needs one.

    ``category`` filters to a single goal category and raises if none match, rather
    than returning an empty list that reads downstream as "this scene has no episodes".
    """
    path = os.path.join(split_dir, "content", scene + _CONTENT_SUFFIX)
    if not os.path.isfile(path):
        raise EpisodeDataError(
            "no content file for scene {!r} at {} — available: {}".format(
                scene, path, ", ".join(available_scenes(split_dir)) or "<none>"
            )
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        content = json.load(handle)

    episodes = parse_content(content, scenes_dir=scenes_dir, source=path)
    if category is not None:
        filtered = tuple(ep for ep in episodes if ep.object_category == category)
        if not filtered:
            raise EpisodeDataError(
                "no episode in {} has object_category={!r}; present: {}".format(
                    path, category, ", ".join(sorted({e.object_category for e in episodes}))
                )
            )
        episodes = filtered

    scene_paths = {ep.scene_path for ep in episodes}
    if len(scene_paths) != 1:
        raise EpisodeDataError(
            "content file {} spans {} meshes ({}) — one content file is one scene".format(
                path, len(scene_paths), ", ".join(sorted(scene_paths))
            )
        )
    return EpisodeDataset(
        scene_label=episodes[0].scene_label,
        scene_path=episodes[0].scene_path,
        episodes=episodes,
    )
