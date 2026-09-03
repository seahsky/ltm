"""The prior pass, assembled: which stops, in what order, heard through the real sensor.

`task/prior_pass.py` walks a scripted route and knows nothing about sound classes or
stores. `task/prior_build.py` and `task/memory_build.py` turn a walked tour into store
rows and know nothing about the simulator. Nothing joined the three to a real scene until
now — `window_pilot.sh`'s own header named this gap explicitly: "no caller of
`task/prior_pass.walk_tour` exists outside its own tests".

**Layered exactly as `task/runner.py` is (ADR-0013).** The planning half
(`plan_scene_tour`, `walk_scene`) takes `world` as `object` and an injected `embed`
callback, so it is Mac-testable against `FakeWorld`. The half that renders real audio
(`render_embedding_at_stop`, `tour_one_scene`, `run_prior_pass`) imports `sim.world` and
`task.models` inside itself, for the reason `run()`'s own docstring gives: a module-level
import would make every Mac test in this file uncollectable.

**Why the tour plan is restricted to `classes` before it is walked, not after.**
`prior_build.observation_for` returns `None` for a category nothing in `classes` anchors
at — the intended contract, per that function's own docstring, is that the CALLER filters
it out, because `walk_tour` appends whatever `observe` returns UNCONDITIONALLY (its own
docstring says exactly that; it is not a bug in `walk_tour`, it is a simple, honest
contract for a module that must not know what a store row is). A plan built from a
scene's whole goal table can reach a living-room stop via `sofa` when only `chair`
anchors any class in the matrix's bank, and passing `observation_for` straight through as
`observe` would then silently write a `None` into `TourRecord.observations` — caught only
much later, by `semantic_from_tour`, as a bare `TypeError` rather than a named defect.
`plan_scene_tour` restricts the CANDIDATES with `categories_with_a_sound` first, so every
stop the plan can produce already has a resolvable class, and `walk_scene` asserts that
invariant rather than trusting it silently.

**One tour per scene serves the whole class bank.** `class_at_category` reads `classes`
at OBSERVE time, per stop, so a scene toured once yields rows for every class in the bank
that anchors at a room the scene has — `bed`'s room hears whichever of `classes` anchors
there, `toilet`'s hears whichever anchors at `toilet`, and so on. The matrix's three
classes (one per room) need exactly one tour per scene, not one per class.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from earshot.task.memory_build import dump_stores
from earshot.task.memory_prior import category_points
from earshot.task.prior_build import (
    categories_with_a_sound,
    class_at_category,
    observation_for,
    stores_from_records,
)
from earshot.task.prior_pass import (
    DEFAULT_LEG_BUDGET,
    TourPlan,
    TourRecord,
    TourStop,
    candidate_stops,
    plan_tour,
    walk_tour,
)
from earshot.types import Xyz

__all__ = [
    "SceneTourOutcome",
    "plan_scene_tour",
    "walk_scene",
    "merge_scene_records",
    "render_embedding_at_stop",
    "tour_one_scene",
    "run_prior_pass",
    "main",
]


def plan_scene_tour(
    dataset: object,
    room_of_category: Mapping[str, str],
    classes: Sequence[str],
    start: Xyz,
    geodesic: Callable[[Xyz, Xyz], Optional[float]],
) -> TourPlan:
    """The room-level route for one scene, restricted to categories `classes` can sound
    from. Pure — no simulator, no encoder, decidable on a machine with no habitat-sim.

    `category_points(dataset)` is every ObjectNav goal position by category —
    `memory_prior`'s own accessor, reused rather than re-derived so this module cannot
    drift from what `points_by_category_for_cell` reads at run time. `plant` is excluded
    by `room_of_category` alone (`vocabulary.ROOM_OF_ANCHOR` carries no row for it); a
    category that IS a room but that `classes` cannot sound from is excluded here.
    """
    all_points = category_points(dataset)
    heard = categories_with_a_sound(all_points.keys(), classes=classes)
    restricted = {category: all_points[category] for category in heard}
    candidates = candidate_stops(restricted, room_of_category)
    return plan_tour(candidates, start, geodesic)


def walk_scene(
    world: object,
    plan: TourPlan,
    *,
    scene: str,
    classes: Sequence[str],
    embed: Callable[[TourStop], Sequence[float]],
    leg_budget: int = DEFAULT_LEG_BUDGET,
    goal_radius: float = 1.0,
) -> TourRecord:
    """`walk_tour`, with the observe seam pre-wired to `observation_for`.

    `embed` is called only for a REACHED stop and must return the store's embedding
    width — `semantic_from_tour` raises on a mismatch at the merge, which is where a
    wiring bug belongs rather than at some later cosine.

    `AssertionError` rather than a silent skip if `observation_for` still returns `None`:
    that is `plan_scene_tour`'s invariant failing, which means THIS module has a bug, not
    that a class was legitimately absent — the whole reason to filter the plan up front
    was to make that case unreachable, and trusting it silently here would let the
    invariant rot without anything noticing.
    """

    def observe(stop: TourStop) -> Mapping[str, object]:
        payload = observation_for(stop, embed(stop), classes=classes)
        if payload is None:
            raise AssertionError(
                "stop at category {!r} (room {!r}) produced no sound under classes={}; "
                "plan_scene_tour should have excluded every such stop from the plan "
                "before this tour was walked".format(
                    stop.category, stop.room, list(classes)
                )
            )
        return payload

    return walk_tour(
        world,
        plan,
        scene=scene,
        observe=observe,
        leg_budget=leg_budget,
        goal_radius=goal_radius,
    )


def merge_scene_records(
    records: Sequence[TourRecord],
) -> Tuple[Any, Any]:
    """Every scene's tour, as the one pair of stores the matrix carves its cells from.

    A one-line wrapper over `prior_build.stores_from_records`, named at this module's own
    vocabulary so a reader of `run_prior_pass` has one import to follow rather than two.
    """
    return stores_from_records(records)


@dataclass(frozen=True)
class SceneTourOutcome:
    """One scene's attempt: the record if it toured, or the reason it did not.

    A scene that failed to LOAD (no mesh, a corrupt navmesh) is a different fact from a
    scene whose tour ran and left legs unreached — `yield_sweep.sh`'s continue-on-failure
    rule, carried here: one bad scene must not cost the rest of the pass, and the reason
    has to be on the record rather than inferred from an absent entry.
    """

    scene: str
    record: Optional[TourRecord]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.record is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "ok": self.ok,
            "complete": None if self.record is None else self.record.complete,
            "rooms_reached": (
                None if self.record is None else list(self.record.rooms_reached)
            ),
            "n_observations": None if self.record is None else len(self.record.observations),
            "error": self.error,
        }


def render_embedding_at_stop(
    handle: Any,
    encoder: Any,
    *,
    stop: TourStop,
    classes: Sequence[str],
    clip_for_class: Callable[[str], np.ndarray],
    bed_for_length: Callable[[int], np.ndarray],
    sample_rate: int,
) -> np.ndarray:
    """The audio embedding of the class this stop should sound as, rendered for real.

    Mirrors `task/clap_gate.py`'s own pattern exactly (`render_through_ir` -> `mix_bed`
    -> `heard_clip_for_clap` -> `audio_embedding`), because it is the same question asked
    at a different pose: what would the agent's own sensor hand CLAP if this class sounded
    from this stop right now. The source moves to `stop.point` — the sound is AT the
    anchor, matching the run-time placement rule — and the listener stays wherever
    `walk_tour`'s follower left it, which is within `goal_radius` of the same point.

    `bed_for_length` rather than one `bed` array: `mix_bed` refuses a length mismatch by
    design, and different classes' ESC-50 recordings are not guaranteed the same length
    (`clap_gate.run_gate` notes this about its own corpus rather than assuming it away).
    The caller picks the bed by THIS clip's own size, once per class, so a driver touring
    several classes in one scene never hands `mix_bed` a bed built for a different clip.

    `class_at_category` is called again here rather than threaded through from the
    caller, because `render_embedding_at_stop` is the box-only half and must be able to
    answer "what does this stop sound like" on its own — the Mac-tested half
    (`plan_scene_tour`) already guarantees this call cannot return `None` for any stop a
    walked plan can produce.
    """
    from earshot.audio.bed import mix_bed
    from earshot.audio.clap import audio_embedding, heard_clip_for_clap
    from earshot.audio.clips import render_through_ir

    sound_class = class_at_category(stop.category, classes=classes)
    if sound_class is None:
        raise AssertionError(
            "render_embedding_at_stop was asked to render category {!r}, which no class "
            "in {} anchors -- plan_scene_tour did not restrict its candidates".format(
                stop.category, list(classes)
            )
        )
    handle.set_source(stop.point)
    observation, _guard = handle.observe()
    ir = handle.audio_of(observation)
    clip = clip_for_class(sound_class)
    signal = render_through_ir(ir, clip)
    heard = mix_bed(signal, bed_for_length(int(clip.size)))
    mono, rate = heard_clip_for_clap(heard, sample_rate)
    return audio_embedding(mono, rate, encoder)


def tour_one_scene(
    world: Any,
    handle: Any,
    encoder: Any,
    *,
    scene: str,
    dataset: object,
    room_of_category: Mapping[str, str],
    classes: Sequence[str],
    clip_for_class: Callable[[str], np.ndarray],
    bed_for_length: Callable[[int], np.ndarray],
    sample_rate: int,
    seed: int,
    leg_budget: int = DEFAULT_LEG_BUDGET,
    goal_radius: float = 1.0,
) -> TourRecord:
    """Plan and walk one scene's prior pass, rendering real audio at every reached stop.

    `world.seed_navmesh(seed)` before picking a start: the same navmesh-agnostic seed
    every other sweep in this tree uses, so a start point is reproducible from the seed
    alone rather than from whichever point the engine's own RNG state happened to be at.
    """
    world.seed_navmesh(seed)
    start = world.random_navigable_point()
    world.set_pose(start)

    def geodesic(a: Xyz, b: Xyz) -> Optional[float]:
        return world.geodesic_distance(a, [b])

    plan = plan_scene_tour(dataset, room_of_category, classes, start, geodesic)

    def embed(stop: TourStop) -> np.ndarray:
        return render_embedding_at_stop(
            handle,
            encoder,
            stop=stop,
            classes=classes,
            clip_for_class=clip_for_class,
            bed_for_length=bed_for_length,
            sample_rate=sample_rate,
        )

    return walk_scene(
        world,
        plan,
        scene=scene,
        classes=classes,
        embed=embed,
        leg_budget=leg_budget,
        goal_radius=goal_radius,
    )


def run_prior_pass(
    *,
    run_dir: str,
    scenes: Sequence[str],
    classes: Sequence[str],
    split: str = "val",
    data_root: str = ".",
    clip_dir: str = "data/anomaly_audio",
    seed: int = 20260821,
    leg_budget: int = DEFAULT_LEG_BUDGET,
    goal_radius: float = 1.0,
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> pathlib.Path:
    """Tour every scene, merge what was heard, and dump the store `run()` will consume.

    Habitat and the model stack are imported inside this function for `runner.run`'s own
    reason: `sim/world.py` imports habitat_sim, so a module-level import would make every
    Mac test in this file uncollectable.

    Continue-on-failure at the SCENE grain, same rule `clap_gate.run_gate` and
    `yield_sweep.sh` already hold: one scene that cannot load, or whose tour comes back
    incomplete, must not cost the rest of the pass. Every scene's outcome is recorded
    (`SceneTourOutcome`), and only COMPLETE tours contribute rows to the merged store —
    `TourRecord.complete`'s own docstring is explicit that a partial tour must never be
    treated as a seen scene, which is exactly what a merge would do by including one.
    """
    say = progress if progress is not None else print

    from earshot.audio.bed import bed_signal
    from earshot.audio.clips import load_anomaly_clip, resolve_anomaly_clip
    from earshot.audio.config import AudioConfig
    from earshot.audio.sensor import AudioSensorHandle
    from earshot.audio.spec import audio_sensor_spec
    from earshot.audio.vocabulary import ROOM_OF_ANCHOR
    from earshot.env_check import assert_env
    from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs
    from earshot.task.episodes import find_scenes_dir, find_split_dir, load_scene
    from earshot.task.models import load_clap_encoder

    say("env_check: probing (clap=True)")
    env = assert_env(clap=True)
    say(env.summary())

    audio_cfg = AudioConfig()
    split_dir = find_split_dir(split, root=data_root)
    scenes_dir = find_scenes_dir(root=data_root)

    say("clip bank: {}".format(", ".join(sorted(classes))))
    clips: Dict[str, np.ndarray] = {}
    for name in sorted(set(classes)):
        path = resolve_anomaly_clip(name, None, clip_dir)
        if not path:
            raise FileNotFoundError(
                "no staged clip for {!r} under {} -- stage it before running the prior "
                "pass, the same clip_dir a matrix sweep will resolve it from".format(
                    name, clip_dir
                )
            )
        clips[name] = load_anomaly_clip(path, audio_cfg.sample_rate)

    def clip_for_class(name: str) -> np.ndarray:
        return clips[name]

    # Keyed by length, not built once: `mix_bed` refuses a mismatch by design, and
    # different classes' ESC-50 recordings are not guaranteed the same length
    # (`clap_gate.run_gate` notes exactly this about its own corpus). Built lazily so a
    # length that never actually occurs never costs a `bed_signal` call.
    beds: Dict[int, np.ndarray] = {}

    def bed_for_length(n_samples: int) -> np.ndarray:
        bed = beds.get(n_samples)
        if bed is None:
            bed = bed_signal(n_samples, audio_cfg.bed_rms)
            beds[n_samples] = bed
        return bed

    lengths = sorted({int(clip.size) for clip in clips.values()})
    if len(lengths) > 1:
        say("NOTE: staged clips have {} distinct lengths {} -- bedded per clip".format(
            len(lengths), lengths
        ))

    say("CLAP: loading")
    encoder = load_clap_encoder()
    say("CLAP: loaded")

    spec, binaural = audio_spec_parts()
    audio_sensor_spec(spec, audio_cfg, binaural)
    audio_uuid = str(spec.uuid)

    outcomes: List[SceneTourOutcome] = []
    complete_records: List[TourRecord] = []
    for label in scenes:
        say("[scene] {}".format(label))
        try:
            dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
            world = World(dataset.scene_path, list(camera_sensor_specs()) + [spec])
        except Exception as exc:  # noqa: BLE001 -- one scene must not cost the rest
            say("  WARN: {} could not load ({}) -- continuing".format(label, exc))
            outcomes.append(SceneTourOutcome(scene=label, record=None, error=str(exc)))
            continue
        try:
            try:
                world.seed_navmesh(seed)
                handle = AudioSensorHandle(
                    world.sensor_handle(audio_uuid),
                    world.observe,
                    world.random_navigable_point(),
                    uuid=audio_uuid,
                )
                record = tour_one_scene(
                    world,
                    handle,
                    encoder,
                    scene=label,
                    dataset=dataset,
                    room_of_category=ROOM_OF_ANCHOR,
                    classes=classes,
                    clip_for_class=clip_for_class,
                    bed_for_length=bed_for_length,
                    sample_rate=audio_cfg.sample_rate,
                    seed=seed,
                    leg_budget=leg_budget,
                    goal_radius=goal_radius,
                )
            except Exception as exc:  # noqa: BLE001
                say("  WARN: {} tour failed ({}) -- continuing".format(label, exc))
                outcomes.append(SceneTourOutcome(scene=label, record=None, error=str(exc)))
                continue
        finally:
            # A `World` per scene, and `n_scenes` of them across a matrix's worth of
            # scenes -- `runner.run` closes its own in a `finally` for the same reason;
            # `clap_gate.run_gate` leaves this to the garbage collector, which is a risk
            # this driver does not take across the longer scene list a matrix pass tours.
            world.close()
        outcomes.append(SceneTourOutcome(scene=label, record=record))
        say("  {}: rooms {} of {} leg(s) reached, {} observation(s){}".format(
            label,
            len(record.rooms_reached),
            len(record.legs),
            len(record.observations),
            "" if record.complete else " -- INCOMPLETE, excluded from the merged store",
        ))
        if record.complete:
            complete_records.append(record)

    if not complete_records:
        raise RuntimeError(
            "every scene's tour was incomplete or failed to load; nothing to merge. "
            "See the per-scene WARN lines above."
        )

    semantic, episodic = merge_scene_records(complete_records)
    say("merged: {} semantic row(s), {} episodic row(s), {} of {} scene(s) complete".format(
        len(semantic), len(episodic), len(complete_records), len(scenes)
    ))

    run_path = pathlib.Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    store_path = run_path / "store.json"
    if store_path.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists. One directory is one run: pass a fresh --run-dir, or "
            "--overwrite if replacing it is the intent.".format(store_path)
        )
    dump_stores(
        str(store_path),
        semantic,
        episodic,
        provenance={
            "split": split,
            "classes": list(classes),
            "seed": seed,
            "scenes_requested": list(scenes),
            "scenes_complete": [o.scene for o in outcomes if o.ok and o.record.complete],
            "scenes_failed": [o.as_dict() for o in outcomes if not o.ok],
        },
    )
    say("wrote {}".format(store_path))
    return store_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk the scripted prior pass over real HM3D scenes and dump the "
                     "store a matrix sweep will consume."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", default=".")
    parser.add_argument(
        "--scenes", required=True,
        help="space-separated scene ids -- the matrix's own class-per-scene assignment "
             "decides which scenes to tour, so there is no default",
    )
    parser.add_argument(
        "--classes", required=True,
        help="space-separated sound classes -- the bank the matrix's assignment names",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--leg-budget", type=int, default=DEFAULT_LEG_BUDGET)
    parser.add_argument("--goal-radius", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(None if argv is None else list(argv))

    run_prior_pass(
        run_dir=args.run_dir,
        scenes=tuple(args.scenes.split()),
        classes=tuple(args.classes.split()),
        split=args.split,
        data_root=args.data_root,
        seed=args.seed,
        leg_budget=args.leg_budget,
        goal_radius=args.goal_radius,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
