"""THE one module in the tree that imports ``habitat_sim`` (ADR-0013).

Everything else reaches the simulator through an injected callable, which is what keeps
ADR-0014's Mac surface most of the tree instead of a corner of it.
``tests/mac/test_layering.py`` asserts the set of ``habitat_sim`` importers is exactly
``{sim/world.py}``, and ``ruff`` (``F`` + ``E9``) is this file's only Mac-side
verification — no Mac can import it, because ``libRLRAudioPropagation.so`` is a
Linux-x64 binary and this laptop cannot even create a GL context.

**Audio-blind.** ``World`` is handed a list of sensor specs it does not interpret and
returns whatever ``get_sensor_observations()`` gives back. It does not know an
``AudioSensorSpec`` from a camera, which is what stops ``import habitat_sim`` spreading
into ``audio/`` for the sensor handle. ``audio/spec.py`` builds the audio spec;
``camera_sensor_specs()`` below builds the cameras, because only this file can
construct a habitat-sim type at all.

**Every spec goes in at construction, and that is a constraint rather than a
preference.** ``Configuration._sanitize_config`` (``habitat_sim/simulator.py:92-112``)
derives ``create_renderer``, ``requires_textures`` and ``load_semantic_mesh`` from the
agent's ``sensor_specifications`` *before* the Simulator is built, and ``add_sensor``
then refuses any modality that was absent at init (``simulator.py:265-284``). Verified
on this Mac against habitat-sim 0.3.3: a Simulator built with an empty spec list raises
``ValueError: Data for SensorType.COLOR sensor was not loaded during Simulator init``
on the next ``add_sensor``. Passing the whole list through one channel is therefore
both the audio-blind shape and the only shape that works.

The one thing that inherits a **cross-version** inference from that check: whether the
box's 2022-era ``RLRAudioPropagationUpdate`` branch accepts an ``AudioSensorSpec``
through ``sensor_specifications`` as well as through ``add_sensor``. ``Agent.__init__``
routes both through the identical ``SensorFactory.create_sensors`` call
(``habitat_sim/agent/agent.py:158-171``), and tickets 04 and 16 proved the
``add_sensor`` form on the branch. ``tests/box/test_world_box.py`` measures the
agent-config form; if it comes back red, the fallback is to add audio specs after
construction, which is one line and costs the audio-blindness this file is built around.

**The observation is one shared call.** RGB, depth and the IR come out of a single
``get_sensor_observations()``, so smoke criterion 1 — render count equals step count —
is measurable on ``n_renders``. ``step()`` deliberately does **not** render: habitat's
own ``sim.step()`` acts and renders together, which would make every step two renders
and the criterion unfalsifiable.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from earshot.audio.guard import assert_habitat_logging_pinned
from earshot.types import Pose, Xyz

# `import earshot` runs `pin_habitat_logging()`, so by the time any module in the
# package is importable the variable is already set. Assert it immediately before the
# import anyway: this is the one file where a missing pin has a consequence (invariant
# 2's log scan goes blind), and an assertion here cannot be bypassed by an entry point
# that forgot, a REPL, or an ad-hoc box script that imported this module directly.
assert_habitat_logging_pinned()

import numpy as np  # noqa: E402
import quaternion  # noqa: E402  MUST precede habitat_sim (habitat-sim issue #1813)

import habitat_sim  # noqa: E402

__all__ = [
    "World",
    "NoRouteError",
    "camera_sensor_specs",
    "yaw_from_quaternion",
    "OBJECTNAV_HM3D",
    "AgentSpec",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
]


class NoRouteError(RuntimeError):
    """The follower could not route to the target from where the agent stands.

    Its own type because the alternative reading — ``None``, meaning *arrived* — is the
    exact confusion that made the old tree's navigation unfalsifiable. The grid-A* it
    replaced found no path on roughly 92% of steps and silently fell back to
    straight-line steering, so "a waypoint was chosen" and "the agent got there" came
    apart with nothing in the code marking where. A caller that wants to re-propose
    catches this; a caller that does not gets a loud failure instead of a slow drift.
    """

# The three actions the greedy follower requires to be present
# (``habitat_sim/nav/greedy_geodesic_follower.py:22-24``). STOP is deliberately absent:
# it is a task decision, not a simulator action, so the runner terminates the episode
# rather than asking the simulator to.
MOVE_FORWARD = "move_forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"


class AgentSpec:
    """The embodiment numbers, so the call sites are named rather than positional.

    Values are habitat-lab's published ObjectNav HM3D benchmark configuration
    (``config/benchmark/nav/objectnav/objectnav_hm3d.yaml``), carried so this run stays
    comparable with the prior record. habitat-sim's own ``AgentConfiguration`` defaults
    differ — 1.5 m tall, 0.1 m radius, 10 degree turns — so accepting them would have
    silently changed the embodiment.
    """

    def __init__(
        self,
        height: float = 0.88,
        radius: float = 0.18,
        step_size_m: float = 0.25,
        turn_angle_deg: float = 30.0,
    ) -> None:
        self.height = float(height)
        self.radius = float(radius)
        self.step_size_m = float(step_size_m)
        self.turn_angle_deg = float(turn_angle_deg)


OBJECTNAV_HM3D = AgentSpec()


def camera_sensor_specs(
    *,
    width: int = 640,
    height: int = 480,
    hfov: float = 79.0,
    eye_height: float = 0.88,
) -> List[Any]:
    """The RGB and depth specs, as habitat-lab's ObjectNav HM3D benchmark sets them.

    Lives here rather than in ``task/`` because a ``CameraSensorSpec`` is a habitat-sim
    type and this is the only file allowed to name one.

    **Depth arrives raw, metric and unclipped, and that is a difference from the old
    tree rather than a setting.** ``min_depth`` 0.5 / ``max_depth`` 5.0 /
    ``normalize_depth`` are habitat-**lab** config fields, applied by its
    ``HabitatSimDepthSensor`` wrapper — none of the three exists on habitat-sim's
    ``CameraSensorSpec`` (checked against the binding, 2026-08-04; ``hasattr`` is False
    for all of them). Setting them here would have been three silently swallowed
    assignments of exactly the kind ticket 12's key validator exists to catch, which is
    why they are named in this docstring rather than in the code.

    The consequence is in our favour. habitat-lab's HM3D ObjectNav config normalises
    depth into [0, 1], and the frontier proposer's occupancy splat assumes metric range:
    under normalised depth a 3 m wall reads 0.3, the height gate marks nearly every
    endpoint occupied, and the map carves almost no free cells (measured on
    ``wcojb4TFT35`` in Run 5, ``habitat_env.py:199-207``). Raw habitat-sim depth is
    already metric, so the clean room does not have that trap. Any range clipping the
    occupancy grid wants is its own to apply, on a frame whose units are known.
    """
    specs: List[Any] = []
    for uuid, sensor_type in (
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [int(height), int(width)]
        spec.position = [0.0, float(eye_height), 0.0]
        spec.hfov = float(hfov)  # implicitly converted to Magnum Deg
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        specs.append(spec)
    return specs


def _vec(point: Xyz) -> Any:
    """``Xyz`` to the float32 array every pathfinder entry point expects."""
    return np.asarray(point.as_tuple(), dtype=np.float32)


def yaw_from_quaternion(rotation: Any) -> float:
    """Heading about the up axis, radians, from a habitat-sim quaternion.

    ``types.Pose`` stores a scalar because everything downstream — the lateral cue, the
    report's ``stopped_at_pose`` — wants a number, and ``types.py`` deliberately holds
    no bearing helper because ticket 09 found the lateral sign inverts between frames.
    This is the conversion, not a bearing: it reads the agent's own rotation and makes
    no claim about the direction to anything else.

    Carried unchanged from ``habitat_env.py:618-620``.
    """
    w, x, y, z = (
        float(rotation.w),
        float(rotation.x),
        float(rotation.y),
        float(rotation.z),
    )
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))


class World:
    """One loaded scene, its agent, its navmesh, and one shared observation call.

    A connector to an external system, which is why this is a class in a tree that is
    otherwise functions: it owns a process-level resource with a lifecycle.
    """

    def __init__(
        self,
        scene: str,
        sensor_specs: Sequence[Any],
        *,
        agent: AgentSpec = OBJECTNAV_HM3D,
        allow_sliding: bool = False,
        gpu_device_id: int = 0,
    ) -> None:
        if not sensor_specs:
            raise ValueError(
                "World needs at least one sensor spec: habitat-sim derives "
                "create_renderer from the list (simulator.py:92) and a Simulator built "
                "with none can never have a camera added afterwards"
            )
        if not os.path.exists(scene):
            raise FileNotFoundError(
                "scene mesh does not exist: {} — episodes carry a path relative to the "
                "scenes directory; resolve it with task.episodes.resolve_scene_path "
                "before constructing a World".format(scene)
            )

        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = scene
        # No scene-dataset config. ObjectNav HM3D v1 resolves `scene_id` as a plain
        # filesystem path and habitat-lab leaves `scene_dataset_config_file` at its
        # "default" — see task.episodes.resolve_scene_path for the full citation chain.
        backend.gpu_device_id = int(gpu_device_id)
        # Semantics stay off: ADR-0007 (materials permanently off), ticket 03 (HM3D's
        # v0.2 texture-based semantics appear to hand the audio context an empty mesh),
        # and the measured all-zeros semantic sensor behind every earlier result.
        # Physics off: nothing in this task moves an object. Sliding off is the
        # ObjectNav benchmark's setting, so a collision stops the agent rather than
        # sliding it along the wall.
        #
        # Set by name and RAISED on absence rather than skipped by `hasattr`. The
        # probes hedged here because they did not yet know the branch; a hedge is the
        # wrong shape now that the setting is load-bearing. `load_semantic_mesh = False`
        # is what keeps ticket 03's empty-mesh path shut, and a renamed field would
        # otherwise turn it back on in silence — the exact failure ticket 12's key
        # validator exists to catch, one layer up. All four fields verified present on
        # the binding, 2026-08-04.
        for field, value in (
            ("load_semantic_mesh", False),
            ("enable_physics", False),
            ("allow_sliding", allow_sliding),
        ):
            if not hasattr(backend, field):
                raise AttributeError(
                    "SimulatorConfiguration has no {!r} on this habitat-sim build. It "
                    "is not optional: skipping it silently would change the scene that "
                    "loads. Find the branch's new name for it and set that.".format(field)
                )
            setattr(backend, field, value)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = agent.height
        agent_cfg.radius = agent.radius
        agent_cfg.sensor_specifications = list(sensor_specs)
        agent_cfg.action_space = {
            MOVE_FORWARD: habitat_sim.agent.ActionSpec(
                MOVE_FORWARD, habitat_sim.agent.ActuationSpec(amount=agent.step_size_m)
            ),
            TURN_LEFT: habitat_sim.agent.ActionSpec(
                TURN_LEFT, habitat_sim.agent.ActuationSpec(amount=agent.turn_angle_deg)
            ),
            TURN_RIGHT: habitat_sim.agent.ActionSpec(
                TURN_RIGHT, habitat_sim.agent.ActuationSpec(amount=agent.turn_angle_deg)
            ),
        }

        self.scene = scene
        self.agent_spec = agent
        self._sim = habitat_sim.Simulator(
            habitat_sim.Configuration(backend, [agent_cfg])
        )
        self._agent_id = 0
        self.n_renders = 0
        self.n_steps = 0

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- the one shared call --------------------------------------------

    def observe(self) -> Dict[str, Any]:
        """Render every sensor once and return the raw observation dict.

        **Raw on purpose.** Ticket 16 found the audio observation is not a numpy array —
        ``getattr(obs, "shape")`` reads ``None`` — so anything wanting a shape must
        ``np.asarray`` it or walk the nesting. Coercing here would either lose that or
        force this module to know which key is audio, and it is audio-blind.

        Wrap it, do not replace it: the runner calls
        ``guarded_observe(world.observe)`` so every render is scanned for the engine
        errors that ``RLRA_*`` swallows.
        """
        observation = self._sim.get_sensor_observations()
        self.n_renders += 1
        return observation

    # -- acting ---------------------------------------------------------

    def step(self, action: str) -> bool:
        """Apply one discrete action. Returns whether it ended in a collision.

        Does **not** render. ``habitat_sim.Simulator.step`` acts and renders in one
        call; using it would make render count exactly twice step count and turn smoke
        criterion 1 into a tautology. The runner steps, then observes.
        """
        if action not in (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT):
            raise ValueError(
                "unknown action {!r}; the agent's space is {} — STOP is a task decision "
                "and never reaches the simulator".format(
                    action, (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT)
                )
            )
        collided = self._sim.get_agent(self._agent_id).act(action)
        self.n_steps += 1
        return bool(collided)

    # -- pose -----------------------------------------------------------

    def pose(self) -> Pose:
        state = self._sim.get_agent(self._agent_id).get_state()
        return Pose(
            position=Xyz.from_sequence(state.position),
            yaw_rad=yaw_from_quaternion(state.rotation),
        )

    def set_pose(
        self,
        position: Xyz,
        rotation: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Seat the agent. ``rotation`` is [x, y, z, w] — the dataset's order.

        habitat-lab's ``quaternion_from_coeff`` (``utils/geometry_utils.py:55-60``)
        reads an episode's ``start_rotation`` as [x, y, z, w] while
        ``numpy-quaternion``'s constructor takes (w, x, y, z). The reorder happens here,
        once, at the only boundary where the dataset's convention meets the
        simulator's.
        """
        state = self._sim.get_agent(self._agent_id).get_state()
        state.position = _vec(position)
        if rotation is not None:
            x, y, z, w = (float(v) for v in rotation)
            state.rotation = quaternion.quaternion(w, x, y, z)
        self._sim.get_agent(self._agent_id).set_state(state)

    def semantic_object_count(self) -> int:
        """How many semantic objects the loaded scene holds. Expected: zero.

        Corroborates by observation what ``load_semantic_mesh = False`` guarantees by
        construction — and it is what makes ticket 08's answer decisive on a box where
        the ``.semantic.glb`` files are all present. "The annotations are absent" was
        never the claim available here; "the annotations are on disk and the scene the
        simulator loaded provably does not contain them" is stronger and directly
        observable. habitat-sim says the same thing in its own log
        (``Simulator.cpp:471``, ``activeSemanticSceneID_ = 0``), but a log line is a
        fragile thing to assert on.
        """
        scene = getattr(self._sim, "semantic_scene", None)
        if scene is None:
            raise AttributeError(
                "Simulator has no semantic_scene on this habitat-sim build, so whether "
                "the scene loaded annotations cannot be read back"
            )
        return len(scene.objects)

    # -- navmesh --------------------------------------------------------

    @property
    def navmesh_loaded(self) -> bool:
        return bool(self._sim.pathfinder.is_loaded)

    def _require_navmesh(self) -> Any:
        pathfinder = self._sim.pathfinder
        if not pathfinder.is_loaded:
            raise RuntimeError(
                "no navmesh for {} — every waypoint the agent proposes is filtered "
                "through it, so a missing navmesh is a broken episode rather than a "
                "degraded one".format(self.scene)
            )
        return pathfinder

    def is_navigable(self, point: Xyz) -> bool:
        return bool(self._require_navmesh().is_navigable(_vec(point)))

    def snap_point(self, point: Xyz) -> Optional[Xyz]:
        """Nearest navigable point, or ``None`` if the navmesh has none nearby.

        habitat-sim signals failure by returning NaNs rather than raising, which reads
        as a coordinate all the way to whoever plots it. Converted to ``None`` here so
        the failure is in the type.
        """
        snapped = self._require_navmesh().snap_point(_vec(point))
        values = [float(v) for v in snapped]
        if any(math.isnan(v) for v in values):
            return None
        return Xyz.from_sequence(values)

    def geodesic_distance(self, start: Xyz, ends: Sequence[Xyz]) -> Optional[float]:
        """Shortest navigable path length to the nearest of ``ends``.

        ``None`` for unreachable rather than ``inf``: habitat-sim leaves the field at
        ``inf`` when ``find_path`` fails, and an ``inf`` that reaches an SPL denominator
        produces a number instead of an error.

        **``PathFinder`` has no ``geodesic_distance`` method** — that name belongs to
        habitat-lab's simulator wrapper (``habitat_simulator.py:528-553``), which builds
        a ``MultiGoalShortestPath``, calls ``find_path`` and reads the field off the
        result. Caught on this Mac before a box trip: the plausible one-liner
        ``self._sim.pathfinder.geodesic_distance(...)`` is an ``AttributeError`` that no
        Mac test could have reached, because nothing here can construct a navmesh.

        **Reads only the fields the box's branch actually has.** This deliberately does
        not return *which* end was nearest. ``MultiGoalShortestPath`` grew a
        ``closest_end_point_index`` field at some point after the 2022-era
        ``RLRAudioPropagationUpdate`` branch: it is present on habitat-sim 0.3.3 and
        **absent on the box**, which is what the first box run of ticket 21 found. If a
        consumer ever needs the index, the branch-safe route is matching ``path.points``
        (which both versions have) against ``ends`` — not the field, and not a
        ``getattr`` fallback that would quietly return ``-1`` on the box forever.
        """
        if not ends:
            return None
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = _vec(start)
        path.requested_ends = np.asarray(
            [end.as_tuple() for end in ends], dtype=np.float32
        )
        found = self._require_navmesh().find_path(path)
        distance = float(path.geodesic_distance)
        if not found or not math.isfinite(distance):
            return None
        return distance

    def random_navigable_point(self) -> Xyz:
        return Xyz.from_sequence(self._require_navmesh().get_random_navigable_point())

    def seed_navmesh(self, seed: int) -> None:
        """Make ``random_navigable_point`` reproducible. A red run that cannot be
        reproduced is not evidence."""
        self._require_navmesh().seed(int(seed))

    # -- steering -------------------------------------------------------

    def follower(self, goal_radius: Optional[float] = None) -> Callable[[Xyz], Optional[str]]:
        """A navmesh point-goal steerer: give it a target, get the next action.

        Returns a callable rather than the ``GreedyGeodesicFollower`` object so
        ``agent/`` can hold it without habitat-sim appearing in its type signatures —
        the injection rule ADR-0013 is built on. ``None`` means arrived (habitat-sim
        signals arrival by returning its ``stop_key``, which defaults to ``None``).

        This replaces the grid-A* the old tree ran, which was inert on the live path:
        no path was found on roughly 92% of steps, so it fell back to straight-line
        steering and looped on forced replans. ``make_greedy_follower`` plans on the
        navmesh and was what fixed locomotion in the first place.
        """
        greedy = self._sim.make_greedy_follower(self._agent_id, goal_radius)

        def next_action(target: Xyz) -> Optional[str]:
            try:
                action = greedy.next_action_along(_vec(target))
            except habitat_sim.errors.GreedyFollowerError as exc:
                raise NoRouteError(
                    "no navmesh route to {} from {}".format(target, self.pose().position)
                ) from exc
            return None if action is None else str(action)

        return next_action

    # -- sensors --------------------------------------------------------

    def sensor_handle(self, uuid: str) -> Any:
        """The live sensor object for ``uuid``, which only exists post-construction.

        ``audio/sensor.py`` needs the ``AudioSensor`` handle to set the source transform
        and to arm the guard; it gets it through here rather than by importing
        habitat-sim itself. Still audio-blind: this looks up a string.
        """
        sensors = self._sim.get_agent(self._agent_id)._sensors
        if uuid not in sensors:
            raise KeyError(
                "no sensor {!r} on the agent; present: {}. A spec that was not in the "
                "list passed to World() cannot be added afterwards "
                "(simulator.py:265-284)".format(uuid, sorted(sensors.keys()))
            )
        return sensors[uuid]
