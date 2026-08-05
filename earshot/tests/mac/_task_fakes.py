"""Fakes for the wiring layer: a world, an audio sensor handle, and a scene.

**A fake licenses nothing about the binding** (ADR-0014). These stand in for the two
objects ``task/runner.py`` duck-types — ``sim.world.World`` and
``audio.sensor.AudioSensorHandle`` — so the episode loop, which is the highest-risk
logic in the tree, can be driven end to end on a machine that cannot load habitat-sim.
What they cannot show is whether the real follower, the real navmesh or the real
renderer behave as assumed; that is the smoke's job (ticket 26), and every method below
carries a citation for the behaviour it imitates.

The one that matters most is ``FakeAudioSensorHandle``'s IR. It is a single impulse whose
amplitude falls with distance **and with facing**, and whose two ears are split by the
source's **agent-frame** lateral offset, derived from ``agent.occupancy.right_xz`` — the
same axis ``audio.lateral.bearing_lateral_sign`` derives independently. That makes the
greedy climb's two inputs real functions of the pose rather than scripted values, which
is what lets a test assert the agent *found* the source instead of asserting that it was
told where the source was.

The facing term is not decoration, and getting it wrong is instructive. With an
omnidirectional gain, turning in place cannot raise the measured level, so
``realizable_investigate_step``'s ``rising -> forward`` branch can never re-arm after a
stall and the agent turns for ever. A real head shadows the far ear and attenuates a
source behind it, which is what makes "turn toward the louder half-plane and try again"
a rule that terminates. Task spec §4.1 is explicit that the rotation-versus-translation
conflation this creates is **instrumented, not fixed** — the per-step ``action`` is what
separates a rotation-driven rise from a translation-driven one after the fact.

It is emphatically **not** a claim about SoundSpaces: a real IR is a scene-dependent
decay tens of thousands of samples long (ticket 06 measured ``[2, 72300]``), the
directivity is an HRTF rather than a cosine, and whether the live cue is agent-frame at
all is ticket 22's box test, not this file's.
"""

import math

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.occupancy import bearing_rel, forward_xz, right_xz
from earshot.audio.guard import AudioContextReport, StepGuardReport
from earshot.task.dataset import AnomalyEpisode, SourcePlacement
from earshot.task.episodes import Episode, ObjectGoal, ViewPoint
from earshot.types import NoRouteError, Pose, Xyz

# The embodiment, matching `sim.world.OBJECTNAV_HM3D` — habitat-lab's published ObjectNav
# HM3D benchmark configuration, which `AgentSpec` carries so the run stays comparable.
STEP_SIZE_M = 0.25
TURN_RAD = math.radians(30.0)

# The greedy follower's turn threshold. habitat-sim's own follower plans on the navmesh;
# this one steers by bearing, which is the same *behaviour* at the seam the runner sees
# (an action, or `None` for arrived, or NoRouteError) and none of the same mechanism.
FOLLOWER_TURN_RAD = math.radians(15.0)
FOLLOWER_ARRIVE_M = 0.3


class FakeWorld:
    """An empty room with a navmesh that accepts everything, and one moving agent.

    Every method the runner reaches for, and nothing else. ``n_renders`` is real, because
    smoke criterion 1 is measured on it and a fake counter that did not increment on
    every observation would make the criterion untestable here.
    """

    def __init__(self, start=Xyz(0.0, 0.0, 0.0), yaw=0.0, blocked=False, wall=None):
        self._pose = Pose(position=start, yaw_rad=float(yaw))
        self.n_renders = 0
        self.n_steps = 0
        self.closed = False
        # `blocked` makes `snap_point` return None for everything, which is the
        # off-navmesh case `reachability.assert_pool` turns into an EmptyPoolError.
        self.blocked = bool(blocked)
        # `wall` is a predicate on the destination position: True means a `move_forward`
        # that would land there does not, and `step` returns habitat's collision flag.
        # The room is empty without one, which is what left the runner's use of that flag
        # unpinned until ticket 26 — every fake forward moved, so a test could not tell
        # whether the runner passed the flag on or dropped it.
        #
        # This blocks hard where habitat SLIDES along the surface. The strict case is the
        # one worth faking: it is what the flag is for (a forward that bought nothing) and
        # a fake that slid would need a wall normal, which is geometry this room does not
        # have. `tests/box/test_world_box.py` owns the real contact behaviour.
        self.wall = wall
        self.draws = []

    # -- observing -------------------------------------------------------

    def observe(self):
        """RGB, depth and (in the handle) the IR from one shared call — ticket 21.

        ``depth`` is ``None`` on purpose. ``FrontierProposer.observe`` skips the splat
        for a missing frame, so the occupancy grid stays all-unknown, ``frontier_cells``
        finds nothing and the compass fan answers every decision step. That makes the
        SEARCH trajectory deterministic without scripting it: a fabricated depth frame
        would be a claim about geometry this fake does not have.
        """
        self.n_renders += 1
        return {"rgb": None, "depth": None}

    # -- acting ----------------------------------------------------------

    def step(self, action):
        """Apply one discrete action, in ``agent/occupancy``'s frame.

        Habitat's forward is ``-z`` at zero yaw and a positive rotation about ``+y``
        turns left (``occupancy.bearing_rel``), so this moves along ``forward_xz(yaw)``
        and ``turn_left`` **adds**. An inverted fake here would hide exactly the defect
        ticket 23 found in the tree it replaced.
        """
        position, yaw = self._pose.position, self._pose.yaw_rad
        collided = False
        if action == "move_forward":
            dx, dz = forward_xz(yaw)
            destination = Xyz(
                position.x + dx * STEP_SIZE_M, position.y, position.z + dz * STEP_SIZE_M
            )
            if self.wall is not None and self.wall(destination):
                collided = True
            else:
                position = destination
        elif action == "turn_left":
            yaw += TURN_RAD
        elif action == "turn_right":
            yaw -= TURN_RAD
        else:
            raise ValueError("unknown action {!r} — STOP never reaches the simulator".format(action))
        self._pose = Pose(position=position, yaw_rad=yaw)
        self.n_steps += 1
        return collided

    # -- pose ------------------------------------------------------------

    def pose(self):
        return self._pose

    def set_pose(self, position, rotation=None):
        yaw = self._pose.yaw_rad
        if rotation is not None:
            # [x, y, z, w], the dataset's order (`episodes.py`'s citation chain).
            x, y, z, w = (float(v) for v in rotation)
            yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
        self._pose = Pose(position=position, yaw_rad=yaw)

    # -- navmesh ---------------------------------------------------------

    def snap_point(self, point):
        return None if self.blocked else point

    def geodesic_distance(self, start, ends):
        """Straight-line ``xz`` distance to the nearest end. An empty room has no detours.

        ``None`` for no ends, which is ticket 21's boundary conversion — habitat-sim
        leaves the field at ``inf`` and an ``inf`` that reaches an SPL denominator
        produces a number instead of an error.
        """
        if not ends:
            return None
        return min(start.horizontal_distance_to(end) for end in ends)

    def random_navigable_point(self):
        """Deterministic draws on a widening ring, so a sweep gets a spread of distances.

        Seeded implicitly by the call count rather than by an RNG: `world.seed_navmesh`
        exists precisely so a red run can be reproduced, and a fake that drew randomly
        would make a failing calibration test irreproducible.
        """
        index = len(self.draws)
        angle = index * 0.7
        radius = 0.5 + 0.5 * index
        point = Xyz(math.cos(angle) * radius, 0.0, math.sin(angle) * radius)
        self.draws.append(point)
        return point

    def seed_navmesh(self, seed):
        self.seed = int(seed)

    # -- sensors ---------------------------------------------------------

    def sensor_handle(self, uuid):
        """The live sensor object, which only exists post-construction.

        ``run_episode`` never asks for it — the handle is injected — but ``run()`` does,
        so the fake carries it and ``test_task_runner`` pins the two APIs against each
        other. A fake that published only what one function happened to call would drift
        from the real class in the direction nothing checks.
        """
        return object()

    # -- steering --------------------------------------------------------

    def follower(self, goal_radius=None):
        """A bearing-following steerer with the seam's three answers.

        ``None`` means arrived (habitat-sim signals it with its ``stop_key``), an action
        means keep going, and ``NoRouteError`` means the target is unreachable — which
        this raises for a target more than 50 m away, so the runner's re-propose path is
        exercised rather than merely present.
        """

        def next_action(target):
            pose = self._pose
            dx, dz = target.x - pose.position.x, target.z - pose.position.z
            distance = math.hypot(dx, dz)
            if distance > 50.0:
                raise NoRouteError("no navmesh route to {}".format(target))
            if distance <= FOLLOWER_ARRIVE_M:
                return None
            bearing = bearing_rel(pose.yaw_rad, dx, dz)
            if abs(bearing) > FOLLOWER_TURN_RAD:
                return "turn_left" if bearing > 0 else "turn_right"
            return "move_forward"

        return next_action

    # -- lifecycle -------------------------------------------------------

    def close(self):
        self.closed = True


class FakeAudioSensorHandle:
    """One impulse per ear, scaled by distance and by the agent-frame lateral offset.

    See the module docstring for what this does and does not license. ``report`` is a
    default ``AudioContextReport`` so the audit record's nesting is exercised; its
    numbers are zeros and mean nothing.
    """

    IR_LENGTH = 64

    def __init__(self, world, source, gain=0.5, visible=True):
        self.world = world
        self.source = source
        self.gain = float(gain)
        self.visible = visible
        self.report = AudioContextReport()
        self.n_source_moves = 0
        self.n_observations = 0

    def set_source(self, source):
        self.source = source
        self.n_source_moves += 1

    def observe(self):
        self.n_observations += 1
        return self.world.observe(), StepGuardReport()

    def audio_of(self, observation):
        del observation
        pose = self.world.pose()
        dx = self.source.x - pose.position.x
        dz = self.source.z - pose.position.z
        distance = math.hypot(dx, dz)
        unit_x, unit_z = dx / max(distance, 1e-9), dz / max(distance, 1e-9)
        forward_x, forward_z = forward_xz(pose.yaw_rad)
        right_x, right_z = right_xz(pose.yaw_rad)
        facing = unit_x * forward_x + unit_z * forward_z  # +1 dead ahead, -1 behind
        lateral = unit_x * right_x + unit_z * right_z  # +1 hard right
        amplitude = self.gain * (1.0 + 0.5 * facing) / (1.0 + distance)
        # The ear split preserves the pair's total power, so the ONLY things that change
        # the measured level are distance and facing. An unnormalised split would make a
        # source abeam read louder than the same source dead ahead, and the climb would
        # be following an artefact of the fake.
        scale = 1.0 / math.sqrt(1.0 + 0.25 * lateral * lateral)
        impulse = np.zeros((2, self.IR_LENGTH), dtype=np.float32)
        impulse[0, 0] = amplitude * (1.0 - 0.5 * lateral) * scale  # left
        impulse[1, 0] = amplitude * (1.0 + 0.5 * lateral) * scale  # right
        return impulse

    def source_is_visible(self):
        return self.visible


def make_view_point(position):
    return ViewPoint(position=position, rotation=(0.0, 0.0, 0.0, 1.0))


def make_goal(position, view_points=None, category=None, object_id=None):
    points = [position] if view_points is None else list(view_points)
    return ObjectGoal(
        position=position,
        view_points=tuple(make_view_point(p) for p in points),
        object_id=object_id,
        object_category=category,
    )


def quaternion_for_yaw(yaw_rad):
    """``[x, y, z, w]`` for a rotation about +y — the ORDER THE DATASET USES.

    habitat-lab's ``quaternion_from_coeff`` reads ``start_rotation`` as [x, y, z, w]
    while ``numpy-quaternion``'s constructor takes (w, x, y, z), and ``World.set_pose``
    is the one boundary where the two conventions meet (``episodes.py``'s citation
    chain). Getting it backwards points the agent somewhere plausible and wrong.
    """
    return [0.0, math.sin(float(yaw_rad) / 2.0), 0.0, math.cos(float(yaw_rad) / 2.0)]


def make_episode(
    *,
    episode_id="0",
    index=0,
    category="chair",
    start=Xyz(0.0, 0.0, 0.0),
    start_yaw=0.0,
    goals=None,
    scene_label="FAKE",
):
    """One ObjectNav episode. ``start_yaw`` matters more than it looks.

    ``run_episode`` seats the agent from ``start_position`` / ``start_rotation`` before
    the first step, so a ``FakeWorld`` constructed at some other yaw is silently
    overwritten — which is correct (the episode owns the start pose) and is exactly the
    kind of thing that reads as a broken cue when a test sets the world's yaw instead.
    """
    return Episode(
        episode_id=episode_id,
        index=index,
        scene_id="hm3d/val/00000-{0}/{0}.basis.glb".format(scene_label),
        scene_label=scene_label,
        scene_path="/nonexistent/{}.basis.glb".format(scene_label),
        object_category=category,
        start_position=start,
        start_rotation=tuple(quaternion_for_yaw(start_yaw)),
        goals=tuple(goals if goals is not None else (make_goal(Xyz(0.0, 0.0, -9.0)),)),
        info={},
    )


def make_anomaly_episode(
    *,
    source=Xyz(0.0, 0.0, -5.0),
    anomaly_object="sofa",
    t_anom=2,
    episode=None,
    anomaly_class="alarm",
):
    """An episode with its source already placed, bypassing the builder.

    The builder's rules are ``test_task_dataset.py``'s subject; a runner test that had to
    satisfy them would be testing two things and diagnosing neither.
    """
    return AnomalyEpisode(
        episode=episode if episode is not None else make_episode(),
        source=SourcePlacement(
            position=source,
            anomaly_object=anomaly_object,
            object_id="7",
            separation_m=4.0,
            height_difference_m=0.0,
            height_difference_to_start_m=0.0,
            same_category=False,
        ),
        anomaly_class=anomaly_class,
        t_anom=int(t_anom),
    )
