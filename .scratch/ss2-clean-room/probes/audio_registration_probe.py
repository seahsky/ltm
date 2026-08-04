"""Which registration form gives habitat-sim's hardcoded ``"audio_sensor"`` a key.

Ticket 25's first box run reached the scene, constructed the ``AudioSensor``, seated the
source, created the audio context — and died on the **first render**:

    simulator.py:765   audio_sensor = self._agent._sensors["audio_sensor"]
    KeyError: 'audio_sensor'

One statement earlier, ``World.sensor_handle(uuid)`` had looked up
``sim.get_agent(0)._sensors`` and *found* the sensor: that method raises with a listing
of the keys it does have, and it did not raise. So two dicts that ought to be the same
dict disagree, and exactly one of these is true:

  (a) they are the same dict and its key is not ``"audio_sensor"`` — the uuid is being
      rewritten somewhere on the agent-config path, in which case the fix is to stop
      reading the uuid back and pin it, because habitat's own lookup is a hardcoded
      string and no read-back can satisfy it; or
  (b) ``wrapper._agent`` is not ``get_agent(0)`` — a stale agent object holding an older
      sensor dict, in which case the fix is the registration form.

Both are cheap to fix and they are different fixes, which is why this measures rather
than shotgunning the pair.

**The two forms are not equally attested.** Every probe on this box that produced a real
IR — ``oneenv_probe`` (ticket 17), ``rendercost_probe`` (ticket 06), ``vram_probe``
(ticket 15) — attached audio with ``sim.add_sensor(spec)`` *after* construction.
``sim/world.py`` is the first thing to pass an ``AudioSensorSpec`` through
``agent_cfg.sensor_specifications``, and its own docstring flagged that as the one
cross-version inference it inherited, named ``tests/box/test_world_box.py`` as the
measurement, and named the fallback in advance. This is that measurement, run directly
rather than through the suite so it prints the internals either verdict needs.

No ``earshot`` import on purpose: the question is about habitat-sim, and a red here that
came through our own stack would be one more layer to rule out.

    python .scratch/ss2-clean-room/probes/audio_registration_probe.py [scene.glb]
"""

import inspect
import os
import sys
import traceback

# Before `import habitat_sim`, which reads these at import time. Quiet, not the tree's
# `Sensor,Assets=Debug` pin: this probe's output is the Python-side dicts, and the
# engine's per-asset chatter buries them.
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"

import numpy as np  # noqa: E402
import habitat_sim  # noqa: E402
from habitat_sim.simulator import Sensor as SensorWrapper  # noqa: E402

# The scene ticket 25's run picked, so a red here is the same red.
DEFAULT_SCENE = (
    "./data/hm3d/scene_datasets/hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
)


def keys_of(sensors):
    """The uuids in an agent's sensor collection, whatever type it turns out to be.

    ``Agent._sensors`` raising ``KeyError`` says "Python dict" and the tree assumes so
    in two places. Assuming it *here* would turn a surprising type into an
    ``AttributeError`` traceback instead of the answer.
    """
    try:
        return sorted(sensors.keys())
    except AttributeError:
        try:
            return sorted(sensors)
        except TypeError:
            return "<{} — neither .keys() nor iterable>".format(type(sensors).__name__)


def cameras():
    """One small colour sensor.

    Not zero: ``Configuration._sanitize_config`` derives ``create_renderer`` from the
    spec list, so an agent with no visual sensor is a different simulator, and the
    difference is not the one under test. Small, because nothing here reads a pixel.
    """
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "rgb"
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.resolution = [64, 64]
    return [spec]


def try_form(scene, in_agent_config):
    """Build one Simulator, register audio the given way, print both dicts, render."""
    label = "sensor_specifications" if in_agent_config else "sim.add_sensor"
    print("\n=============== {} ===============".format(label))

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.load_semantic_mesh = False
    backend.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    spec = habitat_sim.AudioSensorSpec()
    specs = cameras()
    if in_agent_config:
        specs.append(spec)
    agent_cfg.sensor_specifications = specs
    # Read BEFORE construction as well as after: hypothesis (a) is a rewrite, and a
    # rewrite is only visible as a difference between these two prints.
    print("spec.uuid pre-build :", repr(spec.uuid))

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    try:
        if not in_agent_config:
            sim.add_sensor(spec)
        agent = sim.get_agent(0)
        print("spec.uuid post-build:", repr(spec.uuid))
        print("agent._sensors      :", type(agent._sensors).__name__,
              keys_of(agent._sensors))

        # The wrapper dict habitat iterates in `get_sensor_observations`. Name-mangled,
        # so reach for it explicitly and say so if the attribute has moved rather than
        # silently skipping the half of the answer that lives in it.
        wrappers = getattr(sim, "_Simulator__sensors", None)
        if wrappers is None:
            print("wrapper dict        : ABSENT — sensor-ish attributes on Simulator:",
                  [k for k in vars(sim) if "sensor" in k.lower()])
        else:
            print("wrapper dict        :", keys_of(wrappers[0]))
            for uuid, wrapper in wrappers[0].items():
                own = getattr(wrapper, "_agent", None)
                print("   {!r}: wrapper._agent is get_agent(0) -> {} ; its _sensors -> {}"
                      .format(uuid, own is agent,
                              keys_of(own._sensors) if own is not None else "<no _agent>"))

        # A source must be seated before the render: an unplaced source renders a silent
        # IR, and a silent IR is a second failure mode competing with the one under test.
        source = sim.pathfinder.get_random_navigable_point()
        agent._sensors[spec.uuid].setAudioSourceTransform(
            np.asarray(source, dtype=np.float32)
        )
        observation = sim.get_sensor_observations()
        ir = np.asarray(observation[spec.uuid])
        print("RENDER GREEN        : keys={} ir.shape={} peak={:.4g}".format(
            sorted(observation), ir.shape,
            float(np.max(np.abs(ir))) if ir.size else 0.0))
    except Exception:
        # Caught, not raised: the second form is the interesting one when the first is
        # red, and an uncaught first red would never reach it.
        print("RENDER RED          :")
        traceback.print_exc()
    finally:
        sim.close()


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE
    print("habitat_sim", habitat_sim.__version__)
    print("file       ", habitat_sim.__file__)
    print("scene      ", scene)
    print("AudioSensorSpec().uuid default:", repr(habitat_sim.AudioSensorSpec().uuid))

    # The two methods the traceback names. Printed rather than described: the claim
    # "habitat hardcodes the string" is load-bearing for the fix, and the source is the
    # only thing that settles it on this branch.
    for fn in (SensorWrapper.__init__, SensorWrapper._get_audio_observation):
        lines, first = inspect.getsourcelines(fn)
        print("\n--- {} (simulator.py:{}) ---".format(fn.__name__, first))
        print("".join(lines).rstrip())

    try_form(scene, in_agent_config=True)
    try_form(scene, in_agent_config=False)


if __name__ == "__main__":
    main()
