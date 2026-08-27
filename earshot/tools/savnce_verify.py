"""Stage 8 of `savnce_bootstrap.sh`: does this env actually render audio?

Runs **in the `savnce` env, on the box**. It is the only stage that touches the real
artefact, which is why the bootstrap's verdict is this script's verdict.

The one thing worth reading before editing: **criterion 1's patch check is behavioural
here, not a grep.** Unpatched, `habitat_sim.Simulator._get_audio_observation` looks up
the literal key `"audio_sensor"`; SAVN-CE's patch makes it look up `self._spec.uuid` so
more than one audio sensor can exist. So this probe deliberately names its sensor
something else. On an unpatched build the render raises `KeyError`, which is exactly the
failure the patch exists to prevent, and a grep for the patched line would have proved
only that a file contains a string.

The rest is ordinary: imports, the GPU, and one non-silent binaural render. Silence is a
failure, not a warning, for the reason `anommxv` is in this repo's history.
"""

import argparse
import pathlib
import sys
from typing import List, Optional, Tuple

PROBE_UUID = "savnce_probe_audio"  # deliberately NOT "audio_sensor" — see the docstring
SAMPLE_RATE = 44100.0
MIN_SOURCE_GAP_M = 1.0
MAX_SOURCE_GAP_M = 5.0
PLACEMENT_TRIES = 200


def _line(status: str, message: str) -> None:
    sys.stdout.write("  {:<5} {}\n".format(status, message))


def find_scene(data_root: pathlib.Path) -> Optional[pathlib.Path]:
    """Any MP3D `.glb` under the data root. The example scene counts."""
    scenes = sorted(data_root.rglob("*.glb"))
    return scenes[0] if scenes else None


def _check_imports(failures: List[str]) -> Tuple[object, object]:
    import quaternion  # noqa: F401  must precede habitat_sim (habitat-sim issue #1813)
    import numpy

    import habitat_sim
    import habitat_sim.sensor

    layout = getattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType", None)
    if layout is None or not hasattr(layout, "Binaural"):
        failures.append("habitat_sim was built WITHOUT --audio")
    else:
        _line("OK", "habitat_sim {} is audio-capable".format(habitat_sim.__version__))
    # Recorded, not enforced: this is the pin that disagrees with ss2 (ADR-0015).
    _line("OK", "numpy {} (ss2 pins 1.23.5; this env follows SAVN-CE's 1.26.0)".format(numpy.__version__))
    return habitat_sim, numpy


def _check_torch(failures: List[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        failures.append("torch sees no CUDA device")
        return
    major, minor = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    tag = "sm_{}{}".format(major, minor)
    _line("OK", "torch {} on {} ({})".format(torch.__version__, torch.cuda.get_device_name(0), tag))
    if tag not in arch_list:
        failures.append(
            "this torch build has no {} kernels (arch_list={}) — the wheel is wrong for a V100".format(
                tag, ",".join(arch_list)
            )
        )
    else:
        _line("OK", "{} is in the wheel's arch list".format(tag))


def _check_savnce(failures: List[str]) -> None:
    for module in ("savnce", "savnce_baselines", "habitat"):
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 — the message is the deliverable
            failures.append("import {} failed: {}: {}".format(module, type(exc).__name__, exc))
        else:
            _line("OK", "import {}".format(module))


def _set_checked(spec: object, field: str, value: object, failures: List[str]) -> None:
    """`AudioSensorSpec` is bound `py::dynamic_attr`, so a typo attaches silently."""
    if not hasattr(spec, field):
        failures.append("AudioSensorSpec has no field {!r} — habitat-sim moved".format(field))
        return
    setattr(spec, field, value)


def render_audio(habitat_sim: object, numpy: object, scene: pathlib.Path, failures: List[str]) -> Optional[float]:
    """Render one binaural IR and return its peak absolute value, or None on failure.

    Shared with `savnce_probe.py`, which needs the number rather than the verdict.
    """
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene)
    backend.enable_physics = False
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()])
    )
    try:
        spec = habitat_sim.AudioSensorSpec()
        _set_checked(spec, "uuid", PROBE_UUID, failures)
        _set_checked(spec, "enableMaterials", False, failures)  # ADR-0007, and their default too
        if hasattr(spec, "acousticsConfig"):
            spec.acousticsConfig.sampleRate = SAMPLE_RATE
        else:
            failures.append("AudioSensorSpec has no acousticsConfig")
        spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        spec.channelLayout.channelCount = 2
        sim.add_sensor(spec)

        pathfinder = sim.pathfinder
        if not pathfinder.is_loaded:
            failures.append("no navmesh for {} — cannot place agent or source".format(scene.name))
            return None
        listener = pathfinder.get_random_navigable_point()
        source = None
        for _ in range(PLACEMENT_TRIES):
            candidate = pathfinder.get_random_navigable_point()
            gap = float(numpy.linalg.norm(numpy.asarray(candidate) - numpy.asarray(listener)))
            if MIN_SOURCE_GAP_M <= gap <= MAX_SOURCE_GAP_M:
                source = candidate
                break
        if source is None:
            failures.append("no navigable source point within {}-{} m of the listener".format(MIN_SOURCE_GAP_M, MAX_SOURCE_GAP_M))
            return None

        state = sim.get_agent(0).get_state()
        state.position = listener
        sim.get_agent(0).set_state(state)

        sensor = sim.get_agent(0)._sensors[PROBE_UUID]
        sensor.setAudioSourceTransform(numpy.asarray(source, dtype=numpy.float32))

        try:
            observation = sim.get_sensor_observations()[PROBE_UUID]
        except KeyError as exc:
            failures.append(
                "render raised KeyError({}) — this habitat_sim is UNPATCHED. It looks up the "
                "literal 'audio_sensor', so SAVN-CE cannot run more than one audio sensor. "
                "Re-run stage 4 of savnce_bootstrap.sh.".format(exc)
            )
            return None

        impulse = numpy.asarray(observation)
        loudest = float(numpy.abs(impulse).max()) if impulse.size else 0.0
        _line("OK", "scene {} rendered IR shape={} max_abs={:.6g}".format(scene.name, impulse.shape, loudest))
        _line("OK", "the sensor uuid was {!r}, so the multi-audio-sensor patch is EXERCISED".format(PROBE_UUID))
        if loudest <= 0.0:
            failures.append(
                "the IR is silent (max_abs=0). A zero-geometry audio context returns "
                "plausible audio; this is the failure class that invalidated anommxv."
            )
        return loudest
    finally:
        sim.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)

    failures: List[str] = []
    try:
        habitat_sim, numpy = _check_imports(failures)
        _check_torch(failures)
        _check_savnce(failures)
        scene = find_scene(pathlib.Path(args.data_root))
        if scene is None:
            failures.append("no .glb scene under {} — nothing to render".format(args.data_root))
        else:
            render_audio(habitat_sim, numpy, scene, failures)
    except Exception as exc:  # noqa: BLE001 — the traceback is the deliverable
        import traceback

        traceback.print_exc()
        failures.append("{}: {}".format(type(exc).__name__, exc))

    if failures:
        sys.stdout.write("\n  VERIFY_FAILED — {} problem(s):\n".format(len(failures)))
        for problem in failures:
            sys.stdout.write("    - {}\n".format(problem))
        return 1
    sys.stdout.write("\n  VERIFY_OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
