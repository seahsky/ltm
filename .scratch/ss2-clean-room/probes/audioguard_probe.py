#!/usr/bin/env python3
"""Ticket 12 box probe — verify the audio guard against the real binary.

    conda activate ss2
    nrun python3 .scratch/ss2-clean-room/probes/audioguard_probe.py \
        --out runs/ss2-audioguard/report.json

``audio_guard.py`` is fully unit-tested on a Mac against fakes, so what this probe adds
is the three things a fake cannot settle, each of which is currently an inference:

1. **Does the healthy path pass?** Against ticket 04's control — 392,356 verts on
   ``minival/00800-TEEsavR23oF``, non-semantic path — ``arm_audio_context`` must return
   a report, not raise.
2. **Does the guard actually fire?** A guard that has only ever passed is indistinguishable
   from a guard that cannot fail. Four negative controls, each provoking a different
   invariant on the real objects.
3. **Are the calibrated constants right?** Ticket 16 settled both from source before this
   probe ever ran — ``HABITAT_SIM_LOG_PIN`` is correct (``AudioSensor.cpp`` opens
   ``namespace esp::sensor``, so its macros resolve to ``Subsystem::sensor``), and the old
   ``DEFAULT_SEVERITY_RE = r"\\[Error\\]"`` was not merely miscalibrated but structurally
   dead: ``buildMessagePrefix`` emits no severity tag, so it could never match. It is
   replaced by ``HABITAT_LOG_PREFIX_RE``, which tests *provenance*, with severity taken
   from the stream. This probe now confirms that on the binary rather than deriving it.

It also answers the one question ``assert_no_swallowed_keys`` is currently guessing at:
whether a stock construct-and-configure leaves ``vars(spec)`` empty, or whether some
legitimate dynamic attribute has to go on the ``allowed`` list forever.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# MUST precede habitat_sim: HABITAT_SIM_LOG is read at import time, and the whole
# point of invariant 2 is that this cannot be skipped.
from audio_guard import (  # noqa: E402
    HABITAT_LOG_PREFIX_RE,
    HABITAT_SIM_LOG_PIN,
    MIN_SCENE_VERTICES,
    AudioContextError,
    apply_audio_config,
    arm_audio_context,
    assert_no_swallowed_keys,
    bound_field_names,
    capture_habitat_logs,
    pin_habitat_logging,
)

PINNED_LOG = pin_habitat_logging()

# habitat-sim issue #1813: quaternion must import first.
import quaternion  # noqa: E402,F401

REPORT: Dict[str, Any] = {"habitat_sim_log_pin": PINNED_LOG}


def banner(msg: str) -> None:
    print("\n----- {} -----".format(msg), flush=True)


def stage(name: str):
    """Every stage independently guarded — on a RED run the full blocker list is the
    deliverable, and one failure must not hide the ones behind it."""

    def wrap(fn):
        def inner(*args, **kwargs):
            banner(name)
            t0 = time.time()
            try:
                result = fn(*args, **kwargs) or {}
                result.setdefault("ok", True)
            except Exception as exc:
                result = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
                print("  FAILED: {!r}".format(exc), flush=True)
            result["elapsed_s"] = round(time.time() - t0, 3)
            REPORT[name] = result
            return result

        return inner

    return wrap


def _find_scene(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    # Canonical root is data/hm3d/scene_datasets/hm3d/<split> — verified, it is
    # what box_inventory.py counted the 100 val / 10 minival meshes from. The
    # bare data/scene_datasets form is a legacy layout, kept as a fallback.
    patterns = [
        "data/hm3d/scene_datasets/hm3d/minival/**/*.basis.glb",
        "data/hm3d/scene_datasets/hm3d/minival/**/*.glb",
        "data/hm3d/scene_datasets/hm3d/val/**/*.basis.glb",
        "data/scene_datasets/hm3d/minival/**/*.basis.glb",
        "data/scene_datasets/hm3d/minival/**/*.glb",
        "data/scene_datasets/hm3d/val/**/*.basis.glb",
    ]
    for pattern in patterns:
        hits = [p for p in glob.glob(pattern, recursive=True) if "semantic" not in os.path.basename(p)]
        if hits:
            return sorted(hits)[0]
    raise RuntimeError("no HM3D .glb found — pass --scene explicitly")


def _build_sim(scene_path: str, sample_rate: float):
    """Stand up sim + audio sensor exactly as the rebuilt runner will.

    Config goes through ``apply_audio_config`` rather than bare setattr, so this
    exercises the key validator against the real ``AudioSensorSpec`` rather than a fake.
    """
    import habitat_sim

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    for field, value in (("load_semantic_mesh", False), ("enable_physics", False)):
        if hasattr(backend_cfg, field):
            setattr(backend_cfg, field, value)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    spec = habitat_sim.AudioSensorSpec()
    apply_audio_config(
        spec,
        {
            "uuid": "audio_sensor",
            # Permanently off per ADR-0007 — this is the path the clean room runs.
            "enableMaterials": False,
            "acousticsConfig": {"sampleRate": sample_rate},
        },
    )
    spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    spec.channelLayout.channelCount = 2
    sim.add_sensor(spec)
    return sim, spec


PLACEMENT_SEED = 20260803
# A source drawn independently of the listener can land across the scene, and the guard
# asserts a non-silent IR — so an unlucky draw fails the healthy path for a reason that
# has nothing to do with the audio context. Bound the separation instead. Euclidean, not
# geodesic, deliberately: no extra API surface, and the point is only to stay in
# earshot.
SOURCE_MIN_M = 1.0
SOURCE_MAX_M = 8.0
SOURCE_DRAW_TRIES = 64


def _place(sim) -> Dict[str, Any]:
    """Seat the listener and the source, reproducibly.

    Returns the placement so a RED report says where it was standing. Both stages seed
    identically, so stage 3's negative controls run against stage 2's geometry.
    """
    import numpy as np

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    placement: Dict[str, Any] = {"seeded": False, "draws": 0, "fallback": False}
    if sim.pathfinder.is_loaded:
        # Without this the probe is not re-runnable: a RED report could not be
        # reproduced, which is exactly what a negative control needs.
        if hasattr(sim.pathfinder, "seed"):
            sim.pathfinder.seed(PLACEMENT_SEED)
            placement["seeded"] = True
        listener = sim.pathfinder.get_random_navigable_point()
        source = None
        for attempt in range(SOURCE_DRAW_TRIES):
            candidate = sim.pathfinder.get_random_navigable_point()
            separation = float(np.linalg.norm(np.asarray(candidate) - np.asarray(listener)))
            placement["draws"] = attempt + 1
            if SOURCE_MIN_M <= separation <= SOURCE_MAX_M:
                source = candidate
                placement["separation_m"] = round(separation, 3)
                break
        if source is None:
            # Small or oddly-shaped navmesh: take the last draw rather than fail here.
            # The guard's silent-IR check is then the honest report of what happened.
            source = candidate
            placement["fallback"] = True
            placement["separation_m"] = round(separation, 3)
    else:
        listener = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        source = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        placement["navmesh_loaded"] = False
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = listener
    agent.set_state(state)
    audio_sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
    placement["listener"] = [round(float(v), 3) for v in np.asarray(listener).tolist()]
    placement["source"] = [round(float(v), 3) for v in np.asarray(source).tolist()]
    return placement


# ----------------------------------------------------------------------
# stage 1 — the key validator against the real spec
# ----------------------------------------------------------------------


@stage("01_key_validator")
def probe_key_validator() -> Dict[str, Any]:
    import habitat_sim

    spec = habitat_sim.AudioSensorSpec()
    info: Dict[str, Any] = {
        "spec_fields": sorted(bound_field_names(spec)),
        "acoustics_fields": sorted(bound_field_names(spec.acousticsConfig)),
        "channel_layout_fields": sorted(bound_field_names(spec.channelLayout)),
    }
    print("  spec fields: {}".format(", ".join(info["spec_fields"])), flush=True)
    print("  acoustics fields: {}".format(len(info["acoustics_fields"])), flush=True)

    # THE OPEN QUESTION: does a stock construct leave __dict__ empty? If not, whatever
    # is in there has to live on assert_no_swallowed_keys(allowed=...) permanently.
    info["vars_after_construct"] = sorted(vars(spec))
    print("  vars(spec) after bare construct: {}".format(info["vars_after_construct"] or "empty"), flush=True)

    apply_audio_config(spec, {"uuid": "audio_sensor", "enableMaterials": False})
    info["vars_after_configure"] = sorted(vars(spec))
    print("  vars(spec) after configure: {}".format(info["vars_after_configure"] or "empty"), flush=True)
    info["allowed_dynamic_attrs_needed"] = info["vars_after_configure"]

    # Negative control: the branch's own rename (ticket 11) must be rejected, not swallowed.
    try:
        apply_audio_config(habitat_sim.AudioSensorSpec(), {"irTime": 4.0})
        info["rejects_renamed_key"] = False
        print("  *** irTime was NOT rejected — the validator is not working", flush=True)
    except AudioContextError as exc:
        info["rejects_renamed_key"] = True
        info["rejection_message"] = str(exc)[:300]
        print("  irTime rejected OK", flush=True)

    # Negative control: a key attached behind the validator's back is still caught.
    dirty = habitat_sim.AudioSensorSpec()
    dirty.definitelyNotAField = 1.0
    try:
        assert_no_swallowed_keys(dirty)
        info["detects_swallowed_key"] = False
        print("  *** swallowed key NOT detected — vars(spec) is not the right probe", flush=True)
    except AudioContextError:
        info["detects_swallowed_key"] = True
        print("  swallowed key detected OK", flush=True)

    if not (info["rejects_renamed_key"] and info["detects_swallowed_key"]):
        raise RuntimeError("invariant 3 does not hold on the real spec")
    return info


# ----------------------------------------------------------------------
# stage 2 — the healthy path
# ----------------------------------------------------------------------


@stage("02_healthy_path")
def probe_healthy_path(scene: Optional[str], sample_rate: float) -> Dict[str, Any]:
    scene_path = _find_scene(scene)
    print("  scene: {}".format(scene_path), flush=True)
    sim, spec = _build_sim(scene_path, sample_rate)
    info: Dict[str, Any] = {"scene": scene_path}
    try:
        info["placement"] = _place(sim)
        print("  placement: {}".format(info["placement"]), flush=True)
        audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]

        t0 = time.time()
        report = arm_audio_context(
            audio_sensor, lambda: sim.get_sensor_observations()["audio_sensor"]
        )
        info["guard_total_s"] = round(time.time() - t0, 4)
        info["report"] = report.as_dict()
        for key, value in report.as_dict().items():
            print("  {:<20} {}".format(key, value), flush=True)

        # The overhead claim behind "OBJ dump every episode" is currently an estimate
        # (one ~25 MB write against ~0.58 s x 500 steps). Price it.
        t0 = time.time()
        audio_sensor.writeSceneMeshOBJ(os.path.join("/tmp", "audioguard-cost.obj"))
        info["obj_write_s"] = round(time.time() - t0, 4)
        try:
            info["obj_bytes"] = os.path.getsize("/tmp/audioguard-cost.obj")
            os.unlink("/tmp/audioguard-cost.obj")
        except OSError:
            pass
        print("  obj write cost: {} s / {} bytes".format(info["obj_write_s"], info.get("obj_bytes")), flush=True)

        # Ticket 04's control, for a direct comparison rather than a recollection.
        info["matches_ticket_04_control"] = report.n_vertices == 392356
        info["vars_after_render"] = sorted(vars(spec))
    finally:
        sim.close()
    return info


# ----------------------------------------------------------------------
# stage 3 — negative controls: prove the guard can fail
# ----------------------------------------------------------------------


@stage("03_negative_controls")
def probe_negative_controls(scene: Optional[str], sample_rate: float) -> Dict[str, Any]:
    scene_path = _find_scene(scene)
    sim, _spec = _build_sim(scene_path, sample_rate)
    info: Dict[str, Any] = {}
    try:
        info["placement"] = _place(sim)
        audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
        render = lambda: sim.get_sensor_observations()["audio_sensor"]  # noqa: E731

        # (a) An impossible floor must raise. This proves the assertion path is live on
        # the real sensor — it does NOT prove a genuinely empty mesh is detectable,
        # which would need a scene that actually produces one.
        try:
            arm_audio_context(audio_sensor, render, min_vertices=10 ** 9)
            info["floor_fires"] = False
            print("  *** impossible floor did NOT fire", flush=True)
        except AudioContextError as exc:
            info["floor_fires"] = True
            info["floor_message"] = str(exc)[:300]
            print("  impossible floor fires OK", flush=True)

        # (b) Provoke a real ESP_ERROR and confirm two things the source predicts but
        # only the binary can settle: that it lands on fd 2 (Corrade routes Error to
        # std::cerr) and that HABITAT_LOG_PREFIX_RE matches the rendered prefix.
        # `setListenerHRTF` is the reliable one — AudioSensor.cpp:181 logs
        # ESP_ERROR unconditionally on a bad path. The other two are recorded for
        # contrast: setAudioMaterialsJSON only ESP_DEBUGs the path it stored (:169), so
        # it is EXPECTED on fd 1, and it is the cleanest demonstration that the stream
        # split is real rather than an artefact of one call.
        provocations = [
            ("setListenerHRTF", lambda: audio_sensor.setListenerHRTF("/nonexistent/hrtf.wav")),
            ("setAudioMaterialsJSON", lambda: audio_sensor.setAudioMaterialsJSON("/nonexistent/m.json")),
            ("writeSceneMeshOBJ", lambda: audio_sensor.writeSceneMeshOBJ("/nonexistent-dir/x.obj")),
        ]
        samples: List[Dict[str, Any]] = []
        for name, call in provocations:
            raised = None
            with capture_habitat_logs() as captured:
                try:
                    call()
                except Exception as exc:  # a raising binding is a finding, not a failure
                    raised = repr(exc)
            on_stderr = [ln.strip() for ln in captured.stderr.splitlines()
                         if HABITAT_LOG_PREFIX_RE.search(ln)]
            on_stdout = [ln.strip() for ln in captured.stdout.splitlines()
                         if HABITAT_LOG_PREFIX_RE.search(ln)]
            samples.append({
                "provocation": name,
                "raised": raised,
                "stdout_chars": len(captured.stdout),
                "stderr_chars": len(captured.stderr),
                "prefix_re_matched_on_stderr": bool(on_stderr),
                "prefix_re_matched_on_stdout": bool(on_stdout),
                "stderr_matched_lines": on_stderr[:3],
                "stdout_matched_lines": on_stdout[:3],
                # Both tails, so a pattern mismatch is fixable from the report rather
                # than costing a second box trip.
                "stdout_tail": captured.stdout[-600:],
                "stderr_tail": captured.stderr[-600:],
            })
            print("  {:<24} out {:>5}c / err {:>5}c, prefix on stderr: {}".format(
                name, len(captured.stdout), len(captured.stderr), bool(on_stderr)), flush=True)
        info["provocations"] = samples
        # The claim under test: a real ESP_ERROR reaches fd 2 AND the prefix pattern
        # matches it. If this is False the generic arm of invariant 2 is blind again and
        # only FATAL_LOG_SUBSTRINGS is load-bearing.
        info["prefix_re_validated_on_stderr"] = any(
            s["prefix_re_matched_on_stderr"] for s in samples
        )
        if not info["prefix_re_validated_on_stderr"]:
            print("  *** no habitat-prefixed line reached fd 2 — read stderr_tail and fix "
                  "HABITAT_LOG_PREFIX_RE; invariant 2's generic arm is blind", flush=True)

        # (c) The canary. Turning the capture off entirely is not possible here, so
        # instead record which stream carries it — the whole reason stage 2 could have
        # failed spuriously is that this is fd 1, not fd 2.
        with capture_habitat_logs() as captured:
            render()
        canary = ("Vertex count", "[Audio]")
        info["canary_seen_on_second_render"] = any(m in captured.text for m in canary)
        info["canary_on_stdout"] = any(m in captured.stdout for m in canary)
        info["canary_on_stderr"] = any(m in captured.stderr for m in canary)
        info["second_render_stdout_chars"] = len(captured.stdout)
        info["second_render_stderr_chars"] = len(captured.stderr)
        # Expected False: the mesh uploads once per context (newInitialization_), so the
        # "Vertex count" line is a FIRST-render artefact. Recorded to make that explicit
        # — it is why the guard owns the first render rather than running later.
        print("  canary on 2nd render: {} (expected False — mesh uploads once)".format(
            info["canary_seen_on_second_render"]), flush=True)
    finally:
        sim.close()

    if not info.get("floor_fires"):
        raise RuntimeError("the guard did not fire under a forced failure")
    return info


# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=None, help="explicit HM3D .glb path")
    parser.add_argument("--sample-rate", type=float, default=44100.0)
    parser.add_argument("--out", default="runs/ss2-audioguard/report.json")
    args = parser.parse_args()

    print("HABITAT_SIM_LOG pinned to {!r}".format(PINNED_LOG), flush=True)
    print("MIN_SCENE_VERTICES = {}".format(MIN_SCENE_VERTICES), flush=True)
    print("HABITAT_SIM_LOG_PIN = {!r}".format(HABITAT_SIM_LOG_PIN), flush=True)
    print("HABITAT_LOG_PREFIX_RE = {!r}".format(HABITAT_LOG_PREFIX_RE.pattern), flush=True)
    REPORT["habitat_log_prefix_re"] = HABITAT_LOG_PREFIX_RE.pattern
    REPORT["placement_seed"] = PLACEMENT_SEED

    probe_key_validator()
    probe_healthy_path(args.scene, args.sample_rate)
    probe_negative_controls(args.scene, args.sample_rate)

    stages = [k for k in REPORT if k[0].isdigit()]
    REPORT["all_ok"] = all(REPORT[k].get("ok") for k in stages)
    REPORT["verdict"] = "GREEN" if REPORT["all_ok"] else "RED"
    REPORT["blockers"] = [k for k in stages if not REPORT[k].get("ok")]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(REPORT, fh, indent=2, default=str)

    banner("VERDICT: {}".format(REPORT["verdict"]))
    if REPORT["blockers"]:
        for name in REPORT["blockers"]:
            print("  BLOCKER {}: {}".format(name, REPORT[name].get("error")), flush=True)
    print("report written to {}".format(args.out), flush=True)
    return 0 if REPORT["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
