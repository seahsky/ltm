#!/usr/bin/env python3
"""Ticket 04 one-env probe — introspects the audio build and everything layered on it.

Run by ``oneenv_gate.sh`` inside the freshly built ``ss2`` env. Never imports
anything from ``embodied_memory/`` or ``dialogue_memory/``: the whole point of the
clean room is that this result does not depend on the trees being deleted.

Every stage is independently guarded, because on a RED run the *complete* blocker
list is the deliverable — one failing stage must not hide the four behind it.

Emits a human-readable log on stdout and a JSON report to ``--out``.

Python 3.9 (the SoundSpaces pin), stdlib + numpy only at import time.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# ``import quaternion`` MUST precede habitat_sim (habitat-sim issue #1813).
# Done here, at module scope, before anything else can pull habitat_sim in.
try:
    import quaternion  # noqa: F401
    _QUATERNION_OK = True
    _QUATERNION_ERR = None
except Exception as exc:  # pragma: no cover - environment probe
    _QUATERNION_OK = False
    _QUATERNION_ERR = repr(exc)


REPORT: Dict[str, Any] = {}


def banner(msg: str) -> None:
    print("\n----- {} -----".format(msg), flush=True)


def stage(name: str):
    """Decorator: run a probe stage, capture its dict, never let it abort the run."""

    def wrap(fn):
        def inner(*args, **kwargs):
            banner(name)
            t0 = time.time()
            try:
                result = fn(*args, **kwargs) or {}
                result.setdefault("ok", True)
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
                print("  FAILED: {}".format(exc), flush=True)
            result["elapsed_s"] = round(time.time() - t0, 3)
            REPORT[name] = result
            return result

        return inner

    return wrap


def field_dump(obj: Any) -> Dict[str, Any]:
    """Enumerate an object's real (C++-bound) fields and their values.

    ``dir()`` on a freshly constructed pybind object lists exactly the bound
    fields, so this doubles as the authoritative key list a config validator
    needs. Values are repr'd because several are opaque pybind enums.
    """
    out: Dict[str, Any] = {}
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            out[name] = "<unreadable: {!r}>".format(exc)
            continue
        if callable(value):
            continue
        out[name] = repr(value)
    return out


def method_dump(obj_or_cls: Any) -> List[str]:
    """Public callables — used to prove which habitat-sim branch generation this is."""
    names = []
    for name in sorted(dir(obj_or_cls)):
        if name.startswith("_"):
            continue
        try:
            if callable(getattr(obj_or_cls, name)):
                names.append(name)
        except Exception:
            continue
    return names


# ----------------------------------------------------------------------
# stage 0 — interpreter and pins
# ----------------------------------------------------------------------


@stage("00_interpreter")
def probe_interpreter() -> Dict[str, Any]:
    import numpy

    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "quaternion_import_ok": _QUATERNION_OK,
        "quaternion_error": _QUATERNION_ERR,
    }
    # numpy 2.x breaks the 2022-era tree; a torch/transformers install silently
    # upgrading it is the single most likely way this env rots after the build.
    major = int(numpy.__version__.split(".")[0])
    minor = int(numpy.__version__.split(".")[1])
    info["numpy_pin_held"] = (major == 1 and minor < 24)
    for k, v in info.items():
        print("  {:<24} {}".format(k, v), flush=True)
    if not info["numpy_pin_held"]:
        print("  *** numpy pin BROKEN — expected 1.x < 1.24", flush=True)
    if not _QUATERNION_OK:
        raise RuntimeError("numpy-quaternion missing: {}".format(_QUATERNION_ERR))
    return info


# ----------------------------------------------------------------------
# stage 1 — habitat_sim imports and is audio-capable
# ----------------------------------------------------------------------


@stage("01_habitat_sim_audio")
def probe_habitat_sim() -> Dict[str, Any]:
    import habitat_sim
    import habitat_sim.sensor

    layout = getattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType", None)
    if layout is None:
        raise RuntimeError("RLRAudioPropagationChannelLayoutType absent — not an audio build")
    # Probe the ENUM MEMBER, not the class: AudioSensorSpec is bound even in
    # non-audio builds (habitat-sim issue #2340), so a class check passes on a
    # build with no audio in it.
    if not hasattr(layout, "Binaural"):
        raise RuntimeError("channel-layout enum has no Binaural member — built WITHOUT --audio")

    layouts = [n for n in dir(layout) if not n.startswith("_")]
    info = {
        "habitat_sim_version": getattr(habitat_sim, "__version__", "?"),
        "habitat_sim_path": os.path.dirname(habitat_sim.__file__),
        "channel_layouts": layouts,
        "audio_capable": True,
    }
    print("  habitat_sim {} audio-capable OK".format(info["habitat_sim_version"]), flush=True)
    print("  channel layouts: {}".format(", ".join(layouts)), flush=True)
    return info


# ----------------------------------------------------------------------
# stage 2 — branch generation: methods that exist only on RLRAudioPropagationUpdate
# ----------------------------------------------------------------------

# Ticket 04's note: these are a sharper GREEN check than the enum probe, because
# they prove the checkout is the expected branch generation, not a stale one.
_BRANCH_METHODS = [
    "sourceIsVisible",       # free single-ray LOS test — ticket 09 wants it per-step
    "getRayEfficiency",
    "setListenerHRTF",
    "writeIRWave",
    "writeSceneMeshOBJ",     # ticket 12's cheap OBJ-colour proxy depends on this
]

# Ticket 02 found habitat-sim hardcodes ONE source while the engine underneath is
# natively multi-source. If any of these are already bound, the ~40-line patch is
# unnecessary — so record it either way rather than assuming.
_MULTISOURCE_METHODS = ["addSource", "clearSources", "setAudioSourceTransforms"]


@stage("02_branch_generation")
def probe_branch_generation() -> Dict[str, Any]:
    import habitat_sim
    import habitat_sim.sensor

    audio_sensor_cls = getattr(habitat_sim.sensor, "AudioSensor", None)
    if audio_sensor_cls is None:
        raise RuntimeError("habitat_sim.sensor.AudioSensor absent")

    methods = method_dump(audio_sensor_cls)
    present = {m: (m in methods) for m in _BRANCH_METHODS}
    multisource = {m: (m in methods) for m in _MULTISOURCE_METHODS}

    for name, ok in present.items():
        print("  {:<24} {}".format(name, "present" if ok else "ABSENT"), flush=True)
    print("  multi-source surface: {}".format(
        ", ".join(k for k, v in multisource.items() if v) or "none (ticket 02 confirmed)"
    ), flush=True)

    info = {
        "audio_sensor_methods": methods,
        "branch_methods": present,
        "branch_generation_ok": all(present.values()),
        "multisource_methods": multisource,
        "multisource_already_bound": any(multisource.values()),
    }
    if not info["branch_generation_ok"]:
        missing = [k for k, v in present.items() if not v]
        raise RuntimeError(
            "not the expected branch generation — missing {}".format(", ".join(missing))
        )
    return info


# ----------------------------------------------------------------------
# stage 3 — the defaults dump (tickets 01, 03 and 11 all block on this)
# ----------------------------------------------------------------------


@stage("03_defaults_dump")
def probe_defaults() -> Dict[str, Any]:
    import habitat_sim

    spec = habitat_sim.AudioSensorSpec()
    spec_fields = field_dump(spec)
    acoustics = getattr(spec, "acousticsConfig", None)
    acoustics_fields = field_dump(acoustics) if acoustics is not None else {}
    layout = getattr(spec, "channelLayout", None)
    layout_fields = field_dump(layout) if layout is not None else {}

    print("  AudioSensorSpec fields:", flush=True)
    for k, v in spec_fields.items():
        print("    {:<28} {}".format(k, v), flush=True)
    print("  acousticsConfig fields (the parameter sheet, MEASURED):", flush=True)
    for k, v in acoustics_fields.items():
        print("    {:<28} {}".format(k, v), flush=True)
    print("  channelLayout fields:", flush=True)
    for k, v in layout_fields.items():
        print("    {:<28} {}".format(k, v), flush=True)

    info: Dict[str, Any] = {
        "spec_fields": spec_fields,
        "acoustics_fields": acoustics_fields,
        "channel_layout_fields": layout_fields,
        "spec_field_names": sorted(spec_fields),
        "acoustics_field_names": sorted(acoustics_fields),
    }

    # --- the two called-out values ---------------------------------------
    # Ticket 03: transmission's default contradicts itself across three primary
    # sources at the same commit (header says true, pybind docstring and
    # docs/AUDIO.md say false). It is not cosmetic — transmission-on with uniform
    # materials leaks energy through walls and flattens the doorway-occlusion
    # contrast that ticket 03's "gradient survives uniform materials" leans on.
    info["transmission_default"] = acoustics_fields.get("transmission", "<absent>")
    # Ticket 03: enableMaterials should print False — the constructor overwrites
    # the header's `= true` initialiser under #ifdef ESP_BUILD_WITH_AUDIO. True
    # here means the build is not what we think it is.
    info["enableMaterials_default"] = spec_fields.get(
        "enableMaterials", acoustics_fields.get("enableMaterials", "<absent>")
    )
    info["enableMaterials_location"] = (
        "spec" if "enableMaterials" in spec_fields
        else "acousticsConfig" if "enableMaterials" in acoustics_fields
        else "<absent>"
    )
    print("\n  RESOLVED  transmission default   = {}".format(info["transmission_default"]), flush=True)
    print("  RESOLVED  enableMaterials        = {} (on {})".format(
        info["enableMaterials_default"], info["enableMaterials_location"]), flush=True)
    if str(info["enableMaterials_default"]) == "True":
        print("  *** enableMaterials is True — build is NOT what ticket 03 predicted", flush=True)

    # Ticket 11 renamed irTime -> maxIRLength. Record which names this branch has,
    # so the parameter sheet stops being hearsay.
    info["has_maxIRLength"] = "maxIRLength" in acoustics_fields
    info["has_irTime"] = "irTime" in acoustics_fields
    info["has_directRayCount"] = "directRayCount" in acoustics_fields

    # --- the dynamic_attr trap, demonstrated ------------------------------
    # AudioSensorSpec carries py::dynamic_attr (SensorBindings.cpp:395), so a key
    # that does not exist on this branch is SILENTLY attached and never read.
    # RLRAudioPropagationConfiguration does NOT (SensorBindings.cpp:293-295), so
    # the same mistake raises there. Prove both, because the new tree's wrapper has
    # to validate keys on the spec specifically.
    trap: Dict[str, Any] = {}
    probe_key = "irTime" if not info["has_irTime"] else "definitelyNotAField"
    try:
        setattr(spec, probe_key, 4.0)
        trap["spec_swallows_unknown_key"] = True
        trap["spec_probe_key"] = probe_key
        trap["spec_readback"] = repr(getattr(spec, probe_key, None))
    except Exception as exc:
        trap["spec_swallows_unknown_key"] = False
        trap["spec_error"] = repr(exc)
    if acoustics is not None:
        try:
            setattr(acoustics, "definitelyNotAField", 1.0)
            trap["acoustics_swallows_unknown_key"] = True
        except Exception as exc:
            trap["acoustics_swallows_unknown_key"] = False
            trap["acoustics_error"] = repr(exc)
    info["dynamic_attr_trap"] = trap
    print("  dynamic_attr trap: spec swallows unknown key = {}, acousticsConfig swallows = {}".format(
        trap.get("spec_swallows_unknown_key"), trap.get("acoustics_swallows_unknown_key")), flush=True)
    return info


# ----------------------------------------------------------------------
# stage 4 — torch on top, with the GPU actually visible
# ----------------------------------------------------------------------


@stage("04_torch")
def probe_torch() -> Dict[str, Any]:
    # Imported AFTER habitat_sim on purpose: the required order is quaternion ->
    # habitat_sim -> everything else, and an EGL-headless habitat_sim build
    # initialising a GL context before CUDA is exactly where interop breaks.
    import torch

    info = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        props = torch.cuda.get_device_properties(0)
        info["vram_total_gb"] = round(props.total_memory / (1024 ** 3), 2)
        # Allocate for real — is_available() can be true on a box where the
        # context then fails to create.
        t = torch.zeros(1024, 1024, device="cuda")
        info["alloc_smoke_ok"] = bool(float(t.sum().item()) == 0.0)
        del t
        torch.cuda.empty_cache()
    for k, v in info.items():
        print("  {:<24} {}".format(k, v), flush=True)
    if not info["cuda_available"]:
        raise RuntimeError("torch cannot see the GPU after habitat_sim import")
    return info


# ----------------------------------------------------------------------
# stage 5 — CLAP (the only model the clean-room build needs)
# ----------------------------------------------------------------------


@stage("05_clap")
def probe_clap(load_weights: bool = False) -> Dict[str, Any]:
    import scipy
    import transformers
    from transformers import ClapModel, ClapProcessor  # noqa: F401

    info = {
        "transformers": transformers.__version__,
        "scipy": scipy.__version__,
        "clap_symbols_importable": True,
    }
    # scipy.signal.resample_poly is the resampler the CLAP path uses; prove it
    # works under the pinned numpy rather than just that scipy imports.
    import numpy as np
    from scipy.signal import resample_poly

    resampled = resample_poly(np.zeros(4800, dtype=np.float32), 10, 1)
    info["resample_poly_ok"] = int(resampled.shape[0]) == 48000

    if load_weights:
        model = ClapModel.from_pretrained("laion/clap-htsat-fused")
        info["clap_weights_loaded"] = True
        info["clap_param_count_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        del model
    else:
        info["clap_weights_loaded"] = False
        info["note"] = "weights not downloaded (pass --load-clap to fetch ~600 MB)"

    for k, v in info.items():
        print("  {:<24} {}".format(k, v), flush=True)
    return info


# ----------------------------------------------------------------------
# stage 6 — habitat-lab: does it import, and do we need it?
# ----------------------------------------------------------------------


@stage("06_habitat_lab")
def probe_habitat_lab() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import habitat

        info["importable"] = True
        info["habitat_lab_version"] = getattr(habitat, "__version__", "?")
        info["habitat_lab_path"] = os.path.dirname(habitat.__file__)
    except Exception as exc:
        info["importable"] = False
        info["import_error"] = repr(exc)
        print("  habitat-lab NOT importable: {}".format(exc), flush=True)
        # Not a raise: the local source read for ticket 04 established the clean
        # room can drive habitat_sim directly, so this is a data point, not a gate.
        info["ok"] = True
        return info

    print("  habitat-lab {} imports OK".format(info["habitat_lab_version"]), flush=True)
    # The five symbols the old tree actually used, checked one at a time so a
    # partial install is visible rather than collapsing to one ImportError.
    symbols = {
        "habitat.config.read_write": ("habitat.config", "read_write"),
        "habitat.config.default.get_config": ("habitat.config.default", "get_config"),
        "habitat.config.default_structured_configs.HabitatSimSemanticSensorConfig": (
            "habitat.config.default_structured_configs", "HabitatSimSemanticSensorConfig"),
        "habitat.tasks.nav.nav.SPL": ("habitat.tasks.nav.nav", "SPL"),
        "habitat.tasks.nav.nav.SoftSPL": ("habitat.tasks.nav.nav", "SoftSPL"),
        "habitat.tasks.nav.shortest_path_follower.ShortestPathFollower": (
            "habitat.tasks.nav.shortest_path_follower", "ShortestPathFollower"),
    }
    resolved = {}
    for label, (mod_name, attr) in symbols.items():
        try:
            mod = __import__(mod_name, fromlist=[attr])
            resolved[label] = hasattr(mod, attr)
        except Exception as exc:
            resolved[label] = "error: {!r}".format(exc)
    info["symbols"] = resolved
    for k, v in resolved.items():
        print("    {:<64} {}".format(k, v), flush=True)
    return info


# ----------------------------------------------------------------------
# stage 7 — the real GREEN criterion: an audio sensor that renders in a scene
# ----------------------------------------------------------------------


def _find_scene(explicit: Optional[str]) -> str:
    if explicit:
        if not os.path.exists(explicit):
            raise RuntimeError("--scene {} does not exist".format(explicit))
        return explicit
    patterns = [
        "data/hm3d/**/*.basis.glb",
        "data/scene_datasets/**/*.basis.glb",
        "data/hm3d/**/*.glb",
    ]
    for pat in patterns:
        hits = [p for p in glob.glob(pat, recursive=True) if "semantic" not in os.path.basename(p)]
        if hits:
            return sorted(hits)[0]
    raise RuntimeError("no HM3D .glb found — pass --scene explicitly")


@stage("07_live_render")
def probe_live_render(scene: Optional[str], sample_rate: int) -> Dict[str, Any]:
    import numpy as np
    import habitat_sim

    scene_path = _find_scene(scene)
    print("  scene: {}".format(scene_path), flush=True)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    # Materials stay OFF: ticket 03 established they do not resolve on HM3D, and
    # the annotated-semantics path may hand the audio context an empty mesh
    # (ticket 12). This gate proves the sensor renders, not that materials work.
    # Set defensively — a renamed field on this branch must not fail the stage.
    for field, value in (("load_semantic_mesh", False), ("enable_physics", False)):
        if hasattr(backend_cfg, field):
            setattr(backend_cfg, field, value)
        else:
            info_missing = "SimulatorConfiguration has no {}".format(field)
            print("  note: {}".format(info_missing), flush=True)

    # Default agent config (keeps its RGB camera): the headless EGL path is the
    # one the rebuilt runner will actually use, so exercise it here.
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    info: Dict[str, Any] = {"scene": scene_path}
    try:
        spec = habitat_sim.AudioSensorSpec()
        spec.uuid = "audio_sensor"
        spec.enableMaterials = False
        spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        spec.channelLayout.channelCount = 2
        spec.acousticsConfig.sampleRate = int(sample_rate)
        sim.add_sensor(spec)
        info["sensor_constructed"] = True

        audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]

        # Place listener and source on the navmesh so the render is a real room
        # response rather than a degenerate free-field one.
        if sim.pathfinder.is_loaded:
            listener = sim.pathfinder.get_random_navigable_point()
            source = sim.pathfinder.get_random_navigable_point()
            info["navmesh_loaded"] = True
        else:
            listener = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            source = np.array([1.0, 0.0, 1.0], dtype=np.float32)
            info["navmesh_loaded"] = False
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = listener
        agent.set_state(state)
        audio_sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
        info["listener"] = [float(x) for x in listener]
        info["source"] = [float(x) for x in source]

        t0 = time.time()
        obs = sim.get_sensor_observations()["audio_sensor"]
        elapsed = time.time() - t0
        ir = np.asarray(obs, dtype=np.float32)
        info["render_ok"] = True
        # A single timing at defaults. NOT the cost sweep — that is ticket 06,
        # which needs repeats (path tracing is stochastic) and a knob sweep.
        info["first_render_s"] = round(elapsed, 4)
        info["ir_shape"] = list(ir.shape)
        info["ir_nonzero"] = bool(np.any(ir != 0.0))
        info["ir_peak_abs"] = float(np.max(np.abs(ir))) if ir.size else 0.0
        info["ir_rms"] = float(np.sqrt(np.mean(ir ** 2))) if ir.size else 0.0
        if hasattr(audio_sensor, "sourceIsVisible"):
            try:
                info["source_is_visible"] = bool(audio_sensor.sourceIsVisible())
            except Exception as exc:
                info["source_is_visible"] = "error: {!r}".format(exc)
        if hasattr(audio_sensor, "getRayEfficiency"):
            try:
                info["ray_efficiency"] = float(audio_sensor.getRayEfficiency())
            except Exception as exc:
                info["ray_efficiency"] = "error: {!r}".format(exc)

        for k in ("first_render_s", "ir_shape", "ir_nonzero", "ir_peak_abs",
                  "source_is_visible", "ray_efficiency"):
            if k in info:
                print("  {:<24} {}".format(k, info[k]), flush=True)
        if not info["ir_nonzero"]:
            print("  *** IR is all zeros — sensor constructed but rendered silence", flush=True)
    finally:
        sim.close()
    return info


# ----------------------------------------------------------------------
# stage 8 — provenance: what exactly was built
# ----------------------------------------------------------------------


def _git(cwd: str, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git"] + list(args), cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=30,
        )
        if out.returncode != 0:
            return None
        return out.stdout.decode().strip()
    except Exception:
        return None


@stage("08_provenance")
def probe_provenance(sim_dir: Optional[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"sim_dir": sim_dir}
    if not sim_dir or not os.path.isdir(sim_dir):
        info["note"] = "habitat-sim source dir not provided or missing — SHAs unrecorded"
        return info
    info["habitat_sim_branch"] = _git(sim_dir, "rev-parse", "--abbrev-ref", "HEAD")
    info["habitat_sim_sha"] = _git(sim_dir, "rev-parse", "HEAD")
    # Ticket 11: the parameter research was done against the rlr-audio-propagation
    # repo's archived main, but habitat-sim pins a submodule commit that was never
    # checked against it. Recording the SHA makes the parameter sheet falsifiable.
    status = _git(sim_dir, "submodule", "status", "--recursive")
    info["submodule_status"] = status
    rlr_sha = None
    for line in (status or "").splitlines():
        if "rlr-audio-propagation" in line:
            rlr_sha = line.strip().split()[0].lstrip("+-U")
            break
    info["rlr_audio_propagation_sha"] = rlr_sha
    so_path = os.path.join(
        sim_dir, "src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64",
        "libRLRAudioPropagation.so")
    info["prebuilt_so_present"] = os.path.exists(so_path)
    if info["prebuilt_so_present"]:
        info["prebuilt_so_bytes"] = os.path.getsize(so_path)
    for k in ("habitat_sim_branch", "habitat_sim_sha", "rlr_audio_propagation_sha",
              "prebuilt_so_present"):
        print("  {:<28} {}".format(k, info.get(k)), flush=True)
    return info


# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--scene", default=None, help="HM3D .glb (auto-discovered if omitted)")
    ap.add_argument("--sim-dir", default=None, help="habitat-sim source dir, for SHA provenance")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--load-clap", action="store_true",
                    help="actually download the ~600 MB CLAP checkpoint")
    ap.add_argument("--label", default="core", help="tag recorded in the report")
    args = ap.parse_args()

    print("=" * 70, flush=True)
    print("ticket 04 one-env probe — label={}".format(args.label), flush=True)
    print("=" * 70, flush=True)

    probe_interpreter()
    probe_habitat_sim()
    probe_branch_generation()
    probe_defaults()
    probe_torch()
    probe_clap(load_weights=args.load_clap)
    probe_habitat_lab()
    probe_live_render(args.scene, args.sample_rate)
    probe_provenance(args.sim_dir)

    # GREEN = one process, all imports, GPU visible, audio sensor constructible.
    # habitat-lab is deliberately NOT in the gate: ticket 04's source read found
    # the clean-room runner can drive habitat_sim directly.
    gate_stages = [
        "00_interpreter", "01_habitat_sim_audio", "02_branch_generation",
        "03_defaults_dump", "04_torch", "05_clap", "07_live_render",
    ]
    failures = [s for s in gate_stages if not REPORT.get(s, {}).get("ok", False)]
    green = not failures and bool(REPORT.get("07_live_render", {}).get("render_ok"))

    REPORT["_verdict"] = {
        "label": args.label,
        "green": green,
        "gate_stages": gate_stages,
        "failed_stages": failures,
        "habitat_lab_importable": REPORT.get("06_habitat_lab", {}).get("importable"),
        "numpy_pin_held": REPORT.get("00_interpreter", {}).get("numpy_pin_held"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(REPORT, fh, indent=2, sort_keys=True)

    banner("VERDICT [{}]".format(args.label))
    if green:
        print("  GREEN — one env holds habitat-sim(audio) + torch + CLAP, GPU visible,", flush=True)
        print("          audio sensor renders in an HM3D scene.", flush=True)
    else:
        print("  RED — blocker list IS the deliverable:", flush=True)
        for s in failures:
            print("    {}: {}".format(s, REPORT.get(s, {}).get("error")), flush=True)
    print("  habitat-lab importable: {}".format(
        REPORT["_verdict"]["habitat_lab_importable"]), flush=True)
    print("  report: {}".format(args.out), flush=True)
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
