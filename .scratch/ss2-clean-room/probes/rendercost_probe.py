#!/usr/bin/env python3
"""Ticket 06 render-cost probe — how much does live-every-step audio actually cost?

Run by ``rendercost_sweep.sh`` inside the ``ss2`` env built by ticket 04.
Never imports anything from ``embodied_memory/`` or ``dialogue_memory/``.

The whole measurement is a **walk**: place a source, drop the listener a few
metres away, and step it along a navmesh path toward the source, timing every
render. One walk yields all five things this ticket asks for at once —

  * first-call cost (geometry upload + simulate) vs steady-state (simulate only),
    because ticket 03 established the mesh uploads once per context and ticket 04
    could only *infer* the split from log timestamps
  * ms/step under each knob setting
  * whether received energy still climbs as the listener approaches, which is the
    admissibility gate: a preset that is fast but flat is worthless
  * whether cost tracks listener-source distance (per-scene budget stability)
  * LOS / non-LOS labelling, since diffraction-dominated poses are the expensive
    case and a mean over a mixed walk hides that

Timing is deliberately taken through ``get_sensor_observations()`` — the verified
API path from ticket 04 — with the agent configured with **no camera**, so the
number is audio and nothing else. Ticket 04's 0.6013 s was measured with the
default agent, which carries an RGB camera; ``--with-camera-delta`` re-measures
one config with the camera attached so that confound is quantified rather than
argued about.

Python 3.9 (the SoundSpaces pin), stdlib + numpy only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ``import quaternion`` MUST precede habitat_sim (habitat-sim issue #1813).
try:
    import quaternion  # noqa: F401
    _QUATERNION_OK = True
    _QUATERNION_ERR = None
except Exception as exc:  # pragma: no cover - environment probe
    _QUATERNION_OK = False
    _QUATERNION_ERR = repr(exc)

import numpy as np


REPORT: Dict[str, Any] = {}

# Ticket 04's measured defaults, on habitat-sim RLRAudioPropagationUpdate
# @ 4f61e321, rlr-audio-propagation @ 4fd446b4, stock. Every sweep variant is
# expressed as a delta from this, and the probe re-prints the live values so a
# drifted env is visible rather than silently swept against the wrong baseline.
MEASURED_DEFAULTS = {
    "diffraction": 1, "directRayCount": 500, "directSHOrder": 3, "direct": 1,
    "frequencyBands": 4, "globalVolume": 1.0, "indirect": 1,
    "indirectRayCount": 5000, "indirectRayDepth": 200, "indirectSHOrder": 1,
    "maxDiffractionOrder": 10, "maxIRLength": 4.0, "meshSimplification": 0,
    "sampleRate": 44100.0, "sourceRayCount": 200, "sourceRayDepth": 10,
    "temporalCoherence": 0, "threadCount": 1, "transmission": 1, "unitScale": 1.0,
}

# --- pre-registered verdict thresholds ------------------------------------
# Written down BEFORE the numbers come back, so pasting the report resolves the
# ticket instead of opening a fresh judgement call. Rationale in the ticket.
STEP_BUDGET = 500              # steps/episode (ADR-0005 benchmark budget)
MS_AFFORDABLE = 50.0           # <= this: live-every-step holds outright
MS_TOLERABLE = 150.0           # <= this: holds, audio is a visible cost
# A preset is only admissible if the gradient it produces is still climbable.
# Speed is worthless if the energy field stops pointing at the source.
GRADIENT_RHO_MAX = -0.70       # Spearman(energy_dB, geodesic distance)
GRADIENT_DR_MIN_DB = 6.0       # far-to-near dynamic range


def banner(msg: str) -> None:
    print("\n----- {} -----".format(msg), flush=True)


def stage(name: str):
    """Run a stage, capture its dict, never let it abort the run.

    Same contract as ticket 04's probe: on a bad run the *complete* picture is
    the deliverable, so one failing stage must not hide the ones behind it.
    """

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


# ----------------------------------------------------------------------
# statistics — implemented here rather than pulled from scipy, to keep this
# probe's import surface at stdlib + numpy. scipy IS in the env, but importing
# it drags a chain this measurement has no reason to depend on.
# ----------------------------------------------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    srt = a[order]
    i = 0
    while i < n:  # average tied ranks
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(x) != len(y):
        return None
    rx = _rankdata(np.asarray(x))
    ry = _rankdata(np.asarray(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    if denom <= 0:
        return None
    return round(float((rx * ry).sum() / denom), 4)


def energy_db(ir: np.ndarray) -> float:
    """Broadband IR energy in dB. The quantity the controller's climb reads."""
    if ir.size == 0:
        return float("-inf")
    return round(float(10.0 * math.log10(float(np.sum(ir.astype(np.float64) ** 2)) + 1e-20)), 3)


def summarize(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(s),
        "median_ms": round(statistics.median(s) * 1000.0, 2),
        "mean_ms": round(statistics.fmean(s) * 1000.0, 2),
        "min_ms": round(s[0] * 1000.0, 2),
        "max_ms": round(s[-1] * 1000.0, 2),
        "p90_ms": round(s[min(len(s) - 1, int(0.9 * len(s)))] * 1000.0, 2),
        "stdev_ms": round(statistics.pstdev(s) * 1000.0, 2) if len(s) > 1 else 0.0,
    }


# ----------------------------------------------------------------------
# scene + geometry helpers
# ----------------------------------------------------------------------


def find_scenes(explicit: List[str], limit: int) -> List[str]:
    if explicit:
        for p in explicit:
            if not os.path.exists(p):
                raise RuntimeError("--scene {} does not exist".format(p))
        return explicit
    patterns = [
        "data/hm3d/**/*.basis.glb",
        "data/scene_datasets/**/*.basis.glb",
        "data/hm3d/**/*.glb",
    ]
    hits: List[str] = []
    for pat in patterns:
        hits = [p for p in glob.glob(pat, recursive=True)
                if "semantic" not in os.path.basename(p)]
        if hits:
            break
    if not hits:
        raise RuntimeError("no HM3D .glb found — pass --scene explicitly")
    return sorted(hits)[:limit]


def geodesic(sim, a, b) -> float:
    """Geodesic (navmesh) distance, not euclidean — the gradient is walked, not flown."""
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    if not sim.pathfinder.find_path(path):
        return float("inf")
    return float(path.geodesic_distance)


def polyline_length(points: List[np.ndarray]) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        total += float(np.linalg.norm(
            np.asarray(points[i + 1], dtype=np.float32)
            - np.asarray(points[i], dtype=np.float32)))
    return total


def resample_polyline(points: List[np.ndarray], n: int) -> List[np.ndarray]:
    """Spread `n` samples evenly by arc length across the WHOLE path.

    Deliberately not "first n points at fixed spacing". habitat's ShortestPath
    returns corner points only, so fixed spacing plus a truncation samples the
    far end of the approach and throws away the last few metres — which is where
    the gradient is steepest and where ticket 03's contrast question actually
    lives. (Local verification caught this: every config scored FLAT because the
    walk covered 4 m of a 13.6 m path and never got near the source.)
    """
    pts = [np.asarray(p, dtype=np.float32) for p in points]
    if len(pts) < 2 or n < 2:
        return pts[:max(1, n)]
    seglens = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
    total = sum(seglens)
    if total <= 1e-6:
        return [pts[0]]
    out: List[np.ndarray] = []
    for k in range(n):
        target = total * k / (n - 1)
        acc = 0.0
        for i, seg in enumerate(seglens):
            if seg <= 1e-6:
                continue
            if acc + seg >= target or i == len(seglens) - 1:
                t = min(1.0, max(0.0, (target - acc) / seg))
                out.append(pts[i] + (pts[i + 1] - pts[i]) * t)
                break
            acc += seg
    return out


def pick_source_and_start(sim, min_dist: float, tries: int = 200
                          ) -> Tuple[np.ndarray, np.ndarray, float]:
    """A source and a listener start that are genuinely far apart on the navmesh."""
    best = None
    src = sim.pathfinder.get_random_navigable_point()
    for _ in range(tries):
        cand = sim.pathfinder.get_random_navigable_point()
        d = geodesic(sim, cand, src)
        if not math.isfinite(d):
            continue
        if best is None or d > best[1]:
            best = (cand, d)
        if d >= min_dist:
            return np.asarray(src, dtype=np.float32), np.asarray(cand, dtype=np.float32), d
    if best is None:
        raise RuntimeError("no navigable point pair with a finite geodesic path")
    # Fell short of min_dist: return the furthest found and let the report say so.
    return np.asarray(src, dtype=np.float32), np.asarray(best[0], dtype=np.float32), best[1]


# ----------------------------------------------------------------------
# the sim + sensor rig
# ----------------------------------------------------------------------


def build_sim(scene: str, with_camera: bool):
    """One Simulator, configured for audio.

    The agent gets **no camera sensors** by default. ``get_sensor_observations()``
    renders every attached sensor, so ticket 04's 0.6013 s included an RGB render
    it never meant to time. Dropping the camera is what makes the number here a
    pure audio cost.
    """
    import habitat_sim

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene
    # Materials off, semantics off: ticket 03 settled that materials do not
    # resolve on HM3D, and ticket 12 owns the empty-mesh trap. This probe times
    # the geometric path, which is the one the destination actually runs.
    for field, value in (("load_semantic_mesh", False), ("enable_physics", False)):
        if hasattr(backend_cfg, field):
            setattr(backend_cfg, field, value)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    if not with_camera:
        agent_cfg.sensor_specifications = []
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    return sim


def attach_audio(sim, cfg: Dict[str, Any], sample_rate: Optional[int]
                 ) -> Tuple[Any, str]:
    """Attach an audio sensor whose acousticsConfig carries `cfg`.

    Returns the sensor and **the uuid it actually got**, which is deliberately
    not a caller's choice. `AudioSensorSpec` ships `uuid = "audio_sensor"` from
    its C++ constructor, and assigning a different one does not fully take: the
    Python-side `_sensors` dict picks up the new name while the C++ sensor suite
    keeps the old, so `get_sensor_observations()` fails an internal cross-lookup
    with `KeyError('audio_sensor')`. That is what killed the source-count stage
    on the first box run — the sweep survived only because it happened to pass
    the default name. Read the uuid back, never assume it.

    Every acousticsConfig key is validated against the live field list before
    assignment. `AudioSensorSpec` is bound `py::dynamic_attr` (ticket 04 measured
    this: the spec swallows unknown keys, `acousticsConfig` raises), so a typo'd
    knob name would otherwise attach a Python attribute, never be read, and
    produce a perfectly plausible timing for the *default* value. That failure
    mode would silently invalidate the entire sweep.
    """
    import habitat_sim

    spec = habitat_sim.AudioSensorSpec()
    spec.enableMaterials = False
    spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    spec.channelLayout.channelCount = 2
    if sample_rate is not None:
        spec.acousticsConfig.sampleRate = float(sample_rate)

    valid = {n for n in dir(spec.acousticsConfig) if not n.startswith("_")}
    for key, value in cfg.items():
        if key not in valid:
            raise RuntimeError(
                "acousticsConfig has no field {!r} on this build (have: {})".format(
                    key, ", ".join(sorted(valid))))
        setattr(spec.acousticsConfig, key, value)

    sim.add_sensor(spec)
    uuid = str(spec.uuid)
    sensors = sim.get_agent(0)._sensors
    if uuid not in sensors:
        raise RuntimeError(
            "audio sensor registered under an unexpected uuid: spec says {!r}, "
            "agent has {}".format(uuid, sorted(sensors)))
    return sensors[uuid], uuid


def render_once(sim, sensor, uuid: str, listener: np.ndarray, source: np.ndarray
                ) -> Tuple[float, np.ndarray]:
    """One step: move the listener, move the source, render. Returns (seconds, IR).

    Both transforms are set every call because that is what the live runner does
    every step, so their cost belongs inside the measured window.
    """
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.asarray(listener, dtype=np.float32)
    agent.set_state(state)
    sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
    t0 = time.time()
    obs = sim.get_sensor_observations()[uuid]
    elapsed = time.time() - t0
    return elapsed, np.asarray(obs, dtype=np.float32)


def walk_config(sim, cfg: Dict[str, Any], label: str,
                source: np.ndarray, waypoints: List[np.ndarray],
                sample_rate: Optional[int], source_height: float,
                score_gradient: bool = True) -> Dict[str, Any]:
    """Time one config across the whole walk, and score its gradient.

    Returns per-step rows plus the derived summary. The first step is reported
    separately throughout: it carries the one-time geometry upload, and mixing it
    into the steady-state median is exactly the error ticket 04 flagged.

    The caller MUST hand this a simulator carrying no other audio sensor.
    ``get_sensor_observations()`` renders every attached sensor, so a second
    sensor on the same sim silently adds its render to every timing here — which
    produces a smoothly rising cost curve that looks like a real result and is
    not one. (Caught in local verification against a stub, where ``diffraction=0``
    measured as the *slowest* config.)
    """
    sensor, uuid = attach_audio(sim, cfg, sample_rate)
    src = np.asarray(source, dtype=np.float32).copy()
    src[1] = src[1] + source_height

    rows: List[Dict[str, Any]] = []
    for i, wp in enumerate(waypoints):
        elapsed, ir = render_once(sim, sensor, uuid, wp, src)
        row: Dict[str, Any] = {
            "i": i,
            "s": round(elapsed, 4),
            "geodesic_m": round(geodesic(sim, wp, src), 3),
            "energy_db": energy_db(ir),
            "ir_samples": int(ir.shape[-1]) if ir.ndim else 0,
            "ir_nonzero": bool(np.any(ir != 0.0)),
        }
        for name, fn in (("source_is_visible", "sourceIsVisible"),
                         ("ray_efficiency", "getRayEfficiency")):
            if hasattr(sensor, fn):
                try:
                    row[name] = getattr(sensor, fn)()
                    if name == "ray_efficiency":
                        row[name] = round(float(row[name]), 4)
                    else:
                        row[name] = bool(row[name])
                except Exception as exc:
                    row[name] = "error: {!r}".format(exc)
        rows.append(row)

    times = [r["s"] for r in rows]
    steady = times[1:] if len(times) > 1 else times
    finite = [r for r in rows if math.isfinite(r["geodesic_m"]) and r["energy_db"] > -300]
    dists = [r["geodesic_m"] for r in finite]
    energies = [r["energy_db"] for r in finite]

    rho = spearman(energies, dists) if score_gradient else None
    dyn_range = (round(max(energies) - min(energies), 2)
                 if score_gradient and len(energies) >= 2 else None)
    los = [r for r in rows if r.get("source_is_visible") is True]
    nlos = [r for r in rows if r.get("source_is_visible") is False]

    # The non-LOS gradient scored on its own. This is the number the task premise
    # actually rests on: the agent hears an anomaly from another room and has to
    # climb toward a source it cannot see. A config can post a healthy overall
    # gradient purely from its line-of-sight samples while being flat everywhere
    # a wall is in the way — which is exactly what killing `diffraction` or
    # `transmission` would do. Ticket 09 decides on these numbers.
    nlos_ok = [r for r in nlos
               if math.isfinite(r["geodesic_m"]) and r["energy_db"] > -300]
    nlos_rho = (spearman([r["energy_db"] for r in nlos_ok],
                         [r["geodesic_m"] for r in nlos_ok])
                if score_gradient and len(nlos_ok) >= 3 else None)
    nlos_dr = (round(max(r["energy_db"] for r in nlos_ok)
                     - min(r["energy_db"] for r in nlos_ok), 2)
               if score_gradient and len(nlos_ok) >= 2 else None)

    out: Dict[str, Any] = {
        "label": label,
        "config": dict(cfg),
        "rows": rows,
        "first_call": round(times[0], 4) if times else None,
        "steady_state": summarize(steady),
        # Item 4: does cost track distance? If it does, the per-episode budget is
        # pose-dependent and a single ms/step number cannot be quoted.
        "cost_vs_distance_rho": spearman([r["s"] for r in finite], dists),
        "gradient_rho": rho,
        "gradient_dynamic_range_db": dyn_range,
        "gradient_admissible": bool(
            rho is not None and rho <= GRADIENT_RHO_MAX
            and dyn_range is not None and dyn_range >= GRADIENT_DR_MIN_DB),
        "gradient_rho_nlos": nlos_rho,
        "gradient_dynamic_range_nlos_db": nlos_dr,
        "n_nlos_samples": len(nlos_ok),
        "los_steady": summarize([r["s"] for r in los[1:]] if len(los) > 1 else [r["s"] for r in los]),
        "nlos_steady": summarize([r["s"] for r in nlos[1:]] if len(nlos) > 1 else [r["s"] for r in nlos]),
        "ir_samples_min": min((r["ir_samples"] for r in rows), default=0),
        "ir_samples_max": max((r["ir_samples"] for r in rows), default=0),
        "any_silent": any(not r["ir_nonzero"] for r in rows),
    }
    ss = out["steady_state"].get("median_ms")
    print("  {:<28} steady {:>8} ms  first {:>8} ms  rho {:>7}  dr {:>6} dB  {}".format(
        label,
        ss if ss is not None else "?",
        round(out["first_call"] * 1000.0, 1) if out["first_call"] else "?",
        rho if rho is not None else "?",
        dyn_range if dyn_range is not None else "?",
        ("climbable" if out["gradient_admissible"] else "FLAT") if score_gradient
        else "(cost only)",
    ), flush=True)
    return out


# ----------------------------------------------------------------------
# the sweep plan
# ----------------------------------------------------------------------


# Not cost knobs. `transmission` leaks energy through walls and `diffraction`
# bends it around corners — between them they ARE the non-line-of-sight audio
# path. Swept and reported, never auto-adopted. See derive_cheap_preset.
PHYSICS_KNOBS = {"transmission", "diffraction"}


def sweep_plan() -> List[Tuple[str, Dict[str, Any]]]:
    """One knob at a time, off ticket 04's measured defaults.

    ``threadCount`` is included but ticket 04 already cut its ceiling to ~4x
    (the box has 4 cores), so it is not expected to carry the result.
    ``temporalCoherence`` defaults OFF, which makes it a pure win to test —
    nothing is given up by enabling it.
    ``transmission`` is swept on both axes: it is a cost knob AND a contrast
    knob, because leaking energy through walls works directly against the
    doorway-occlusion contrast the gradient leans on.
    """
    return [
        ("baseline_defaults", {}),
        ("maxIRLength=1.0", {"maxIRLength": 1.0}),
        ("maxIRLength=0.5", {"maxIRLength": 0.5}),
        ("indirectRayCount=1000", {"indirectRayCount": 1000}),
        ("indirectRayCount=500", {"indirectRayCount": 500}),
        ("indirectRayDepth=50", {"indirectRayDepth": 50}),
        ("directRayCount=100", {"directRayCount": 100}),
        ("threadCount=4", {"threadCount": 4}),
        ("temporalCoherence=1", {"temporalCoherence": 1}),
        ("transmission=0", {"transmission": 0}),
        ("diffraction=0", {"diffraction": 0}),
    ]


def walk_geometry(scene: str, min_dist: float, step_m: float, walk_steps: int
                  ) -> Dict[str, Any]:
    """Pick the source and the walk ONCE, on a throwaway sim.

    Every config must be timed over the *same* poses. Re-sampling a random walk
    per config would compare two different journeys and call the difference a
    knob effect. habitat's navigable-point sampler carries no reproducibility
    guarantee across simulators, so the points are computed here and replayed.
    """
    import habitat_sim

    sim = build_sim(scene, with_camera=False)
    try:
        if not sim.pathfinder.is_loaded:
            raise RuntimeError("navmesh not loaded for {}".format(scene))
        source, start, dist = pick_source_and_start(sim, min_dist)
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = source
        if not sim.pathfinder.find_path(path):
            raise RuntimeError("no navmesh path from start to source")
        pts = list(path.points)
        length = polyline_length(pts)
        # `walk_steps` is a CAP, not a count: use enough samples to keep spacing
        # under step_m, but never so few that the walk skips the near field.
        n = int(max(4, min(walk_steps, math.ceil(length / max(step_m, 1e-3)) + 1)))
        waypoints = resample_polyline(pts, n)
        return {
            "source": np.asarray(source, dtype=np.float32),
            "start": np.asarray(start, dtype=np.float32),
            "geodesic_m": float(dist),
            "path_length_m": round(length, 3),
            "reached_min_dist": bool(dist >= min_dist),
            "step_spacing_m": round(length / max(1, n - 1), 3),
            "waypoints": waypoints,
        }
    finally:
        sim.close()


def measure_config(scene: str, cfg: Dict[str, Any], label: str, geom: Dict[str, Any],
                   sample_rate: Optional[int], source_height: float,
                   with_camera: bool = False, score_gradient: bool = True
                   ) -> Dict[str, Any]:
    """One config, one fresh simulator.

    The scene reload per config is the price of the isolation described in
    ``walk_config`` — it costs seconds and buys a timing that means what it says.
    """
    sim = build_sim(scene, with_camera=with_camera)
    try:
        return walk_config(sim, cfg, label, geom["source"],
                           geom["waypoints"], sample_rate, source_height,
                           score_gradient=score_gradient)
    finally:
        sim.close()


def derive_cheap_preset(results: List[Dict[str, Any]], baseline_ms: float
                        ) -> Dict[str, Any]:
    """Combine every knob that was both faster AND left the gradient climbable.

    Derived from the measurements rather than guessed up front, so the preset is
    a result of this sweep instead of an input to it. Combined effects are not
    additive, which is exactly why the combination is then measured too.
    """
    cheap: Dict[str, Any] = {}
    for r in results:
        if r["label"] == "baseline_defaults" or not r.get("config"):
            continue
        # PHYSICS_KNOBS are excluded on purpose. They are cheaper, and on a walk
        # that is mostly line-of-sight they will still score "climbable" — but
        # they decide whether a source is audible THROUGH A WALL at all, which is
        # the premise of the whole anomaly-response task, and ticket 09 owns that
        # call (it is where ADR-0003's floor constraint gets retired or kept).
        # Folding them into a preset labelled "cheap" would smuggle a task-design
        # decision in as a performance tweak.
        if any(k in PHYSICS_KNOBS for k in r["config"]):
            continue
        ms = r.get("steady_state", {}).get("median_ms")
        if ms is None or not r.get("gradient_admissible"):
            continue
        if ms < baseline_ms * 0.95:  # a real win, not noise
            cheap.update(r["config"])
    return cheap


# ----------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------


@stage("00_interpreter")
def probe_interpreter() -> Dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "quaternion_import_ok": _QUATERNION_OK,
    }
    for k, v in info.items():
        print("  {:<24} {}".format(k, v), flush=True)
    if not _QUATERNION_OK:
        raise RuntimeError("numpy-quaternion missing: {}".format(_QUATERNION_ERR))
    return info


@stage("01_defaults_recheck")
def probe_defaults_recheck() -> Dict[str, Any]:
    """Re-read the defaults and diff against ticket 04's measured table.

    Cheap, and it is the only thing standing between a drifted env and a sweep
    whose entire baseline column is wrong.
    """
    import habitat_sim

    spec = habitat_sim.AudioSensorSpec()
    live: Dict[str, Any] = {}
    for name in sorted(dir(spec.acousticsConfig)):
        if name.startswith("_"):
            continue
        value = getattr(spec.acousticsConfig, name)
        if not callable(value):
            live[name] = value

    drift = {}
    for key, expected in MEASURED_DEFAULTS.items():
        if key not in live:
            drift[key] = {"expected": expected, "actual": "<absent>"}
        elif float(live[key]) != float(expected):
            drift[key] = {"expected": expected, "actual": live[key]}

    print("  live acousticsConfig fields: {}".format(len(live)), flush=True)
    if drift:
        print("  *** DRIFT from ticket 04's measured defaults:", flush=True)
        for k, v in drift.items():
            print("      {:<24} expected {} got {}".format(k, v["expected"], v["actual"]), flush=True)
    else:
        print("  matches ticket 04's measured defaults exactly", flush=True)
    return {
        "live_defaults": {k: repr(v) for k, v in live.items()},
        "drift_from_ticket_04": drift,
        "defaults_stable": not drift,
    }


@stage("02_sweep")
def probe_sweep(scenes: List[str], walk_steps: int, step_m: float, min_dist: float,
                sample_rate: Optional[int], source_height: float,
                with_camera_delta: bool) -> Dict[str, Any]:
    per_scene: List[Dict[str, Any]] = []
    geoms: Dict[str, Dict[str, Any]] = {}

    # --- pass 1: one knob at a time, per scene ----------------------------
    for scene in scenes:
        print("\n  === scene: {} ===".format(scene), flush=True)
        scene_out: Dict[str, Any] = {"scene": scene}
        try:
            t_load = time.time()
            geom = walk_geometry(scene, min_dist, step_m, walk_steps)
            scene_out["scene_load_s"] = round(time.time() - t_load, 2)
            geoms[scene] = geom
            scene_out["source"] = [float(x) for x in geom["source"]]
            scene_out["start"] = [float(x) for x in geom["start"]]
            scene_out["start_geodesic_m"] = round(geom["geodesic_m"], 3)
            scene_out["reached_min_dist"] = geom["reached_min_dist"]
            scene_out["walk_steps"] = len(geom["waypoints"])
            scene_out["step_spacing_m"] = geom["step_spacing_m"]
            scene_out["path_length_m"] = geom["path_length_m"]
            print("  walk: {} steps spanning {:.2f} m path ({:.2f} m apart), "
                  "geodesic {:.2f} m{}".format(
                      len(geom["waypoints"]), geom["path_length_m"],
                      geom["step_spacing_m"], geom["geodesic_m"],
                      "" if geom["reached_min_dist"] else "  *** short of --min-dist"),
                  flush=True)

            scene_out["results"] = [
                measure_config(scene, cfg, label, geom, sample_rate, source_height)
                for label, cfg in sweep_plan()
            ]
        except Exception as exc:
            scene_out["error"] = repr(exc)
            scene_out["traceback"] = traceback.format_exc(limit=6)
            print("  scene FAILED: {}".format(exc), flush=True)

        # The camera delta: ticket 04 timed the default agent, which carries an
        # RGB sensor, so its 0.6013 s was audio + RGB. Re-measure baseline with
        # the camera attached so that confound becomes a number.
        if with_camera_delta and scene in geoms:
            try:
                cam = measure_config(scene, {}, "baseline_WITH_camera", geoms[scene],
                                     sample_rate, source_height, with_camera=True,
                                     score_gradient=False)
                scene_out["with_camera"] = {
                    "steady_state": cam["steady_state"],
                    "first_call": cam["first_call"],
                    "note": "get_sensor_observations() renders RGB too; the delta "
                            "against baseline_defaults is what ticket 04's 0.6013 s "
                            "included and this ticket's number excludes",
                }
            except Exception as exc:
                scene_out["with_camera"] = {"error": repr(exc)}
        per_scene.append(scene_out)

    # --- pass 2: one cheap preset, derived once, measured everywhere ------
    # Derived from the first scene that produced results, then applied unchanged
    # to every scene. A per-scene preset would not be comparable across scenes,
    # and the recommendation this ticket owes is a single preset.
    first = next((s for s in per_scene if s.get("results")), None)
    cheap_cfg: Dict[str, Any] = {}
    if first:
        base = next((r for r in first["results"]
                     if r["label"] == "baseline_defaults"), None)
        base_ms = (base or {}).get("steady_state", {}).get("median_ms")
        if base_ms:
            cheap_cfg = derive_cheap_preset(first["results"], base_ms)
    if cheap_cfg:
        print("\n  derived cheap preset (from {}): {}".format(
            os.path.basename(first["scene"]), cheap_cfg), flush=True)
        for scene_out in per_scene:
            scene = scene_out["scene"]
            if scene not in geoms or not scene_out.get("results"):
                continue
            try:
                scene_out["results"].append(measure_config(
                    scene, cheap_cfg, "cheap_preset", geoms[scene],
                    sample_rate, source_height))
            except Exception as exc:
                scene_out["cheap_preset_error"] = repr(exc)
    else:
        print("\n  no knob was both faster and gradient-admissible —"
              " no cheap preset to derive", flush=True)

    return {"per_scene": per_scene, "cheap_preset": cheap_cfg}


@stage("03_source_count")
def probe_source_count(scenes: List[str], repeats: int, sample_rate: Optional[int],
                       source_height: float, min_dist: float) -> Dict[str, Any]:
    """Item 5, as far as a STOCK build can answer it.

    Ticket 02 found the engine is natively multi-source with per-source IRs, but
    ticket 04 confirmed against the built binary that habitat-sim exposes none of
    it (``multi-source surface: none``). So the patched cost is NOT measurable
    here, and this stage does not pretend otherwise.

    What it does measure is the **sequential upper bound** — N sources rendered
    as N renders with the source moved, which needs no patch and is ticket 02's
    workaround 2. That brackets the answer: the patched cost lies somewhere in
    [1x, Nx], and if Nx is already affordable the ~40-line patch is not on the
    critical path at all.
    """
    per_scene = []
    for scene in scenes[:1]:  # one scene is enough to bracket a ratio
        sim = build_sim(scene, with_camera=False)
        out: Dict[str, Any] = {"scene": scene}
        try:
            source, start, _ = pick_source_and_start(sim, min_dist)
            sensor, uuid = attach_audio(sim, {}, sample_rate)
            src = np.asarray(source, dtype=np.float32).copy()
            src[1] += source_height
            offsets = [np.zeros(3, dtype=np.float32),
                       np.array([1.5, 0.0, 0.0], dtype=np.float32),
                       np.array([0.0, 0.0, 1.5], dtype=np.float32)]

            render_once(sim, sensor, uuid, start, src)  # warm the context
            counts: Dict[str, Any] = {}
            for n in (1, 2, 3):
                times = []
                for _ in range(repeats):
                    t0 = time.time()
                    for k in range(n):
                        pos = src + offsets[k]
                        if sim.pathfinder.is_loaded:
                            snapped = sim.pathfinder.snap_point(pos)
                            if np.all(np.isfinite(snapped)):
                                pos = np.asarray(snapped, dtype=np.float32)
                                pos[1] += source_height
                        sensor.setAudioSourceTransform(np.asarray(pos, dtype=np.float32))
                        sim.get_sensor_observations()
                    times.append(time.time() - t0)
                counts["n{}".format(n)] = summarize(times)
                print("  {} source(s) sequential: {} ms median".format(
                    n, counts["n{}".format(n)]["median_ms"]), flush=True)
            out["sequential"] = counts
            one = counts["n1"]["median_ms"]
            out["scaling_vs_1"] = {
                k: round(v["median_ms"] / one, 3) for k, v in counts.items()} if one else None
            out["note"] = ("sequential upper bound on a STOCK build; the patched "
                           "single-simulate cost is bounded above by these numbers "
                           "and below by n1")
        except Exception as exc:
            out["error"] = repr(exc)
            print("  FAILED: {}".format(exc), flush=True)
        finally:
            sim.close()
        per_scene.append(out)
    return {"per_scene": per_scene}


# ----------------------------------------------------------------------


def build_verdict() -> Dict[str, Any]:
    """Apply the pre-registered thresholds. No judgement left to the reader."""
    sweep = REPORT.get("02_sweep", {}).get("per_scene", [])
    rows: List[Dict[str, Any]] = []
    for scene in sweep:
        for r in scene.get("results", []):
            ms = r.get("steady_state", {}).get("median_ms")
            if ms is None:
                continue
            rows.append({
                "scene": os.path.basename(scene.get("scene", "?")),
                "label": r["label"],
                "steady_ms": ms,
                "admissible": r.get("gradient_admissible"),
                "rho": r.get("gradient_rho"),
                "dr_db": r.get("gradient_dynamic_range_db"),
                "rho_nlos": r.get("gradient_rho_nlos"),
                "dr_nlos_db": r.get("gradient_dynamic_range_nlos_db"),
                "n_nlos": r.get("n_nlos_samples"),
                "episode_s": round(ms * STEP_BUDGET / 1000.0, 1),
            })

    # Collapse per-scene rows to per-config. A preset only counts as admissible
    # if its gradient is climbable in EVERY scene it was measured in — a config
    # that works in one room and goes flat in the next is not a recommendation.
    # Cost is quoted at the WORST scene, so the budget is not read off the
    # friendliest geometry.
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    configs: List[Dict[str, Any]] = []
    for label, group in by_label.items():
        worst = max(group, key=lambda r: r["steady_ms"])
        configs.append({
            "label": label,
            "scenes": len(group),
            "steady_ms_worst": worst["steady_ms"],
            "episode_s_worst": worst["episode_s"],
            "admissible_everywhere": all(bool(g["admissible"]) for g in group),
            "admissible_scenes": sum(1 for g in group if g["admissible"]),
            "rho_worst": max((g["rho"] for g in group if g["rho"] is not None),
                             default=None),
        })

    admissible = [c for c in configs if c["admissible_everywhere"]]
    best = min(admissible, key=lambda c: c["steady_ms_worst"]) if admissible else None
    if best is not None:  # present the verdict in the same keys the table uses
        best = dict(best, steady_ms=best["steady_ms_worst"],
                    episode_s=best["episode_s_worst"])

    # On a bad run the blocker list is the deliverable, not a vague verdict.
    blockers: List[str] = []
    if REPORT.get("_scene_discovery_error"):
        blockers.append("scene discovery: {}".format(REPORT["_scene_discovery_error"]))
    for name, res in REPORT.items():
        if isinstance(res, dict) and res.get("ok") is False:
            blockers.append("{}: {}".format(name, res.get("error")))
    for scene in sweep:
        if scene.get("error"):
            blockers.append("{}: {}".format(
                os.path.basename(scene.get("scene", "?")), scene["error"]))

    if best is None:
        verdict = "INDETERMINATE"
        partial = [c for c in configs if c["admissible_scenes"]]
        if blockers:
            reason = "the measurement did not complete — {} blocker(s): {}".format(
                len(blockers), " | ".join(blockers[:4]))
        elif partial:
            reason = ("{} config(s) were gradient-admissible in some scenes but not "
                      "all — no preset can be recommended across scenes".format(
                          len(partial)))
        else:
            reason = ("no configuration produced a climbable gradient — the walk "
                      "was degenerate (check start_geodesic_m and walk_steps)")
    elif best["steady_ms"] <= MS_AFFORDABLE:
        verdict = "LIVE_EVERY_STEP_HOLDS"
        reason = "{} at {} ms/step (worst scene) = {} s per {}-step episode".format(
            best["label"], best["steady_ms"], best["episode_s"], STEP_BUDGET)
    elif best["steady_ms"] <= MS_TOLERABLE:
        verdict = "LIVE_EVERY_STEP_TOLERABLE"
        reason = ("{} at {} ms/step (worst scene) = {} s per episode; holds, but "
                  "audio is a visible cost".format(
                      best["label"], best["steady_ms"], best["episode_s"]))
    else:
        verdict = "THROTTLE_REQUIRED"
        reason = ("cheapest gradient-admissible config is {} at {} ms/step "
                  "(worst scene) = {} s per episode; the map's destination needs "
                  "amending to the throttled variant".format(
                      best["label"], best["steady_ms"], best["episode_s"]))

    return {
        "verdict": verdict,
        "reason": reason,
        "best_admissible": best,
        "blockers": blockers,
        "per_config": sorted(configs, key=lambda c: c["steady_ms_worst"]),
        "table": sorted(rows, key=lambda r: (r["scene"], r["steady_ms"])),
        "thresholds": {
            "step_budget": STEP_BUDGET,
            "ms_affordable": MS_AFFORDABLE,
            "ms_tolerable": MS_TOLERABLE,
            "gradient_rho_max": GRADIENT_RHO_MAX,
            "gradient_dynamic_range_min_db": GRADIENT_DR_MIN_DB,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--scene", action="append", default=[],
                    help="HM3D .glb (repeatable; auto-discovered if omitted)")
    ap.add_argument("--max-scenes", type=int, default=2,
                    help="scenes to auto-discover — 2 gives the scene-size comparison")
    ap.add_argument("--walk-steps", type=int, default=20)
    ap.add_argument("--step-m", type=float, default=0.5,
                    help="navmesh spacing between listener samples")
    ap.add_argument("--min-dist", type=float, default=6.0,
                    help="minimum start-to-source geodesic distance")
    ap.add_argument("--source-height", type=float, default=1.0,
                    help="metres above the navmesh to place the source")
    ap.add_argument("--sample-rate", type=int, default=None,
                    help="override sampleRate (default: leave at the build's 44100)")
    ap.add_argument("--source-repeats", type=int, default=5)
    ap.add_argument("--with-camera-delta", action="store_true",
                    help="re-measure baseline with an RGB camera attached, to "
                         "quantify the confound in ticket 04's 0.6013 s")
    ap.add_argument("--skip-source-count", action="store_true")
    args = ap.parse_args()

    print("=" * 74, flush=True)
    print("ticket 06 render-cost probe — is live-every-step audio affordable?", flush=True)
    print("=" * 74, flush=True)

    probe_interpreter()
    probe_defaults_recheck()

    try:
        scenes = find_scenes(args.scene, args.max_scenes)
        print("\nscenes: {}".format(", ".join(scenes)), flush=True)
    except Exception as exc:
        scenes = []
        REPORT["_scene_discovery_error"] = repr(exc)
        print("\nscene discovery FAILED: {!r}".format(exc), flush=True)

    if scenes:
        probe_sweep(scenes, args.walk_steps, args.step_m, args.min_dist,
                    args.sample_rate, args.source_height, args.with_camera_delta)
        if not args.skip_source_count:
            probe_source_count(scenes, args.source_repeats, args.sample_rate,
                               args.source_height, args.min_dist)

    REPORT["_verdict"] = build_verdict()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(REPORT, fh, indent=2, sort_keys=True, default=str)

    banner("VERDICT")
    v = REPORT["_verdict"]
    print("  {}".format(v["verdict"]), flush=True)
    print("  {}".format(v["reason"]), flush=True)
    print("\n  {:<18} {:<24} {:>9} {:>11} {:>7} {:>9} {:>7}".format(
        "scene", "config", "ms/step", "s/episode", "rho", "rho_nlos", "climb"), flush=True)
    for r in v["table"]:
        print("  {:<18} {:<24} {:>9} {:>11} {:>7} {:>9} {:>7}".format(
            r["scene"][:18], r["label"][:24], r["steady_ms"], r["episode_s"],
            r["rho"] if r["rho"] is not None else "?",
            r["rho_nlos"] if r.get("rho_nlos") is not None else "-",
            "yes" if r["admissible"] else "no"), flush=True)
    print("\n  rho_nlos is the non-line-of-sight gradient — the number that says "
          "whether\n  the agent can climb toward a source it cannot see. "
          "'-' means the walk had\n  fewer than 3 non-LOS samples, not that the "
          "gradient failed.", flush=True)
    print("\n  report: {}".format(args.out), flush=True)
    return 0 if v["verdict"] != "INDETERMINATE" else 1


if __name__ == "__main__":
    sys.exit(main())
