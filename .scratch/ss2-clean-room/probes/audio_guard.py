#!/usr/bin/env python3
"""Ticket 12 — the audio context guard.

A zero-geometry RLR audio context still returns plausible-looking audio. Nothing in
habitat-sim stops it: ``loadMesh`` submits whatever the joined scene mesh contains,
logs ``Vertex count : N`` at ESP_DEBUG, and never compares N to zero. That is the same
failure class that invalidated the ``anommxv`` headline — a whole matrix ran before
anyone noticed the interrupt was firing on the wrong thing.

This module is what makes that state unreachable rather than merely unlikely. It is
deliberately free of any ``habitat_sim`` import: every simulator object it touches is
injected, so the whole guard unit-tests on a Mac (see ``test_audio_guard.py``) and only
``audioguard_probe.py`` needs the box.

Three invariants, and one correction to how ticket 12 originally stated them.

1. **Non-empty audio mesh.** Load-bearing, and the backstop for the other two — every
   way the mesh upload can fail ends in a short vertex count. Read via
   ``AudioSensor::writeSceneMeshOBJ``, which is ``RLRA_WriteSceneMeshOBJ(...) ==
   RLRA_Success`` and therefore reports the geometry the *engine* holds, not the
   geometry habitat thinks it handed over.

   **This cannot run "at context creation".** Read at ``4f61e321``, the mesh upload is
   lazy: ``createAudioSimulator()`` sets ``newInitialization_`` and the first
   ``runSimulation()`` consumes it. At construction there is no mesh to assert on. So
   ``arm_audio_context`` performs the first render *itself* rather than asking the
   caller to sequence it correctly.

2. **RLRA errors.** habitat-sim already compares every ``RLRA_*`` return against
   ``RLRA_Success`` — but the handler is ``ESP_ERROR() << ...; return;`` and
   ``runSimulation`` is ``void``, so the failure is detected in C++ and *discarded
   before Python*. There is no return code, no exception, no flag. The only way to see
   it without patching the bindings is to read the log, and habitat-sim logs from C++
   to **file descriptor 2** — which ``contextlib.redirect_stderr`` does not touch. A
   guard built on ``redirect_stderr`` captures nothing and passes vacuously, which is
   this ticket's own failure mode. Hence the fd-level capture, and the canary below.

3. **Unknown spec keys.** ``AudioSensorSpec`` carries ``py::dynamic_attr()``, so
   ``spec.irTime = 4.0`` (renamed to ``maxIRLength`` on this branch) attaches a fresh
   Python attribute and is never read. ``RLRAudioPropagationConfiguration`` does not,
   so the same mistake raises there — the validator belongs on the spec and nowhere
   else. ``vars(spec)`` detects it *exactly*: a ``def_readwrite`` field is a data
   descriptor on the type and never lands in the instance ``__dict__``, so a non-empty
   ``__dict__`` is precisely the set of swallowed keys.

Python 3.9 (the SoundSpaces pin), stdlib only.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "AudioContextError",
    "AudioContextReport",
    "KNOWN_DYNAMIC_ATTRS",
    "MIN_SCENE_VERTICES",
    "apply_audio_config",
    "arm_audio_context",
    "assert_no_swallowed_keys",
    "bound_field_names",
    "capture_fd_stderr",
    "count_obj_vertices",
    "pin_habitat_logging",
]


class AudioContextError(RuntimeError):
    """The audio context is not in a state any result may be quoted from."""


# ----------------------------------------------------------------------
# calibration constants
# ----------------------------------------------------------------------

# Ticket 04's control is 392,356 verts on minival/00800-TEEsavR23oF, non-semantic path.
# The floor is deliberately not `> 0`: the literal invariant would pass a degenerate
# three-vertex mesh, which produces the same direct-path-only IR as an empty one. Two
# orders of magnitude below the measured control is far enough below any real HM3D
# scene to never fire spuriously, and far enough above zero to catch the degenerate case.
MIN_SCENE_VERTICES = 10_000

# habitat-sim reads this at import (esp/core/Logging.h). The grammar is
# `SUBSYSTEM[,SUBSYSTEM]*=LEVEL` joined by `:`, levels VeryVerbose|Debug|Warning|Error
# with aliases Verbose=Debug and Quiet=Error. `Sensor` covers AudioSensor.cpp's own
# ESP_DEBUG/ESP_ERROR; `Assets` covers ResourceManager.cpp, where the empty-mesh cast
# failure lives. Pinned rather than inherited because an operator setting
# HABITAT_SIM_LOG=quiet to reduce noise would otherwise silently disarm invariant 2.
#
# BOX-VERIFIED BY audioguard_probe.py, not assumed: the subsystem a given ESP_DEBUG
# resolves to is derived from its C++ namespace, and this is the inference. The probe
# asserts the vertex line actually appears under this value.
HABITAT_SIM_LOG_PIN = "Sensor,Assets=Debug"

# Verbatim from the source at 4f61e321. These are the failures that produce a plausible
# IR over a broken context, so each one is fatal rather than merely logged.
FATAL_LOG_SUBSTRINGS: Tuple[str, ...] = (
    # ResourceManager.cpp:2937-2943 — joinSemanticHierarchy's cast fails and returns
    # bare, so the node is skipped and the joined mesh comes back empty.
    "Could not get the GenericSemanticMeshData",
    # AudioSensor.cpp — the RLRA_* handlers. Each logs and returns void.
    "Error while running audio simulation",
    "Error setting audio source position",
    "Error setting audio listener transform",
)

# habitat-sim's log prefix carries a severity marker. The exact rendering is NOT read
# from source here — it is built by buildMessagePrefix() plus Corrade's own severity
# output — so this pattern is a default that audioguard_probe.py VALIDATES on the box by
# provoking a real ESP_ERROR and asserting it matches. Treat a probe failure here as a
# finding about the pattern, not about the run.
DEFAULT_SEVERITY_RE = re.compile(r"\[Error\]")

# If the capture comes back with none of these, either the fd redirect did not take or
# logging is turned down far enough to hide the errors invariant 2 exists to catch.
# Either way the log scan proved nothing, and saying so is the whole point.
LOG_CANARY_SUBSTRINGS: Tuple[str, ...] = ("Vertex count", "[Audio]")


# ----------------------------------------------------------------------
# fd-level stderr capture
# ----------------------------------------------------------------------


class _StderrCapture:
    """Holds the text captured from file descriptor 2. Populated on context exit."""

    def __init__(self) -> None:
        self.text: str = ""


class capture_fd_stderr:
    """Redirect **file descriptor 2** to a temp file for the duration of the block.

    Not ``contextlib.redirect_stderr``: that rebinds ``sys.stderr``, a Python-level
    object the C++ logger never touches. habitat-sim writes to fd 2 directly, so only an
    ``os.dup2`` on the descriptor sees it.

    The original fd is restored in ``__exit__`` whether or not the block raised.
    """

    def __init__(self) -> None:
        self._saved_fd: Optional[int] = None
        self._tmp_fd: Optional[int] = None
        self._tmp_path: Optional[str] = None
        self.captured = _StderrCapture()

    def __enter__(self) -> _StderrCapture:
        # Flush Python's own buffer first, or its pending bytes land in the capture.
        try:
            sys.stderr.flush()
        except Exception:
            pass
        self._saved_fd = os.dup(2)
        self._tmp_fd, self._tmp_path = tempfile.mkstemp(prefix="audioguard-stderr-", suffix=".log")
        os.dup2(self._tmp_fd, 2)
        return self.captured

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        if self._saved_fd is not None:
            os.dup2(self._saved_fd, 2)
            os.close(self._saved_fd)
            self._saved_fd = None
        if self._tmp_fd is not None:
            os.close(self._tmp_fd)
            self._tmp_fd = None
        if self._tmp_path is not None:
            try:
                with open(self._tmp_path, "r", errors="replace") as fh:
                    self.captured.text = fh.read()
            finally:
                try:
                    os.unlink(self._tmp_path)
                except OSError:
                    pass
                self._tmp_path = None
        return False  # never swallow


# ----------------------------------------------------------------------
# invariant 3 — the key validator (AudioSensorSpec only)
# ----------------------------------------------------------------------


def bound_field_names(obj: Any) -> frozenset:
    """The object's real C++-bound fields.

    ``dir()`` on a pybind object lists exactly the bound members, so introspecting is
    strictly better than a hardcoded list: when the branch renames a field (``irTime``
    to ``maxIRLength``, ticket 11) the rename surfaces as a rejected key instead of a
    silently swallowed one.
    """
    names = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        names.add(name)
    return frozenset(names)


# MEASURED on the box 2026-08-03 (ticket 15's budget run, which is what surfaced
# it — this is the answer to ticket 16's stage-1 question). A stock
# `AudioSensorSpec()` does NOT leave `vars(spec)` empty: the real constructor
# attaches `__noise_model_kwargs` as a genuine instance attribute, while
# `noise_model_kwargs` is a separate bound field. Without this exclusion the
# guard raises on every healthy spec, so invariant 3 would be a false positive
# forever. The unit-test fakes do not reproduce it, which is precisely the class
# of gap ticket 16 exists to close.
KNOWN_DYNAMIC_ATTRS: Tuple[str, ...] = ("__noise_model_kwargs",)


def assert_no_swallowed_keys(spec: Any, allowed: Iterable[str] = ()) -> None:
    """Fail if anything was attached to the spec that is not a bound field.

    Exact, not heuristic: ``def_readwrite`` installs a data descriptor on the type, so a
    real field never reaches the instance ``__dict__``. Whatever is in there is the set
    of keys ``py::dynamic_attr`` swallowed.

    ``allowed`` exists because a legitimate dynamic attribute on this branch would
    otherwise be a permanent false positive. ``KNOWN_DYNAMIC_ATTRS`` is always
    excluded on top of it, so a caller passing its own ``allowed`` cannot
    accidentally re-open a false positive that has already been measured.
    """
    stray = sorted(set(vars(spec)) - set(allowed) - set(KNOWN_DYNAMIC_ATTRS))
    if stray:
        raise AudioContextError(
            "AudioSensorSpec swallowed {} unknown key(s): {}. py::dynamic_attr attaches "
            "these silently and nothing ever reads them — the config did not take. "
            "Bound fields on this branch: {}".format(
                len(stray), ", ".join(stray), ", ".join(sorted(bound_field_names(spec)))
            )
        )


def apply_audio_config(spec: Any, config: Mapping[str, Any]) -> Any:
    """Apply a nested config to an ``AudioSensorSpec``, rejecting unknown keys first.

    Nested mappings route to the matching sub-object (``acousticsConfig``,
    ``channelLayout``); everything else is set on the spec. Every key is checked against
    that object's introspected field list *before* anything is written, so a typo cannot
    half-apply a config.

    Returns the same spec, configured.
    """
    _validate_against(spec, config, path="spec")
    for key, value in config.items():
        target = getattr(spec, key)
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                setattr(target, sub_key, sub_value)
        else:
            setattr(spec, key, value)
    # Belt and braces: an unknown key cannot have got past _validate_against, but this
    # also catches anything a caller attached to the spec before handing it over.
    assert_no_swallowed_keys(spec)
    return spec


def _validate_against(obj: Any, config: Mapping[str, Any], path: str) -> None:
    valid = bound_field_names(obj)
    unknown = sorted(set(config) - valid)
    if unknown:
        raise AudioContextError(
            "unknown audio config key(s) on {}: {}. Valid on this branch: {}".format(
                path, ", ".join(unknown), ", ".join(sorted(valid))
            )
        )
    for key, value in config.items():
        if isinstance(value, Mapping):
            _validate_against(getattr(obj, key), value, path="{}.{}".format(path, key))


# ----------------------------------------------------------------------
# invariant 1 — the mesh read
# ----------------------------------------------------------------------


def count_obj_vertices(path: str) -> int:
    """Count geometric vertices in a Wavefront OBJ.

    ``v `` only — ``vn ``/``vt `` are normals and texture coords and would inflate the
    count. Streamed in binary because the control mesh is ~392k verts / tens of MB.
    """
    count = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"v ") or line.startswith(b"v\t"):
                count += 1
    return count


def _peak_abs(ir: Any) -> float:
    """Largest absolute sample in an IR, without requiring numpy."""
    try:
        return float(max(abs(float(x)) for x in _flatten(ir)))
    except ValueError:  # empty
        return 0.0


def _flatten(value: Any):
    if isinstance(value, (str, bytes)):
        return
    try:
        iterator = iter(value)
    except TypeError:
        yield value
        return
    for item in iterator:
        for leaf in _flatten(item):
            yield leaf


# ----------------------------------------------------------------------
# the guard
# ----------------------------------------------------------------------


@dataclass
class AudioContextReport:
    """What the guard measured. Log this per episode — it is the audit trail."""

    n_vertices: int = 0
    obj_written: bool = False
    ir_peak_abs: float = 0.0
    ir_shape: Optional[Sequence[int]] = None
    ray_efficiency: Optional[float] = None
    source_is_visible: Optional[bool] = None
    log_chars: int = 0
    log_canary_seen: bool = False
    fatal_log_lines: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_vertices": self.n_vertices,
            "obj_written": self.obj_written,
            "ir_peak_abs": self.ir_peak_abs,
            "ir_shape": list(self.ir_shape) if self.ir_shape is not None else None,
            "ray_efficiency": self.ray_efficiency,
            "source_is_visible": self.source_is_visible,
            "log_chars": self.log_chars,
            "log_canary_seen": self.log_canary_seen,
            "fatal_log_lines": list(self.fatal_log_lines),
        }


def arm_audio_context(
    audio_sensor: Any,
    render: Callable[[], Any],
    *,
    min_vertices: int = MIN_SCENE_VERTICES,
    obj_dir: Optional[str] = None,
    keep_obj: bool = False,
    severity_re: "re.Pattern" = DEFAULT_SEVERITY_RE,
    fatal_substrings: Sequence[str] = FATAL_LOG_SUBSTRINGS,
    canary_substrings: Sequence[str] = LOG_CANARY_SUBSTRINGS,
    require_log_canary: bool = True,
) -> AudioContextReport:
    """Perform the first render on a fresh audio context, then assert it is real.

    The guard owns the first render rather than trusting the caller to sequence one,
    because the mesh does not exist until ``runSimulation`` has run once and an
    assertion made before that would pass over nothing.

    ``render`` is a zero-argument callable that triggers exactly one render and returns
    the IR (in the runner, ``lambda: sim.get_sensor_observations()["audio_sensor"]``).

    Every check runs before anything raises, and one ``AudioContextError`` carries all
    of them — a broken context usually trips several, and the first failure alone is
    rarely the diagnosis.
    """
    report = AudioContextReport()
    failures: List[str] = []

    with capture_fd_stderr() as captured:
        ir = render()
    log = captured.text
    report.log_chars = len(log)

    # --- invariant 2: the log scan, and whether it proved anything --------------
    report.log_canary_seen = any(marker in log for marker in canary_substrings)
    if require_log_canary and not report.log_canary_seen:
        failures.append(
            "stderr capture returned {} chars and none of {} — either the fd-2 redirect "
            "did not take or HABITAT_SIM_LOG is turned down below Debug. Invariant 2 is "
            "unverified, not satisfied; call pin_habitat_logging() before importing "
            "habitat_sim.".format(report.log_chars, list(canary_substrings))
        )
    for line in log.splitlines():
        if any(marker in line for marker in fatal_substrings) or severity_re.search(line):
            report.fatal_log_lines.append(line.strip())
    if report.fatal_log_lines:
        failures.append(
            "habitat-sim logged {} error line(s) during the first render — every RLRA_* "
            "failure is handled by ESP_ERROR + bare return and is invisible to Python, "
            "so this log is the only channel:\n  {}".format(
                len(report.fatal_log_lines), "\n  ".join(report.fatal_log_lines[:10])
            )
        )

    # --- invariant 1: the mesh the engine actually holds ------------------------
    obj_path = os.path.join(
        obj_dir or tempfile.gettempdir(), "audioguard-scene-{}.obj".format(os.getpid())
    )
    try:
        report.obj_written = audio_sensor.writeSceneMeshOBJ(obj_path) is True
        if not report.obj_written:
            failures.append(
                "writeSceneMeshOBJ returned falsy — RLRA_WriteSceneMeshOBJ did not "
                "return RLRA_Success, so the audio context could not produce its own "
                "geometry and the vertex count is unknowable"
            )
        else:
            report.n_vertices = count_obj_vertices(obj_path)
            if report.n_vertices < min_vertices:
                failures.append(
                    "audio context holds {} vertices, below the {} floor. loadMesh "
                    "submits whatever it is given and only ESP_DEBUG-logs the count, so "
                    "a zero-geometry context renders a direct-path-only IR that looks "
                    "entirely plausible. Ticket 04's control is 392,356 verts.".format(
                        report.n_vertices, min_vertices
                    )
                )
    finally:
        if not keep_obj:
            try:
                os.unlink(obj_path)
            except OSError:
                pass

    # --- corroboration: the render did something --------------------------------
    report.ir_shape = getattr(ir, "shape", None)
    report.ir_peak_abs = _peak_abs(ir)
    if report.ir_peak_abs <= 0.0:
        failures.append("first render returned a silent IR (peak abs 0.0)")

    # Recorded, never asserted: RLRA_GetIndirectRayEfficiency's value over a
    # zero-geometry context is unknown (the .so is closed), so there is no honest
    # threshold. Ticket 04 measured 0.548 on the control — worth watching, not gating.
    report.ray_efficiency = _safe_call(audio_sensor, "getRayEfficiency", float)
    # Likewise sourceIsVisible: with no geometry nothing occludes, so it reads True
    # everywhere. Diagnostic, and only meaningful against a known-occluded pair.
    report.source_is_visible = _safe_call(audio_sensor, "sourceIsVisible", bool)

    if failures:
        raise AudioContextError(
            "audio context failed {} invariant(s):\n\n{}".format(
                len(failures), "\n\n".join("- " + f for f in failures)
            )
        )
    return report


def _safe_call(obj: Any, name: str, cast: Callable[[Any], Any]) -> Optional[Any]:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return cast(fn())
    except Exception:
        return None


# ----------------------------------------------------------------------
# logging pin — must run before habitat_sim is imported
# ----------------------------------------------------------------------


def pin_habitat_logging(level: str = HABITAT_SIM_LOG_PIN) -> str:
    """Pin ``HABITAT_SIM_LOG`` so invariant 2 cannot be disarmed by the environment.

    habitat-sim reads the variable at import, so this raises rather than no-ops if
    ``habitat_sim`` is already loaded — silently doing nothing is exactly the class of
    failure this module exists to remove.
    """
    if "habitat_sim" in sys.modules:
        raise AudioContextError(
            "habitat_sim is already imported — HABITAT_SIM_LOG is read at import time, "
            "so pinning it now would have no effect. Call pin_habitat_logging() first."
        )
    os.environ["HABITAT_SIM_LOG"] = level
    return level
