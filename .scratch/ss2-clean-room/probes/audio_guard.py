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
   at the **file-descriptor** level — which ``contextlib.redirect_stderr`` does not
   touch. A guard built on ``redirect_stderr`` captures nothing and passes vacuously,
   which is this ticket's own failure mode. Hence the fd-level capture, and the canary.

   **The log is split across two descriptors, and ticket 16 corrected this module on
   the point.** ``ESP_DEBUG`` expands to ``Corrade::Utility::Debug`` (``Logging.h:326``),
   whose ``defaultOutput()`` is ``&std::cout`` (Corrade ``Debug.cpp:525``), while
   ``ESP_WARNING``/``ESP_ERROR`` expand to ``Warning``/``Error``, whose
   ``defaultOutput()`` is ``&std::cerr`` (``:526-527``). So the ``Vertex count`` canary
   (``AudioSensor.cpp:499``, an ``ESP_DEBUG``) arrives on **fd 1** and every RLRA
   failure on **fd 2**. Capturing fd 2 alone sees an empty stream over a perfectly
   healthy render — the canary would have failed every run.

   That split is also the *only* severity signal there is. ``buildMessagePrefix``
   (``Logging.cpp:149-152``) renders ``"[HH:MM:SS:uuuuuu]:[Subsystem] file(line)::func : "``
   and Corrade adds no severity tag, so nothing in the text says "Error". **The stream
   is the severity**: fd 2 carries Warning and Error and nothing else.

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
    "HABITAT_LOG_PREFIX_RE",
    "KNOWN_DYNAMIC_ATTRS",
    "MIN_SCENE_VERTICES",
    "apply_audio_config",
    "arm_audio_context",
    "assert_no_swallowed_keys",
    "bound_field_names",
    "capture_habitat_logs",
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
# with aliases Verbose=Debug and Quiet=Error. Pinned rather than inherited because an
# operator setting HABITAT_SIM_LOG=quiet to reduce noise would otherwise silently
# disarm invariant 2.
#
# SOURCE-VERIFIED by ticket 16, where it was an inference. The subsystem an ESP_DEBUG
# resolves to comes from C++ namespace lookup on `espLoggingSubsystem()`
# (`Logging.h:312`), and `ESP_ADD_SUBSYSTEM_FN(sensor)` installs that function in
# `esp::sensor` — which is exactly the namespace `AudioSensor.cpp` opens (`:12-13`), so
# its ESP_DEBUG/ESP_ERROR resolve to `Subsystem::sensor`, rendered "Sensor". Likewise
# ResourceManager.cpp is `esp::assets` -> "Assets", where the empty-mesh cast failure
# lives. Note the default level is already Verbose (== Debug, `Logging.h:167`), so on an
# untouched env this pin changes nothing; its whole job is to survive an operator who
# turned logging down.
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

# The subsystem names, verbatim from `subsystemNames` (Logging.h) — the enum and the
# array are static_assert'd to the same length in Logging.cpp:30, so this list is the
# whole set.
_SUBSYSTEM_NAMES: Tuple[str, ...] = (
    "Default", "Gfx", "Scene", "Sim", "Physics", "Nav", "Metadata",
    "Geo", "IO", "URDF", "Core", "Assets", "Sensor", "Agent",
)

# This matches habitat-sim's log PREFIX — it does not detect severity, because no such
# marker exists. `buildMessagePrefix` (Logging.cpp:149-152) formats
# "[{h}:{m}:{s}:{us}]:[{Subsystem}] {file}({line})::{func} : " and Corrade prepends no
# severity tag of its own, so the earlier `r"\[Error\]"` pattern could never match
# anything and its arm of invariant 2 was silently dead.
#
# What it is FOR: on fd 2 the severity is already known (Corrade routes only Warning and
# Error there), so the one thing left to establish is whether a line came from
# habitat-sim at all rather than from some third party that also writes to stderr. That
# is provenance, and the prefix is an exact test for it.
HABITAT_LOG_PREFIX_RE = re.compile(
    r"\]:\[(?:" + "|".join(_SUBSYSTEM_NAMES) + r")\] \S+\(\d+\)::"
)

# If the capture comes back with none of these, either the fd redirect did not take or
# logging is turned down far enough to hide the errors invariant 2 exists to catch.
# Either way the log scan proved nothing, and saying so is the whole point.
#
# Both of these are ESP_DEBUG output and therefore arrive on fd 1, not fd 2:
# "Vertex count" is AudioSensor.cpp:499 and "[Audio] " is its `logHeader_` (AudioSensor.h:45).
LOG_CANARY_SUBSTRINGS: Tuple[str, ...] = ("Vertex count", "[Audio]")


# ----------------------------------------------------------------------
# fd-level log capture
# ----------------------------------------------------------------------


class _CapturedLogs:
    """Holds the text captured per file descriptor. Populated on context exit.

    ``stdout`` and ``stderr`` are kept apart because the split *is* the severity
    signal — see invariant 2 in the module docstring. ``text`` concatenates them for
    substring scans that do not care which stream a line came from; it does not
    preserve interleaving, and nothing should depend on ordering across streams.
    """

    def __init__(self) -> None:
        self.streams: Dict[int, str] = {}

    @property
    def stdout(self) -> str:
        return self.streams.get(1, "")

    @property
    def stderr(self) -> str:
        return self.streams.get(2, "")

    @property
    def text(self) -> str:
        return self.stdout + self.stderr


class capture_habitat_logs:
    """Redirect **file descriptors 1 and 2** to temp files for the duration of the block.

    Both descriptors, because habitat-sim splits its own log across them: ``ESP_DEBUG``
    is a ``Corrade::Utility::Debug`` and goes to ``std::cout``, ``ESP_WARNING`` and
    ``ESP_ERROR`` go to ``std::cerr``. Capturing fd 2 alone misses the ``Vertex count``
    canary entirely and reports an empty log over a healthy render.

    Not ``contextlib.redirect_stdout``/``redirect_stderr``: those rebind Python-level
    objects the C++ logger never touches. habitat-sim writes to the descriptors
    directly, so only an ``os.dup2`` on them sees it.

    Both original fds are restored in ``__exit__`` whether or not the block raised.
    """

    _PY_STREAMS = {1: "stdout", 2: "stderr"}

    def __init__(self, fds: Sequence[int] = (1, 2)) -> None:
        self._fds: Tuple[int, ...] = tuple(fds)
        self._saved: Dict[int, int] = {}
        self._tmp: Dict[int, Tuple[int, str]] = {}
        self.captured = _CapturedLogs()

    def _flush_python(self) -> None:
        # Flush Python's own buffers first, or its pending bytes land in the capture —
        # which on fd 1 would mean the caller's own print() output.
        for name in self._PY_STREAMS.values():
            try:
                getattr(sys, name).flush()
            except Exception:
                pass

    def __enter__(self) -> _CapturedLogs:
        self._flush_python()
        for fd in self._fds:
            self._saved[fd] = os.dup(fd)
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix="audioguard-fd{}-".format(fd), suffix=".log"
            )
            self._tmp[fd] = (tmp_fd, tmp_path)
            os.dup2(tmp_fd, fd)
        return self.captured

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._flush_python()
        for fd, saved_fd in self._saved.items():
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        self._saved.clear()
        for fd, (tmp_fd, tmp_path) in self._tmp.items():
            os.close(tmp_fd)
            try:
                with open(tmp_path, "r", errors="replace") as fh:
                    self.captured.streams[fd] = fh.read()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        self._tmp.clear()
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
    stdout_chars: int = 0
    stderr_chars: int = 0
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
            # Split out because it is diagnostic on its own: a healthy first render is
            # expected to be all fd 1 and an empty fd 2, and the reverse would mean the
            # canary moved.
            "stdout_chars": self.stdout_chars,
            "stderr_chars": self.stderr_chars,
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
    habitat_prefix_re: "re.Pattern" = HABITAT_LOG_PREFIX_RE,
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

    with capture_habitat_logs() as captured:
        ir = render()
    report.stdout_chars = len(captured.stdout)
    report.stderr_chars = len(captured.stderr)
    report.log_chars = report.stdout_chars + report.stderr_chars

    # --- invariant 2: the log scan, and whether it proved anything --------------
    # The canary is ESP_DEBUG output and lands on fd 1; the scan is over both streams
    # anyway, so a future habitat-sim that reroutes its Debug output still trips it.
    report.log_canary_seen = any(marker in captured.text for marker in canary_substrings)
    if require_log_canary and not report.log_canary_seen:
        failures.append(
            "fd capture returned {} chars on stdout and {} on stderr, and none of {} — "
            "either the fd redirect did not take or HABITAT_SIM_LOG is turned down below "
            "Debug. Invariant 2 is unverified, not satisfied; call pin_habitat_logging() "
            "before importing habitat_sim.".format(
                report.stdout_chars, report.stderr_chars, list(canary_substrings)
            )
        )
    # Two ways a line is fatal, and they cover different gaps. The substrings name the
    # specific failures that produce a plausible IR over a broken context, wherever they
    # are logged. The stream rule is the general one: Corrade routes ONLY Warning and
    # Error to fd 2, so habitat-sim writing there at all during the first render is by
    # construction a severity event — the prefix match is what keeps third-party stderr
    # noise from counting as one.
    for stream_fd, stream_text in ((1, captured.stdout), (2, captured.stderr)):
        for line in stream_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            fatal = any(marker in line for marker in fatal_substrings)
            if not fatal and stream_fd == 2 and habitat_prefix_re.search(line):
                fatal = True
            if fatal and stripped not in report.fatal_log_lines:
                report.fatal_log_lines.append(stripped)
    if report.fatal_log_lines:
        failures.append(
            "habitat-sim logged {} error/warning line(s) during the first render — every "
            "RLRA_* failure is handled by ESP_ERROR + bare return and is invisible to "
            "Python, so this log is the only channel:\n  {}".format(
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
