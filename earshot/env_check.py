"""``assert_env()`` — ticket 17's runtime assertion. One implementation, two callers.

``earshot/tools/bootstrap_ss2.sh`` runs ``python -m earshot.env_check --strict`` as its
verdict; the runtime entry point calls ``assert_env()`` before an episode. Splitting the
build from the assertion is the whole reason this is importable Python rather than more
bash: an assertion stranded in the gate cannot run at episode time, which is exactly
when a drifted env produces results instead of an error.

## Capability-shaped, never provenance-shaped

Ticket 13 is the whole argument. ``transformers`` reported **4.57.6 both before and
after** the fix, and ``ClapModel`` imported cleanly the entire time it was a
``DummyObject`` that raised only on instantiation. A version-set comparison would have
printed green through the whole failure. So every probe here does the thing: allocates
on the GPU and reads the result back, resolves the enum **member**, instantiates the
model and reads a finite logit.

## A probe that did not run is not a pass

This is the same ticket's *other* bug, and it is the one this module is shaped around.
``oneenv_gate.sh``'s torch layer skipped on mere importability and reported success —
a layer that computed the right answer and then did not use it. So:

- ``ProbeStatus.NOT_RUN`` exists and is **never** green. An import that raised, a probe
  that could not complete, a check that was skipped: all the same verdict.
- ``judge()`` takes the set of probes it **expects**, and a missing name is red. A probe
  that silently stopped being emitted cannot pass by absence.
- ``judge()`` is **pure**. Ticket 19: *given a failing probe result, does ``assert_env``
  raise* is the highest-value assertion in this module and it needs no box at all, so
  the probing and the judging are separate functions and the Mac suite injects results.

## The layer, and what it costs

``_tree.LAYER_IMPORTS`` puts this module at ``()``: it imports nothing inside
``earshot``, because it has to run when the tree is otherwise unusable. Two consequences,
both deliberate:

- ``EnvReport`` is defined here rather than in ``report/``, and
  ``report/artifacts.write_env_report`` takes a mapping.
- ``--strict`` **prints**; it does not write ``env_report.json``. ADR-0013 makes
  ``report/artifacts.py`` the only writer in the tree, and the wiring layer does
  ``write_env_report(run_dir, assert_env().as_dict())``.

Importing this module has already run ``pin_habitat_logging()`` via ``earshot/__init__``,
which is what dissolves ticket 17's ordering contradiction (ADR-0013): the enum probe is
free to import habitat-sim because the pin cannot have been missed — a late import of
this package after habitat-sim is loaded raises in ``__init__`` before reaching here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "EnvCheckError",
    "ProbeStatus",
    "Probe",
    "EnvReport",
    "REQUIRED_PROBES",
    "CLAP_PROBE",
    "judge",
    "run_probes",
    "assert_green",
    "assert_env",
    "parse_pins",
    "parse_resolved",
    "compare_resolved_against_constraints",
    "ProvenanceComparison",
    "main",
]

# provenance: box — the 2022-era habitat-sim tree dies on numpy 2.x, and ticket 04
# measured the working set at 1.23.5. The pin is `< 1.24` rather than `== 1.23.5`
# because this is the assertion, not the constraints file: a patch release inside 1.23
# is fine and a hard equality here would fail a healthy env for a reason that is not a
# defect.
NUMPY_MAX_EXCLUSIVE = (1, 24)

# provenance: box — transformers gates its torch backend on >= 2.1.0, which is ticket
# 13's exact failure: torch 2.0.1 against transformers 4.57.6 turned ClapModel into a
# DummyObject. The capability probe below is the real check; this metadata bound is here
# because it is free and because it names the number that moved.
TORCH_MIN_VERSION = (2, 1)

PINNED_PROBE = "pinned_versions_match"

# `earshot/tools/ss2-constraints.txt`, relative to this file — this module sits at layer
# () and imports nothing inside the package, so it cannot ask another module where the
# tree is.
CONSTRAINTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tools", "ss2-constraints.txt"
)

REQUIRED_PROBES: FrozenSet[str] = frozenset(
    {
        "numpy_below_1_24",
        "torch_min_version",
        "torch_cuda_allocation",
        "habitat_sim_audio_enum_member",
        PINNED_PROBE,
    }
)

# Requested, not required. Ticket 17 puts the CLAP assertion where the model is
# constructed — "paid only by runs that use it" — and ticket 22 made `audio/clap.py`
# pure (the encoder is injected), so in the clean room that construction happens in
# `task/`, which may import this module. The bootstrap asks for it explicitly.
CLAP_PROBE = "clap_instantiable"

# provenance: box — ticket 13's known-good CLAP checkpoint.
CLAP_MODEL_ID = "laion/clap-htsat-unfused"


class EnvCheckError(RuntimeError):
    """The environment cannot run an episode. Raised before the episode, never during."""


class ProbeStatus(Enum):
    """Three states, and the third is the point.

    ``NOT_RUN`` is what a skipped or un-completable probe returns. It is never green,
    because ticket 13's gate reported success from exactly that position.
    """

    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class Probe:
    """One capability, asked and answered.

    ``measured`` is a tuple of string pairs rather than a dict so the dataclass is
    genuinely immutable, and stringified so no numpy scalar or torch dtype leaks into
    the JSON artefact.
    """

    name: str
    status: ProbeStatus
    detail: str = ""
    measured: Tuple[Tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is ProbeStatus.PASS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "measured": {key: value for key, value in self.measured},
        }


def _measured(**values: Any) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in values.items()))


@dataclass(frozen=True)
class EnvReport:
    """What ``assert_env`` found. Lands as ``env_report.json`` via the wiring layer.

    ``environment`` records which interpreter and which conda prefix answered — the
    forensic half. Ticket 13's diagnosis cost a whole ticket partly because nothing on
    disk said which env a result came from.
    """

    probes: Tuple[Probe, ...] = ()
    missing: Tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)

    @property
    def failed(self) -> Tuple[str, ...]:
        """Every probe that is not a PASS — ``FAIL`` and ``NOT_RUN`` alike."""
        return tuple(probe.name for probe in self.probes if not probe.ok)

    @property
    def green(self) -> bool:
        return not self.missing and not self.failed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "green": self.green,
            "missing": list(self.missing),
            "failed": list(self.failed),
            "probes": [probe.as_dict() for probe in self.probes],
            "environment": dict(self.environment),
        }

    def summary(self) -> str:
        lines = ["env_check: {}".format("GREEN" if self.green else "RED")]
        for probe in self.probes:
            lines.append(
                "  [{:<7}] {:<32} {}".format(probe.status.value, probe.name, probe.detail)
            )
        for name in self.missing:
            lines.append(
                "  [missing] {:<32} expected and never emitted — a probe that stopped "
                "running cannot pass by absence".format(name)
            )
        for key, value in sorted(self.environment.items()):
            lines.append("  env  {:<32} {}".format(key, value))
        return "\n".join(lines)


# ----------------------------------------------------------------------
# the judge — pure, and the reason the probing is a separate function
# ----------------------------------------------------------------------


def judge(probes: Sequence[Probe], expected: Iterable[str]) -> EnvReport:
    """Turn probe results into a verdict. No imports, no I/O, no environment.

    This is ticket 19's Mac-with-injected-results row, and it is the highest-value
    assertion in the module because it is where ticket 13's bug lived: the layer that
    computed the right answer and then did not act on it.

    A duplicate probe name raises rather than shadowing. Two probes disagreeing under
    one name is a caller bug that would otherwise resolve to whichever ran last.
    """
    seen: Dict[str, Probe] = {}
    for probe in probes:
        if probe.name in seen:
            raise ValueError(
                "two probes named {!r} — one of them would silently shadow the "
                "other".format(probe.name)
            )
        seen[probe.name] = probe
    missing = tuple(sorted(name for name in expected if name not in seen))
    return EnvReport(probes=tuple(probes), missing=missing, environment=describe_environment())


def describe_environment() -> Dict[str, str]:
    """Which interpreter and which env answered. The one place this module reads ``os``.

    ``test_no_env_flags.py`` exempts this file because "reading the resolved environment
    is its whole job" — and an exemption spent on nothing would be an inert pin, the
    same class ticket 17 named for a constraint on a package that is never installed.
    These four are the identity of the thing under test: ``CUDA_VISIBLE_DEVICES``
    changes what the allocation probe even means, and ``HABITAT_SIM_LOG`` is the pin the
    audio guard's canary depends on.

    **Memory is deliberately NOT here**, and the reason is this function's own rule. The
    box inventory behind ``docs/race-box-runbook.md`` §1 recorded GLIBC, cores, GPU, disk
    and mesh coverage but never RAM, which is the resource ``bootstrap_ss2.sh`` then
    oversubscribed by sizing its build from ``nproc`` — a crash rather than a failure,
    because the host dies and no report is sent. The obvious fix was to record it here.
    ADR-0013's graph refused it: ``env_check`` imports nothing intra-package, because it
    is what a half-built tree runs to find out what it is missing, and the layering gate
    caught the attempt. Reading memory would mean either a second copy of
    ``tools/build_jobs.py``'s cgroup arithmetic or a leaf that is no longer a leaf.
    So the measurement lives with the build that needs it, in ``runs/ss2-bootstrap``, and
    this docstring is the pointer rather than a silence someone rediscovers.
    """
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "habitat_sim_log": os.environ.get("HABITAT_SIM_LOG", ""),
    }


# ----------------------------------------------------------------------
# the probes — each one does the thing rather than asking about it
# ----------------------------------------------------------------------


def _version_tuple(text: str) -> Tuple[int, ...]:
    """Leading numeric components of a version string. ``2.2.2+cu118`` -> ``(2, 2, 2)``."""
    parts: List[int] = []
    for chunk in re.split(r"[.\-+]", text.strip()):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts)


def probe_numpy_below_1_24() -> Probe:
    """Metadata, and the one probe that is honestly metadata-shaped.

    numpy's ABI break is a version fact, not a capability: the 2022-era habitat-sim tree
    fails to *import* against 2.x, so there is nothing to exercise that the import does
    not already decide.
    """
    try:
        import numpy
    except Exception as exc:  # pragma: no cover - exercised on a broken env only
        return Probe(
            "numpy_below_1_24",
            ProbeStatus.NOT_RUN,
            "numpy did not import: {}".format(exc),
        )
    version = getattr(numpy, "__version__", "")
    ok = _version_tuple(version) < NUMPY_MAX_EXCLUSIVE
    return Probe(
        "numpy_below_1_24",
        ProbeStatus.PASS if ok else ProbeStatus.FAIL,
        "numpy {}{}".format(
            version,
            "" if ok else " — a pip layer defeated ss2-constraints.txt; find the "
            "install that pulled it",
        ),
        _measured(numpy=version),
    )


def probe_torch_min_version() -> Probe:
    """The pin that moved in ticket 13, asserted as the number it is."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on a broken env only
        return Probe(
            "torch_min_version",
            ProbeStatus.NOT_RUN,
            "torch did not import: {}".format(exc),
        )
    version = getattr(torch, "__version__", "")
    ok = _version_tuple(version) >= TORCH_MIN_VERSION
    return Probe(
        "torch_min_version",
        ProbeStatus.PASS if ok else ProbeStatus.FAIL,
        "torch {}{}".format(
            version,
            "" if ok else " — transformers disables its torch backend below {}, which "
            "turns ClapModel into a DummyObject that imports fine".format(
                ".".join(str(part) for part in TORCH_MIN_VERSION)
            ),
        ),
        _measured(torch=version),
    )


def probe_torch_cuda_allocation() -> Probe:
    """Allocate on the GPU, run an op, read the result back.

    ``torch.cuda.is_available()`` is not this check. A wheel built without kernels for
    the device's architecture reports available and then raises *"no kernel image is
    available for execution on the device"* on the first real op — which is why the
    probe multiplies and copies back rather than asking.

    The device's compute capability is **recorded, not required**. Demanding ``(7, 0)``
    would fail a healthy env on any other card for a reason that is not a defect; what
    has to hold is that this wheel has kernels for whatever card is present, and on our
    box that card is the V100's sm_70.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on a broken env only
        return Probe(
            "torch_cuda_allocation",
            ProbeStatus.NOT_RUN,
            "torch did not import: {}".format(exc),
        )
    if not torch.cuda.is_available():
        return Probe(
            "torch_cuda_allocation",
            ProbeStatus.FAIL,
            "torch.cuda.is_available() is False — no GPU visible to this process "
            "(CUDA_VISIBLE_DEVICES={!r})".format(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        )
    try:
        capability = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        tensor = torch.ones(64, 64, device="cuda")
        value = float((tensor @ tensor).sum().item())
    except Exception as exc:
        return Probe(
            "torch_cuda_allocation",
            ProbeStatus.FAIL,
            "a real allocation failed on the visible GPU: {}".format(exc),
        )
    expected = 64.0 * 64.0 * 64.0
    ok = abs(value - expected) < 1e-3
    return Probe(
        "torch_cuda_allocation",
        ProbeStatus.PASS if ok else ProbeStatus.FAIL,
        "{} sm_{}{} matmul readback {}".format(
            name, capability[0], capability[1], "correct" if ok else "WRONG"
        ),
        _measured(
            device=name,
            capability="{}.{}".format(*capability),
            readback=value,
            expected=expected,
        ),
    )


def probe_habitat_sim_audio_enum_member() -> Probe:
    """Resolve the audio sensor enum **member**, not the class.

    ``AudioSensorSpec`` is bound even in non-audio habitat-sim builds
    (habitat-sim #2340), so its presence proves nothing — ticket 12 found the same shape
    in ``transformers``' ``DummyObject``, where the class imports and only instantiation
    raises. The member is what a non-audio build does not have, and it costs seconds
    against the 90 a full build check costs.
    """
    try:
        # MUST precede habitat_sim and nothing here uses it — see the measurement in
        # `sim/world.py`'s import block. Bare `import habitat_sim` aborts the interpreter
        # with `free(): invalid pointer`, which no `except` can catch, so the honest
        # NOT_RUN branch below would never be reached. Imported inside this probe rather
        # than relied on from `probe_torch_min_version`: a probe that only works when
        # another probe ran first is an ordering dependency between two functions with
        # no call edge, and `assert_env` is not the only caller — `tests/box/` runs them
        # one at a time.
        import torch  # noqa: F401

        import habitat_sim
    except Exception as exc:  # pragma: no cover - exercised on a broken env only
        return Probe(
            "habitat_sim_audio_enum_member",
            ProbeStatus.NOT_RUN,
            "habitat_sim did not import: {}".format(exc),
        )
    try:
        member = habitat_sim.SensorType.AUDIO
    except Exception as exc:
        return Probe(
            "habitat_sim_audio_enum_member",
            ProbeStatus.FAIL,
            "SensorType.AUDIO is absent — this habitat-sim was built without "
            "--audio: {}".format(exc),
        )
    return Probe(
        "habitat_sim_audio_enum_member",
        ProbeStatus.PASS,
        "SensorType.AUDIO resolved on habitat_sim {}".format(
            getattr(habitat_sim, "__version__", "?")
        ),
        _measured(
            habitat_sim=getattr(habitat_sim, "__version__", "?"),
            sensor_type_audio=member,
        ),
    )


def probe_clap_instantiable(model_id: str = CLAP_MODEL_ID) -> Probe:
    """Instantiate CLAP and read a finite logit. Ticket 13's assertion, verbatim in job.

    Importability proves nothing here: ``transformers`` substitutes a ``DummyObject``
    when its torch backend is disabled, and that object imports cleanly and raises only
    when constructed. So the probe constructs.

    Requested rather than required — 153.5 M params and 0.713 GB of VRAM, paid only by
    runs that use CLAP. ``model_id`` is a parameter so the box suite can force the
    failure arm (ADR-0014: a detector ships both arms) by pointing it at nothing.
    """
    try:
        import torch
        from transformers import ClapModel
    except Exception as exc:  # pragma: no cover - exercised on a broken env only
        return Probe(
            CLAP_PROBE, ProbeStatus.NOT_RUN, "transformers/torch did not import: {}".format(exc)
        )
    try:
        model = ClapModel.from_pretrained(model_id)
        model.eval()
        n_params = sum(int(p.numel()) for p in model.parameters())
        with torch.no_grad():
            features = model.get_text_features(
                input_ids=torch.zeros((1, 8), dtype=torch.long),
                attention_mask=torch.ones((1, 8), dtype=torch.long),
            )
        finite = bool(torch.isfinite(features).all().item())
    except Exception as exc:
        return Probe(
            CLAP_PROBE,
            ProbeStatus.FAIL,
            "ClapModel could not be instantiated or produced no logit: {}".format(exc),
        )
    return Probe(
        CLAP_PROBE,
        ProbeStatus.PASS if finite else ProbeStatus.FAIL,
        "ClapModel {} produced a {}finite feature vector".format(
            model_id, "" if finite else "NON-"
        ),
        _measured(model=model_id, n_params=n_params, shape=tuple(features.shape), finite=finite),
    )


def probe_pinned_versions() -> Probe:
    """Is this the `ss2` env at all, or merely *an* env that can import the same names?

    **The gap the box found on 2026-08-05.** A run launched from `ltm-embodied` — torch
    2.8.0+cu128, habitat-sim 0.3.3 — passed three of the four probes above. Only numpy
    caught it, and only by luck: `TORCH_MIN_VERSION` is a **floor** while the pin is
    `torch==2.2.2`, and the habitat probe asks whether the audio enum member resolves,
    which 0.3.3 answers yes to. Had that env carried numpy 1.23, the whole gate would
    have gone green and the episode would have produced numbers off a stack no
    measurement on this map was ever taken on.

    That is ticket 13's defect in its third costume: first a version-blind *skip*, then a
    check that passed on mere importability, now a **floor where the pin is exact**. The
    capability probes cannot see it by construction — the wrong env is not incapable, it
    is *different* — so this one is deliberately provenance-shaped.

    It does not replace them. Ticket 17's rule was that enforcement is capability-shaped
    *because* a version check would have printed green through the whole of ticket 13,
    and that still holds for everything above; this is the one question a capability
    cannot answer, asked alongside rather than instead.

    Reuses `compare_resolved_against_constraints` verbatim, so the `2.2.2+cu118` local
    version is handled the way the bootstrap already handles it — one rule, not two.

    **habitat-sim is out of this probe's reach twice over, and both are deliberate.** It
    is not in the constraints file at all — a source install at SHA `4f61e321`, which a
    freeze records as a path and cannot compare — and reading `habitat_sim.__version__`
    here would spend ADR-0013's one-importer exemption a second time, which
    `test_layering` correctly refused. Its version is already in the report:
    `probe_habitat_sim_audio_enum_member` holds the exemption and prints it. Pinning that
    version is a separate question needing a measurement from the good env, and a guessed
    constant would fail it — worse than an honest gap.
    """
    try:
        from importlib import metadata
    except Exception as exc:  # pragma: no cover - 3.8 and below only
        return Probe(PINNED_PROBE, ProbeStatus.NOT_RUN,
                     "importlib.metadata unavailable: {}".format(exc))

    text = _read(CONSTRAINTS_PATH)
    if text is None:
        return Probe(PINNED_PROBE, ProbeStatus.NOT_RUN,
                     "cannot read {} — the pins are unverified, which is not "
                     "satisfied".format(CONSTRAINTS_PATH))
    pins = parse_pins(text)
    if not pins:
        return Probe(PINNED_PROBE, ProbeStatus.NOT_RUN,
                     "{} parsed to no pins at all".format(CONSTRAINTS_PATH))

    resolved = {}
    for name in pins:
        try:
            resolved[name] = metadata.version(name)
        except Exception:
            continue  # absent: `compare_resolved_against_constraints` reports it as inert
    comparison = compare_resolved_against_constraints(pins, resolved)

    measured = _measured(
        n_pinned=str(comparison.n_pinned),
        **{name: version for name, version in sorted(resolved.items())}
    )
    if comparison.skew or comparison.inert:
        detail = "; ".join(
            ["{} pinned {} but resolved {}".format(*row) for row in comparison.skew]
            + ["{} pinned but not installed".format(name) for name in comparison.inert]
        )
        return Probe(
            PINNED_PROBE,
            ProbeStatus.FAIL,
            "{} — check which env is active before chasing a pip layer".format(detail),
            measured,
        )
    return Probe(
        PINNED_PROBE,
        ProbeStatus.PASS,
        "{} pin(s) match ss2-constraints.txt".format(comparison.n_pinned),
        measured,
    )


def run_probes(*, clap: bool = False) -> List[Probe]:
    """Every required probe, plus CLAP when requested. The box half of the split."""
    probes = [
        probe_numpy_below_1_24(),
        probe_torch_min_version(),
        probe_torch_cuda_allocation(),
        probe_habitat_sim_audio_enum_member(),
        probe_pinned_versions(),
    ]
    if clap:
        probes.append(probe_clap_instantiable())
    return probes


def expected_probes(*, clap: bool = False) -> FrozenSet[str]:
    return REQUIRED_PROBES | ({CLAP_PROBE} if clap else frozenset())


def assert_green(report: EnvReport) -> EnvReport:
    """Raise unless the report is green. Pure, so the raise itself is Mac-testable.

    Split out of ``assert_env`` deliberately. A test that stages its own ``raise`` to
    check this behaviour passes whether or not the code raises at all — the exact
    class of vacuous test an adversarial pass on ticket 23 kept finding. With the
    judgement and the raise both pure, the Mac suite exercises the real function on an
    injected red report, and ``run_probes`` is the only part that needs a box.
    """
    if not report.green:
        raise EnvCheckError(
            "the environment cannot run an episode:\n{}".format(report.summary())
        )
    return report


def assert_env(*, clap: bool = False) -> EnvReport:
    """Run the probes, judge them, raise if the env cannot run an episode.

    Returns the report so the caller can record it — ``task/`` writes it as
    ``env_report.json`` through ``report/artifacts.write_env_report``.
    """
    return assert_green(judge(run_probes(clap=clap), expected_probes(clap=clap)))


# ----------------------------------------------------------------------
# bootstrap-time provenance — the pure half ticket 19 puts on the Mac
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceComparison:
    """Did the pin actually take? A different job from every probe above.

    A capability probe cannot see an **inert pin** — a constraint on a package nothing
    installs enforces nothing and says nothing while doing it, which is ticket 13's
    version-blind skip wearing the constraints file's clothes. Only comparing the
    resolved set against the pinned one finds it.

    ``skew`` is the other half: the resolver ignored a pin it did honour the name of.
    """

    inert: Tuple[str, ...] = ()
    skew: Tuple[Tuple[str, str, str], ...] = ()
    n_pinned: int = 0

    @property
    def ok(self) -> bool:
        return not self.inert and not self.skew

    def summary(self) -> str:
        lines = [
            "provenance: {} ({} pinned, {} resolved)".format(
                "every pin took" if self.ok else "PROBLEM",
                self.n_pinned,
                self.n_pinned - len(self.inert),
            )
        ]
        for name in self.inert:
            lines.append(
                "  INERT PIN   {:<20} constrained but never installed — either the "
                "name is misspelled or nothing in the recipe pulls it".format(name)
            )
        for name, want, got in self.skew:
            lines.append(
                "  SKEW        {:<20} pinned {:<12} resolved {}".format(name, want, got)
            )
        return "\n".join(lines)


def _canonical(name: str) -> str:
    """PEP 503 normalisation. pip spells distribution names inconsistently across
    versions, so the comparison is on the canonical form rather than on whatever the
    file happens to write."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def parse_pins(text: str) -> Dict[str, str]:
    """``ss2-constraints.txt`` -> ``{canonical name: version}``. Comments stripped."""
    pins: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" in line:
            name, _, version = line.partition("==")
            pins[_canonical(name)] = version.strip()
    return pins


def parse_resolved(text: str) -> Dict[str, str]:
    """``pip freeze`` -> ``{canonical name: version}``.

    Lines without ``==`` are dropped on purpose: an editable or VCS install has no
    version to compare, and habitat-sim is a source install a freeze records as
    ``habitat-sim @ file:///root/ss2-build/…``, which is not reinstallable and not
    comparable.
    """
    resolved: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or "==" not in line:
            continue
        name, _, version = line.partition("==")
        resolved[_canonical(name)] = version.strip()
    return resolved


def compare_resolved_against_constraints(
    pins: Mapping[str, str], resolved: Mapping[str, str]
) -> ProvenanceComparison:
    """The comparison, as a pure function over two parsed sets.

    ``torch==2.2.2`` matches a resolved ``2.2.2+cu118``: PEP 440 makes ``+cu118`` a
    local version, so the pin is honoured and the two strings differ. Comparing them
    raw would report skew on the one package this env is most careful about.
    """
    inert: List[str] = []
    skew: List[Tuple[str, str, str]] = []
    for name in sorted(pins):
        want = pins[name]
        got = resolved.get(name)
        if got is None:
            inert.append(name)
        elif got.split("+", 1)[0] != want:
            skew.append((name, want, got))
    return ProvenanceComparison(tuple(inert), tuple(skew), len(pins))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as stream:
            return stream.read()
    except OSError:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m earshot.env_check``. Prints; never writes (ADR-0013's one writer)."""
    parser = argparse.ArgumentParser(
        prog="python -m earshot.env_check",
        description="the ss2 environment's runtime assertion (ticket 17)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero unless every expected probe passed",
    )
    parser.add_argument(
        "--clap",
        action="store_true",
        help="also instantiate CLAP and read a logit (153.5M params, ~0.7 GB VRAM)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--provenance",
        action="store_true",
        help="compare a pip freeze against the constraints file instead of probing",
    )
    parser.add_argument("--constraints", help="path to ss2-constraints.txt")
    parser.add_argument("--freeze", help="path to a pip freeze")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.provenance:
        if not args.constraints or not args.freeze:
            parser.error("--provenance needs both --constraints and --freeze")
        pins_text = _read(args.constraints)
        freeze_text = _read(args.freeze)
        if pins_text is None:
            print("provenance: cannot read {} — UNVERIFIED".format(args.constraints))
            return 1 if args.strict else 0
        if freeze_text is None:
            # Deliberately not silent: unverified is not the same as verified, and the
            # bootstrap's own comment says so about a failed `pip freeze`.
            print("provenance: cannot read {} — UNVERIFIED".format(args.freeze))
            return 1 if args.strict else 0
        comparison = compare_resolved_against_constraints(
            parse_pins(pins_text), parse_resolved(freeze_text)
        )
        print(json.dumps(comparison.summary()) if args.json else comparison.summary())
        return 0 if comparison.ok or not args.strict else 1

    report = judge(run_probes(clap=args.clap), expected_probes(clap=args.clap))
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True) if args.json else report.summary())
    return 0 if report.green or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
