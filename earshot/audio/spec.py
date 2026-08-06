"""The one place an ``AudioSensorSpec`` is configured, and the preset it is given.

ADR-0013 says this module is "THE only ``AudioSensorSpec()`` call site", so that
requirement 2's key validator is structural rather than remembered — a bare ``setattr``
elsewhere has nowhere to happen.

**One correction, forced by the ADR's own one-importer rule.** ``AudioSensorSpec`` is a
habitat-sim type, and ``sim/world.py`` is the only module in the tree permitted to name
one (``tests/mac/test_layering.py`` enforces it, and ``audio/`` importing the simulator
is exactly the edge ADR-0013 exists to keep absent). So this module cannot *construct*
the spec. It is handed a bare one and is the only place that **configures** one, which
delivers the property the ADR was after: every field that reaches an audio spec goes
through ``apply_audio_config`` + ``assert_no_swallowed_keys``, on one line, once.
``sim.world.audio_spec_parts()`` is the constructor, and it sets nothing.

That matters because of what a mistake costs here. ``AudioSensorSpec`` is bound
``py::dynamic_attr`` (``SensorBindings.cpp:395``, measured on the box in ticket 04): a
field name that does not exist on this branch — ``irTime``, renamed to ``maxIRLength``;
``updateDt``, ``dumpWaveFiles``, ``outputDirectory``, all gone — is **silently attached
as a Python attribute and never read**. There is no error, and the render proceeds with
the default. ``acousticsConfig`` is bound without it and raises instead, so the
validator belongs on the spec specifically and nowhere else.

The second reason this module exists is the preset. Ticket 06 measured the stock
defaults at **1723 ms/step** — 14 minutes of audio per 500-step episode — against
**27.2 ms/step** for the derived cheap preset, a 63x difference that costs nothing in
gradient quality. "Construct an ``AudioSensorSpec`` and use it as-is" is therefore a
trap, and the tree closes it by having exactly one configuration path.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from earshot.audio.config import AudioConfig
from earshot.audio.guard import apply_audio_config, assert_no_swallowed_keys

__all__ = [
    "ACOUSTICS_PRESET",
    "AUDIO_SENSOR_UUID",
    "audio_sensor_spec",
    "audio_config_mapping",
]

# provenance: box — the uuid habitat-sim's own render path REQUIRES, and it must be
# ASSIGNED. `Simulator._get_audio_observation` (simulator.py:765 on this build) reads
#
#     audio_sensor = self._agent._sensors["audio_sensor"]
#
# with the name as a literal, so this is not a preference: a sensor registered under any
# other key raises `KeyError('audio_sensor')` on the first observation, from inside
# habitat, before our code sees anything.
#
# **The constructor default is `'audio'`, NOT this.** Measured 2026-08-05 by
# `.scratch/ss2-clean-room/probes/audio_registration_probe.py`, which printed
# `AudioSensorSpec().uuid default: 'audio'` and then reproduced the KeyError twice — once
# with the spec passed through `agent_cfg.sensor_specifications` and once through
# `sim.add_sensor`, byte-identical in both, ruling out the registration form.
#
# This comment previously said the opposite — "read back, never assumed, never
# overridden", attributing to ticket 06 a rule that reading the uuid back is the safe
# move. That rule is what broke ticket 25's first box run: the read-back yields `'audio'`,
# the one name habitat can never find. Ticket 06's actual finding survives and is
# narrower — assigning a name *other than* `"audio_sensor"` also fails, because the
# Python-side `_sensors` dict picks up the new name while the hardcoded lookup does not.
# Both halves point the same way: this exact string, assigned, asserted.
AUDIO_SENSOR_UUID = "audio_sensor"

# provenance: box — ticket 06's derived `cheap_preset`, measured on the V100 across two
# HM3D scenes: 1723 ms/step at the defaults, 27.2 ms/step here, with the energy gradient
# still climbable (Spearman rho -0.98/-0.99, and -0.95/-0.98 on the non-line-of-sight
# split). The knobs, and why each is in the set:
#
#   indirectRayCount  5000 -> 500   the dominant lever, roughly linear in ray count
#   indirectRayDepth   200 -> 50    ~3.7x on its own
#   threadCount          1 -> 4     ~2.4x on the box's 4 cores — a real lever, but not
#                                   the order of magnitude the map's "free speed knob"
#                                   framing implied (ticket 04 called that correction)
#   temporalCoherence    0 -> 1     defaults OFF, so enabling it gives up nothing
#
# NOT in the set, deliberately: `maxIRLength` and `directRayCount` are not cost knobs at
# all (the IR cap bounds the output buffer, not the tracing), and `transmission` /
# `diffraction` are PHYSICS knobs — between them they are the non-line-of-sight audio
# path, which is the premise of the whole anomaly-response task. They are cheaper with
# the sound switched off; folding them into something labelled "cheap" would smuggle a
# task-design decision in as a performance tweak.
ACOUSTICS_PRESET: Dict[str, Any] = {
    "indirectRayCount": 500,
    "indirectRayDepth": 50,
    "threadCount": 4,
    "temporalCoherence": 1,
}


def audio_config_mapping(
    config: AudioConfig,
    binaural_layout: Any,
    acoustics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The nested mapping ``apply_audio_config`` validates and applies. Pure.

    Separated from the spec so the *contents* of the configuration can be asserted on a
    Mac without a spec object to hang them on, and so a diff shows what changed rather
    than which line moved.

    ``enableMaterials`` is set explicitly to ``False`` even though ticket 04 measured it
    already constructing that way. ADR-0007 makes materials permanently off, and this is
    the line that says so; inheriting it would leave the decision recorded only in a
    document. It also lives on the **spec**, not on ``acousticsConfig`` — SoundSpaces'
    own example sets it on the wrong object and raises (ticket 03/04).

    ``channelLayout`` is likewise set rather than inherited. Binaural/2 is the measured
    default on this branch, and it is also the assumption the entire lateral cue rests
    on, so it is asserted at the one place that can.
    """
    # `acoustics` replaces the whole preset and is the low-level escape hatch the tests
    # use. `config.indirect_ray_count` overrides ONE key of whichever base is in force and
    # is what a run sets, because it is the only preset entry that trades accuracy for
    # speed rather than speed alone — see `AudioConfig.indirect_ray_count` for the
    # measurement that made it a knob. Applied after, so it wins over either base and a
    # run's ray count is always the number `run_config` records.
    base = dict(ACOUSTICS_PRESET if acoustics is None else acoustics)
    if config.indirect_ray_count is not None:
        base["indirectRayCount"] = int(config.indirect_ray_count)
    return {
        "enableMaterials": False,
        "channelLayout": {"type": binaural_layout, "channelCount": 2},
        "acousticsConfig": dict(
            base,
            sampleRate=float(config.sample_rate),
            globalVolume=float(config.global_volume),
        ),
    }


def audio_sensor_spec(
    spec: Any,
    config: AudioConfig,
    binaural_layout: Any,
    acoustics: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Configure a bare ``AudioSensorSpec`` and return it. The only such path in the tree.

    ``spec`` and ``binaural_layout`` come from ``sim.world.audio_spec_parts()``: the two
    habitat-sim names this module needs and is not allowed to say. Everything else is
    ours.

    Every key is validated against the live field list *before* anything is written
    (``apply_audio_config``), and the spec's ``__dict__`` is checked afterwards
    (``assert_no_swallowed_keys``) — belt and braces, and the second catches anything a
    caller attached before handing the spec over.

    The post-conditions are asserted rather than assumed, because each has a silent
    failure mode: an unset channel layout gives a lateral sign of exactly zero at every
    pose, and an unapplied preset is a 63x slower render that still produces a perfectly
    plausible IR.

    The uuid is **assigned**, for the reason ``AUDIO_SENSOR_UUID`` documents: the
    constructor default is ``'audio'`` and habitat's own render path looks up the literal
    ``"audio_sensor"``. Assigned here rather than in ``sim/world.py`` because that module
    is audio-blind by construction (ADR-0013) and this function is already "the only such
    path in the tree" — one place that touches audio spec fields, not two.
    """
    # Before `apply_audio_config`, so a build that rejects the assignment fails on the
    # uuid rather than on whichever config key happens to be checked first.
    spec.uuid = AUDIO_SENSOR_UUID
    if str(getattr(spec, "uuid", None)) != AUDIO_SENSOR_UUID:
        raise ValueError(
            "spec.uuid is {!r} after assignment, not {!r}. habitat-sim's "
            "_get_audio_observation looks the sensor up under that literal name, so a "
            "spec that will not take it renders nothing — the whole audio path is "
            "unreachable on this build".format(getattr(spec, "uuid", None),
                                               AUDIO_SENSOR_UUID)
        )

    mapping = audio_config_mapping(config, binaural_layout, acoustics)
    apply_audio_config(spec, mapping)
    assert_no_swallowed_keys(spec)

    layout = getattr(spec, "channelLayout")
    if getattr(layout, "channelCount", None) != 2:
        raise ValueError(
            "channelLayout.channelCount is {!r} after configuration, not 2 — the "
            "binaural cue in audio/lateral.py needs two ears and reads exactly two "
            "rows".format(getattr(layout, "channelCount", None))
        )
    # `==`, not `is`. pybind11 casts a C++ enum back to Python by constructing a wrapper
    # rather than returning the interned class attribute, so reading a `def_readwrite`
    # enum field can yield an object that compares equal and is not identical. An
    # identity check here would fail on the box against a spec that took the value
    # correctly — the opposite of the failure this assertion is for.
    if getattr(layout, "type", None) != binaural_layout:
        raise ValueError(
            "channelLayout.type did not take: it is {!r}, not the Binaural member "
            "passed in. A Mono or Ambisonic observation reaches audio/clips.as_binaural "
            "as the wrong shape, and a lateral sign of zero at every pose is a "
            "controller that never turns".format(getattr(layout, "type", None))
        )
    applied = getattr(spec, "acousticsConfig")
    for key, value in mapping["acousticsConfig"].items():
        actual = getattr(applied, key, None)
        if actual != value:
            raise ValueError(
                "acousticsConfig.{} is {!r} after configuration, not the {!r} that was "
                "applied. Ticket 06 measured the stock defaults at 1723 ms/step against "
                "27.2 ms/step for this preset, and an unapplied knob renders a "
                "plausible IR at the slow default".format(key, actual, value)
            )
    return spec
