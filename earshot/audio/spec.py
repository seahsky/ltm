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

# provenance: box — the sensor's own default uuid, from the C++ constructor. **Read
# back, never assumed, and never overridden**: ticket 06 measured that assigning a
# different uuid does not fully take — the Python-side `_sensors` dict picks up the new
# name while the C++ sensor suite keeps the old, and `get_sensor_observations()` then
# fails an internal cross-lookup with `KeyError('audio_sensor')`. That killed a box
# stage once. `sensor.AudioSensorHandle` reads `spec.uuid` off the spec rather than
# trusting this constant, and this exists to name what it should find.
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
    return {
        "enableMaterials": False,
        "channelLayout": {"type": binaural_layout, "channelCount": 2},
        "acousticsConfig": dict(
            ACOUSTICS_PRESET if acoustics is None else acoustics,
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
    """
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
