"""Fakes for the audio layer. Shared, so the assumptions are stated once.

**A fake licenses nothing about the binding** (ADR-0014). Ticket 12's guard passed 27
tests against fakes and then raised on the first real ``AudioSensorSpec``, because the
real constructor attaches ``__noise_model_kwargs`` and no fake did. So every fake here
carries, at its definition, a citation for the behaviour it imitates and a note on what
it cannot show.

The one that matters most is ``FakeAudioSensorSpec``, which reproduces
``py::dynamic_attr`` semantics **structurally** rather than by convention: declared
fields are data descriptors on the *type*, so they never reach the instance
``__dict__``, and anything else lands there silently. That is precisely what
``def_readwrite`` does and precisely why ``vars(spec)`` detects a swallowed key exactly.
A fake that used plain instance attributes would put every legitimate field in
``vars()``, and ``assert_no_swallowed_keys`` would raise on a healthy spec — a green
test written against it would be measuring the wrong thing.
"""

import weakref

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

# ----------------------------------------------------------------------
# the spec side
# ----------------------------------------------------------------------


class _Field:
    """A data descriptor holding its values off-instance. pybind's ``def_readwrite``.

    Values live in a ``WeakKeyDictionary`` keyed by the instance, so the instance
    ``__dict__`` stays empty however many declared fields are assigned — which is the
    property ``guard.assert_no_swallowed_keys`` reads.
    """

    def __init__(self, default=None):
        self._default = default
        self._values = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self._values.get(instance, self._default)

    def __set__(self, instance, value):
        self._values[instance] = value


class FakeAcousticsConfig:
    """``acousticsConfig``, with ticket 04's MEASURED defaults from the box.

    Plain instance attributes, unlike the spec: the real
    ``RLRAudioPropagationConfiguration`` is bound **without** ``py::dynamic_attr``
    (``SensorBindings.cpp:293-295``), so an unknown key there raises rather than being
    swallowed. This fake cannot raise the way pybind does, which is why
    ``audio/spec.py`` validates every key against the live field list *before* writing
    anything rather than relying on the object to complain.
    """

    def __init__(self):
        self.diffraction = 1
        self.directRayCount = 500
        self.directSHOrder = 3
        self.direct = 1
        self.frequencyBands = 4
        self.globalVolume = 1.0
        self.indirect = 1
        self.indirectRayCount = 5000
        self.indirectRayDepth = 200
        self.indirectSHOrder = 1
        self.maxDiffractionOrder = 10
        self.maxIRLength = 4.0
        self.meshSimplification = 0
        self.sampleRate = 44100.0
        self.sourceRayCount = 200
        self.sourceRayDepth = 10
        self.temporalCoherence = 0
        self.threadCount = 1
        self.transmission = 1
        self.unitScale = 1.0


class FakeChannelLayout:
    """``channelLayout``. Binaural / 2 is the measured default on this branch."""

    def __init__(self, layout_type=None):
        self.type = BINAURAL if layout_type is None else layout_type
        self.channelCount = 2


class FakeAudioSensorSpec:
    """``AudioSensorSpec``, including the two things a naive fake gets wrong.

    1. **``vars()`` is not empty on a fresh spec.** The real constructor attaches
       ``__noise_model_kwargs`` as a genuine instance attribute (measured on the box in
       ticket 15's run, which is what answered ticket 16's stage-1 question), which is
       why ``guard.KNOWN_DYNAMIC_ATTRS`` exists. Reproduced, so a test passing against
       this fake is exercising that exclusion too.
    2. **``uuid`` is ``"audio_sensor"`` from C++**, and assigning another name does not
       fully take — the Python ``_sensors`` dict picks it up while the C++ suite keeps
       the old one (ticket 06). Reproduced as an ordinary field, since the divergence is
       on the simulator side and no Mac can show it.
    """

    uuid = _Field("audio_sensor")
    enableMaterials = _Field(False)
    acousticsConfig = _Field(None)
    channelLayout = _Field(None)
    noise_model_kwargs = _Field(None)

    def __init__(self):
        self.acousticsConfig = FakeAcousticsConfig()
        self.channelLayout = FakeChannelLayout()
        self.noise_model_kwargs = {}
        # Not a declared field, so it lands in `vars()` exactly like the real one.
        setattr(self, "__noise_model_kwargs", {})


# Stands in for `habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural`. An
# opaque sentinel rather than the string "Binaural": `audio/spec.py` compares it by
# identity, and a string would let an accidental equality pass.
BINAURAL = object()


# ----------------------------------------------------------------------
# the sensor side
# ----------------------------------------------------------------------


def synthetic_ir(n_samples=512, left=1.0, right=1.0, seed=7):
    """A ``(2, L)`` IR with a controllable interaural level difference.

    Decaying noise rather than a single spike: an impulse convolves to a scaled copy of
    the clip, which would hide a transform-length error that a spread response exposes.
    """
    rng = np.random.default_rng(seed)
    envelope = np.exp(-np.linspace(0.0, 6.0, n_samples))
    base = rng.standard_normal(n_samples) * envelope
    return np.stack([base * float(left), base * float(right)]).astype(np.float32)


class FakeAudioSensor:
    """The ``AudioSensor`` handle: the four calls the tree makes on it.

    ``writeSceneMeshOBJ`` writes a mesh over the guard's 10,000-vertex floor and returns
    ``True``, because a handle that fails to arm is a different test.
    ``sourceIsVisible`` returns what it is told — on the real binary over a
    zero-geometry context it reads ``True`` everywhere, which is exactly why the guard
    records it and never asserts on it.
    """

    def __init__(self, n_vertices=20_000, visible=True, raises_on_visible=False):
        self.n_vertices = int(n_vertices)
        self.visible = visible
        self.raises_on_visible = raises_on_visible
        self.source_transforms = []

    def setAudioSourceTransform(self, xyz):
        self.source_transforms.append([float(v) for v in np.asarray(xyz).reshape(-1)])

    def sourceIsVisible(self):
        if self.raises_on_visible:
            raise RuntimeError("the engine declined")
        return self.visible

    def getRayEfficiency(self):
        return 0.548  # ticket 04's measured control

    def writeSceneMeshOBJ(self, path):
        with open(path, "w") as handle:
            handle.write("".join("v {} 0 0\n".format(i) for i in range(self.n_vertices)))
        return True


class FakeWorld:
    """The injected ``observe`` callable, and the log the guard has to see.

    Prints the canary on **stdout**, because that is where it really lands: ``ESP_DEBUG``
    is a ``Corrade::Utility::Debug`` whose default output is ``std::cout``
    (``Debug.cpp:525``), and ticket 16 measured 916 chars on fd 1 against 0 on fd 2 for a
    healthy render. A fake that printed it to stderr would let a guard capturing the
    wrong descriptor pass.
    """

    def __init__(self, ir=None, uuid="audio_sensor"):
        self.ir = synthetic_ir() if ir is None else ir
        self.uuid = uuid
        self.n_renders = 0

    def observe(self):
        self.n_renders += 1
        print("[Audio] Vertex count : 392356")
        return {"rgb": np.zeros((4, 4, 3), dtype=np.uint8), self.uuid: self.ir}


# ----------------------------------------------------------------------
# the CLAP side
# ----------------------------------------------------------------------


class FakeClapEncoder:
    """A CLAP stand-in: fixed vectors keyed by prompt text, and one for the audio.

    Deterministic, so a test states which prompt should win rather than measuring which
    one does. It says nothing about real CLAP cosines — those are the calibration's
    subject, and that calibration ran on the box at EER 0.00.
    """

    def __init__(self, audio_vector, text_vectors, dim=8):
        self.dim = dim
        self.audio_vector = np.asarray(audio_vector, dtype=np.float32)
        self.text_vectors = {
            key: np.asarray(value, dtype=np.float32) for key, value in text_vectors.items()
        }
        self.seen_texts = []
        self.seen_rates = []

    def encode_audio(self, waveform, sample_rate):
        self.seen_rates.append(int(sample_rate))
        return self.audio_vector

    def encode_text(self, text):
        self.seen_texts.append(text)
        return self.text_vectors.get(text, np.zeros(self.dim, dtype=np.float32))


def one_hot(index, dim=8, scale=1.0):
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = float(scale)
    return vector
