"""``AudioSensorHandle`` — the live sensor, armed at construction (requirement 1b).

A connector to an external system, which is why this is a class in a layer that is
otherwise functions: it owns a handle with a lifecycle and an invariant that has to hold
from the first render onward.

**Constructing it arms the guard, and the two cannot be separated.** That is
requirement 1(b) taken literally, and it is forced by a measurement rather than chosen
for tidiness: the audio mesh upload is **lazy** — read at ``4f61e321``,
``createAudioSimulator()`` sets ``newInitialization_`` and the *first* ``runSimulation()``
consumes it — so at construction there is no mesh to assert on. ``arm_audio_context``
therefore performs the first render itself. A handle that existed unarmed would be a
handle someone could observe through, and a zero-geometry audio context returns
plausible-looking audio that nothing else in the tree can tell from the real thing.

**The source is set before that first render, in the same constructor.** RLR renders an
IR *for a source*; arming with none would either produce the silent IR the guard rejects
or, worse, a plausible one from a stale transform. So the source position is a
constructor argument, not a later call, and the ordering is structural.

No ``habitat_sim`` import: the sensor object and the observation callable are injected
(ADR-0013), which is what makes this module testable against a fake on a Mac. What a
Mac cannot tell you is whether the real ``setAudioSourceTransform`` and
``sourceIsVisible`` behave as assumed — ``tests/box/`` owns that.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from earshot.audio.guard import (
    AudioContextError,
    AudioContextReport,
    StepGuardReport,
    arm_audio_context,
    guarded_observe,
)
from earshot.audio.spec import AUDIO_SENSOR_UUID
from earshot.types import Xyz

__all__ = ["AudioSensorHandle"]


class AudioSensorHandle:
    """One armed audio sensor: its source, its observations, its analyst-only LOS test."""

    def __init__(
        self,
        sensor: Any,
        observe: Callable[[], Dict[str, Any]],
        source: Xyz,
        *,
        uuid: str = AUDIO_SENSOR_UUID,
        obj_dir: Optional[str] = None,
    ) -> None:
        """Seat the source, take the first render, and assert the context is real.

        ``observe`` is the world's one shared observation call — the same one that
        returns RGB and depth, because there is no separate audio render
        (``oneenv_probe.py:629``). That is what makes smoke criterion 1's "render count
        equals step count" measurable at all.

        ``uuid`` is read from the spec by the caller and passed in rather than assumed —
        but the safety in that is the *agreement*, not the reading.
        ``audio.spec.audio_sensor_spec`` **assigns** ``AUDIO_SENSOR_UUID`` and asserts it
        took, because the constructor default is ``'audio'`` while habitat's own
        ``_get_audio_observation`` looks up the literal ``"audio_sensor"``. Passing the
        value through means this handle and the world's sensor lookup cannot disagree
        about which sensor they hold; it is not a licence for the spec to carry whatever
        name it was born with.
        """
        self._sensor = sensor
        self._observe = observe
        self.uuid = str(uuid)
        self.source: Optional[Xyz] = None
        self.n_source_moves = 0

        self.set_source(source)
        self.report: AudioContextReport = arm_audio_context(
            sensor, lambda: self._audio_of(self._observe()), obj_dir=obj_dir
        )

    # -- the source ------------------------------------------------------

    def set_source(self, source: Xyz) -> None:
        """Move the positioned source. Exactly one exists per episode (ADR-0009).

        habitat-sim's binding takes a float32 array, and the coordinate is a world
        position in the scene's own frame — the same frame ``World.pose()`` reports in,
        which is what lets ``lateral.bearing_lateral_sign`` compare the two.

        The engine's own handler for a failed ``RLRA_SetAudioSourceTransform`` is
        ``ESP_ERROR`` followed by a bare return into a ``void``, so a failure here has no
        return code and no exception — it is visible only in the log the guard scans on
        the next render (``FATAL_LOG_SUBSTRINGS`` carries "Error setting audio source
        position" for exactly this).
        """
        self._sensor.setAudioSourceTransform(
            np.asarray(source.as_tuple(), dtype=np.float32)
        )
        self.source = source
        self.n_source_moves += 1

    # -- observing -------------------------------------------------------

    def observe(self) -> Tuple[Dict[str, Any], StepGuardReport]:
        """One guarded render: the full observation dict and what the guard saw.

        The light half of the split (ADR-0013). ``arm_audio_context`` established that
        the context is real once; this establishes it has not *stopped* being real,
        which ticket 16 measured to be a different claim — ``[Audio]`` is logged on every
        ``runSimulation`` so the canary stays armed all episode, and the closed engine
        can write an un-prefixed error block to fd 2 while ``RLRA_SetListenerHRTF``
        still returns ``Success``. A context that degrades at step 300 is invisible to
        everything else in the tree.

        Returns the whole dict, not the IR: RGB, depth and the IR come out of this one
        call, and the frontier proposer wants the depth frame from the same render the
        onset detector measures.
        """
        return guarded_observe(self._observe)

    def audio_of(self, observation: Dict[str, Any]) -> Any:
        """The IR out of an observation dict, by uuid, with a diagnosable failure."""
        return self._audio_of(observation)

    def _audio_of(self, observation: Dict[str, Any]) -> Any:
        try:
            return observation[self.uuid]
        except (KeyError, TypeError) as exc:
            raise AudioContextError(
                "no {!r} in the observation (keys: {}). The audio spec did not reach "
                "the sensor suite, or its uuid was reassigned — which does not fully "
                "take on this branch, so the Python `_sensors` dict and the C++ suite "
                "disagree".format(
                    self.uuid,
                    sorted(observation) if isinstance(observation, dict) else type(observation),
                )
            ) from exc

    # -- analyst-only ----------------------------------------------------

    def source_is_visible(self) -> Optional[bool]:
        """Line of sight from the listener to the source. **Analyst-only** (§3.3).

        Free — one ``RLRA_TraceRayAnyHit`` — and the best available diagnostic for why a
        gradient climb stalled, which is why §3.2 records it at every step.

        **The controller must never read it.** It is computed from the ground-truth
        source position, so feeding it to the decision rule plants a hidden oracle inside
        the realizable arm, which is the one thing ADR-0011 exists to avoid. The
        realizable arm would still climb, and its report would still validate, so nothing
        downstream would catch it — this docstring and the report's type are the whole
        defence.

        ``None`` when the branch does not expose it, rather than a guess: with no
        geometry nothing occludes and it reads ``True`` everywhere, so a fabricated
        ``True`` is indistinguishable from the broken-context case.
        """
        probe = getattr(self._sensor, "sourceIsVisible", None)
        if probe is None:
            return None
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 - a diagnostic must never end an episode
            return None
