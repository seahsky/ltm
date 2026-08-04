"""The lateral cue, and the frame convention that silently inverted underneath it.

``lateral_sign`` is carried verbatim from ``audio.py``: the fold-invariant left/right
cue from the interaural level difference, ``+1`` for a source to the **right**, ``-1``
left, ``0`` ambiguous. It is the cue SoundSpaces spatialises reliably (level, not time
— the near-zero interaural time difference is engine-weak, which is why ``estimate_doa``
does not carry).

**The code did not change and the convention inverted anyway.** The RIR grid was
rendered at *identity listener yaw*, so the cue it produced was **world-frame**, and the
fusion arc compensated for that with ``heard == -right(world-bearing)``. Live rendering
uses the agent's real transform, so the same arithmetic on the same samples now returns
an **agent-frame** cue. Carried across with the old compensation, the controller turns
the wrong way on every stall — and it looks like a mediocre climb rather than a bug,
which is why ticket 09 found it from source rather than from a run.

So the convention is **measured, not asserted**: ``tests/box/test_lateral_box.py``
renders the same source with the agent facing it and then turned 180 degrees away. The
agent frame predicts the sign flips; the world frame predicts it does not. That pair is
decisive, and a fake cannot settle it — this is the one assumption in ticket 22 that a
green Mac suite licenses nothing about (ADR-0014).

``types.py`` deliberately holds no bearing helper for this reason, and points here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from earshot.types import Pose, Xyz

__all__ = [
    "LEFT_EAR",
    "RIGHT_EAR",
    "LATERAL_LEFT",
    "LATERAL_RIGHT",
    "LATERAL_AMBIGUOUS",
    "ILD_DEAD_ZONE",
    "interaural_level_difference",
    "lateral_sign",
    "bearing_lateral_sign",
]

# The channel order the cue is read in. Named rather than inlined as ``b[0]``/``b[1]``
# because it is an assumption about the binding, not a fact about our code: it is
# pinned, together with the frame, by the box test. An inverted ear order and a
# world-frame cue are two different bugs with the same symptom at a single pose, and
# only the turned-around pose tells them apart.
LEFT_EAR = 0
RIGHT_EAR = 1

LATERAL_RIGHT = 1
LATERAL_LEFT = -1
LATERAL_AMBIGUOUS = 0

# Below this |ILD| the cue is not a direction. Carried verbatim (``1e-6``), and it is
# not a tuned parameter: it separates "the two ears are numerically identical" — a
# source dead ahead, a diotic bed, a silent render — from a real if small imbalance.
ILD_DEAD_ZONE = 1e-6


def interaural_level_difference(binaural: Any) -> float:
    """``(rms_right - rms_left) / (rms_right + rms_left)``, in ``[-1, +1]``.

    Normalised by the total rather than left as a difference, which is what makes it
    fold-invariant: the cue is the same at 1 m and at 8 m, so a controller reading it
    does not have to know how loud the source is to know which way to turn.

    ``0.0`` for a silent frame, because a ratio of two zeros is not a direction.
    """
    signal = np.asarray(binaural, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != 2:
        raise ValueError(
            "the lateral cue needs a (2, L) binaural signal, got {}".format(signal.shape)
        )
    left = float(np.sqrt(np.mean(np.square(signal[LEFT_EAR]))))
    right = float(np.sqrt(np.mean(np.square(signal[RIGHT_EAR]))))
    total = left + right
    if total <= 1e-12:
        return 0.0
    return (right - left) / total


def lateral_sign(binaural: Any) -> int:
    """Which half-plane the sound is in: ``+1`` right, ``-1`` left, ``0`` ambiguous.

    Agent-frame under live rendering — see the module docstring. The controller's stall
    rule turns toward the louder half-plane on this number directly, with **no
    compensation term**: ticket 23's port must not re-apply the grid era's
    ``heard == -right(world-bearing)``.
    """
    ild = interaural_level_difference(binaural)
    if abs(ild) < ILD_DEAD_ZONE:
        return LATERAL_AMBIGUOUS
    return LATERAL_RIGHT if ild > 0.0 else LATERAL_LEFT


def bearing_lateral_sign(pose: Pose, source: Xyz) -> int:
    """The sign the agent frame *predicts*, from ground truth. **Analyst-only.**

    Privileged in exactly the way ``sourceIsVisible()`` is (§3.3): it is computed from
    the true source position, so **the controller must never read it**. Feeding it to
    the decision rule plants a hidden oracle inside the realizable arm — the one thing
    ADR-0011 exists to avoid — and it would do it silently, because the realizable arm
    would still climb.

    It exists so the box test can state a prediction the renderer either matches or
    refutes, and so the audit record can carry "which way the source actually was"
    beside "which way the agent heard it". Without it, a stalled climb is undiagnosable.

    Habitat's frame: y is up, the agent's forward is ``-z`` and its right is ``+x`` at
    zero yaw, and ``Pose.yaw_rad`` is a right-handed rotation about ``+y`` (the
    extraction in ``sim/world.yaw_from_quaternion``). So the agent's right axis in world
    coordinates is ``(cos yaw, 0, -sin yaw)``, and the lateral component of the offset
    to the source is its dot product with that axis.
    """
    dx = source.x - pose.position.x
    dz = source.z - pose.position.z
    lateral = dx * math.cos(pose.yaw_rad) - dz * math.sin(pose.yaw_rad)
    if abs(lateral) < ILD_DEAD_ZONE:
        return LATERAL_AMBIGUOUS
    return LATERAL_RIGHT if lateral > 0.0 else LATERAL_LEFT
