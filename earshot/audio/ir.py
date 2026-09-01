"""The anechoic IR policy: what "no reverberation" means for a rendered impulse response.

Carried verbatim from ``tests/box/test_sounding_window_box.py``'s ``_anechoic_like``
(ADR-0019's own control fixture), and promoted here because ADR-0018's matrix needs it
as a runtime policy rather than a test-only helper: ``IrPolicy.ANECHOIC`` routes every
rendered IR through this function before it is convolved, so the ablation can ask
whether the audible reverb tail buys any success-when-silent -- pilot-2 found win-alarm's
silent phase audible in 356 of 356 episodes and win-burst's in 0 of 356, with SWS 0.115
against 0.112 -- rather than only whether a test fixture can reproduce a clip tail.

**Scaled to the real IR's peak, not to unit gain.** Every comparison against this control
runs on a curve normalised by its own settled level, where the scale cancels exactly --
so matching the peak keeps the printed absolute levels in the same range as the room
being compared against, and changes nothing about what a comparison concludes.

**The floor is not cosmetic.** A zero-peak IR would make ``render_through_ir`` return
silence and the onset would never fire -- attrition that reads as ordinary "the sound
wasn't loud enough" rather than what it actually is, a policy that broke the sensor.

numpy only, and no ``earshot`` imports: this is a leaf under the ``audio`` layer, for the
same reason ``audio/clips.py``'s module docstring gives -- a module-level ``scipy``
import would make every Mac test in this layer uncollectable, and this file needs no
scipy call to begin with.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["anechoic_like"]

# The floor the box fixture used. Not zero: a silent or failed render's IR peaks at
# exactly 0.0, and flooring there would return true silence -- the one input this
# function exists to guard against.
_PEAK_FLOOR = 1e-12


def anechoic_like(impulse: Any) -> np.ndarray:
    """A ``(2, 1)`` IR at the same peak as ``impulse`` -- a room with no reverberation.

    Scaled to the real IR's peak only so the printed absolute levels stay in the same
    range; every comparison against it is on a curve normalised by its own settled level,
    where the scale cancels exactly.

    Pure: ``impulse`` is read, never written, and the return is a fresh array.

    Raises ``ValueError`` on an empty ``impulse`` -- ``np.max`` of an empty array raises
    its own ``ValueError`` with no reference to the caller, so this names the caller
    instead of leaving that to a generic numpy message.
    """
    values = np.asarray(impulse, dtype=np.float32)
    if values.size == 0:
        raise ValueError("anechoic_like: impulse is empty, nothing to take a peak of")
    peak = float(np.max(np.abs(values)))
    return np.full((2, 1), max(peak, _PEAK_FLOOR), dtype=np.float32)
