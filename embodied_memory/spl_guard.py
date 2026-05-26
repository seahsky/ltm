"""
Guard Habitat SPL / SoftSPL measures against a zero start-to-goal geodesic.

The lifelong/revisit eval seeds the LTM with a COLD episode that starts the
agent ON the goal viewpoint, so the step-0 caption provably captures the goal.
Habitat's SoftSPL then evaluates ``1 - distance_to_goal / start_end_distance``
with ``start_end_distance == 0`` and raises ``ZeroDivisionError`` mid-step,
aborting the seeding run before it can consolidate the sighting. The cold
episode's SPL is never read by Gate A (which measures WARM revisits only), so we
only need it to RUN and seed memory. This wraps ``update_metric`` so the
degenerate division yields ``metric = 0.0`` instead of crashing.

Stdlib-only and habitat-free so it unit-tests without the sim installed; the
real measure classes are passed in by ``habitat_env`` at env-build time.
"""

from __future__ import annotations

from typing import Iterable

_GUARD_FLAG = "_ltm_zero_start_guarded"


def guard_zero_start_distance(measure_classes: Iterable[type]) -> int:
    """Wrap each measure class's ``update_metric`` so a ``ZeroDivisionError``
    (start-on-goal cold episode) sets ``self._metric = 0.0`` instead of
    propagating.

    Idempotent per class — the guard flag is checked on the class's OWN
    ``__dict__`` (not via inheritance), so a subclass that overrides
    ``update_metric`` (e.g. ``SoftSPL`` over ``SPL``) is still patched even when
    its parent was already guarded. Returns the number of classes newly patched.
    """
    n_patched = 0
    for cls in measure_classes:
        if cls.__dict__.get(_GUARD_FLAG, False):
            continue
        original = cls.update_metric

        def _guarded(self, *args, _original=original, **kwargs):
            try:
                return _original(self, *args, **kwargs)
            except ZeroDivisionError:
                # Cold seeding episode started on the goal (start_end == 0).
                # Its SPL is unused by Gate A; keep the episode alive to seed.
                self._metric = 0.0
                return None

        cls.update_metric = _guarded
        setattr(cls, _GUARD_FLAG, True)
        n_patched += 1
    return n_patched
