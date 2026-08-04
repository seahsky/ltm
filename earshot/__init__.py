"""The clean room. Importing it pins ``HABITAT_SIM_LOG`` and does nothing else.

``HABITAT_SIM_LOG`` is read at habitat-sim import time, so a late pin is a silent
no-op — which is why ``pin_habitat_logging`` **raises** when ``habitat_sim`` is
already in ``sys.modules`` rather than quietly doing nothing. Putting the call here
means every path into the package runs it first *by construction*: there is no
convention for an entry point to forget and no ad-hoc box script that can bypass it
(ADR-0013).

Keep this file to the one call. ``audio.guard`` is stdlib-only, so this import costs
nothing measurable; anything heavier here would be paid by every ``python -m`` in the
tree, including the ones that never touch the simulator.
"""

from .audio.guard import pin_habitat_logging

pin_habitat_logging()
