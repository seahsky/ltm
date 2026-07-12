"""
TDD for the G0.4 realizable-localization kill-switch (ADR-0001): is the RIR
grid's energy field climbable toward the source? Energy-gradient climbing only
works if a greedy ascent of live binaural loudness reaches the source instead of
stalling on a spurious loud spot.

Two PURE functions carry the logic (the render/grid read is RACE integration):
  - gradient_climbability(energies, positions, source): spearman of cell energy
    vs distance-to-source (want strongly NEGATIVE — closer is louder) + the count
    of local ENERGY MAXIMA excluding the global source peak (the traps a greedy
    ascent falls into).
  - energy_gradient_verdict(stats): GO / BORDERLINE / STOP.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_diagnose_energy_gradient.py
"""
from __future__ import annotations

import math
import sys

from embodied_memory.scripts.diagnose_energy_gradient import (
    energy_gradient_verdict,
    gradient_climbability,
)


def case_monotone_field_is_perfectly_climbable():
    # Cells on a line at x=1..5, source at the origin; energy decreases smoothly
    # with distance (closer = louder). Energy vs distance is perfectly anti-
    # correlated (spearman = -1) and there is no spurious peak.
    positions = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]]
    energies = [10.0, 8.0, 6.0, 4.0, 2.0]
    source = [0.0, 0.0, 0.0]
    s = gradient_climbability(energies, positions, source, k=2)
    assert s["n_cells"] == 5, s
    assert math.isclose(s["spearman"], -1.0, abs_tol=1e-9), s
    assert s["n_local_maxima"] == 0, s          # only the global (source) peak
    assert math.isclose(s["frac_local_maxima"], 0.0), s
    print("  case_monotone_field_is_perfectly_climbable: OK")


def case_spurious_peak_is_a_trap():
    # Same line, but a bump at x=4 (energy 7) that is louder than its neighbors
    # yet not the global peak (x=1, energy 10) — a false summit a greedy ascent
    # would stall on.
    positions = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0]]
    energies = [10.0, 6.0, 5.0, 7.0, 3.0, 1.0]
    source = [0.0, 0.0, 0.0]
    s = gradient_climbability(energies, positions, source, k=2)
    assert s["n_local_maxima"] == 1, s          # x=4 bump; global x=1 excluded
    print("  case_spurious_peak_is_a_trap: OK")


def case_verdict_thresholds():
    # A clean, strongly-negative gradient with no spurious peaks is climbable.
    assert energy_gradient_verdict({"spearman": -0.9, "frac_local_maxima": 0.0}) == "GO"
    assert energy_gradient_verdict({"spearman": -0.4, "frac_local_maxima": 0.05}) == "GO"  # boundary
    # No usable gradient (energy barely tracks distance) -> not localizable.
    assert energy_gradient_verdict({"spearman": -0.1, "frac_local_maxima": 0.0}) == "STOP"
    # Strong gradient but riddled with false summits a greedy climb stalls on.
    assert energy_gradient_verdict({"spearman": -0.9, "frac_local_maxima": 0.30}) == "STOP"
    # In-between -> BORDERLINE.
    assert energy_gradient_verdict({"spearman": -0.30, "frac_local_maxima": 0.0}) == "BORDERLINE"
    print("  case_verdict_thresholds: OK")


def main() -> int:
    print("running G0.4 energy-gradient tests…")
    case_monotone_field_is_perfectly_climbable()
    case_spurious_peak_is_a_trap()
    case_verdict_thresholds()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
