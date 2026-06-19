"""
TDD for the S2 audio-DOA head (memory_bridge._audio_doa_bonus / _world_right_sign).
Pure functions — no FAISS index, no Habitat. Pins the world-frame INVERTED sign
convention (diagnose_audio_doa_calib: GO, "heard==-right(world-bearing)") against
ACTIVE HARM, plus the zero-sum + energy-gate + no-op-when-off contracts.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_audio_doa_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from embodied_memory.memory_bridge import _audio_doa_bonus, _world_right_sign  # noqa: E402


def case_world_right_sign_convention():
    # RIGHT <=> dx<0 (mirrors right_sign_from_bearing(atan2(dx, dz)))
    assert _world_right_sign(-2.0, 2.0) == 1      # world-right
    assert _world_right_sign(2.0, 2.0) == -1      # world-left
    assert _world_right_sign(0.0, 2.0) == 0       # abeam/ahead
    print("  case world_right_sign_convention: OK")


def case_zero_sum():
    # whatever the geometry, bonuses must sum to ~0 (the over-fire guard — the
    # all-positive caption-centrality form over-fired +84%)
    xys = [[2.0, 2.0], [-3.0, 2.0], [1.0, 5.0]]
    b = _audio_doa_bonus(xys, 0.0, 0.0, lateral_sign=1, energy=0.2, weight=0.05, energy_ref=0.2)
    assert abs(sum(b)) < 1e-9, b
    print("  case zero_sum: OK")


def case_positive_convention_boosts_correct_demotes_wrong():
    # source at world (2,2): world_right_sign=-1 (left) -> pinned heard=-(-1)=+1.
    # The correct candidate near the source (2,2) must be BOOSTED and a wrong one
    # on the opposite world-side (-3,2) DEMOTED. A flipped convention would do the
    # reverse (active harm) — caught here where a byte-identity test cannot.
    heard = -_world_right_sign(2.0, 2.0)          # the pinned inverted convention
    assert heard == 1
    xys = [[2.0, 2.0], [-3.0, 2.0]]               # [correct, wrong]
    b = _audio_doa_bonus(xys, 0.0, 0.0, lateral_sign=heard, energy=0.2, weight=0.05, energy_ref=0.2)
    assert b[0] > 0.0 > b[1], b
    assert abs(sum(b)) < 1e-9, b
    print("  case positive_convention_boosts_correct_demotes_wrong: OK")


def case_no_op_when_sign_absent_or_weight_zero():
    xys = [[2.0, 2.0], [-3.0, 2.0]]
    assert _audio_doa_bonus(xys, 0, 0, lateral_sign=0, energy=0.2, weight=0.05, energy_ref=0.2) == [0.0, 0.0]
    assert _audio_doa_bonus(xys, 0, 0, lateral_sign=None, energy=0.2, weight=0.05, energy_ref=0.2) == [0.0, 0.0]
    assert _audio_doa_bonus(xys, 0, 0, lateral_sign=1, energy=0.2, weight=0.0, energy_ref=0.2) == [0.0, 0.0]
    print("  case no_op_when_sign_absent_or_weight_zero: OK")


def case_energy_gate_scales_magnitude():
    xys = [[2.0, 2.0], [-3.0, 2.0]]
    heard = -_world_right_sign(2.0, 2.0)
    loud = _audio_doa_bonus(xys, 0, 0, heard, energy=0.2, weight=0.05, energy_ref=0.2)    # g=1.0
    quiet = _audio_doa_bonus(xys, 0, 0, heard, energy=0.05, weight=0.05, energy_ref=0.2)  # g=0.25
    assert abs(quiet[0]) < abs(loud[0]), (quiet, loud)
    assert abs(sum(quiet)) < 1e-9 and abs(sum(loud)) < 1e-9
    print("  case energy_gate_scales_magnitude: OK")


def main() -> int:
    cases = [
        case_world_right_sign_convention,
        case_zero_sum,
        case_positive_convention_boosts_correct_demotes_wrong,
        case_no_op_when_sign_absent_or_weight_zero,
        case_energy_gate_scales_magnitude,
    ]
    print(f"running {len(cases)} audio_doa_head cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
