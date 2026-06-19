"""
TDD for fetch_anomaly_clips.select_clip (pure — no network). The download path is
integration-only (needs internet) and is exercised on RACE.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_fetch_anomaly_clips.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_anomaly_clips as fc  # noqa: E402


def _rows():
    return [
        {"filename": "1-b.wav", "category": "crying_baby"},
        {"filename": "1-a.wav", "category": "crying_baby"},
        {"filename": "2-x.wav", "category": "clock_alarm"},
        {"filename": "3-y.wav", "category": "glass_breaking"},
    ]


def case_select_clip_sorted_deterministic():
    # sorted -> '1-a.wav' before '1-b.wav', so index 0 is stable across CSV order
    assert fc.select_clip(_rows(), "crying_baby", 0) == "1-a.wav"
    assert fc.select_clip(_rows(), "crying_baby", 1) == "1-b.wav"
    print("  case select_clip_sorted_deterministic: OK")


def case_select_clip_index_wraps():
    assert fc.select_clip(_rows(), "crying_baby", 2) == "1-a.wav"   # wraps (2 % 2 == 0)
    print("  case select_clip_index_wraps: OK")


def case_select_clip_absent_category():
    assert fc.select_clip(_rows(), "siren", 0) is None
    assert fc.select_clip([], "crying_baby", 0) is None
    print("  case select_clip_absent_category: OK")


def case_class_map_covers_locked_classes():
    assert set(fc.CLASS_TO_ESC50) == {"baby_cry", "alarm", "glass_break"}
    assert fc.CLASS_TO_ESC50["baby_cry"] == "crying_baby"
    assert fc.CLASS_TO_ESC50["alarm"] == "clock_alarm"
    assert fc.CLASS_TO_ESC50["glass_break"] == "glass_breaking"
    print("  case class_map_covers_locked_classes: OK")


def main() -> int:
    cases = [
        case_select_clip_sorted_deterministic,
        case_select_clip_index_wraps,
        case_select_clip_absent_category,
        case_class_map_covers_locked_classes,
    ]
    print(f"running {len(cases)} fetch_anomaly_clips cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
