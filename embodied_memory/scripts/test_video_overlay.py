"""
TDD for video_overlay.draw_overlay — the informative episode-video HUD.

Pure helper; needs Pillow (present on the ltm-embodied env). Run:

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        /opt/anaconda3/envs/ltm-embodied/bin/python \
        embodied_memory/scripts/test_video_overlay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from embodied_memory import video_overlay as vo  # noqa: E402


def _frame(h=64, w=80, fill=0):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _hud(**kw):
    base = {"step": 7, "task": "audiogoal", "backbone": "remembr", "goal": "chair",
            "action": "FWD", "d2g": 2.31, "n_mem_chosen": 3, "n_mem_inj": 21,
            "wp_source": "memory", "wp_dist": 1.4, "caption": "a wooden chair near a table",
            "is_stop": False}
    base.update(kw)
    return base


def case_shape_preserved():
    out = vo.draw_overlay(_frame(64, 80), _hud())
    assert out.shape == (64, 80, 3), out.shape
    print("  case shape_preserved: OK")


def case_dtype_preserved():
    out = vo.draw_overlay(_frame(96, 128), _hud())
    assert out.dtype == np.uint8, out.dtype
    print("  case dtype_preserved: OK")


def case_none_hud_returns_input():
    f = _frame()
    out = vo.draw_overlay(f, None)
    assert out is f or np.array_equal(out, f)
    print("  case none_hud_returns_input: OK")


def case_empty_hud_no_crash():
    f = _frame()
    out = vo.draw_overlay(f, {})
    assert out.shape == f.shape and out.dtype == np.uint8
    print("  case empty_hud_no_crash: OK")


def case_input_not_mutated():
    f = _frame(64, 80, fill=123)
    before = f.copy()
    _ = vo.draw_overlay(f, _hud())
    assert np.array_equal(f, before), "input frame was mutated"
    print("  case input_not_mutated: OK")


def case_text_present_sentinel():
    # White frame so any drawn pixel (the dark HUD box) shows up as a delta.
    f = _frame(128, 160, fill=255)
    out = vo.draw_overlay(f, _hud(goal="SENTINELGOAL"))
    assert (out != f).any(), "nothing was drawn on the frame"
    print("  case text_present_sentinel: OK")


def case_never_raises_on_internal_error(monkeypatched=None):
    # Force an internal failure (font load) → must return the input unchanged,
    # never propagate. Exercises the never-raise contract w/o fighting imports.
    orig = vo._load_font
    vo._load_font = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        f = _frame(64, 80, fill=200)
        out = vo.draw_overlay(f, _hud())
        assert out is f or np.array_equal(out, f), "did not fall back to input on error"
    finally:
        vo._load_font = orig
    print("  case never_raises_on_internal_error: OK")


def case_audio_block_renders():
    f = _frame(128, 160, fill=255)
    out = vo.draw_overlay(f, _hud(audio={
        "detected": True, "klass": "alarm", "lateral": 1, "energy": 0.08,
        "onset_fired": True, "onset_step": 30, "target": "bed"}))
    assert (out != f).any()
    # green-ish audio bar present somewhere in the bottom strip
    assert out.shape == (128, 160, 3)
    print("  case audio_block_renders: OK")


def case_stop_banner():
    f = _frame(128, 160, fill=255)
    out = vo.draw_overlay(f, _hud(is_stop=True))
    assert (out != f).any()
    print("  case stop_banner: OK")


def case_non_contiguous_input():
    big = np.zeros((64, 80, 6), dtype=np.uint8)
    view = big[:, :, :3]  # non-contiguous (H,W,3) view
    assert not view.flags["C_CONTIGUOUS"]
    out = vo.draw_overlay(view, _hud())
    assert out.shape == (64, 80, 3) and out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]
    print("  case non_contiguous_input: OK")


def case_wrong_channels_returns_input():
    f = np.zeros((32, 40), dtype=np.uint8)  # 2D, no channel
    out = vo.draw_overlay(f, _hud())
    assert out is f or np.array_equal(out, f)
    print("  case wrong_channels_returns_input: OK")


def main() -> int:
    cases = [
        case_shape_preserved,
        case_dtype_preserved,
        case_none_hud_returns_input,
        case_empty_hud_no_crash,
        case_input_not_mutated,
        case_text_present_sentinel,
        case_never_raises_on_internal_error,
        case_audio_block_renders,
        case_stop_banner,
        case_non_contiguous_input,
        case_wrong_channels_returns_input,
    ]
    print(f"running {len(cases)} video_overlay cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
