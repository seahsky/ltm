#!/usr/bin/env python
"""Unit tests for make_demo_sidebyside — the pure frame compositing (no video I/O).

Run: /opt/anaconda3/envs/ltm-embodied/bin/python embodied_memory/scripts/test_make_demo_sidebyside.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embodied_memory.scripts.make_demo_sidebyside import (  # noqa: E402
    add_label_bar,
    pad_to_length,
    stack_clips,
    stack_pair,
)


def _frame(h, w, val):
    return np.full((h, w, 3), val, dtype=np.uint8)


def case_pad_to_length_freezes_last():
    a, b = _frame(2, 2, 1), _frame(2, 2, 2)
    out = pad_to_length([a, b], 4)
    assert len(out) == 4
    assert np.array_equal(out[2], b) and np.array_equal(out[3], b)  # last frozen


def case_pad_to_length_no_truncate():
    a, b, c = _frame(2, 2, 1), _frame(2, 2, 2), _frame(2, 2, 3)
    out = pad_to_length([a, b, c], 2)  # target < len -> unchanged
    assert len(out) == 3


def case_pad_empty_safe():
    assert pad_to_length([], 5) == []


def case_stack_pair_shape_with_gap():
    out = stack_pair(_frame(4, 4, 10), _frame(4, 4, 20), gap_px=8)
    assert out.shape == (4, 4 + 8 + 4, 3), out.shape


def case_stack_pair_no_gap():
    out = stack_pair(_frame(4, 5, 10), _frame(4, 6, 20), gap_px=0)
    assert out.shape == (4, 11, 3), out.shape


def case_stack_pair_matches_height():
    # different heights -> letterboxed to the taller
    out = stack_pair(_frame(4, 4, 10), _frame(6, 4, 20), gap_px=0)
    assert out.shape[0] == 6, out.shape


def case_stack_pair_preserves_lr_content():
    out = stack_pair(_frame(4, 4, 10), _frame(4, 4, 20), gap_px=2, gap_color=(0, 0, 0))
    assert np.all(out[:, :4] == 10), "left half preserved"
    assert np.all(out[:, -4:] == 20), "right half preserved"
    assert np.all(out[:, 4:6] == 0), "gap bar"


def case_stack_clips_pads_to_max_len():
    left = [_frame(3, 3, i) for i in range(3)]
    right = [_frame(3, 3, i) for i in range(5)]
    out = stack_clips(left, right, gap_px=1)
    assert len(out) == 5, len(out)  # padded to the longer clip
    assert out[0].shape == (3, 3 + 1 + 3, 3)


def case_add_label_bar_grows_height():
    out = add_label_bar(_frame(4, 16, 50), "OFF", "ON", bar_h=30)
    assert out.shape == (4 + 30, 16, 3), out.shape
    # the original frame content sits below the bar, unchanged
    assert np.all(out[30:] == 50)


def case_add_label_bar_zero_height_passthrough_like():
    out = add_label_bar(_frame(4, 8, 7), "a", "b", bar_h=0)
    assert out.shape == (4, 8, 3)


def main():
    case_pad_to_length_freezes_last()
    case_pad_to_length_no_truncate()
    case_pad_empty_safe()
    case_stack_pair_shape_with_gap()
    case_stack_pair_no_gap()
    case_stack_pair_matches_height()
    case_stack_pair_preserves_lr_content()
    case_stack_clips_pads_to_max_len()
    case_add_label_bar_grows_height()
    case_add_label_bar_zero_height_passthrough_like()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
