"""
Sanity test for ``episode_order.pin_no_shuffle`` — pins habitat's episode
iterator to NOT shuffle so a multi-scene revisit run yields each
(scene, category) group's COLD seed episode before its WARM revisits (the
analyzer assigns visit order by processing order; a shuffled iterator would
mislabel warm/cold and could run a warm visit before its cold sighting was
ever indexed in the LTM).

Stdlib-only (uses a fake config namespace) — runs locally without habitat.

Invoke with::

    python embodied_memory/scripts/test_episode_order.py
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import episode_order  # noqa: E402


def _cfg_with_iterator_options(shuffle=True):
    """A minimal stand-in for habitat's nested config:
    config.habitat.dataset.episode_iterator_options.shuffle"""
    opts = types.SimpleNamespace(shuffle=shuffle)
    dataset = types.SimpleNamespace(episode_iterator_options=opts)
    habitat = types.SimpleNamespace(dataset=dataset)
    return types.SimpleNamespace(habitat=habitat)


def case_sets_shuffle_false():
    cfg = _cfg_with_iterator_options(shuffle=True)
    ok = episode_order.pin_no_shuffle(cfg)
    assert ok is True, ok
    assert cfg.habitat.dataset.episode_iterator_options.shuffle is False
    print("  case sets_shuffle_false: OK")


def case_missing_key_is_noop():
    # A config lacking episode_iterator_options must not crash — return False.
    cfg = types.SimpleNamespace(habitat=types.SimpleNamespace(dataset=types.SimpleNamespace()))
    ok = episode_order.pin_no_shuffle(cfg)
    assert ok is False, ok
    print("  case missing_key_is_noop: OK")


def case_no_habitat_attr_is_noop():
    ok = episode_order.pin_no_shuffle(types.SimpleNamespace())
    assert ok is False, ok
    print("  case no_habitat_attr_is_noop: OK")


def main() -> int:
    print("episode_order.pin_no_shuffle sanity tests")
    case_sets_shuffle_false()
    case_missing_key_is_noop()
    case_no_habitat_attr_is_noop()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
