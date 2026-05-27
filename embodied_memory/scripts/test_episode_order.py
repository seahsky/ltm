"""
Sanity test for ``episode_order.pin_episode_order`` — pins habitat's episode
iterator for deterministic cold-before-warm ordering (``shuffle = False`` AND
``group_by_scene = True``) so a multi-scene revisit run yields each
(scene, category) group's COLD seed episode before its WARM revisits (the
analyzer assigns visit order by processing order; a shuffled or scene-interleaved
iterator would mislabel warm/cold and could run a warm visit before its cold
sighting was ever indexed in the LTM).

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


def _cfg_with_iterator_options(shuffle=True, group_by_scene=False):
    """A minimal stand-in for habitat's nested config:
    config.habitat.dataset.episode_iterator_options.{shuffle,group_by_scene}"""
    opts = types.SimpleNamespace(shuffle=shuffle, group_by_scene=group_by_scene)
    dataset = types.SimpleNamespace(episode_iterator_options=opts)
    habitat = types.SimpleNamespace(dataset=dataset)
    return types.SimpleNamespace(habitat=habitat)


def case_pins_order():
    cfg = _cfg_with_iterator_options(shuffle=True, group_by_scene=False)
    ok = episode_order.pin_episode_order(cfg)
    assert ok is True, ok
    assert cfg.habitat.dataset.episode_iterator_options.shuffle is False
    assert cfg.habitat.dataset.episode_iterator_options.group_by_scene is True
    print("  case pins_order (shuffle False + group_by_scene True): OK")


def case_missing_key_is_noop():
    # A config lacking episode_iterator_options must not crash — return False.
    cfg = types.SimpleNamespace(habitat=types.SimpleNamespace(dataset=types.SimpleNamespace()))
    ok = episode_order.pin_episode_order(cfg)
    assert ok is False, ok
    print("  case missing_key_is_noop: OK")


def case_no_habitat_attr_is_noop():
    ok = episode_order.pin_episode_order(types.SimpleNamespace())
    assert ok is False, ok
    print("  case no_habitat_attr_is_noop: OK")


def main() -> int:
    print("episode_order.pin_episode_order sanity tests")
    case_pins_order()
    case_missing_key_is_noop()
    case_no_habitat_attr_is_noop()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
