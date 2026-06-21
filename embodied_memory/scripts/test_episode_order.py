"""
Sanity test for ``episode_order.pin_episode_order`` — pins habitat's episode
iterator for deterministic cold-before-warm ordering (``shuffle = False`` AND
``group_by_scene = True``) so a multi-scene revisit run yields each
(scene, category) group's COLD seed episode before its WARM revisits (the
analyzer assigns visit order by processing order; a shuffled or scene-interleaved
iterator would mislabel warm/cold and could run a warm visit before its cold
sighting was ever indexed in the LTM).

REGRESSION CONTEXT (2026-06-21): the original code+test targeted the FICTIONAL
key ``config.habitat.dataset.episode_iterator_options`` — which does NOT exist on
the real habitat ``DatasetConfig``. habitat reads the iterator options from
``config.habitat.environment.iterator_options`` (an ``IteratorOptionsConfig`` whose
``shuffle`` default is True). So the pin was a SILENT NO-OP on every real run and
``shuffle`` stayed True — which let a 1-episode non-LOS Tier-3 caption run grab a
random (warm) episode instead of the cold seed at index 0. These cases now assert
against the REAL key (with a back-compat case for the legacy/fictional one).

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


def _cfg_real(shuffle=True, group_by_scene=False):
    """A minimal stand-in for the REAL habitat nested config:
    config.habitat.environment.iterator_options.{shuffle,group_by_scene}
    (this is where habitat.Env actually reads the options — see
    habitat/core/env.py: ``config.environment.iterator_options.items()``)."""
    opts = types.SimpleNamespace(shuffle=shuffle, group_by_scene=group_by_scene)
    environment = types.SimpleNamespace(iterator_options=opts)
    # also carry an (empty) dataset so the helper's dataset-key fallback can't
    # accidentally satisfy the pin and mask a regression on the real key.
    dataset = types.SimpleNamespace()
    habitat = types.SimpleNamespace(environment=environment, dataset=dataset)
    return types.SimpleNamespace(habitat=habitat)


def _cfg_legacy(shuffle=True, group_by_scene=False):
    """The legacy/fictional layout ``config.habitat.dataset.episode_iterator_options``
    (no ``environment.iterator_options``). Kept so the helper degrades gracefully
    on an alternate habitat layout, but this is NOT the real one."""
    opts = types.SimpleNamespace(shuffle=shuffle, group_by_scene=group_by_scene)
    dataset = types.SimpleNamespace(episode_iterator_options=opts)
    habitat = types.SimpleNamespace(dataset=dataset)
    return types.SimpleNamespace(habitat=habitat)


def case_pins_real_environment_key():
    """The decisive regression case: on the REAL key the pin must actually flip
    shuffle False + group_by_scene True (the old code no-oped here)."""
    cfg = _cfg_real(shuffle=True, group_by_scene=False)
    ok = episode_order.pin_episode_order(cfg)
    assert ok is True, ok
    assert cfg.habitat.environment.iterator_options.shuffle is False, \
        cfg.habitat.environment.iterator_options.shuffle
    assert cfg.habitat.environment.iterator_options.group_by_scene is True, \
        cfg.habitat.environment.iterator_options.group_by_scene
    print("  case pins_real_environment_key (shuffle False + group_by_scene True): OK")


def case_pins_legacy_dataset_key():
    """Back-compat: on a config that only exposes the legacy dataset key the pin
    still works (so we never regress an alternate habitat layout)."""
    cfg = _cfg_legacy(shuffle=True, group_by_scene=False)
    ok = episode_order.pin_episode_order(cfg)
    assert ok is True, ok
    assert cfg.habitat.dataset.episode_iterator_options.shuffle is False
    assert cfg.habitat.dataset.episode_iterator_options.group_by_scene is True
    print("  case pins_legacy_dataset_key: OK")


def case_real_key_preferred_over_legacy():
    """If BOTH keys are present (unlikely, but defensive), the real environment
    key MUST be pinned (that is the one habitat reads)."""
    real_opts = types.SimpleNamespace(shuffle=True, group_by_scene=False)
    legacy_opts = types.SimpleNamespace(shuffle=True, group_by_scene=False)
    environment = types.SimpleNamespace(iterator_options=real_opts)
    dataset = types.SimpleNamespace(episode_iterator_options=legacy_opts)
    habitat = types.SimpleNamespace(environment=environment, dataset=dataset)
    cfg = types.SimpleNamespace(habitat=habitat)
    ok = episode_order.pin_episode_order(cfg)
    assert ok is True, ok
    assert real_opts.shuffle is False, "real environment key must be pinned"
    assert real_opts.group_by_scene is True
    print("  case real_key_preferred_over_legacy: OK")


def case_missing_key_is_noop():
    # A config lacking BOTH iterator-option keys must not crash — return False.
    cfg = types.SimpleNamespace(
        habitat=types.SimpleNamespace(
            dataset=types.SimpleNamespace(),
            environment=types.SimpleNamespace(),
        )
    )
    ok = episode_order.pin_episode_order(cfg)
    assert ok is False, ok
    print("  case missing_key_is_noop: OK")


def case_no_habitat_attr_is_noop():
    ok = episode_order.pin_episode_order(types.SimpleNamespace())
    assert ok is False, ok
    print("  case no_habitat_attr_is_noop: OK")


def main() -> int:
    print("episode_order.pin_episode_order sanity tests")
    case_pins_real_environment_key()
    case_pins_legacy_dataset_key()
    case_real_key_preferred_over_legacy()
    case_missing_key_is_noop()
    case_no_habitat_attr_is_noop()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
