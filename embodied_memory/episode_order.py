"""
Pin habitat's episode iterator for deterministic cold-before-warm ordering.

The lifelong/revisit eval runs multiple scenes in one process (``--scene all``).
The analyzer assigns each episode a *visit order* by the order the runner
processed it — 0 = "cold" (first sighting of a category in a scene), >=1 =
"warm". For that labelling to be correct, and for a warm visit to run only
*after* its cold sighting was indexed in the persisting LTM, habitat must yield
each (scene, category) group's cold episode first. Two iterator options
guarantee that, so we pin BOTH rather than trust habitat's defaults:

  * ``shuffle = False``       — keep each scene's episodes in dataset order
                                (the builder writes the cold seed first).
  * ``group_by_scene = True`` — process one scene's episodes contiguously before
                                the next, so a scene's cold seed precedes its
                                warm visits even across a multi-scene run.

The single-scene smoke happened to order correctly; multi-scene must guarantee
it regardless of habitat defaults.

Habitat-free (operates on the passed config object) so it unit-tests without the
sim; the caller (``habitat_env._build_env``) invokes it inside ``read_write``.
"""

from __future__ import annotations


def _iterator_options(config):
    """Return the live iterator-options object habitat will read, or None.

    The REAL key on current habitat-lab is
    ``config.habitat.environment.iterator_options`` — that is where
    ``habitat.Env`` reads the options (``config.environment.iterator_options.items()``)
    and its ``IteratorOptionsConfig.shuffle`` default is True. An earlier version
    of this helper targeted ``config.habitat.dataset.episode_iterator_options``,
    which does NOT exist on ``DatasetConfig`` — so the pin was a silent no-op and
    ``shuffle`` stayed True on every real run. We prefer the real environment key
    and keep the legacy dataset key as a graceful fallback for alternate layouts.
    """
    # Preferred (real) location.
    try:
        opts = config.habitat.environment.iterator_options
        if opts is not None:
            return opts
    except Exception:
        pass
    # Legacy / fictional fallback (kept so an alternate habitat layout still pins).
    try:
        opts = config.habitat.dataset.episode_iterator_options
        if opts is not None:
            return opts
    except Exception:
        pass
    return None


def pin_episode_order(config) -> bool:
    """Pin the episode iterator for cold-before-warm ordering: set
    ``shuffle = False`` and ``group_by_scene = True``.

    Targets the REAL habitat key ``config.habitat.environment.iterator_options``
    (falling back to the legacy ``config.habitat.dataset.episode_iterator_options``
    if only that is present). Returns True if both options were set, False if the
    config exposes neither iterator-options key (an older/newer habitat layout) —
    the caller treats False as a harmless no-op, never an error. Must be called
    inside a ``read_write(config)`` block when the config is a frozen omegaconf
    object.
    """
    opts = _iterator_options(config)
    if opts is None:
        return False
    try:
        opts.shuffle = False
        opts.group_by_scene = True
    except Exception:
        return False
    return True
