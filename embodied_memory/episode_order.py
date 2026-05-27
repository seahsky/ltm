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


def pin_episode_order(config) -> bool:
    """Pin ``episode_iterator_options`` for cold-before-warm ordering: set
    ``shuffle = False`` and ``group_by_scene = True``.

    Returns True if both options were set, False if the config lacks the
    ``episode_iterator_options`` key (an older/newer habitat layout) — the caller
    treats False as a harmless no-op, never an error. Must be called inside a
    ``read_write(config)`` block when the config is a frozen omegaconf object.
    """
    try:
        opts = config.habitat.dataset.episode_iterator_options
    except Exception:
        return False
    try:
        opts.shuffle = False
        opts.group_by_scene = True
    except Exception:
        return False
    return True
