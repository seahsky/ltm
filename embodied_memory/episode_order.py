"""
Pin habitat's episode iterator to NOT shuffle.

The lifelong/revisit eval runs multiple scenes in one process (``--scene all``).
The analyzer assigns each episode a *visit order* by the order the runner
processed it — 0 = "cold" (first sighting of a category in a scene), >=1 =
"warm". For that labelling to be correct, and for a warm visit to run only
*after* its cold sighting was indexed in the persisting LTM, habitat must yield
each (scene, category) group's cold episode first. Habitat's ``group_by_scene``
keeps scenes contiguous; setting ``shuffle = False`` keeps each scene's episodes
in dataset order (the builder writes cold first). The single-scene smoke happened
to order correctly; multi-scene must guarantee it regardless of habitat defaults.

Habitat-free (operates on the passed config object) so it unit-tests without the
sim; the caller (``habitat_env._build_env``) invokes it inside ``read_write``.
"""

from __future__ import annotations


def pin_no_shuffle(config) -> bool:
    """Set ``config.habitat.dataset.episode_iterator_options.shuffle = False``.

    Returns True if the option was set, False if the config lacks that key
    (an older/newer habitat layout) — the caller treats False as a harmless
    no-op, never an error. Must be called inside a ``read_write(config)`` block
    when the config is a frozen omegaconf object.
    """
    try:
        opts = config.habitat.dataset.episode_iterator_options
    except Exception:
        return False
    try:
        opts.shuffle = False
    except Exception:
        return False
    return True
