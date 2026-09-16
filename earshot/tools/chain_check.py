"""Does the chained `M^E` on disk match the scenes a resumed sweep is about to skip.

**WHY A RESUME NEEDS THIS AT ALL.** The `dream` arm appends `M^E` to one file, scene by
scene, and `--resume` skips a scene directory that already holds a finished run. Those
two facts can disagree. A scene whose episodes completed but whose `--dream-memory-out`
never landed -- the process killed between the last episode write and the memory dump --
leaves a finished directory and a memory file that never saw it. Resuming then skips the
scene, and `M^E` is missing a house for the rest of the sweep with nothing on any audit
saying so.

That is `matrix-1`'s silent scene drop, which cost two of nineteen scenes and was found
only by a review written afterwards. The difference here is that the memory file already
records its own provenance, so the disagreement is CHEAP to detect: compare the scene
list the file carries against the scenes the driver is about to skip, in order.

Exit 0 when they agree, 2 when they do not, and print both lists either way. Read-only,
no GPU, milliseconds.

    python -m earshot.tools.chain_check runs/<tag>/dream/memory.json --expect "sceneA sceneB"
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple

from earshot.task.dream_store import read_provenance

__all__ = ["compare", "main"]


def compare(
    carried: Sequence[str], expected: Sequence[str]
) -> Tuple[bool, str]:
    """`(agree, why)`. Order matters: the chain's order is part of its result.

    `M^E` is built by appending, so scene k's rows were written against a memory holding
    scenes 0..k-1. Two chains over the same scenes in a different order are two different
    memories, and `ablation_sweep.sh` says so in its own header.
    """
    carried = tuple(str(scene) for scene in carried)
    expected = tuple(str(scene) for scene in expected)
    if carried == expected:
        return True, "the memory carries exactly the {} finished scene(s), in order".format(
            len(carried)
        )
    if not carried and expected:
        return False, (
            "the memory file carries NO scene list but {} scene(s) are finished on "
            "disk. Skipping them would restart M^E from empty and every later scene "
            "would be scored against a memory those scenes never entered.".format(
                len(expected)
            )
        )
    if len(carried) != len(expected):
        return False, (
            "the memory carries {} scene(s) and {} are finished on disk. The "
            "difference is what a resume would silently drop from M^E.".format(
                len(carried), len(expected)
            )
        )
    first = next(
        index for index, pair in enumerate(zip(carried, expected)) if pair[0] != pair[1]
    )
    return False, (
        "the chains diverge at position {}: the memory has {!r} where the finished "
        "scenes have {!r}. Same length, different order, so this is not a dropped "
        "scene but a different memory.".format(first, carried[first], expected[first])
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m earshot.tools.chain_check",
        description=(
            "Compare a chained M^E's recorded scene list against the scenes a resumed "
            "sweep would skip. Exits 2 when they disagree."
        ),
    )
    parser.add_argument("memory", help="the chain's memory.json")
    parser.add_argument(
        "--expect",
        default="",
        help=(
            "the finished scenes, space-separated, IN SWEEP ORDER. Empty means the "
            "chain has not started, which a missing memory file matches."
        ),
    )
    args = parser.parse_args(argv)

    carried = read_provenance(args.memory)
    expected = tuple(args.expect.split())
    agree, why = compare(carried, expected)

    print("  chain check: {}".format(args.memory))
    print("    memory carries:  {}".format(" ".join(carried) if carried else "(nothing)"))
    print("    finished on disk: {}".format(" ".join(expected) if expected else "(nothing)"))
    print("    {}".format(why))
    if agree:
        return 0
    print(
        "    REFUSING THE RESUME. Re-run this arm from a fresh --tag, or delete the\n"
        "    unfinished scene directories AND the memory file to start the chain over.\n"
        "    A chain that silently lost a scene cannot be detected afterwards."
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
