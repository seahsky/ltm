"""The tour controls must exist on BOTH entry points into the tour, and survive a re-exec.

`prior_pass.sh` grew `--start-draws` (PR #88) and `--max-tour-dy` (PR #90) because two
scenes could not be toured without them. `matrix_sweep.sh` calls `task/prior_driver.py`
directly rather than through `prior_pass.sh`, never grew either flag, and so every matrix
sweep would have run the tour that `prior-2` measured and died at its own coverage gate --
with both fixes sitting in the repo, merged, tested, and unreachable from the one script
that needed them.

Nothing caught it because nothing asserted the two halves agree. This file does, on the
scripts AS SHIPPED: the flag names are read out of the case blocks and out of
`prior_driver`'s own parser, never pasted here, so a knob added to one side and forgotten
on the other fails on a Mac in a second instead of on the box in an hour.

Three arms:
  * every tour flag is offered by both scripts;
  * every tour flag `matrix_sweep.sh` forwards is one `prior_driver` actually accepts
    (the typo case -- a wrong name is a hard failure only once the box runs it);
  * every tour flag survives the self-update re-exec, because a re-exec that drops
    `--max-tour-dy` spends the night building a DIFFERENT store than the one asked for
    and says nothing.
"""

import pathlib
import re
import unittest

from _interpreter import assert_interpreter  # noqa: F401

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
SWEEP = TOOLS / "matrix_sweep.sh"
PRIOR_PASS = TOOLS / "prior_pass.sh"

# The knobs that change what the tour DOES, and therefore what lands in the store. Not
# every flag: `--scenes` is computed by the sweep and passed by hand to the standalone
# pass, and `--run-dir`/`--tag` name the output rather than the work.
TOUR_FLAGS = ("--classes", "--seed", "--leg-budget", "--goal-radius",
              "--start-draws", "--max-tour-dy")


def case_flags(path: pathlib.Path) -> set:
    """The long options a script's `case` block accepts, read from the script itself."""
    return set(re.findall(r"^\s*(--[a-z0-9-]+)\)", path.read_text(), flags=re.M))


def forwarded_to_prior_driver(source: str) -> set:
    """The long options the sweep hands to `prior_driver`, from that invocation only."""
    start = source.index("python -m earshot.task.prior_driver")
    end = source.index("PRIOR_STATUS", start)
    return set(re.findall(r"(--[a-z0-9-]+)", source[start:end]))


def prior_driver_options() -> set:
    """`prior_driver`'s own parser, asked rather than transcribed."""
    from earshot.task.prior_driver import build_parser
    return {
        option
        for action in build_parser()._actions
        for option in action.option_strings
        if option.startswith("--")
    }


class TestTheTourFlagsReachBothScripts(unittest.TestCase):
    def test_every_tour_flag_is_offered_by_both_entry_points(self):
        sweep, standalone = case_flags(SWEEP), case_flags(PRIOR_PASS)
        for flag in TOUR_FLAGS:
            self.assertIn(flag, standalone, "{} is not on prior_pass.sh".format(flag))
            self.assertIn(
                flag, sweep,
                "{} controls the tour and matrix_sweep.sh does not offer it, so the "
                "sweep cannot run the tour a standalone pass measured".format(flag),
            )

    def test_the_sweep_forwards_only_flags_prior_driver_accepts(self):
        forwarded = forwarded_to_prior_driver(SWEEP.read_text())
        accepted = prior_driver_options()
        self.assertTrue(forwarded, "the prior_driver invocation was not found")
        unknown = sorted(forwarded - accepted)
        self.assertEqual(
            unknown, [],
            "matrix_sweep.sh passes {} to prior_driver, which does not accept "
            "them".format(unknown),
        )

    def test_the_tour_flags_survive_the_self_update_re_exec(self):
        source = SWEEP.read_text()
        start = source.index("exec bash \"$0\"")
        end = source.index("\nelse", start)
        re_exec = source[start:end]
        for flag in TOUR_FLAGS:
            self.assertIn(
                flag, re_exec,
                "the re-exec drops {}, so a sweep that self-updated would build a "
                "different store than the one asked for".format(flag),
            )

    def test_prior_only_stops_before_the_cells(self):
        """The dry run is only a dry run if it exits above the cell loop."""
        source = SWEEP.read_text()
        self.assertIn("--prior-only)", source)
        exit_at = source.index("--prior-only: stopping after the gate")
        cells_at = source.index("banner \"[5/6]")
        self.assertLess(
            exit_at, cells_at,
            "--prior-only must stop above the cell loop, not inside it",
        )


if __name__ == "__main__":
    unittest.main()
