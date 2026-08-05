"""Per-test outcomes from a unittest suite, and the control comparison a leak claim needs.

    python -m earshot.tools.suite_result --run --start-dir earshot/tests/box --out a.json
    python -m earshot.tools.suite_result --compare control.json hermetic.json

**Why this exists.** The first run of `hermeticity_gate.sh` on the box stopped with

    FATAL: the box suite is red without the old trees — that is a leak

and it was not a leak. Two of 45 box tests failed on `clap_instantiable`, which loads
`laion/clap-htsat-unfused` through transformers and touches nothing the reset deletes;
the cause is a torch/checkpoint interaction that has nothing to do with hermeticity.
The message asserted a causal claim the run could not support.

**A red without the old trees is evidence of a leak only if the same test is green with
them.** So the gate now runs the suite twice, once on either side of the move, and the
comparison here is what it reads:

  * `leaks` — green in the control, red without the trees. Fatal, and the only outcome
    that means what the original message said.
  * `pre_existing` — red in both. Loud, recorded in the run's artefact, and **not** a
    hermeticity failure. It is a sick environment, which is a different problem with a
    different owner.
  * `recovered` — red in the control, green without the trees. Reported, because a test
    that passes only once its own tree is gone is worth a look.
  * `vanished` / `appeared` — the suite did not run the same tests both times, which
    makes the comparison meaningless rather than clean, so it is stated.

`compare()` is pure, so the question *given a control and a hermetic result, is this a
leak* is answerable on the Mac with injected dicts — ticket 19's third row again.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import unittest
from typing import Any, Dict, Mapping, Optional, Sequence

__all__ = ["PASS", "run_suite", "compare", "main"]

PASS = "pass"
FAIL = "fail"
ERROR = "error"
SKIP = "skip"


class _RecordingResult(unittest.TestResult):
    """Outcomes by test id. Nothing is printed here; the tests' own output still is.

    ADR-0014: box tests print their measurements, and ticket 16's numbers are what made
    tickets 09, 15 and 17 decidable. Running the suite programmatically must not swallow
    that, so this records outcomes and leaves stdout alone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.outcomes: Dict[str, str] = {}

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes[test.id()] = PASS

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes[test.id()] = FAIL

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes[test.id()] = ERROR

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.outcomes[test.id()] = SKIP

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.outcomes[test.id()] = PASS

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.outcomes[test.id()] = FAIL


def run_suite(start_dir: str, *, pattern: str = "test*.py") -> Dict[str, Any]:
    """Discover and run, mirroring `python -m unittest discover <start_dir>`.

    `top_level_dir` is left at the discovery default (the start directory), which is what
    the CLI does — the box tests reach `earshot` through `PYTHONPATH`, not through the
    discovery root, and changing that here would run a different suite from the one the
    box gate runs.

    A module that fails to import becomes a `_FailedTest` and lands as an `error`, which
    is the correct signal: an import that only works while the old trees are present is
    exactly the leak this is looking for.
    """
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern=pattern)
    result = _RecordingResult()
    suite.run(result)
    outcomes = dict(result.outcomes)
    return {
        "start_dir": start_dir,
        "outcomes": outcomes,
        "n_tests": len(outcomes),
        "n_green": sum(1 for v in outcomes.values() if v == PASS),
        "green": all(v in (PASS, SKIP) for v in outcomes.values()) and bool(outcomes),
    }


def compare(control: Mapping[str, Any], hermetic: Mapping[str, Any]) -> Dict[str, Any]:
    """Which failures are attributable to the move, and which were already there."""
    c = dict(control.get("outcomes") or {})
    h = dict(hermetic.get("outcomes") or {})

    def bad(outcome):
        return outcome in (FAIL, ERROR)

    shared = sorted(set(c) & set(h))
    leaks = [t for t in shared if not bad(c[t]) and bad(h[t])]
    pre_existing = [t for t in shared if bad(c[t]) and bad(h[t])]
    recovered = [t for t in shared if bad(c[t]) and not bad(h[t])]
    return {
        "leaks": leaks,
        "pre_existing": pre_existing,
        "recovered": recovered,
        "vanished": sorted(set(c) - set(h)),
        "appeared": sorted(set(h) - set(c)),
        "n_control": len(c),
        "n_hermetic": len(h),
        # `comparable` is separate from `leaks` being empty: if the two runs did not
        # collect the same tests, "no leaks" is an absence of evidence rather than
        # evidence of absence, and the gate must not read it as the second.
        "comparable": bool(c) and bool(h) and set(c) == set(h),
    }


def _load(path: str) -> Dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run a suite and write its outcomes")
    mode.add_argument("--compare", nargs=2, metavar=("CONTROL", "HERMETIC"))
    parser.add_argument("--start-dir", default="earshot/tests/box")
    parser.add_argument("--out", help="where to write the JSON (default: stdout)")
    args = parser.parse_args(argv)

    if args.run:
        payload = run_suite(args.start_dir)
        text = json.dumps(payload, indent=2)
        if args.out:
            pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        print("  {}: {}/{} green".format(args.start_dir, payload["n_green"],
                                         payload["n_tests"]), file=sys.stderr)
        # Always 0: the caller wants the outcomes, and the verdict is the comparison's.
        return 0

    control, hermetic = (_load(p) for p in args.compare)
    verdict = compare(control, hermetic)
    print(json.dumps(verdict, indent=2))
    if not verdict["comparable"]:
        print("NOT COMPARABLE: the two runs did not collect the same tests "
              "(vanished {}, appeared {})".format(len(verdict["vanished"]),
                                                  len(verdict["appeared"])),
              file=sys.stderr)
        return 2
    if verdict["leaks"]:
        print("LEAKS ({}): {}".format(len(verdict["leaks"]),
                                      ", ".join(verdict["leaks"])), file=sys.stderr)
        return 1
    if verdict["pre_existing"]:
        print("no leaks; {} pre-existing failure(s), unrelated to the move: {}".format(
            len(verdict["pre_existing"]), ", ".join(verdict["pre_existing"])),
            file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
