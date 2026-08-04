"""The interpreter refusal. Every module in this suite imports it, first.

**Version skew is a third divergence axis**, alongside fake-versus-real and
Mac-versus-box, and neither ticket 19 nor ADR-0013 listed it. Measured 2026-08-04:
this Mac's default ``python3`` is **3.14.3**; the box is **3.9.19**. A suite green on
3.14 licenses nothing about whether the code even *imports* on the box. ADR-0013 names
two 3.9 constraints — ``int | None`` in an annotation needs ``from __future__ import
annotations``, and ``typing.get_type_hints()`` raises on those dataclasses — and both
are invisible under 3.14. ``match``, runtime PEP 604 unions and resolved ``list[str]``
annotations would all sail through here and break there.

**The refusal matters more than the pinned env.** It is the capability-shaped
discipline of ADR-0014 applied to the suite itself: the suite must not be able to pass
silently on the wrong Python the way ticket 13's gate passed silently on the wrong
torch. So this raises at import time, which turns every test module in the suite into
a collection error rather than letting a subset quietly run.

Why an imported module and not ``tests/mac/__init__.py``: ``unittest discover
earshot/tests/mac`` sets ``top_level_dir`` to the start directory, puts it on
``sys.path``, and imports each ``test_*.py`` as a **top-level** module — so the
package ``__init__.py`` is never executed and a refusal placed there would never run.
Verified against CPython's ``TestLoader._find_tests``, not assumed.
``test_suite_hygiene.py`` asserts every ``test_*.py`` here imports this module, so a
new file cannot opt out by forgetting.
"""

import sys

# The box is 3.9.19; `actions/setup-python` provides 3.9 up to 3.9.25 (checked against
# actions/python-versions' versions-manifest.json on 2026-08-04, 32 entries, linux-x64
# present — so the refusal did NOT have to widen, which ADR-0014 flagged as its one
# known way to weaken). Patch drift inside 3.9 is ABI-stable, so the minor is the pin.
REQUIRED_PYTHON = (3, 9)


def assert_interpreter() -> None:
    actual = sys.version_info[:2]
    if actual != REQUIRED_PYTHON:
        raise RuntimeError(
            "the Mac suite refuses to run on Python {}.{} — it is pinned to {}.{}, "
            "because the box is 3.9 and a suite green on another interpreter licenses "
            "nothing about whether this code imports there. Create the env and use it:"
            "\n    conda create -y -n earshot-mac python=3.9"
            "\n    conda run -n earshot-mac pip install -r earshot/tools/mac-requirements.txt"
            "\n    conda run -n earshot-mac python -m unittest discover earshot/tests/mac"
            .format(actual[0], actual[1], *REQUIRED_PYTHON)
        )


assert_interpreter()
