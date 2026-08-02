# 17 — Does the clean room pin the `ss2` env, or is the recipe enough?

Type: grilling
Status: open
Blocked by: none

## Question

The env the entire destination runs on is reproduced by re-running a resolver, not by restoring a pinned state.
Does the clean room ship an actual pin, and if so what kind — or is the written recipe plus a loud version-pair assertion the answer?

## Why it matters

`oneenv_gate.sh` installs from version *ranges* (`transformers>=4.40,<5`, `numpy>=1.16.1,<1.24`), so every rebuild re-resolves against whatever PyPI serves that day.

That is not hypothetical. It is exactly how the env broke: `transformers` drifted to 4.57.6 against a frozen `torch==2.0.1`, disabled its own PyTorch backend, and turned `ClapModel` into a dummy object that imports fine and cannot instantiate (ticket 13).

Ticket 13 fixed the **instance** (move the torch pin) and the **detection** (the gate now fails loudly when the resolved pair does not actually work).
Neither touches the **class**.

## What changed since the fog patch was written

Two things, both from tickets 13 and 14, and they cut in opposite directions:

- **The working set is now known** (ticket 13): python 3.9.19 / numpy 1.23.5 / torch 2.2.2+cu118 / transformers 4.57.6 / scipy 1.13.1, on habitat-sim `4f61e321` stock. So there is now something concrete to pin *to*.
- **It is written down and has a home** (ticket 14): `docs/race-box-runbook.md` section 3 carries that table and section 4 carries the build recipe, both stating outright that this is a record of what worked once, not a lockfile.

The fog patch asked whether this was "part of 14 rather than its own ticket".
Ticket 14 answered no: writing the recipe down does not make it a pin.

## What makes it a real question rather than an obvious yes

- A `pip freeze` of this env is **not** a reproducible artifact on its own. habitat-sim is compiled from source against a git branch and a submodule SHA, so a lockfile covers only part of the surface and the expensive part is the part it misses.
- Half the failure surface is already covered by ticket 13's loud gate. A pin that duplicates a working assertion buys less than it costs to maintain.
- Python 3.9 already bounds the worst drift axis: transformers 5.x requires >= 3.10, so resolution is capped at the 4.x line.
- Against all that: the box is a three-month reservation, and a pod that comes back without `ss2` is a rebuild with a resolver that has moved on.

## What would resolve it

A decision, plus wherever it lands in the clean room's layout:

- pin / do not pin, and if pinning, **which artifact** (a constraints file, a full `pip freeze`, a conda `environment.yml`, or just the version table plus the assertion) and **what asserts it at runtime**;
- whether the habitat-sim branch SHA + submodule SHA are pinned alongside, since they are the half a Python lockfile cannot reach;
- and whether the `ss2` rebuild path is a documented recipe (status quo) or an executable script the clean room owns.

Related: the clean room's bootstrap script is where a pin would naturally live, and that script does not exist yet.
