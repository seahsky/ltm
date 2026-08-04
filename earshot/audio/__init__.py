"""Audio: the sensor spec, the guard, onset, the bed, CLAP, and the lateral cue.

Deliberately does **not** import ``sim`` (ADR-0013). Every simulator object this
layer touches arrives injected — the sensor handle and an ``observe`` callable — which
is what keeps ``import habitat_sim`` in exactly one file and this whole layer
Mac-testable.

Kept empty on purpose: ``earshot/__init__.py`` imports ``audio.guard`` to pin the
logging, so anything expensive here would be paid on every entry into the package.
"""
