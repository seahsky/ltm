"""Memory: the stores for ADR-0018's generalization matrix, and DREAM's consolidation.

`store.py` holds two frozen tables, `SemanticStore` and `EpisodicStore`, and the pure
filters (`without_class`, `without_scene`) that carve the four matrix cells out of them.
`consolidate.py` is DREAM §III.D (eq. 8-13): it decides which segments of a finished
episode are worth keeping, and it is what fills the hierarchical long-term memory the
paper's eq. 14 defines.

The two modules do not import each other, and the layer table still reads
`"memory": ("types",)`. Eq. 12's novelty ranges over all three levels of `M^L`, whose row
types differ, so `novelty` takes vectors rather than rows and needs nothing from
`store.py`.

Neither store nor filter reaches for `earshot.audio.vocabulary`: the whole point of the heard/
unheard split is that the semantic store must LEARN a sound-room association rather than
read it off the placement table, and `tests/mac/test_audio_vocabulary.py` fences that.

Kept empty on purpose, in the shape `agent/__init__.py` and `audio/__init__.py` already
set: `earshot/__init__.py` runs on every entry into the package, so anything imported
here is paid for by every process that touches the tree, including `ruff` and the
structural tests. Callers import the submodule directly — `from earshot.memory.store
import SemanticStore` — exactly as the rest of the tree imports `earshot.audio.onset` or
`earshot.agent.detector` rather than their packages.
"""
