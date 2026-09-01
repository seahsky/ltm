"""Memory: the semantic and episodic stores for ADR-0018's generalization matrix.

Two frozen tables, `SemanticStore` and `EpisodicStore`, and the pure filters
(`without_class`, `without_scene`) that carve the four matrix cells out of them. Neither
store nor filter reaches for `earshot.audio.vocabulary`: the whole point of the heard/
unheard split is that the semantic store must LEARN a sound-room association rather than
read it off the placement table, and `tests/mac/test_audio_vocabulary.py` fences that.

Kept empty on purpose, in the shape `agent/__init__.py` and `audio/__init__.py` already
set: `earshot/__init__.py` runs on every entry into the package, so anything imported
here is paid for by every process that touches the tree, including `ruff` and the
structural tests. Callers import the submodule directly — `from earshot.memory.store
import SemanticStore` — exactly as the rest of the tree imports `earshot.audio.onset` or
`earshot.agent.detector` rather than their packages.
"""
