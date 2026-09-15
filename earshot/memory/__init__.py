"""Memory: the stores for ADR-0018's generalization matrix, and DREAM's consolidation.

`store.py` holds two frozen tables, `SemanticStore` and `EpisodicStore`, and the pure
filters (`without_class`, `without_scene`) that carve the four matrix cells out of them.
`consolidate.py` is DREAM §III.D (eq. 8-13): it decides which segments of a finished
episode are worth keeping. `longterm.py` is §III.E (eq. 14-18): the three levels those
retained segments go into, and the `G` that abstracts one level into the next.
`retrieve.py` is §III.F (eq. 19-24): the query, the three per-level retrievals, and the
`omega_t` that weighs them -- the mechanism the paper's central claim is about.

`consolidate.py` imports neither of the others — eq. 12's novelty ranges over rows whose
types differ, so it takes vectors. `longterm.py` imports `store.py`, because DREAM's `M^K`
(eq. 18) IS `SemanticStore` rather than a second sound-to-object table, and that one edge
is why `LAYER_IMPORTS["memory"]` reads `("memory", "types")` rather than `("types",)`.
`audio` and `agent` are still absent from it, and those are the absences that do the work.

**"Episodic" means two different things in here, so read the names carefully.**
`store.EpisodicStore` is ADR-0018's matrix axis: where a category was literally seen on a
prior tour, as coordinates. `longterm.ExperienceStore` is DREAM's episodic EXPERIENCE
memory (eq. 15), which holds no coordinates at all by design. Nothing converts between
them.

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
