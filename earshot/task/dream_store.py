"""`M^E` on disk, so a DREAM memory can outlive one `python -m earshot` invocation.

**WHY THIS MODULE EXISTS.** `dream-1` ran 19 scenes and built 19 memories. The sweep
invokes the runner once per scene -- which is what gives every scene its own directory,
its own log, its own smoke gate and its own place in the continue-on-failure rule -- so
`run()`'s carry-over never crossed a scene boundary. `tools/dream_report.py` section D
measured it: every one of the 19 scenes started from an empty `M^E`, and the largest
memory any episode ever saw held 83 rows built from that scene alone.

That is the upstream half of why `omega_t` was inert. `tests/mac/test_dream_omega_reach.py`
pins the downstream half: `omega` reaches behaviour only through `memory_consistency`,
which renormalises `omega^E` against `omega^P`, so it can only act when `M^P` is
non-empty. And `M^P` is `abstract(M^E, min_support)` grouped by `(c^audio, c^obj, c^sce)`
-- two successes must share a triple before a pattern exists at all. Fifteen episodes in
one room at `dream-1`'s 32.6% reach rate rarely produce two. Nineteen scenes' successes
at one `(sound, object)` triple routinely do.

So this file carries `M^E` from one invocation to the next, and the sweep chains the
scenes through it. Nothing about `G` changes: the route to a populated `M^P` is more
experience, not a looser abstraction.

**`M^E` IS PERSISTED AND `M^P` IS NOT.** `dream.consolidate_episode` already REBUILDS the
whole pattern store from the whole experience store on every episode rather than appending
to it, so `M^P` is a pure function of `M^E` and `min_support` and has no independent state
to save. Writing it too would create a second source of truth that a hand-edited file
could put out of step with the first. `load_memory` rebuilds it with the caller's
`min_support`, which is the knob the audit already records.

**`M^K` IS NOT PERSISTED EITHER, AND THAT IS NOT AN OVERSIGHT.** Nothing in the DREAM path
writes the semantic level -- `consolidate_episode` passes `context.memory.knowledge`
through untouched -- so there is never anything of it to save that the caller did not
already have. `load_memory` takes the knowledge store as an argument for that reason: the
file describes what a run LEARNED, and a prior the run was GIVEN belongs to whoever gave
it.

The format is JSON with a refused-on-mismatch `format_version`, the convention
`task/memory_build.py` established and for the reason it gives: a store read under the
wrong schema answers every query confidently and there is no symptom to notice. It is not
compact -- a thousand rows of a 1024-wide `h^av` is tens of megabytes -- and that is the
trade this repo has taken before, because an artefact a human can open is worth more here
than an artefact that is small.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from earshot.memory.longterm import (
    ExperienceEntry,
    ExperienceStore,
    LongTermMemory,
    Outcome,
    abstract,
)
from earshot.memory.store import SemanticStore

__all__ = [
    "DREAM_MEMORY_FORMAT_VERSION",
    "DreamMemoryError",
    "dump_memory",
    "load_memory",
    "memory_summary",
    "read_provenance",
]

# Bumped when the on-disk shape changes in a way an old reader would misread.
DREAM_MEMORY_FORMAT_VERSION = 1


class DreamMemoryError(ValueError):
    """A memory file that cannot be read, with the path and the reason named."""


def _entry_as_dict(entry: ExperienceEntry) -> Dict[str, Any]:
    return {
        "context": [float(value) for value in np.asarray(entry.context).reshape(-1)],
        "trajectory": [
            float(value) for value in np.asarray(entry.trajectory).reshape(-1)
        ],
        "target_concept": str(entry.target_concept),
        "sound_concept": str(entry.sound_concept),
        # `None` survives the round trip as `null`. It is the ORDINARY value -- the runs
        # use `NullRoomLabeler`, which always abstains -- and writing "" or "unknown"
        # here would make an abstention look like a room and change `G`'s grouping.
        "room_concept": (
            None if entry.room_concept is None else str(entry.room_concept)
        ),
        "reached": bool(entry.outcome.reached),
        "final_gap_m": float(entry.outcome.final_gap_m),
    }


def _entry_from_dict(row: Any, *, path: str, index: int) -> ExperienceEntry:
    if not isinstance(row, dict):
        raise DreamMemoryError(
            "{}: experience row {} is {}, not an object".format(
                path, index, type(row).__name__
            )
        )
    missing = [
        key
        for key in (
            "context", "trajectory", "target_concept", "sound_concept",
            "room_concept", "reached", "final_gap_m",
        )
        if key not in row
    ]
    if missing:
        raise DreamMemoryError(
            "{}: experience row {} is missing {}. A row short of a field is a truncated "
            "write, not a row to be defaulted -- a zeroed h^av would be a retrieval key "
            "pointing nowhere.".format(path, index, ", ".join(missing))
        )
    return ExperienceEntry(
        context=np.asarray(row["context"], dtype=np.float32),
        trajectory=np.asarray(row["trajectory"], dtype=np.float32),
        target_concept=str(row["target_concept"]),
        outcome=Outcome(
            reached=bool(row["reached"]), final_gap_m=float(row["final_gap_m"])
        ),
        sound_concept=str(row["sound_concept"]),
        room_concept=(
            None if row["room_concept"] is None else str(row["room_concept"])
        ),
    )


def dump_memory(
    path: str, memory: LongTermMemory, *, scenes: Sequence[str] = ()
) -> pathlib.Path:
    """Write `M^E` to `path`, overwriting. Returns the path written.

    OVERWRITING IS DELIBERATE HERE and is the one place in this tree that does it. The
    report artefacts are append-only because each is a record of one episode that
    happened; this file is a running state that is REPLACED at each scene, and a
    never-overwrite rule would make the second scene of a sweep fail.

    `scenes` is provenance only -- which scenes have contributed so far, in order. It is
    never read back into the memory, because a scene list that disagreed with the rows
    would be a second source of truth about the same thing. It is there so a human
    opening the file can see how far the chain got.
    """
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": DREAM_MEMORY_FORMAT_VERSION,
        "scenes": [str(scene) for scene in scenes],
        "experience": [_entry_as_dict(entry) for entry in memory.experience.entries],
    }
    # Written whole and then renamed, so a run killed mid-write leaves the PREVIOUS
    # scene's memory intact rather than a half-file the next scene would refuse.
    staging = target.with_name(target.name + ".partial")
    staging.write_text(json.dumps(payload), encoding="utf-8")
    staging.replace(target)
    return target


def load_memory(
    path: str, *, min_support: int, knowledge: Optional[SemanticStore] = None
) -> LongTermMemory:
    """Read `M^E` from `path` and rebuild `M^P` over it. Never returns a partial memory.

    `min_support` has no default for the reason `abstract` gives: the number of times
    something must recur before it counts is the whole content of the word, so the caller
    states it and the audit records it. Passing a different `min_support` than the run
    that wrote the file is legal and is how a retention knob gets swept over a fixed
    memory.

    `knowledge` defaults to an empty `SemanticStore`, which is what every DREAM run has
    had so far. Nothing in the DREAM path writes that level, so it is an input and never
    an output.
    """
    target = pathlib.Path(path)
    if not target.is_file():
        raise DreamMemoryError(
            "{} does not exist. A missing memory file is NOT an empty memory: a run that "
            "silently started from nothing would be indistinguishable from the "
            "per-scene reset this file exists to remove.".format(path)
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except ValueError as error:
        raise DreamMemoryError("{} is not readable JSON: {}".format(path, error))
    version = payload.get("format_version")
    if version != DREAM_MEMORY_FORMAT_VERSION:
        raise DreamMemoryError(
            "{} was written at memory format version {!r}; this reader is version {}. "
            "Rebuild the memory rather than reading it under the wrong schema.".format(
                path, version, DREAM_MEMORY_FORMAT_VERSION
            )
        )
    rows = payload.get("experience")
    if not isinstance(rows, list):
        raise DreamMemoryError(
            "{}: 'experience' is {}, not a list".format(path, type(rows).__name__)
        )
    entries: List[ExperienceEntry] = [
        _entry_from_dict(row, path=path, index=index)
        for index, row in enumerate(rows)
    ]
    experience = ExperienceStore(entries=tuple(entries))
    return LongTermMemory(
        experience=experience,
        pattern=abstract(experience, min_support=int(min_support)),
        knowledge=SemanticStore() if knowledge is None else knowledge,
    )


def read_provenance(path: str) -> tuple:
    """The scene list a memory file carries, or `()` if there is no file yet.

    Separate from `load_memory` on purpose: the memory and the provenance are two facts
    and only one of them is state the agent retrieves against. `run()` reads this to
    append its own scene before writing, so the chain records its own order -- which is
    the thing a reader needs to know a carried-over sweep is reproducible at all.

    `()` on a missing file rather than a raise, because the FIRST scene of a chain has no
    provenance and that is not an error. `load_memory` raises on the same missing file,
    and the asymmetry is deliberate: a missing memory is a broken chain, a missing
    provenance is the start of one.
    """
    target = pathlib.Path(path)
    if not target.is_file():
        return ()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except ValueError:
        return ()
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        return ()
    return tuple(str(scene) for scene in scenes)


def memory_summary(memory: LongTermMemory) -> str:
    """One line for a run log: how much memory this scene inherited.

    Printed by the runner because a chain that silently broke -- a driver that stopped
    passing the path, a file the next scene could not read -- looks exactly like a chain
    that worked, until the sweep is over and section D says every scene started empty.
    """
    return "M^E {} row(s), M^P {} pattern(s), M^K {} entr(ies)".format(
        len(memory.experience), len(memory.pattern), len(memory.knowledge.entries)
    )
