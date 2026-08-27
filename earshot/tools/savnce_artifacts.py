"""Checksums for the SAVN-CE artefacts we did not build (ADR-0015).

The episode dataset and the released checkpoints arrive from a Google Drive **folder**,
which can change under us silently and leaves no version in the file. This repo pins
everything else it depends on, so it pins these too.

**What is hashed and what is not.** Checkpoints and episode `json.gz` files are hashed:
they are small, they decide the number, and a wrong one is invisible. MP3D scene meshes
are recorded by size and count only. Hashing tens of gigabytes of `.glb` before every
run would cost more than it protects, and a corrupt mesh fails loudly at load rather
than quietly at eval.

`sha256: null` means "not recorded yet". `record` fills it, `verify` checks it. A null
entry is not a pass and not a failure: it is an unrecorded artefact, and `verify` says
so rather than pretending either way.
"""

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence

MANIFEST = pathlib.Path(__file__).resolve().parent / "savnce-artifacts.json"
READ_CHUNK = 1024 * 1024

# Relative to the data root. Globs, because the checkpoint and split file names are
# theirs to choose and we should notice a new one rather than ignore it.
HASHED_GLOBS = (
    "pretrained_ckpts/**/*.pth",
    "datasets/savnce_dataset/mp3d/*/*/*.json.gz",
    "datasets/savnce_dataset/mp3d/*/*/content/*.json.gz",
)
SIZED_DIRS = ("scene_datasets/mp3d", "sounds")


# Severities, because "this file changed under us" and "this file is new and unrecorded"
# are not the same event and must not share an exit code. Blocking every first run on the
# second one would guarantee the manifest gets bypassed, which is worse than not having
# it. Both are always PRINTED — an unrecorded artefact that nobody ever sees is the
# silent no-op this repo keeps being bitten by.
HARD = "HARD"
SOFT = "SOFT"


class Mismatch(NamedTuple):
    path: str
    expected: str
    actual: str
    severity: str = HARD

    @property
    def is_hard(self) -> bool:
        return self.severity == HARD


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan(data_root: pathlib.Path) -> Dict[str, object]:
    """Everything the manifest tracks, measured from disk. Pure apart from the reads."""
    hashed: Dict[str, str] = {}
    for pattern in HASHED_GLOBS:
        for path in sorted(data_root.glob(pattern)):
            if path.is_file():
                hashed[str(path.relative_to(data_root))] = sha256_file(path)
    sized: Dict[str, Dict[str, int]] = {}
    for relative in SIZED_DIRS:
        directory = data_root / relative
        if not directory.is_dir():
            continue
        files = [item for item in directory.rglob("*") if item.is_file()]
        sized[relative] = {"files": len(files), "bytes": sum(item.stat().st_size for item in files)}
    return {"sha256": hashed, "sizes": sized}


def compare(manifest: Dict[str, object], observed: Dict[str, object]) -> List[Mismatch]:
    """Recorded versus present. Unrecorded artefacts are reported, never assumed good."""
    recorded = manifest.get("sha256") or {}
    present = observed.get("sha256") or {}
    mismatches: List[Mismatch] = []
    if not isinstance(recorded, dict) or not isinstance(present, dict):
        return [Mismatch("<manifest>", "an object", "something else", HARD)]
    for path, expected in sorted(recorded.items()):
        actual = present.get(path)
        if expected is None:
            mismatches.append(Mismatch(path, "not yet recorded", "present" if actual else "absent", SOFT))
        elif actual is None:
            mismatches.append(Mismatch(path, expected, "absent", HARD))
        elif actual != expected:
            mismatches.append(Mismatch(path, expected, actual, HARD))
    for path in sorted(set(present) - set(recorded)):
        mismatches.append(Mismatch(path, "not in the manifest", present[path], SOFT))
    return mismatches


def _load_manifest() -> Dict[str, object]:
    if not MANIFEST.is_file():
        return {"sha256": {}, "sizes": {}}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("record", "verify"))
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)

    data_root = pathlib.Path(args.data_root)
    if not data_root.is_dir():
        sys.stderr.write("FATAL: {} is not a directory\n".format(data_root))
        return 2
    observed = scan(data_root)

    if args.mode == "record":
        MANIFEST.write_text(json.dumps(observed, indent=2, sort_keys=True), encoding="utf-8")
        hashed = observed["sha256"]
        sys.stdout.write("recorded {} checksums to {}\n".format(len(hashed), MANIFEST.name))
        for relative, size in sorted((observed["sizes"] or {}).items()):
            sys.stdout.write("  {}: {} files, {:.1f} GB\n".format(relative, size["files"], size["bytes"] / 1e9))
        return 0

    mismatches = compare(_load_manifest(), observed)
    if not mismatches:
        sys.stdout.write("artefacts verify clean ({} checksums)\n".format(len(observed["sha256"])))
        return 0
    hard = [item for item in mismatches if item.is_hard]
    sys.stdout.write(
        "artefact manifest: {} discrepancy(ies), {} of them blocking\n".format(len(mismatches), len(hard))
    )
    for item in mismatches:
        sys.stdout.write(
            "  [{}] {}\n    expected {}\n    actual   {}\n".format(item.severity, item.path, item.expected, item.actual)
        )
    if not hard:
        sys.stdout.write(
            "  nothing blocking. Record them: python -m earshot.tools.savnce_artifacts record --data-root ...\n"
        )
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
