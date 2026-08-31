"""The anomaly signal: stage it, resolve it, normalise it, measure it, render it.

Carried from ``audio_task.py`` (``resolve_anomaly_clip``, ``normalize_clip``) and
``audio.py`` (``rms``), plus the ESC-50 fetch that used to be
``embodied_memory/scripts/fetch_anomaly_clips.py``. Task spec §7 lists all three as
carrying "re-homed into a new audio module".

**``render_through_ir`` is new here, and it is where the grid died.** The old tree
looked a binaural IR up in a precomputed grid and convolved with ``scipy.signal.
fftconvolve``; ADR-0009 renders the IR live in the simulator every step, so the lookup
is gone and the convolution is not. §7 calls the per-step orchestration "rewritten
rather than ported", and ADR-0013's tree named no module for it — it lands here and in
``bed.heard_signal``, because what it operates on is the clip.

Two things follow from ADR-0014's Mac surface and are load-bearing rather than
stylistic:

- **numpy only at import time.** ``earshot/tools/mac-requirements.txt`` is numpy and
  ruff; scipy is in the box's ``ss2`` env and nowhere else. So the convolution is
  numpy's own FFT rather than ``scipy.signal.fftconvolve``, and the two functions that
  genuinely need scipy (wav read, resample) import it inside the call. A module-level
  ``import scipy`` here would make every Mac test in this layer uncollectable.
- **No silent fallback to synthetic audio.** The old ``build_anomaly_clip`` fell back to
  a seeded broadband burst whenever the path was missing, which is how a run could
  calibrate CLAP on a real recording and then classify a burst. ``load_anomaly_clip``
  raises; ``synthetic_burst`` still exists and has to be asked for by name.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ANOMALY_CLASSES",
    "CLASS_TO_ESC50",
    "BENIGN_TO_ESC50",
    "ESC50_BASE",
    "rms",
    "normalize_clip",
    "as_binaural",
    "render_through_ir",
    "resolve_anomaly_clip",
    "load_anomaly_clip",
    "synthetic_burst",
    "select_esc50_clip",
    "select_esc50_clips",
    "corpus_clip_paths",
    "fetch_esc50_clips",
    "fetch_esc50_corpus",
]

# The three FSD50K/ESC-50-backed emergency classes, locked. Carried verbatim; the
# room-conditioned *ambiguous* classes live with the prior that gives them meaning, in
# `normality.py`.
ANOMALY_CLASSES: Tuple[str, ...] = ("baby_cry", "alarm", "glass_break")

# ESC-50 (Piczak 2015, CC BY-NC). Real recordings of the exact classes.
ESC50_BASE = "https://github.com/karoldvl/ESC-50/raw/master"
CLASS_TO_ESC50: Dict[str, str] = {
    "baby_cry": "crying_baby",
    "alarm": "clock_alarm",
    "glass_break": "glass_breaking",
}
# The "routine, do not respond" negatives the open-set gate must REJECT. Staged into
# their own directory so they can never be resolved as an anomaly clip by accident.
BENIGN_TO_ESC50: Dict[str, str] = {
    "footsteps": "footsteps",
    "coughing": "coughing",
    "knock": "door_wood_knock",
    "vacuum": "vacuum_cleaner",
}


# ----------------------------------------------------------------------
# signal primitives
# ----------------------------------------------------------------------


def rms(signal: Any) -> float:
    """Root-mean-square over every sample and channel. Carried verbatim.

    The one measurement §3.1 and §3.2 are written in terms of, so it has exactly one
    definition in the tree: the onset threshold, the bed level, the calibration
    distribution and the per-step record are all this number.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def normalize_clip(clip: Any, target_db: float = -20.0) -> np.ndarray:
    """Mono float32, RMS-normalised to ``target_db`` dBFS. Carried verbatim.

    Done once per run, so the anomaly's level is a property of the run rather than of
    which ESC-50 recording happened to be staged. Silence stays silence.
    """
    x = np.asarray(clip, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0)
    x = x.reshape(-1).astype(np.float32)
    current = rms(x)
    if current <= 1e-8:
        return x
    target = 10.0 ** (float(target_db) / 20.0)
    return (x * (target / current)).astype(np.float32)


def as_binaural(ir: Any) -> np.ndarray:
    """The sensor observation as a ``(2, L)`` float32 array, or a loud failure.

    **The audio observation is not a numpy array** — ticket 16 measured
    ``getattr(ir, "shape")`` reading ``None`` over a perfectly good ``[2, 72300]`` IR,
    which is why ``guard.py`` walks the nesting rather than asking for a shape. It
    converts cleanly; what it does not do is arrive as one.

    Shape is asserted rather than reshaped. A mono or ambisonic observation reaching
    here means ``spec.py``'s channel layout did not take, and silently averaging it into
    two ears would produce a lateral sign of exactly zero at every pose — a controller
    that never turns, looking like a mediocre climb rather than a broken config.
    """
    array = np.asarray(ir, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(
            "audio observation is {}, not (2, L) binaural — check that "
            "audio.spec.audio_sensor_spec set channelLayout to Binaural/2 on the spec "
            "this sensor was built from".format(array.shape)
        )
    return array


def render_through_ir(ir: Any, clip: Any) -> np.ndarray:
    """Convolve a mono clip through a binaural IR: ``(2, N)``, N the clip's length.

    This is what the agent hears from the anomaly at this pose — the received signal,
    not the impulse response. §3.1 and §3.2 are written about a *signal* whose RMS is
    comparable with the bed's, and an IR is not in that domain.

    Trimmed to the clip's length rather than the full ``L + N - 1``, which is
    ``render_at_pose``'s ``max_len`` convention carried across: a fixed output length is
    what makes the per-step RMS comparable between steps, and the IR's own width is
    scene- and pose-dependent (ticket 06 measured ``[2, 72300]`` against a 4.0 s
    ``maxIRLength`` cap, so **any fixed-width assumption about the IR is wrong**).

    numpy's FFT, not ``scipy.signal.fftconvolve``: scipy is not on the Mac side, and
    this function has to be Mac-testable because it is the domain every downstream
    assertion is written in. The spectrum of the clip is recomputed each call — cacheing
    it would need a fixed transform size, and the IR's width changes with the pose.
    """
    impulse = as_binaural(ir)
    signal = np.asarray(clip, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("anomaly clip is empty — nothing to render through the IR")
    length = impulse.shape[1] + signal.size - 1
    n_fft = 1 << int(max(1, length - 1)).bit_length()
    spectrum = np.fft.rfft(signal, n_fft)
    rendered = np.fft.irfft(np.fft.rfft(impulse, n_fft, axis=1) * spectrum, n_fft, axis=1)
    return np.ascontiguousarray(rendered[:, : signal.size], dtype=np.float32)


# ----------------------------------------------------------------------
# staging and loading
# ----------------------------------------------------------------------


def resolve_anomaly_clip(
    anomaly_class: Optional[str],
    explicit_path: Optional[str] = None,
    clip_dir: str = "data/anomaly_audio",
) -> Optional[str]:
    """Which ``.wav`` to render: explicit path wins, else ``<clip_dir>/<class>.wav``.

    Carried verbatim. Pure path logic apart from one ``isfile``, so it is safe to call
    before anything audio is wired. ``None`` means nothing is staged — the caller
    decides what that means, and ``load_anomaly_clip`` refuses to guess.
    """
    if explicit_path:
        return explicit_path
    if anomaly_class:
        candidate = os.path.join(clip_dir, "{}.wav".format(anomaly_class))
        if os.path.isfile(candidate):
            return candidate
    return None


def load_anomaly_clip(
    path: str, sample_rate: int, target_norm_rms_db: float = -20.0
) -> np.ndarray:
    """A mono, RMS-normalised clip at ``sample_rate``. Raises if the file is not there.

    **The silent synthetic fallback does not carry.** ``build_anomaly_clip`` returned a
    seeded broadband burst whenever the path was missing or unreadable, so a run whose
    ESC-50 staging had failed produced a plausible episode in which CLAP — calibrated on
    real recordings — classified a noise burst. That is this map's recurring failure
    class, and the whole of the fix is that this function raises. ``synthetic_burst``
    still exists for tests and diagnostics and has to be named.

    scipy is imported inside the call: it is in the box's env and not on the Mac side,
    and everything above this line has to stay importable there.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "no anomaly clip at {!r}. Stage the ESC-50 recordings with "
            "`python -m earshot.audio.clips` on a host with internet, or pass an "
            "explicit path. There is deliberately no synthetic fallback: CLAP is "
            "calibrated on real audio, so a substituted burst would classify against a "
            "distribution it was never measured on.".format(path)
        )
    from scipy.io import wavfile

    file_rate, data = wavfile.read(path)
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / 32768.0
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.reshape(-1).astype(np.float32)
    if int(file_rate) != int(sample_rate):
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(int(file_rate), int(sample_rate))
        data = resample_poly(
            data, int(sample_rate) // divisor, int(file_rate) // divisor
        ).astype(np.float32)
    return normalize_clip(data, target_norm_rms_db)


def synthetic_burst(
    sample_rate: int, seconds: float = 0.5, target_norm_rms_db: float = -20.0
) -> np.ndarray:
    """A deterministic broadband burst (seed 0). Never a stand-in for a real clip.

    Kept because a signal with known statistics is what the Mac tests and the
    calibration arithmetic want, and because it is the honest way to exercise the
    render path with no dataset on disk. It is not what an episode plays.
    """
    rng = np.random.default_rng(0)
    n = int(float(sample_rate) * float(seconds))
    envelope = np.minimum(1.0, np.linspace(0.0, 4.0, n))
    burst = (rng.standard_normal(n).astype(np.float32) * envelope).astype(np.float32)
    return normalize_clip(burst, target_norm_rms_db)


# ----------------------------------------------------------------------
# the ESC-50 fetch
# ----------------------------------------------------------------------


def select_esc50_clip(
    rows: Sequence[Dict[str, Any]], esc_category: str, index: int = 0
) -> Optional[str]:
    """The ``index``-th filename for a category, sorted for determinism, index wrapping.

    Pure, and separated from the download for exactly that reason: which recording a
    class resolves to is a property of the run that has to be reproducible, and it is
    the only part of the fetch a Mac can test.
    """
    files = sorted(str(r.get("filename")) for r in rows if r.get("category") == esc_category)
    if not files:
        return None
    return files[index % len(files)]


def select_esc50_clips(
    rows: Sequence[Dict[str, Any]], esc_category: str, n: int, start: int = 0
) -> List[str]:
    """The first ``n`` distinct filenames for a category from ``start``, sorted, wrapping.

    The plural of ``select_esc50_clip`` and pure for the same reason. It exists because the
    separation gate and the recording-level robustness axis both need SEVERAL recordings of
    one class: ESC-50 ships 40 per category, and a gate measured on one recording per class
    cannot tell a class CLAP understands from a recording CLAP happens to like.

    Distinct rather than merely ``n`` draws: asking for more than the category holds returns
    everything it holds, once each, instead of silently repeating a file and inflating ``n``
    with duplicates that would read as independent samples.
    """
    files = sorted(str(r.get("filename")) for r in rows if r.get("category") == esc_category)
    if not files:
        return []
    count = min(int(n), len(files))
    return [files[(int(start) + offset) % len(files)] for offset in range(count)]


def corpus_clip_paths(class_name: str, corpus_dir: str) -> List[str]:
    """Every staged recording for a class, as ``<corpus_dir>/<class>/<index>.wav``, sorted.

    Returns an empty list when nothing is staged. The caller decides what that means, on the
    same rule as ``resolve_anomaly_clip``: this module does not guess, and
    ``load_anomaly_clip`` refuses to substitute.
    """
    directory = os.path.join(corpus_dir, str(class_name))
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".wav")
    )


def fetch_esc50_corpus(
    out_dir: str,
    class_map: Dict[str, str],
    classes: Optional[Sequence[str]] = None,
    n_per_class: int = 8,
    start: int = 0,
    timeout: int = 120,
) -> Dict[str, List[str]]:
    """Stage ``n_per_class`` recordings per class into ``<out_dir>/<class>/<index>.wav``.

    The multi-recording sibling of ``fetch_esc50_clips``, laid out one directory per class so
    a class's recordings can be split heard-from-unheard later by index without re-staging.

    Raises on an unknown class or a category ESC-50 has no rows for, and raises when a class
    yields fewer recordings than asked, rather than staging a short class and letting the
    per-class ``n`` differ silently across the vocabulary -- an uneven ``n`` is exactly what
    makes a per-class recall table unreadable.
    """
    import csv
    import io
    import urllib.request

    wanted = list(class_map) if classes is None else list(classes)

    def fetch(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.read()

    rows = list(
        csv.DictReader(io.StringIO(fetch("{}/meta/esc50.csv".format(ESC50_BASE)).decode()))
    )
    staged: Dict[str, List[str]] = {}
    for name in wanted:
        category = class_map.get(name)
        if category is None:
            raise KeyError("unknown class {!r}; known: {}".format(name, sorted(class_map)))
        filenames = select_esc50_clips(rows, category, n_per_class, start)
        if not filenames:
            raise LookupError(
                "ESC-50 has no rows for category {!r} (class {!r})".format(category, name)
            )
        if len(filenames) < int(n_per_class):
            raise LookupError(
                "ESC-50 category {!r} (class {!r}) holds only {} recordings, {} were asked "
                "for; a class staged short would enter the per-class table with a different "
                "n from its neighbours".format(
                    category, name, len(filenames), n_per_class
                )
            )
        directory = os.path.join(out_dir, name)
        os.makedirs(directory, exist_ok=True)
        written: List[str] = []
        for index, filename in enumerate(filenames):
            destination = os.path.join(directory, "{:02d}.wav".format(index))
            with open(destination, "wb") as handle:
                handle.write(fetch("{}/audio/{}".format(ESC50_BASE, filename)))
            written.append(destination)
        staged[name] = written
        print("  {} <- ESC-50 {} x{} -> {}/".format(name, category, len(written), directory))
    return staged


def fetch_esc50_clips(
    out_dir: str,
    class_map: Optional[Dict[str, str]] = None,
    classes: Optional[Sequence[str]] = None,
    index: int = 0,
    timeout: int = 120,
) -> List[str]:
    """Download one ESC-50 recording per class into ``<out_dir>/<class>.wav``.

    Needs internet, so it is a staging step run once on the box and never on the live
    path. Returns the paths written. Raises on an unknown class or a category with no
    rows rather than warning and continuing: a partial staging is what
    ``load_anomaly_clip`` is now built to refuse.
    """
    import csv
    import io
    import urllib.request

    mapping = dict(CLASS_TO_ESC50 if class_map is None else class_map)
    wanted = list(mapping) if classes is None else list(classes)

    def fetch(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.read()

    rows = list(
        csv.DictReader(io.StringIO(fetch("{}/meta/esc50.csv".format(ESC50_BASE)).decode()))
    )
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for name in wanted:
        category = mapping.get(name)
        if category is None:
            raise KeyError(
                "unknown class {!r}; known: {}".format(name, sorted(mapping))
            )
        filename = select_esc50_clip(rows, category, index)
        if filename is None:
            raise LookupError(
                "ESC-50 has no rows for category {!r} (class {!r})".format(category, name)
            )
        destination = os.path.join(out_dir, "{}.wav".format(name))
        with open(destination, "wb") as handle:
            handle.write(fetch("{}/audio/{}".format(ESC50_BASE, filename)))
        written.append(destination)
        print("  {} <- ESC-50 {}/{} -> {}".format(name, category, filename, destination))
    return written


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage ESC-50 clips for the anomaly task.")
    parser.add_argument("--out-dir", default="data/anomaly_audio")
    parser.add_argument("--benign-out-dir", default="data/benign_audio")
    parser.add_argument("--index", type=int, default=0, help="which clip per class (0..39)")
    parser.add_argument("--include-benign", action="store_true")
    # The sounding class vocabulary is staged as a CORPUS -- several recordings per class in
    # its own directory -- because the separation gate has to be able to tell a class CLAP
    # understands from one recording CLAP happens to like. `--vocabulary` stages the
    # candidate set AND `ABSENT_CLASSES`, which are the gate's forced-failure arm and are
    # useless staged separately: a gate run needs both or it has one arm.
    parser.add_argument(
        "--vocabulary",
        action="store_true",
        help="stage the candidate sounding vocabulary + the absent classes as a corpus",
    )
    parser.add_argument("--corpus-dir", default="data/sound_corpus")
    parser.add_argument("--absent-dir", default="data/absent_corpus")
    parser.add_argument("--n-per-class", type=int, default=8)
    # HELD-OUT RECORDINGS. The prune picks its vocabulary using the staged clips, so any
    # accuracy re-measured on those same clips is selection on the outcome. ESC-50 ships 40
    # recordings per class and a run stages 8, so `--clip-start 8` gives a disjoint set and
    # the only unbiased number this design can produce without new audio.
    parser.add_argument("--clip-start", type=int, default=0)
    args = parser.parse_args(None if argv is None else list(argv))

    if args.vocabulary:
        from earshot.audio.vocabulary import ABSENT_CLASSES, CANDIDATE_VOCABULARY

        candidates = {entry.name: entry.esc50_category for entry in CANDIDATE_VOCABULARY}
        print(
            "candidate vocabulary ({} classes, recordings {}..{}) -> {}".format(
                len(candidates),
                args.clip_start,
                args.clip_start + args.n_per_class - 1,
                args.corpus_dir,
            )
        )
        fetch_esc50_corpus(
            args.corpus_dir, candidates, None, args.n_per_class, start=args.clip_start
        )
        absent = {name: name for name in ABSENT_CLASSES}
        print("absent classes ({}) -> {}".format(len(absent), args.absent_dir))
        fetch_esc50_corpus(
            args.absent_dir, absent, None, args.n_per_class, start=args.clip_start
        )
        return 0

    fetch_esc50_clips(args.out_dir, CLASS_TO_ESC50, list(CLASS_TO_ESC50), args.index)
    if args.include_benign:
        fetch_esc50_clips(
            args.benign_out_dir, BENIGN_TO_ESC50, list(BENIGN_TO_ESC50), args.index
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
