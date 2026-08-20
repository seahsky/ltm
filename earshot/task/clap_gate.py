"""The CLAP separation gate on live HM3D reverb: render, classify, and measure both arms.

`CONTEXT.md` defines the **CLAP separation gate** as the measurement that must clear before
anything reads the **inferred goal class**, and ADR-0017 makes the inferred class the whole
reason the pivot to sound-source finding is worth running. This module is the half of the
gate that needs a simulator. The arithmetic is `audio/separation.py`, which is Mac-testable,
so the numbers printed here are checkable without a box.

**Why this run exists at all.** `audio/clap.py` ships `ANOMALY_GATE_DELTA` and
`ANOMALY_GATE_TAU` with a caveat written into the source: they were calibrated on a grid
render convolved OFFLINE, and "the domain should match" is an inference. Worse, the one arc
that exercised the gate live had it reject 0 of 8 -- the signature of a gate that
discriminates nothing. So nothing in the new task may depend on the inferred class until a
run measures the separation on the renderer that will actually produce it.

**One render serves the whole vocabulary.** The IR is a property of the pose and the scene,
not of the clip, so a pose is rendered ONCE and every candidate recording is convolved
through that one IR offline. That is what makes a gate over 20 classes at several recordings
each affordable: the simulator cost scales with poses, not with the vocabulary.

**The signal is the one the agent would classify.** Bed included, ears averaged, via the same
`mix_bed` and `heard_clip_for_clap` the live path uses. A gate measured on a dry clip would
answer a question the agent never asks -- which is the exact mistake the offline calibration
made.

**Both arms, or it is red.** In-vocabulary recordings are the healthy arm.
`vocabulary.ABSENT_CLASSES` are the induced failure: classes never placed in the prompt bank,
so an open-set rule that accepts them is accepting anything. `separation.summarise` raises
when either arm is missing rather than reporting the half that ran.

**Rows are written as they are measured.** A crash three scenes in leaves three scenes of
usable data and a report that says so, rather than nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from earshot.audio.bed import bed_signal, mix_bed
from earshot.audio.clap import NORMAL_PROMPTS, heard_clip_for_clap
from earshot.audio.clips import as_binaural, corpus_clip_paths, load_anomaly_clip
from earshot.audio.config import AudioConfig
from earshot.audio.separation import GateRow, summarise
from earshot.audio.vocabulary import ABSENT_CLASSES, CANDIDATE_VOCABULARY, prompts

__all__ = [
    "GateConfig",
    "render_batch_through_ir",
    "score_rows_at_pose",
    "run_gate",
]


@dataclass(frozen=True)
class GateConfig:
    """One gate run. Frozen for the reason `RunConfig` is: it lands in the artefact verbatim.

    The three counts multiply, so their product is the cost: `n_sources * n_poses` renders per
    scene and `n_sources * n_poses * (n_classes + n_absent) * n_recordings` convolutions plus
    CLAP encodes. Defaults are sized to finish a 20-scene sweep inside an hour on the V100 and
    every one of them is a flag, because the honest response to a thin result is more samples
    rather than a softer bar.
    """

    run_dir: str
    data_root: str = "."
    split: str = "val"
    scenes: Tuple[str, ...] = ()
    corpus_dir: str = "data/sound_corpus"
    absent_dir: str = "data/absent_corpus"
    n_sources: int = 2
    n_poses: int = 6
    n_recordings: int = 4
    n_bands: int = 4
    seed: int = 20260820
    audio: AudioConfig = field(default_factory=AudioConfig)
    overwrite: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "data_root": self.data_root,
            "split": self.split,
            "scenes": list(self.scenes),
            "corpus_dir": self.corpus_dir,
            "absent_dir": self.absent_dir,
            "n_sources": self.n_sources,
            "n_poses": self.n_poses,
            "n_recordings": self.n_recordings,
            "n_bands": self.n_bands,
            "seed": self.seed,
            "audio": dataclasses.asdict(self.audio),
        }


def render_batch_through_ir(ir: Any, clips: Sequence[Any]) -> List[np.ndarray]:
    """Convolve many clips through ONE binaural IR, reusing the IR's spectrum.

    Identical output to calling `clips.render_through_ir(ir, clip)` per clip -- that identity
    is asserted by `tests/mac/test_clap_gate_render.py` and is the only reason this
    duplicate exists. Without it the gate recomputes the IR's FFT once per candidate
    recording, which at 20 classes is 80 redundant transforms per pose and turns a
    45-minute sweep into a four-hour one.

    Clips are grouped by length so `n_fft` is shared only among clips that genuinely share
    it. ESC-50 recordings are all 5.0 s so in practice there is one group, but assuming that
    would be a fixed-width assumption of exactly the kind ticket 06 measured to be wrong
    about the IR.
    """
    impulse = as_binaural(ir)
    signals = [np.asarray(clip, dtype=np.float32).reshape(-1) for clip in clips]
    for index, signal in enumerate(signals):
        if signal.size == 0:
            raise ValueError("clip {} is empty — nothing to render through the IR".format(index))

    rendered: List[Optional[np.ndarray]] = [None] * len(signals)
    by_length: Dict[int, List[int]] = {}
    for index, signal in enumerate(signals):
        by_length.setdefault(int(signal.size), []).append(index)

    for size, indices in by_length.items():
        length = impulse.shape[1] + size - 1
        n_fft = 1 << int(max(1, length - 1)).bit_length()
        impulse_spectrum = np.fft.rfft(impulse, n_fft, axis=1)
        for index in indices:
            spectrum = np.fft.rfft(signals[index], n_fft)
            out = np.fft.irfft(impulse_spectrum * spectrum, n_fft, axis=1)
            rendered[index] = np.ascontiguousarray(out[:, :size], dtype=np.float32)

    for index, value in enumerate(rendered):
        if value is None:
            raise AssertionError("clip {} was never rendered".format(index))
    return [value for value in rendered if value is not None]


def score_rows_at_pose(
    ir: Any,
    distance_m: float,
    scene: str,
    *,
    known: Sequence[Tuple[str, int, np.ndarray]],
    absent: Sequence[Tuple[str, int, np.ndarray]],
    bed: Any,
    sample_rate: int,
    encode_audio: Callable[[np.ndarray, int], np.ndarray],
    prompt_embeddings: Dict[str, np.ndarray],
    normal_embeddings: Sequence[np.ndarray],
) -> List[GateRow]:
    """Every row this pose produces: each staged recording rendered, bedded, and CLAP-scored.

    `known` and `absent` are `(class_name, recording_index, clip)` triples. Text embeddings are
    passed in already computed: there are 25 prompts and tens of thousands of audio encodes, so
    re-encoding text per pose would be most of the GPU time for none of the information.

    The audio embedding is unit-normalised here rather than inside the dot product, matching
    `clap._unit`, so a cosine printed by the gate is the same number the live gate would read.
    """
    entries = [(name, index, clip, True) for name, index, clip in known]
    entries += [(name, index, clip, False) for name, index, clip in absent]
    if not entries:
        raise ValueError("no staged recordings to score at this pose")

    rendered = render_batch_through_ir(ir, [clip for _n, _i, clip, _k in entries])
    rows: List[GateRow] = []
    for (name, recording_index, _clip, in_vocabulary), signal in zip(entries, rendered):
        heard = mix_bed(signal, bed)
        mono, rate = heard_clip_for_clap(heard, sample_rate)
        audio = np.asarray(encode_audio(mono, rate), dtype=np.float32).reshape(-1)
        audio = audio / (float(np.linalg.norm(audio)) + 1e-8)
        scores = {
            prompt_name: float(np.dot(audio, embedding))
            for prompt_name, embedding in prompt_embeddings.items()
        }
        normal_cosine = max(float(np.dot(audio, item)) for item in normal_embeddings)
        rows.append(
            GateRow(
                true_class=name,
                in_vocabulary=in_vocabulary,
                distance_m=float(distance_m),
                scene=str(scene),
                recording_index=int(recording_index),
                scores=scores,
                normal_cosine=normal_cosine,
            )
        )
    return rows


def _load_corpus(
    directory: str, names: Sequence[str], n_recordings: int, cfg: GateConfig
) -> List[Tuple[str, int, np.ndarray]]:
    """`(class, recording_index, clip)` for every staged recording, or a raise naming the gap.

    Raises on a class with nothing staged rather than skipping it. A vocabulary silently short
    of a class produces a confusion matrix missing a row, and a reader has no way to tell that
    from a class CLAP never predicted.
    """
    loaded: List[Tuple[str, int, np.ndarray]] = []
    for name in names:
        paths = corpus_clip_paths(name, directory)
        if not paths:
            raise FileNotFoundError(
                "nothing staged for {!r} under {} — run "
                "`python -m earshot.audio.clips --vocabulary` first. A class staged short "
                "would leave a hole in the confusion matrix that reads like a class CLAP "
                "never predicted.".format(name, directory)
            )
        if len(paths) < int(n_recordings):
            raise FileNotFoundError(
                "{!r} has {} staged recording(s) under {} but {} were asked for; an uneven "
                "per-class n makes the recall table unreadable".format(
                    name, len(paths), directory, n_recordings
                )
            )
        for index, path in enumerate(paths[: int(n_recordings)]):
            loaded.append(
                (
                    name,
                    index,
                    load_anomaly_clip(
                        path, cfg.audio.sample_rate, cfg.audio.target_norm_rms_db
                    ),
                )
            )
    return loaded


def run_gate(cfg: GateConfig, progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Render, classify and summarise. Writes `rows.jsonl` and `separation.json`, returns the report.

    Habitat and the model stack are imported inside this function for `runner.run`'s reason:
    `sim/world.py` imports habitat_sim, so a module-level import would make every Mac test in
    this file's suite uncollectable.
    """
    say = progress if progress is not None else print

    from earshot.env_check import assert_env

    say("env_check: probing (clap=True)")
    env = assert_env(clap=True)
    say(env.summary())

    from earshot.task.episodes import (
        available_scenes,
        find_scenes_dir,
        find_split_dir,
        load_scene,
    )

    split_dir = find_split_dir(cfg.split, root=cfg.data_root)
    scenes_dir = find_scenes_dir(root=cfg.data_root)

    wanted = list(cfg.scenes)
    if not wanted:
        wanted = []
        for label in available_scenes(split_dir):
            try:
                dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
            except Exception:
                continue
            if os.path.exists(dataset.scene_path):
                wanted.append(label)
    if not wanted:
        raise RuntimeError("no scene in split {!r} has a mesh on this machine".format(cfg.split))
    say("{} scene(s): {}".format(len(wanted), " ".join(wanted)))

    run_dir = pathlib.Path(cfg.run_dir)
    rows_path = run_dir / "rows.jsonl"
    report_path = run_dir / "separation.json"
    if rows_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            "{} already holds a gate run. One directory is one run: mixing two invocations "
            "is how yield-1 reported a pool of two runs as one. Use a fresh --run-dir.".format(
                run_dir
            )
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    candidate_names = [entry.name for entry in CANDIDATE_VOCABULARY]
    known_clips = _load_corpus(cfg.corpus_dir, candidate_names, cfg.n_recordings, cfg)
    absent_clips = _load_corpus(cfg.absent_dir, list(ABSENT_CLASSES), cfg.n_recordings, cfg)
    say("corpus: {} in-vocabulary clip(s), {} absent clip(s)".format(
        len(known_clips), len(absent_clips)
    ))

    clip_lengths = {int(clip.size) for _n, _i, clip in known_clips + absent_clips}
    if len(clip_lengths) != 1:
        say("NOTE: staged clips have {} distinct lengths {} — the bed is built per length"
            .format(len(clip_lengths), sorted(clip_lengths)))
    beds = {
        size: bed_signal(size, cfg.audio.bed_rms) for size in sorted(clip_lengths)
    }

    from earshot.task.models import load_clap_encoder

    encoder = load_clap_encoder()
    say("CLAP: loaded")

    def unit(vector: Any) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        return values / (float(np.linalg.norm(values)) + 1e-8)

    prompt_bank = prompts()
    prompt_embeddings = {
        name: unit(encoder.encode_text(text)) for name, text in prompt_bank.items()
    }
    normal_embeddings = [unit(encoder.encode_text(text)) for text in NORMAL_PROMPTS]
    say("prompt bank: {} candidate prompts, {} normal prompts".format(
        len(prompt_embeddings), len(normal_embeddings)
    ))

    from earshot.audio.sensor import AudioSensorHandle
    from earshot.audio.spec import audio_sensor_spec
    from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs
    from earshot.task.runner import calibration_poses

    spec, binaural = audio_spec_parts()
    audio_sensor_spec(spec, cfg.audio, binaural)
    audio_uuid = str(spec.uuid)

    rows: List[GateRow] = []
    failed: List[str] = []
    with rows_path.open("w", encoding="utf-8") as sink:
        for label in wanted:
            say("[scene] {}".format(label))
            try:
                dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
                world = World(dataset.scene_path, list(camera_sensor_specs()) + [spec])
            except Exception as exc:  # noqa: BLE001 — one scene must not cost the rest
                say("  WARN: {} could not load ({}) — continuing".format(label, exc))
                failed.append(label)
                continue
            try:
                world.seed_navmesh(cfg.seed)
                handle: Optional[Any] = None
                for source_index in range(int(cfg.n_sources)):
                    source = world.random_navigable_point()
                    if handle is None:
                        handle = AudioSensorHandle(
                            world.sensor_handle(audio_uuid),
                            world.observe,
                            source,
                            uuid=audio_uuid,
                        )
                    else:
                        handle.set_source(source)
                    try:
                        poses = calibration_poses(
                            world, source, cfg.audio.audible_band_m, cfg.n_poses
                        )
                    except Exception as exc:  # noqa: BLE001
                        say("  source {}: no band poses ({}) — skipped".format(source_index, exc))
                        continue
                    for pose in poses:
                        distance = world.geodesic_distance(pose, [source])
                        if distance is None or not np.isfinite(float(distance)):
                            # An unroutable pose is dropped, never defaulted to zero: a
                            # distance of zero would enter the band curve as the near field.
                            continue
                        world.set_pose(pose)
                        observation, _guard = handle.observe()
                        ir = handle.audio_of(observation)
                        size = int(known_clips[0][2].size)
                        pose_rows = score_rows_at_pose(
                            ir,
                            float(distance),
                            label,
                            known=known_clips,
                            absent=absent_clips,
                            bed=beds[size],
                            sample_rate=cfg.audio.sample_rate,
                            encode_audio=encoder.encode_audio,
                            prompt_embeddings=prompt_embeddings,
                            normal_embeddings=normal_embeddings,
                        )
                        for row in pose_rows:
                            sink.write(
                                json.dumps(
                                    {
                                        "true_class": row.true_class,
                                        "in_vocabulary": row.in_vocabulary,
                                        "distance_m": row.distance_m,
                                        "scene": row.scene,
                                        "recording_index": row.recording_index,
                                        "scores": dict(row.scores),
                                        "normal_cosine": row.normal_cosine,
                                    }
                                )
                                + "\n"
                            )
                        sink.flush()
                        rows.extend(pose_rows)
                    say("  source {}: {} pose(s), {} row(s) so far".format(
                        source_index, len(poses), len(rows)
                    ))
            finally:
                close = getattr(world, "close", None)
                if callable(close):
                    close()

    if not rows:
        raise RuntimeError(
            "the gate produced no rows over {} scene(s); NOT_RUN is red, so this is a "
            "failure rather than an empty result".format(len(wanted))
        )

    affinities = {entry.name: entry.affinity for entry in CANDIDATE_VOCABULARY}
    # The anchor map goes in because ANCHOR accuracy is the number the task rests on: the
    # agent navigates to an object, so a class confused for a sibling of the same anchor
    # costs it nothing. Reporting only class top-1 understates the system, sometimes by a
    # lot -- every one of `snoring`'s misses in clapsmoke-3 landed on `breathing`.
    anchors = {entry.name: entry.anchor_object for entry in CANDIDATE_VOCABULARY}
    report = summarise(rows, affinities=affinities, anchors=anchors, n_bands=cfg.n_bands)
    payload: Dict[str, Any] = dict(report.as_dict())
    payload["config"] = cfg.as_dict()
    payload["scenes_run"] = [label for label in wanted if label not in failed]
    payload["scenes_failed"] = failed
    with report_path.open("w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)

    say(_format_report(report, failed))
    return payload


def _format_report(report: Any, failed: Sequence[str]) -> str:
    """The measurements, printed. Box tests print their numbers; so does this."""
    lines: List[str] = []
    lines.append("")
    lines.append("=== CLAP separation gate ===")
    lines.append(
        "closed-set top-1: {:.3f} over {} rows, {} classes (chance {:.3f})".format(
            report.top1_accuracy, report.n_rows, report.n_classes, report.chance_accuracy
        )
    )
    if report.anchor_top1_accuracy is not None:
        lines.append(
            "ANCHOR top-1:     {:.3f}  <- THE TASK NUMBER: the agent navigates to an object, "
            "so a sibling confusion at the same anchor costs it nothing".format(
                report.anchor_top1_accuracy
            )
        )
    lines.append("mean true-class margin: {:+.4f}".format(report.mean_true_margin))
    if report.per_anchor:
        lines.append("")
        lines.append("-- per anchor object --")
        for item in sorted(report.per_anchor, key=lambda entry: -entry.accuracy):
            confusion = (
                "{} x{}".format(item.top_confusion[0], item.top_confusion[1])
                if item.top_confusion
                else "-"
            )
            lines.append(
                "  {:12s} n={:5d}  classes={:2d}  accuracy={:.3f}  top-confusion={}".format(
                    item.anchor, item.n, item.n_classes, item.accuracy, confusion
                )
            )
    lines.append("")
    lines.append("-- top-1 by distance band (THE curve; a scalar cannot say this) --")
    for band in report.per_band:
        lines.append(
            "  {:5.2f}-{:5.2f} m  n={:5d}  top1={:.3f}  margin={:+.4f}".format(
                band.near_m, band.far_m, band.n, band.top1_accuracy, band.mean_true_margin
            )
        )
    lines.append("")
    lines.append("-- per class (recall, and what it was mistaken for) --")
    for item in sorted(report.per_class, key=lambda entry: -entry.recall):
        confusion = (
            "{} x{}".format(item.top_confusion[0], item.top_confusion[1])
            if item.top_confusion
            else "-"
        )
        lines.append(
            "  {:18s} {:8s} n={:4d}  recall={:.3f}  margin={:+.4f}  top-confusion={}".format(
                item.name, item.affinity, item.n, item.recall, item.mean_true_margin, confusion
            )
        )
    lines.append("")
    lines.append("-- forced-failure arm (absent classes were never in the prompt bank) --")
    lines.append(
        "  EER {:.3f} at threshold {:+.4f}  |  in-vocab n={}  absent n={}".format(
            report.rejection.eer,
            report.rejection.threshold_at_eer,
            report.rejection.n_in_vocabulary,
            report.rejection.n_absent,
        )
    )
    lines.append(
        "  absent rejected {:.3f}  |  in-vocab falsely rejected {:.3f}".format(
            report.rejection.rejection_rate, report.rejection.false_rejection_rate
        )
    )
    lines.append("")
    lines.append(
        "  EER near 0.500 means the two arms are on top of each other, which is what a "
        "gate that discriminates nothing looks like."
    )
    if failed:
        lines.append("")
        lines.append("SCENES THAT FAILED TO LOAD ({}): {}".format(len(failed), " ".join(failed)))
        lines.append("  Continuing is not passing — this run is incomplete.")
    lines.append("")
    return "\n".join(lines)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure CLAP's separation on live HM3D reverb, both arms."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenes", default="", help="space-separated; default every scene with a mesh")
    parser.add_argument("--corpus-dir", default="data/sound_corpus")
    parser.add_argument("--absent-dir", default="data/absent_corpus")
    parser.add_argument("--n-sources", type=int, default=2)
    parser.add_argument("--n-poses", type=int, default=6)
    parser.add_argument("--n-recordings", type=int, default=4)
    parser.add_argument("--n-bands", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--indirect-ray-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(None if argv is None else list(argv))

    audio = AudioConfig()
    if args.indirect_ray_count is not None:
        audio = dataclasses.replace(audio, indirect_ray_count=int(args.indirect_ray_count))

    cfg = GateConfig(
        run_dir=args.run_dir,
        data_root=args.data_root,
        split=args.split,
        scenes=tuple(args.scenes.split()) if args.scenes else (),
        corpus_dir=args.corpus_dir,
        absent_dir=args.absent_dir,
        n_sources=args.n_sources,
        n_poses=args.n_poses,
        n_recordings=args.n_recordings,
        n_bands=args.n_bands,
        seed=args.seed,
        audio=audio,
        overwrite=args.overwrite,
    )
    run_gate(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
