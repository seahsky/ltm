"""The model connectors the wiring layer constructs, and the seam they satisfy.

``audio/clap.py`` is pure: it takes an object exposing ``encode_audio(waveform,
sample_rate)`` and ``encode_text(text)`` and does the cosines itself. That injection is
what keeps the whole audio layer Mac-testable and numpy-only, and it is also why the
concrete encoder has had **no home** until now — ADR-0013's tree names ``vlm.py`` for
the captioner and names nothing for CLAP, because ticket 22 dissolved the constructor
the assertion used to live in.

It lands here, in ``task/``, for the reason ADR-0013 gives for ``task/`` existing: it is
the only layer allowed to import everything, and a connector to an external system is
exactly what it wires. That is a **disclosed addition to the ADR's tree**, not a
correction to it — the ADR's layer graph is unchanged and the seam is unchanged; what
changes is that the seam now ships with both sides rather than one, which is the property
``audio/clap.py``'s own docstring argues for.

**torch and transformers are imported inside the constructor.** They are in the box's
``ss2`` env and nowhere near ``earshot/tools/mac-requirements.txt``, and every module in
this package has to stay importable on a Mac — the same discipline ``audio/clips.py``
applies to scipy. The import is at *construction*, which happens once at startup
(requirement 9), so nothing here is a lazy-loading seam inside the episode loop.

**There is no fallback encoder.** A stub that returned plausible vectors would let a run
produce anomaly classifications with no model behind them, which is the failure class
``clips.load_anomaly_clip`` refuses in the signal domain. ``load_clap_encoder`` raises.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "ClapEncoder",
    "load_clap_encoder",
    "CLAP_MODEL_ID",
    "CLAP_LOCAL_DIR",
    "CLAP_SAMPLE_RATE",
    "resample_ratio",
    "resolve_clap_source",
    "stage_clap_safetensors",
    "ClipEncoder",
    "load_clip_encoder",
    "CLIP_MODEL_ID",
    "CLIP_LOCAL_DIR",
    "resolve_clip_source",
    "stage_clip_safetensors",
]

# provenance: box — ticket 13's known-good checkpoint, the one `env_check`'s
# `probe_clap_instantiable` constructs and the one the Gate-0b calibration that produced
# `ANOMALY_GATE_DELTA` / `ANOMALY_GATE_TAU` was measured against. Changing it invalidates
# those two constants, which is why it is named in one place.
CLAP_MODEL_ID = "laion/clap-htsat-unfused"

# provenance: box -- where `stage_clap_safetensors` writes the converted checkpoint. Named
# here and duplicated ONCE in `env_check.py`, which ADR-0013's layer graph forbids from
# importing anything intra-package; the duplication is deliberate and both sites say so.
CLAP_LOCAL_DIR = "models/clap-htsat-unfused"

# What the staged directory says about ITSELF. `stage_clap_safetensors` already wrote this
# file; `resolve_clap_source` did not read it, and that was the bug: any model_id resolved to
# whatever was staged, so `env_check`'s forced-failure arm asked for a model that does not
# exist, got the real checkpoint, and reported PASS. A directory that cannot name the model it
# holds is not a checkpoint for that model.
STAGED_MARKER = "PROVENANCE.json"

# provenance: box -- `ClapFeatureExtractor.sampling_rate` on this checkpoint. The extractor
# REFUSES any other rate; it does not resample. See `resample_ratio`.
CLAP_SAMPLE_RATE = 48000

# DREAM's `f_v` (eq. 6). ViT-B/32 for three reasons, in order of how much they matter:
#
#   * its image embedding is 512-d, the SAME width as `laion/clap-htsat-unfused`'s audio
#     embedding, so `f_fuse`'s concatenation (eq. 7) is balanced rather than three
#     quarters visual. An unbalanced fuse makes `q_t` a visual query wearing an audio
#     hat, and the `omega_t` shift the paper's central hypothesis is about would be
#     measuring the fuse rather than the memory;
#   * 88 M params beside CLAP's 153.5 M, and ticket 15 measured 0.713 GiB of CLAP against
#     26.45 GiB of margin on the V100. DREAM encodes EVERY step, unlike CLAP which fires
#     once per onset, so the per-step cost lands directly inside criterion 7's 0.5 s
#     ceiling -- `matrix-2` ran at 0.3278-0.3356 s worst-step, so the headroom is real but
#     it is not large, and `tools/clip_gate.sh` measures it rather than assuming it;
#   * it is the checkpoint this tree has already used. `step4-coarse-affordance` built a
#     CLIP zero-shot room classifier on it and measured real cosines (~0.30), so the
#     staging path and the failure modes are known rather than new.
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# provenance: box -- where `stage_clip_safetensors` writes the checkpoint, beside CLAP's.
CLIP_LOCAL_DIR = "models/clip-vit-base-patch32"


def resample_ratio(from_rate: int, to_rate: int = CLAP_SAMPLE_RATE):
    """The `(up, down)` integer pair for a polyphase resample, in lowest terms.

    Split out from the resample itself so the arithmetic is Mac-testable: scipy lives in the
    box's `ss2` env and nowhere near `mac-requirements.txt`, but getting 44100 -> 48000 wrong
    would silently change every duration CLAP sees, and that is worth a test.

    44100 -> 48000 is 160/147. `(1, 1)` when the rates already agree, so a caller can skip
    the resample without a second comparison.
    """
    import math

    source, target = int(from_rate), int(to_rate)
    if source <= 0 or target <= 0:
        raise ValueError(
            "sample rates must be positive, got from_rate={} to_rate={}".format(source, target)
        )
    divisor = math.gcd(source, target)
    return target // divisor, source // divisor



def resolve_clap_source(model_id: str = CLAP_MODEL_ID, local_dir: str = CLAP_LOCAL_DIR) -> str:
    """The staged local copy when it is complete, otherwise the Hub id.

    Prefers local because `laion/clap-htsat-unfused` ships ONLY `pytorch_model.bin`, and
    transformers >= 4.52 refuses to `torch.load` a `.bin` unless torch >= 2.6
    (CVE-2025-32434). The box pins torch 2.2.2+cu118 because cu118 is the last CUDA line
    where the V100's sm_70 is a first-class target, so upgrading torch to satisfy
    transformers would cost the GPU. Converting the checkpoint once costs neither.

    Completeness is checked, not assumed: a directory holding a half-written conversion
    would otherwise be preferred over the Hub and fail later with a confusing error.

    **IDENTITY is checked too, and the first version did not check it.** Returning the staged
    directory for ANY `model_id` made `env_check`'s forced-failure arm vacuous: a probe asked
    for `earshot/definitely-not-a-model` got the real local checkpoint and reported a finite
    feature vector, so the arm that exists to prove the probe can fail reported PASS. The box
    gate caught it, which is ADR-0014's rule working exactly as intended.

    `STAGED_MARKER` records which model the directory was converted from. A directory without
    it is a directory that cannot say what it holds, so it is not used. `stage_clap_safetensors`
    writes it, and the marker is part of the completeness check, so an older staged copy is
    simply re-staged on the next run rather than trusted on faith.
    """
    return _staged_source(model_id, local_dir)


def _staged_source(model_id: str, local_dir: str) -> str:
    """The identity-and-completeness check both resolvers run. Pure over the filesystem.

    Extracted when CLIP arrived rather than copied: the identity half is incident-driven
    (`resolve_clap_source`'s docstring has the account -- a probe asking for
    `earshot/definitely-not-a-model` got the real checkpoint and reported PASS), and a
    second copy of it is a second place for that arm to go vacuous.
    """
    import json
    import os

    needed = (
        "model.safetensors",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        STAGED_MARKER,
    )
    if not all(os.path.isfile(os.path.join(local_dir, name)) for name in needed):
        return model_id
    try:
        with open(os.path.join(local_dir, STAGED_MARKER), encoding="utf-8") as handle:
            staged = str(json.load(handle).get("model_id", ""))
    except (ValueError, OSError):
        return model_id
    return local_dir if staged == model_id else model_id


def resolve_clip_source(model_id: str = CLIP_MODEL_ID, local_dir: str = CLIP_LOCAL_DIR) -> str:
    """The staged local copy when it is complete AND says it holds `model_id`, else the Hub id.

    Same contract and same guards as `resolve_clap_source`, including the identity check
    that keeps `env_check`'s forced-failure arm from being vacuous.
    """
    return _staged_source(model_id, local_dir)


def stage_clap_safetensors(model_id: str = CLAP_MODEL_ID, out_dir: str = CLAP_LOCAL_DIR) -> str:
    """Convert the Hub checkpoint to safetensors under `out_dir`. Idempotent. Returns the dir.

    A staging step, run once on the box like `python -m earshot.audio.clips`, never on the
    live path. It calls `torch.load` DIRECTLY rather than through transformers: the CVE
    guard is transformers' policy about untrusted checkpoints, and this checkpoint is the
    one `CLAP_MODEL_ID` names and the bootstrap already fetched.

    Every tensor is cloned before saving. `safetensors.save_file` refuses tensors that share
    storage, which tied embeddings produce, and a clone is the documented fix rather than a
    superstition.
    """
    import json
    import os
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    if resolve_clap_source(model_id, out_dir) == out_dir:
        print("  CLAP already staged at {} - nothing to do".format(out_dir))
        return out_dir

    source = snapshot_download(model_id)
    checkpoint = os.path.join(source, "pytorch_model.bin")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            "{} has no pytorch_model.bin at {}. If the Hub repo gained a model.safetensors "
            "this conversion is unnecessary and `resolve_clap_source` should point at the "
            "Hub id again.".format(model_id, source)
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tensors = {
        key: value.contiguous().clone()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    dropped = sorted(set(state) - set(tensors))
    if dropped:
        # Printed, never swallowed: a dropped key is a weight the model would initialise
        # randomly, which is the silent-fabrication class this repo keeps finding.
        print("  WARNING: {} non-tensor key(s) not carried: {}".format(len(dropped), dropped))

    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    for name in sorted(os.listdir(source)):
        if name.endswith((".json", ".txt")):
            shutil.copyfile(os.path.join(source, name), os.path.join(out_dir, name))

    with open(os.path.join(out_dir, "PROVENANCE.json"), "w", encoding="utf-8") as sink:
        json.dump(
            {
                "model_id": model_id,
                "converted_from": checkpoint,
                "n_tensors": len(tensors),
                "dropped_non_tensor_keys": dropped,
                "why": (
                    "transformers >= 4.52 refuses torch.load on a .bin below torch 2.6 "
                    "(CVE-2025-32434); the box pins torch 2.2.2+cu118 for the V100 sm_70"
                ),
            },
            sink,
            indent=2,
            sort_keys=True,
        )
    print("  CLAP staged: {} tensor(s) -> {}/model.safetensors".format(len(tensors), out_dir))
    return out_dir


def stage_clip_safetensors(model_id: str = CLIP_MODEL_ID, out_dir: str = CLIP_LOCAL_DIR) -> str:
    """Stage the CLIP checkpoint under `out_dir` as safetensors. Idempotent. Returns the dir.

    **It does not assume which format the Hub ships, and that is the difference from
    `stage_clap_safetensors`.** CLAP's stager can say flatly that `laion/clap-htsat-unfused`
    ships only `pytorch_model.bin`, because that was checked on the box and the docstring
    records it. Nobody has checked this repo for `openai/clip-vit-base-patch32`, so the
    snapshot is ASKED rather than predicted: a `model.safetensors` already there is copied,
    and only a `.bin`-only snapshot is converted. Writing "CLIP ships safetensors" here
    without having run it would be exactly the fabrication the tree keeps catching.

    The conversion branch calls `torch.load` directly for the reason CLAP's does: the CVE
    guard is transformers' policy about untrusted checkpoints, and this is the checkpoint
    `CLIP_MODEL_ID` names.
    """
    import json
    import os
    import shutil

    from huggingface_hub import snapshot_download

    if resolve_clip_source(model_id, out_dir) == out_dir:
        print("  CLIP already staged at {} - nothing to do".format(out_dir))
        return out_dir

    source = snapshot_download(model_id)
    os.makedirs(out_dir, exist_ok=True)
    shipped = os.path.join(source, "model.safetensors")
    if os.path.isfile(shipped):
        shutil.copyfile(shipped, os.path.join(out_dir, "model.safetensors"))
        n_tensors = None
        converted_from = shipped
        how = "copied the safetensors the Hub already ships"
    else:
        import torch
        from safetensors.torch import save_file

        converted_from = os.path.join(source, "pytorch_model.bin")
        if not os.path.isfile(converted_from):
            raise FileNotFoundError(
                "{} has neither model.safetensors nor pytorch_model.bin at {}. Nothing "
                "here can stage it, and `resolve_clip_source` will keep returning the Hub "
                "id -- which transformers will then refuse below torch 2.6 "
                "(CVE-2025-32434).".format(model_id, source)
            )
        state = torch.load(converted_from, map_location="cpu", weights_only=True)
        tensors = {
            key: value.contiguous().clone()
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        dropped = sorted(set(state) - set(tensors))
        if dropped:
            # Printed, never swallowed: a dropped key is a weight the model would
            # initialise randomly, which is the silent-fabrication class this repo keeps
            # finding.
            print("  WARNING: {} non-tensor key(s) not carried: {}".format(
                len(dropped), dropped))
        save_file(
            tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"}
        )
        n_tensors = len(tensors)
        how = "converted pytorch_model.bin"

    for name in sorted(os.listdir(source)):
        if name.endswith((".json", ".txt")):
            shutil.copyfile(os.path.join(source, name), os.path.join(out_dir, name))

    with open(os.path.join(out_dir, STAGED_MARKER), "w", encoding="utf-8") as sink:
        json.dump(
            {
                "model_id": model_id,
                "converted_from": converted_from,
                "n_tensors": n_tensors,
                "how": how,
                "why": (
                    "DREAM eq. 6 needs f_v and the tree had no visual encoder; staged "
                    "locally for the same CVE-2025-32434 reason as CLAP"
                ),
            },
            sink,
            indent=2,
            sort_keys=True,
        )
    print("  CLIP staged ({}) -> {}/model.safetensors".format(how, out_dir))
    return out_dir


class ClipEncoder:
    """A connector: ``openai/clip-vit-base-patch32`` behind the one call `vlm/encode.py` needs.

    A class in a tree that is otherwise functions, on the rule this repo states for it --
    classes are for connectors to external systems, and this owns a loaded model with a
    lifecycle and a device. The deliberate twin of `ClapEncoder`.

    **It runs under `torch.no_grad()` and `eval()`, and that is a cost decision rather
    than a correctness one.** DREAM encodes EVERY step (eq. 5), not once per onset the way
    CLAP does, so this sits inside criterion 7's 0.5 s per-step ceiling on every one of a
    run's ~180 steps. `matrix-2` measured a worst step of 0.3278-0.3356 s, so there is
    headroom and it is not generous.
    """

    def __init__(self, model_id: str = CLIP_MODEL_ID, device: Optional[str] = None) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model_id = str(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The staged safetensors copy when present, else the Hub id -- and the identity
        # check in `_staged_source` is what stops a wrong `model_id` silently resolving to
        # whatever happens to be on disk.
        self.source = resolve_clip_source(self.model_id)
        self._processor = CLIPProcessor.from_pretrained(self.source)
        self._model = CLIPModel.from_pretrained(self.source).to(self.device)
        self._model.eval()

    def encode_image(self, frame: Any) -> Any:
        """The image embedding for one RGB frame, as the processor's own preprocessing wants it.

        The frame arrives as habitat-sim renders it: HxWx3 or HxWx4 `uint8`. The alpha
        channel is dropped here rather than at the call site, because a caller that forgot
        would hand CLIP a 4-channel array and get a resize error several frames into an
        episode instead of a shape error on the first.
        """
        import numpy as np

        values = np.asarray(frame)
        if values.ndim == 3 and values.shape[-1] == 4:
            values = values[..., :3]
        inputs = self._processor(images=values, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return features.squeeze(0).float().cpu().numpy()


def load_clip_encoder(model_id: str = CLIP_MODEL_ID) -> ClipEncoder:
    """Construct the encoder, or raise with the same diagnosis `load_clap_encoder` pays for.

    The failure named is not "transformers is missing": it is ``transformers`` gating its
    torch backend on ``torch >= 2.1`` and substituting a ``DummyObject`` that **imports
    cleanly and raises only when constructed**.
    """
    try:
        return ClipEncoder(model_id)
    except Exception as exc:
        raise RuntimeError(
            "could not construct CLIP ({}): {}. transformers disables its torch backend "
            "below torch 2.1 and substitutes a DummyObject that imports fine and raises "
            "on construction -- check the torch/transformers pair, not just that both are "
            "installed (ticket 13).".format(model_id, exc)
        ) from exc


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Stage the CLAP and CLIP checkpoints locally as safetensors (run once on the "
            "box). Both, because DREAM needs f_u AND f_v and a sweep that staged one of "
            "them fails halfway through its first episode."
        )
    )
    parser.add_argument("--model-id", default=CLAP_MODEL_ID)
    parser.add_argument("--out-dir", default=CLAP_LOCAL_DIR)
    parser.add_argument("--clip-model-id", default=CLIP_MODEL_ID)
    parser.add_argument("--clip-out-dir", default=CLIP_LOCAL_DIR)
    parser.add_argument(
        "--clap-only", action="store_true",
        help="stage CLAP and skip CLIP (what every caller predating DREAM's f_v wanted)",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    stage_clap_safetensors(args.model_id, args.out_dir)
    print("  resolve_clap_source now returns: {}".format(resolve_clap_source(args.model_id, args.out_dir)))
    if not args.clap_only:
        stage_clip_safetensors(args.clip_model_id, args.clip_out_dir)
        print("  resolve_clip_source now returns: {}".format(
            resolve_clip_source(args.clip_model_id, args.clip_out_dir)))
    return 0



class ClapEncoder:
    """A connector: ``laion/clap-htsat-unfused`` behind the two calls ``clap.py`` needs.

    A class in a tree that is otherwise functions, on the rule this repo states for it —
    classes are for connectors to external systems, and this owns a loaded model with a
    lifecycle and a device.

    Both methods return a 512-d vector; ``audio/clap.py`` normalises and dots them, so
    nothing here decides anything. The waveform is the **heard clip** —
    ``clap.heard_clip_for_clap`` collapses the binaural signal the onset fired on into
    mono — rather than a re-render, because the gate and the onset must be answering
    questions about the same sound.
    """

    def __init__(self, model_id: str = CLAP_MODEL_ID, device: Optional[str] = None) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        self._torch = torch
        self.model_id = str(model_id)
        # `cuda` when there is one, because the audio path already pays for the render
        # and 153.5 M params on the CPU would land inside the per-step budget criterion 7
        # audits. Ticket 15 measured CLAP at 0.713 GiB against 26.45 GiB of margin.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The staged safetensors copy when present, else the Hub id. See
        # `resolve_clap_source` for why the local copy has to exist on this box.
        self.source = resolve_clap_source(self.model_id)
        self._processor = ClapProcessor.from_pretrained(self.source)
        self._model = ClapModel.from_pretrained(self.source).to(self.device)
        # `audios=` is deprecated for removal in transformers 4.59 in favour of `audio=`.
        # Resolved ONCE from the signature rather than guessed: transformers wraps
        # `__call__` with a decorator that uses functools.wraps, so `inspect.signature`
        # follows `__wrapped__` and reports the real parameter names. `audios` is the
        # fallback because it is what every version in the declared >=4.40,<5 range
        # accepts, warning included.
        import inspect

        try:
            names = inspect.signature(type(self._processor).__call__).parameters
        except (TypeError, ValueError):
            names = {}
        self._audio_kwarg = "audio" if "audio" in names else "audios"
        self._model.eval()

    def encode_audio(self, waveform: Any, sample_rate: int) -> Any:
        """The audio embedding for one mono waveform, resampled to CLAP's own rate.

        **``ClapProcessor`` does NOT resample.** The previous version of this docstring said
        it did, and that claim was wrong from the day it was written: `ClapFeatureExtractor`
        raises ``ValueError`` on any rate other than its 48 kHz. Nothing caught it because
        the weights had never been loaded on the box -- `bootstrap_ss2.sh` reported
        ``clap_weights_loaded False`` and ``--clap`` defaults off -- so the first caller to
        reach this line was the separation gate.

        The renderer stays at 44100. `AudioConfig.sample_rate` is the branch's own
        ``sampleRate`` and ESC-50 is 44.1 kHz, so the heard signal's domain is the TASK's;
        CLAP is one consumer of it and converts at its own boundary. Moving the renderer to
        48 kHz to suit a text-audio encoder would change every IR in the tree.
        """
        samples = self._as_1d(waveform)
        rate = int(sample_rate)
        if rate != CLAP_SAMPLE_RATE:
            from scipy.signal import resample_poly

            up, down = resample_ratio(rate, CLAP_SAMPLE_RATE)
            samples = resample_poly(samples, up, down)
            rate = CLAP_SAMPLE_RATE
        inputs = self._processor(
            **{self._audio_kwarg: samples},
            sampling_rate=rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_audio_features(**inputs)
        return features[0].detach().cpu().numpy()

    def encode_text(self, text: str) -> Any:
        """The text embedding for one prompt."""
        inputs = self._processor(text=[str(text)], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return features[0].detach().cpu().numpy()

    @staticmethod
    def _as_1d(waveform: Any) -> Any:
        """Flatten to mono samples without importing numpy at module scope.

        ``heard_clip_for_clap`` already averages the ears, so this is a guard against a
        caller that passed the binaural signal straight through rather than a conversion
        anything is meant to rely on.
        """
        values = getattr(waveform, "reshape", None)
        return waveform if values is None else waveform.reshape(-1)


def load_clap_encoder(model_id: str = CLAP_MODEL_ID) -> ClapEncoder:
    """Construct the encoder, or raise with the diagnosis ticket 13 paid for.

    The failure this message exists for is not "transformers is missing": it is
    ``transformers`` gating its torch backend on ``torch >= 2.1`` and substituting a
    ``DummyObject`` that **imports cleanly and raises only when constructed**. So the
    error names the pair rather than the package, and ``env_check``'s
    ``probe_clap_instantiable`` asks the same question at startup by constructing too.
    """
    try:
        return ClapEncoder(model_id)
    except Exception as exc:
        raise RuntimeError(
            "could not construct CLAP ({}): {}. transformers disables its torch backend "
            "below torch 2.1 and substitutes a DummyObject that imports fine and raises "
            "on construction — check the torch/transformers pair, not just that both are "
            "installed (ticket 13). `python -m earshot.env_check --clap --strict` is the "
            "same probe.".format(model_id, exc)
        ) from exc


if __name__ == "__main__":
    raise SystemExit(_main())
