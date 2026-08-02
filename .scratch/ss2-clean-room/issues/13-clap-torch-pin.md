# 13 — CLAP cannot run in the `ss2` env: which pin moves?

Type: task
Status: claimed
Assignee: Sky
Blocked by: none

## Question

In the `ss2` env, transformers 4.57.6 has disabled its PyTorch backend because it requires `torch>=2.1` and found `torch==2.0.1+cu117`, so `ClapModel` is a dummy and the anomaly classifier cannot instantiate.
Which pin moves — torch up, or transformers down — and does the fix hold the rest of the env?

## Why it matters

Ticket 04 is GREEN and stays GREEN: habitat-sim(audio) + torch-on-GPU + the CLAP stack coexist in one interpreter and the numpy pin holds through every layer.
That was the hard question and it is answered.

This is the soft one. CLAP is not optional — it is the anomaly onset trigger and 3-way classifier, and the map's destination requires an anomaly sound to fire and be responded to.
Right now the import succeeds and the model does not exist:

```
Disabling PyTorch because PyTorch >= 2.1 is required but found 2.0.1+cu117
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be
available and only tokenizers, configuration and file/data utilities can be used.
```

Ticket 04's gate reported `clap_symbols_importable: True` against that, which is a **false positive of exactly the kind the gate avoids elsewhere** — issue #2340 taught it to probe the enum member rather than the class for audio capability, and the same lesson was not applied to CLAP. Whatever fix lands here, the check has to become "a CLAP forward pass produces a logit", not "the symbol imports".

The pin that failed guarded the wrong end. `transformers>=4.30,<5` capped the top to stop a major bump changing the API; the break came from the **bottom** being loose enough to resolve 4.57, against a `torch==2.0.1` that ticket 04's own comment already flagged as an unverified V100-*era* guess.

## What would resolve it

Pick one of two, apply it in `oneenv_gate.sh`, re-run, and prove CLAP actually infers:

- **(a) Bump torch — recommended.** The `2.0.1` pin was a guess, never a requirement. The box's driver is `580.159.03`, which is modern enough to run any cu11x/cu12x runtime, and the V100 is compute 7.0 — comfortably inside the supported arch list for the torch 2.1–2.4 range. Nothing in the stack depends on torch 2.0 specifically; habitat-sim does not import torch at all. Set `SS2_TORCH_SPEC` / `SS2_TORCH_INDEX` and keep a current transformers.
- **(b) Pin transformers down.** `transformers>=4.30,<4.46` keeps CLAP (which landed in 4.27) and drops the `torch>=2.1` requirement. Smaller download, more moving parts frozen, and it leaves the box on a torch pin nobody has justified.

Either way, two things must be re-asserted after the change, because both are the reason this env is delicate:
- **`numpy<1.24` still holds.** The constraint file already applies to every install, but the whole 2022-era habitat-sim tree dies on numpy 2.x, so it gets asserted, not assumed.
- **The audio probe still passes after the swap.** Ticket 04's layering discipline — re-probe audio after every install layer — exists because a later `pip install` quietly resolving a dependency is the most likely way this env breaks.

GREEN = one env, audio probe still passing, and a real `ClapModel` forward pass on a real waveform returning finite logits on the GPU (`--load-clap`, ~600 MB).

## Note

Cheap to fold into whatever box session runs next rather than paying a separate 25-minute build — the gate is idempotent and skips the habitat-sim build when an audio-capable `habitat_sim` already imports, so this is a pip layer and a probe, not a rebuild.

## Note added by ticket 05 (resolved) — option (a) now has direct evidence, and a sharper risk

**Take option (a), bump torch.** This ticket argued it from the driver version. The inventory measured it on the same box: the `ltm-embodied` env runs **torch 2.8.0+cu128 with `cuda=True` on this exact V100**, and the driver's max supported CUDA is **13.0**. So "a modern torch runs here" stops being an inference about driver compatibility and becomes an observation. Option (b), pinning transformers down, would freeze the stack around a torch pin that is now positively known to be unnecessary.

**But do not copy `ltm-embodied`'s torch version.** That env also carries **numpy 1.26.4**, which is above the `<1.24` pin the whole 2022-era habitat-sim tree depends on. The box's proof-of-modern-torch arrives bundled with exactly the numpy that would kill `ss2`. So the real question this ticket answers is narrower than "does torch 2.8 work here" — it is **the highest torch that still resolves against numpy 1.23.5**, which is what the existing `numpy<1.24` constraint file has to be asserted against after the install, not assumed.

That points at the torch 2.1–2.4 range this ticket already named, for the reason it named (>= 2.1 satisfies transformers, and the V100's compute 7.0 is inside the supported arch list) rather than at the newest wheel available. `nvcc` is absent on the box, which is fine — these are binary wheels and habitat-sim's audio build never needed it.

Also relevant to the GREEN check: only **8.2 GB of 32 GB VRAM was free** at inventory time. CLAP is ~600 MB so this does not threaten the forward pass, but if the `--load-clap` probe OOMs, check `nvidia-smi` before suspecting the pin.

## Progress 2026-08-02 — decision made and implemented; box verification pending

The pin choice is settled and the code is landed and unit-tested on the Mac.
What is **not** yet done is the box run, so this ticket stays `claimed`: the whole lesson of ticket 04's false positive is that a GREEN is measured, not argued, and it would be precisely the wrong move to close this by asserting the fix works.

### The answer: option (a), torch up — `torch==2.2.2` from the **cu118** index

Ticket 05 argued this from an observation (torch 2.8.0+cu128 runs on this V100). Three further facts were checked against primary sources, and they narrow "somewhere in 2.1–2.4" to one wheel:

1. **transformers' gate is `version.parse(torch) >= 2.1.0`, exactly.** Source-verified in `transformers/utils/import_utils.py` @ v4.57.1 — `_torch_available` is set, then overwritten by that comparison, and the failure is a `logger.warning`. Nothing else. So 2.2.x clears it with a minor version of headroom rather than sitting on the boundary.
2. **`torch-2.2.2+cu118-cp39-cp39-linux_x86_64.whl` exists.** Verified in the cu118 index. cp39 wheels run through 2.6.0 on that index; 2.7+ switches to `manylinux_2_28` naming. So the Python 3.9 pin is not the binding constraint here, though it is worth knowing it bites at 2.7.
3. **torch declares no numpy dependency at all.** Verified in the 2.2.2 release metadata: `requires_dist` is filelock / typing-extensions / sympy / networkx / jinja2 / fsspec / the nvidia-* set / triton. No numpy. So this install *physically cannot* move the `numpy<1.24` pin — the constraint file is a belt, not the trousers.

**This corrects the question ticket 05 handed down.** It framed the task as finding "the highest torch that still *resolves* against numpy 1.23.5". Given fact 3, that is not a resolution question at all — every torch resolves against every numpy, because there is no declared edge. It is an **ABI-era** question, and that inverts the search direction: take the **oldest** torch that clears the transformers gate, not the highest, because torch 2.2.x predates numpy 2.0 and was therefore *built against* numpy 1.x. numpy 1.23.5 is its native ABI rather than a tolerated downgrade. Nothing in this stack needs a feature from torch 2.3+, so the headroom buys nothing and the version gap costs risk.

cu118 rather than cu121+: the V100 is compute 7.0, and cu118 is the last CUDA line where sm_70 is a first-class target rather than legacy. Ticket 05 already proved the driver is not the constraint, so there is no reason to reach.

### One correction to this ticket's own diagnosis

The ticket says: *"The pin that failed guarded the wrong end. `transformers>=4.30,<5` capped the top... the break came from the bottom being loose."*

Half right, and the wrong half is load-bearing. **On this env the `<5` cap is inert**: transformers 5.x declares `requires_python >= 3.10`, and the SoundSpaces pin is Python 3.9, so the *interpreter* already caps resolution at the 4.x line. The cap was not guarding the wrong end — it was not guarding anything.

What actually broke the env is narrower and more general: **a floating dependency was resolved against a fixed one, with nothing asserting the resulting pair works.** `transformers` drifted up to the newest release py39 permits (4.57.6) while `torch` sat frozen at a guess, and the gate had no check capable of noticing. Fixing the version fixes today. Fixing the *check* fixes the class — which is why the probe change below matters more than the pin change.

### Three defects found while implementing

- **The torch layer's skip was version-blind** (`oneenv_gate.sh` step 5: `if python -c "import torch"`). The box's `ss2` env already has torch 2.0.1 installed, so this fix would have been a **silent no-op** — the gate would re-run, skip the layer, and faithfully reproduce the failure it was meant to repair. Now gated on the version transformers actually requires. Tested against 2.0.1 / 2.2.2 / 2.10.0 / absent; note 2.10 vs 2.1 is why the comparison is tuple-based and not string-based.
- **`clap_symbols_importable: True` was structurally incapable of being False.** `from transformers import ClapModel` succeeds with the backend disabled because transformers substitutes a `DummyObject` that only raises on *instantiation*. Replaced with a capability probe: `is_torch_available()`, plus a dummy-object check on `ClapModel.__module__`, both always-on and both fatal. Tested with a faked disabled-backend transformers — the stage now goes RED with an error naming this ticket, where it previously went GREEN.
- **`ClapProcessor(audios=...)` is deprecated for removal in 4.59.** Since transformers is deliberately left floating within its py39-bounded band, the forward pass tries `audio=` and falls back to `audios=`, so the probe spans the whole range the resolver can legally pick.

### What "GREEN" now means, and it is checked in two tiers

- **Always on, free:** backend enabled + `ClapModel` is not a dummy. This is the check that would have caught the original break, and it costs nothing, so nobody can skip it.
- **On `SS2_LOAD_CLAP=1`:** a real `ClapModel` forward pass on a real waveform (a deterministic tone plus noise — an all-zero buffer can survive a broken normalisation silently), shaped as the 3-way zero-shot anomaly classification the destination actually needs, asserting `logits_per_audio` is `[1, 3]` and finite on CUDA. Finiteness and shape only; classification *accuracy* is the anomaly gate's problem, not this ticket's.

The gate's verdict block now prints which of the two ran, so a GREEN can never again be ambiguous about what it proved of CLAP.

Also added: an explicit post-install numpy assertion (the ticket asked for asserted-not-assumed), which fails loudly on the 1.26.4 that `ltm-embodied` carries on this same box.

### Box step — this is all that is left

The gate is idempotent and skips the ~25-minute habitat-sim build when an audio-capable `habitat_sim` already imports, so this is a pip layer, a probe and a ~600 MB download:

```bash
nrun bash .scratch/ss2-clean-room/probes/oneenv_gate.sh
```

with `SS2_LOAD_CLAP=1` exported, since ticket-13 GREEN is the logit. **Remember the self-update gotcha** — the script git-pulls itself, so the first invocation after this commit runs the *old* copy; it needs a second invocation to take effect.

Expected: step 5 reports `found torch 2.0.1+cu117 (need >= 2.1)` then installs 2.2.2+cu118; steps 6–7 confirm numpy still 1.23.x, the audio probe still passes, and CLAP returns three finite logits. If `--load-clap` OOMs, check `nvidia-smi` first — CLAP is 600 MB against the 8.2 GB that was free, so an OOM points at ticket 15's unaccounted VRAM, not at the pin.
