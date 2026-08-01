# 13 — CLAP cannot run in the `ss2` env: which pin moves?

Type: task
Status: open
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
