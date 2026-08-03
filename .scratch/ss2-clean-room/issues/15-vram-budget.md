# 15 — What holds 24 GB of VRAM, and what is the clean room's VRAM budget?

Type: task
Status: closed
Assignee: Sky
Blocked by: none
Resolved: 2026-08-03

## Question

On the RACE box, 8,249 MiB is free of 32,768 MiB — roughly 24 GB is held by something nobody has accounted for.
What is holding it, and once it is released, does the clean room's stack fit in 32 GB with margin?

## Why it matters

Ticket 05 found this and explicitly declined to act on it.
Ticket 10 declined it too, but for a different reason: it is not a cleanup item at all, so it does not belong on a reset checklist. It is a correctness threat, and it has two consumers already in flight.

**Ticket 06 is timing audio renders underneath it right now.** If 24 GB is held by a zombie process rather than by anything intentional, 06's numbers are measured under memory pressure, and 06's verdict decides whether the map's destination — audio rendered live at every step — is reachable as named. A cost measurement taken on a contended box is a validity problem, not a rounding error.

**ADR-0008 is already counting VRAM.** Dropping the 7B planner was justified in part as freeing ~15 GB. That arithmetic assumes a known starting point, and 8 GB free is not one. On a 32 GB V100 the clean room has to hold: the audio context and its scene geometry, the Qwen2-VL-2B captioner (kept by ADR-0008 for the caption-grounded detector), and — if ticket 09 rules that scene-conditioned normality survives — CLIP for the room classifier, plus the CLAP anomaly classifier whichever way ticket 13 resolves its torch pin.

Nobody has added those up against a measured free figure.

## What would resolve it

On the box:

1. **`nvidia-smi` with process attribution** — what holds the 24 GB. Expect one of: a stale python from an earlier run, a leaked notebook kernel, or something intentional nobody documented.
2. **Release it and re-measure** the free figure at rest.
3. **Add up the clean room's stack** against that figure, as a table with a measured number per component, not an estimate:
   - the SoundSpaces audio context + scene geometry (ticket 04 loaded a real HM3D scene; 392,356 verts)
   - Qwen2-VL-2B captioner
   - CLAP (whichever pin ticket 13 lands on)
   - CLIP, **conditional on ticket 09** — 07 removed CLIP from the agent, and the ADR-0002 room classifier is the only route by which it returns
4. **State the margin**, and say plainly whether the stack fits with the audio sim co-resident or whether something has to be lazy-loaded or dropped.

Note the precedent: the L3 milestone hit exactly this wall and solved it at the config level — swapping the 7B planner for `microsoft/Phi-3.5-mini-instruct` in one process let OWLv2 base and large both run on cuda co-resident with no OOM. The VRAM fix was the durable win of that milestone even though the detector arc closed as a negative. Same class of problem, and the clean room should not rediscover it during a build.

Two things this ticket is **not**:

- It is not a deletion or tidiness item. Ticket 10's box sweep rules on those, and this was promoted out of it deliberately.
- It does not decide whether CLIP is in the stack. Ticket 09 owns that; this ticket prices both answers.

Deliverable: the process attribution, the released free figure, and a component-by-component VRAM budget with a stated margin.

---

## Resolution (2026-08-03)

**Not a leak. It was a live 13.7-day `nrun` job, and the clean room fits in the card with ~26 GiB to spare — VRAM was never a constraint on anything in this map.**

Probe: `.scratch/ss2-clean-room/probes/vram_probe.py` (`--attribute` / `--release` / `--budget` / `--summarize`), teardown: `.scratch/ss2-clean-room/probes/kill_nrun.sh`. Data on the box at `runs/ss2-vram/{attribution,attribution-after,budget}.json`.

### 1. What held it

PID 153833, `state: R (running)`, uid `riftuser`, RSS 4 GB, `cwd=/home/riftuser/ltm`, interpreter from the **`ltm-embodied`** env:

```
bash scripts/notify-run.sh bash scripts/race-r1-objectnav.sh --tag r1v1 --split val   (69112, elapsed 13-17:28)
  └── bash scripts/race-r1-objectnav.sh                                               (69120)
        └── python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr
              --setting 1 --split val --scene all --n-episodes 2000 --max-steps 500
              --semantic-frontier-backend blip2 --value-model Salesforce/blip2-itm-vit-g
              --out-dir runs/r1v1-s1plus                                              (153833, elapsed 5-20:04)
```

**Every hypothesis in this ticket's own body was wrong.** Not a stale python, not a leaked notebook kernel, and not namespace blindness — `fuser -v /dev/nvidia*` named the same single PID the driver did, and only **5 MiB** of the 24,402 MiB in use was unattributed. It was the third option, "something intentional nobody documented", which this ticket listed last and framed as least likely. The distinguishing fact is that it was *still running*, not stale: the R1 S1+ arm at `episode_1497.json` of 2000, ~75% through.

Two things about it that matter beyond attribution:

- **The wrapper had been up since ~2026-07-20**, eight days longer than the python, because the driver runs arms sequentially. So the box has been contended since 2026-07-20, not since 2026-07-28.
- **It was the last consumer of the tree this map exists to delete**, running `embodied_memory.run_hm3d_pol` out of the box's single working tree. Killing it *unblocks* ticket 10 rather than blocking it (see knock-ons).

### 2. Released

`kill_nrun.sh --yes` signalled the workload (and the intermediate driver), leaving the wrapper to exit through its own `EXIT` trap and email its report. Result: **0 MiB used, 32,495 MiB free**.

The ~273 MiB gap against the 32,768 nameplate is ECC reserve (ECC is Enabled). So the card's **usable ceiling is 32,495 MiB = 31.73 GiB**, and every margin below is quoted against that.

### 3. The budget

Measured with `cudaMemGetInfo` deltas, in runner load order, on `minival/00800-TEEsavR23oF`, RGB+depth at 480x640, torch 2.2.2+cu118:

| component | delta GiB | cumulative GiB |
| --- | ---: | ---: |
| CUDA context | 0.363 | 0.363 |
| habitat-sim + HM3D scene + RGB/depth | 0.021 | 0.384 |
| first RGB/depth render | 0.112 | 0.496 |
| audio sensor (spec added) | **0.000** | 0.496 |
| audio first render | **0.000** | 0.496 |
| CLAP weights (`laion/clap-htsat-fused`) | 0.615 | 1.111 |
| CLAP forward | 0.156 | 1.268 |
| CLIP (`clip-vit-base-patch32`) | **FAILED** — see below | — |
| Qwen2-VL-2B-Instruct, fp16 (R2 only) | 4.279 | 5.547 |

### 4. The margin

| stack | resident | margin vs 31.73 GiB |
| --- | ---: | ---: |
| smoke (sim + audio + CLAP) | **1.268 GiB** | **30.73 GiB** |
| full (+ captioner) | **5.547 GiB** | **26.45 GiB** |

**Nothing has to be lazy-loaded and nothing has to be dropped.** The clean room uses ~17% of the card at its heaviest measured configuration, ~4% for the smoke.

### What this ticket found that it was not asked for

**The audio spine is VRAM-free.** Zero at spec-add and zero at first render, with a real IR returned — shape `[2, 72300]`, binaural, 1.506 s at 48 kHz (consistent with ticket 04's 1.64 s against the 4.0 s `maxIRLength` cap, and with the map's "no fixed-width IR buffer" requirement). RLR propagation is entirely CPU-side, so **the entire cost of "live audio at every step" is ticket 06's CPU budget and VRAM never bounds it.** Shape proves a render happened, not that the mesh was non-empty — that remains `arm_audio_context`'s vertex floor (tickets 12/16).

**ADR-0008's VRAM sentence is both miscited and immaterial.** It reads "Ticket 04's GREEN measured 31.73 GB free on the box." That figure is `props.total_memory` (`oneenv_probe.py:341`) — ECC-adjusted *capacity*, which equals free only at idle, and the GPU was not idle on 2026-08-01 (8.2 GB free). But the deeper point is that the correction does not matter: dropping the 7B planner "frees ~15 GB", and 5.547 + 15 = 20.5 GiB **still fits inside 31.73 with margin**. Capacity never forced any part of ADR-0008. Its own heading already says the decision rests on evidence rather than freed VRAM, so the fix is to delete the sentence, not restate it. Done in this commit.

**The allocator undercount is now quantified.** CLAP costs **0.771 GiB** driver-side against ticket 13's **0.713 GB** allocator figure — 8% under on its own, and **0.421 GiB under** once the CUDA context no allocator ever sees is included. Any future budget summed from `max_memory_allocated()` is optimistic by roughly a third of a gig per process, always in the same direction.

**CLIP cannot load in `ss2`, for a reason that outlives this ticket.** transformers 4.57.6 refuses `torch.load` below torch 2.6 (CVE-2025-32434) unless the checkpoint is safetensors; the cached CLIP is a `.bin`, and ticket 13 pinned torch at 2.2.2+cu118 for the V100. Qwen2-VL loaded fine because it ships safetensors. This is a live constraint on ticket 09, not a probe bug.

**`assert_no_swallowed_keys` had a false positive that would have fired on every healthy run** — and fixing it answers ticket 16's stage-1 question. The real `AudioSensorSpec` constructor attaches `__noise_model_kwargs` as a genuine instance attribute (`noise_model_kwargs` is a separate bound field), so `vars(spec)` is **not** empty on the binary. Now `KNOWN_DYNAMIC_ATTRS`, subtracted on top of any caller-supplied `allowed` so it cannot be re-opened. The 27 unit tests were green because the fakes never reproduced it — precisely the gap ticket 16 exists to close. +3 tests, 30 green, one asserting a real typo still fails alongside it.

### Corrections to this ticket's own framing

- "**Ticket 06 is timing audio renders underneath it right now**" was false at measurement time — the process table held exactly one entry, so 06's sweep was not resident. The validity threat is **retrospective**: any 06 timing taken from 2026-07-20 onward was contended. Recorded on ticket 06.
- The ticket treats the margin as the open question. It is not close, and the more useful finding is the shape of the answer: **VRAM is retired as a design consideration for this map entirely**, including for the `--consume`-style follow-ons. The binding resources are CPU (ticket 06) and the torch pin (ticket 13/17).

### Two bugs found on the way, both of which would have cost a box trip

- **Scene glob missed a path level** in `vram_probe.py` *and* `audioguard_probe.py`: the canonical root is `data/hm3d/scene_datasets/hm3d/<split>`, which is what `box_inventory.py` counted the 100 val / 10 minival meshes from. Both fixed, and the finder now fails diagnostically by listing what does exist. `audioguard_probe.py` has never run on the box and would have hit this identically on ticket 16's trip.
- **The simulator row read +0.000 GiB on the first attempt** because an `AgentConfiguration()` with no visual sensors makes habitat-sim log `CreateSceneInstance success ... without renderer` and allocate nothing. The budget now attaches RGB+depth and warns if the cumulative never rises above the CUDA context.

### Deliverables

- `vram_probe.py` — attribution / release / budget / summarize. `--attribute` is stdlib-only and read-only, so it is the "run `nvidia-smi` before anything that wants real VRAM" check the runbook asks for, with process attribution attached.
- `kill_nrun.sh` — signals the workload rather than the wrapper, so the wrapper's `EXIT` trap still emails a complete run report; dry-run by default, never auto-`SIGKILL`s, protects the `ss2` probes by cmdline.
- `runs/ss2-vram/{attribution,attribution-after,budget}.json` on the box, plus a `killed-<ts>.txt` record of what was torn down.

### Left open deliberately

`--resolution` defaults to 480x640. Ticket 09 has not specified the runner's sensor resolution and framebuffer VRAM scales with it, so the sim rows are quoted **at that resolution** rather than as resolution-independent facts. At 0.133 GiB the sensitivity is irrelevant to the margin either way.
