# 05 — RACE box inventory

Type: task
Status: resolved
Assignee: Sky
Blocked by: none

## Question

What is actually on the RACE V100 box right now, and what of it is worth keeping through the reset?

## Why it matters

The reset is being executed, not just planned, and the box is now the only execution environment.
Deleting or rebuilding blind is how hours get lost re-downloading 1.2 GB of HM3D or rebuilding habitat-sim for no reason.

It also feeds ticket 04 directly: if an audio-capable habitat-sim build already exists on the box and is sound, that build step is minutes rather than an hour.

## What would resolve it

Record, concretely:
- conda envs present, their Python and numpy versions, and which of them import habitat_sim
- whether any existing habitat-sim build is audio-capable, and on which branch/commit
- disk free, and what the large consumers are
- HM3D copies: where, which splits, whether semantics are present, whether meshes exist for all 20 val scenes (a prior run found only 2 of 20 had meshes)
- model weights already downloaded, and their sizes
- GPU: driver, CUDA version, VRAM free
- CPU core count, since `threadCount` is a free speed knob currently set to 1
- the `soundspaces-spike` build dir state

Deliverable: a short inventory file in the new tree, plus a keep/rebuild/delete call on each item.

## Answer

**Ran on `riftvm` 2026-08-01, exit 0 in 21s** (`python3 .scratch/ss2-clean-room/probes/box_inventory.py`, repo at `e334285`). Full report: `runs/ss2-box-inventory/inventory.{json,md}` on the box.

Nothing blocks the map. Every load-bearing assumption held, one long-standing problem turns out to be already fixed, and two other tickets get real evidence.

| | |
| --- | --- |
| GLIBC | **2.39** (>= 2.29) — the prebuilt `.so` loads. The map called this load-bearing and it had never been measured. |
| CPU | **4 cores** — confirms ticket 04's cut of `threadCount` to a ~4x ceiling. |
| GPU | Tesla V100-SXM3-32GB, driver **580.159.03**, driver max CUDA **13.0**, `nvcc` absent |
| VRAM | **8249 MiB free of 32768** — see the one loose end below |
| Disk | **680.1G free of 773.9G** — the reset has no disk-pressure justification |
| HM3D val meshes | **20/20** |

### Envs

| env | python | numpy | habitat_sim | torch |
| --- | --- | --- | --- | --- |
| `ss2` | 3.9.19 | 1.23.5 | 0.2.2 **audio=True** | 2.0.1+cu117, cuda=True |
| `soundspaces-spike` | 3.9.19 | 1.23.5 | 0.2.2 audio=True | none |
| `ltm-embodied` | 3.9.23 | 1.26.4 | 0.3.3 audio=False | **2.8.0+cu128, cuda=True** |
| `miniconda3` (base) | 3.13.13 | absent | absent | none |

`ss2` is intact and is the env ticket 06's sweep runs in, so that probe can go now.

### Three findings that matter beyond this ticket

1. **HM3D val mesh coverage is 20/20.** The prior record has this at **2 of 20** — it is why `race-scaleup-matrix.sh` grew a mesh preflight and why the R1 smoke ran on `val_mini`. That constraint is gone. `val` holds 100 `.basis.glb` / 36 `.semantic.glb` at 9.3G, `minival` 10 / 4 at 1.1G. Goes to **ticket 08**.
2. **The box already runs a modern torch on this V100.** `ltm-embodied` carries **torch 2.8.0+cu128 with `cuda=True`**, and the driver's max CUDA is 13.0. Ticket 13's recommended option (a) "bump torch" was argued from the driver version; this is the same claim measured on the same GPU. Goes to **ticket 13**, with a sharpened risk noted there.
3. **No MP3D anywhere on the box.** Every listed split is HM3D. Moving datasets means a fresh download, not a re-point. Goes to **ticket 08**.

### Keep / rebuild / delete

- **keep** — GLIBC, HM3D val meshes, disk, GPU.
- **keep** — `ss2`. Ticket 04 built it clean and it is the map's execution env.
- **ticket 10's call** — `soundspaces-spike` (env + 5.5G `~/soundspaces-build`), `ltm-embodied`, `~/.cache/pip` 5.7G, `~/.cache/huggingface` 23.9G, `runs/` 880.3M. This inventory deletes nothing and recommends nothing here beyond reporting the sizes.
- **inferred, worth a look** — `data/` is 19.9G and its splits total ~19.7G with `val` and `versioned_data` **both** reporting 9.3G at 100/36. If those were hardlinked, the parent `du` would have counted them once and read ~10.4G, so this reads as a genuine **~9.3G duplicate copy**. Stated as an inference off `du` arithmetic, not a verified fact — ticket 10 should confirm before acting, and with 680G free there is no urgency.

### One loose end

**Only 8.2 GB of 32 GB VRAM is free** with nothing meant to be running. Not blocking anything in this map — the destination needs CLAP at ~600 MB and an EGL context, and memory (the 7B planner, the VLM) is out of scope — but ~24 GB is held by something unaccounted for. Worth an `nvidia-smi` before any run that wants real VRAM, and it belongs on ticket 10's list.

### Honest note on ordering

Both this ticket and ticket 04 argued this should run **first**, worth ~1 hour. It ran after — `ss2` already exists with torch installed, so 04 had already paid the full build. No harm done (04 went GREEN), and the ordering argument was sound; it simply was not followed. The CUDA finding still lands, just on ticket 13 rather than on 04's wheel choice.

## Comments

### 2026-08-01 — probe built and handed to the box; run it BEFORE ticket 04's gate

This ticket has no off-box half.
Every item on the list is a fact about the V100 and none of it is answerable by reading source, so this session built the probe and handed it over rather than pretending to an answer.

`.scratch/ss2-clean-room/probes/box_inventory.py` — stdlib only, no env of its own, `python3` off PATH is enough:

```
nrun python3 .scratch/ss2-clean-room/probes/box_inventory.py
```

~2–5 min, almost all of it importing `habitat_sim` once per candidate env.
Writes `runs/ss2-box-inventory/inventory.{json,md}` and prints the markdown.
**Resolve this ticket by pasting `inventory.md` back here.**

**It is strictly read-only** — installs nothing, deletes nothing, writes only under `--out-dir`.
That is a deliberate property, not a default: this runs on the box holding the only copy of the HM3D download, and the ticket exists to stop the reset deleting or rebuilding blind.
The keep/rebuild/delete column it prints is a *recommendation*; ticket 10 owns the reset spec and nothing here acts on it.

Coverage against the eight items asked for, plus what each is really for:

| item | how it is read |
| --- | --- |
| conda envs, Python + numpy, which import `habitat_sim` | `conda info --json`, falling back to scanning `~/{mini,ana,miniforge}conda3/envs` because `conda` is off PATH after a RACE pod restart. Each env is probed through **its own `bin/python`** rather than `conda activate`, so no shell-init state can skew the answer. |
| audio-capable? which branch? | enum **member** `RLRAudioPropagationChannelLayoutType.Binaural` (issue #2340 — the spec class is bound even in non-audio builds, so the class proves nothing), plus the five branch-generation methods from ticket 04's note. |
| disk free + large consumers | named candidates only. A blind `du` over `$HOME` on a box with a conda tree and 1.2 GB of meshes is minutes of nothing useful. |
| HM3D copies, splits, semantics, mesh coverage | per-split counts of `.basis.glb` / `.semantic.glb` / `.semantic.txt`, plus the explicit cross-check of episode scenes against meshes. |
| model weights | HF hub cache per repo, torch cache, `models/`. |
| GPU driver / CUDA / VRAM | `nvidia-smi`, and the driver's **max supported CUDA** specifically. |
| CPU cores | `os.cpu_count()`, reported against `threadCount=1`. |
| `soundspaces-spike` build dir state | git HEAD, branch, dirty flag, **submodule SHAs**, whether `build/` exists, and whether `libRLRAudioPropagation*.so` is actually on disk. |

Verified locally before handover, to the same bar ticket 04 set for itself: a bare Mac with no GPU, no glibc and no HM3D still produces a valid report with every section either populated or carrying an explicit `error`, rather than crashing.
That run also caught a real portability bug — `du --block-size=1` is GNU-only and returns nothing on BSD, which silently zeroed every size; it now uses POSIX `du -skx`.

#### Three things it does beyond the list

1. **GLIBC version, checked first and loudly.**
   `libRLRAudioPropagation.so` is a prebuilt Linux-x64 binary needing GLIBC >= 2.29, which the map's Notes call load-bearing but nothing has ever measured.
   One line, and a `BLOCKER` verdict if it fails, because that would invalidate the map's execution-environment assumption where it stands.
2. **Opportunistic `AudioSensorSpec` defaults dump.**
   Only fires if an env already holds an audio-capable build.
   Ticket 06 *blocks* on those defaults and ticket 04 plans an hour-long clean build to get them.
   This does not usurp 04 — an existing env is exactly the "unknown drift" 04 refuses to trust, so **04 stays authoritative** — but agreement between the two is a strong prior, and disagreement is itself a finding.
3. **Demonstrates rather than assumes the branch generation**, by recording which of `maxIRLength` / `irTime` / `directRayCount` exist (ticket 11's rename).

#### Ordering: this should run before ticket 04's gate

Not a blocking edge — 04 runs fine without it — but running 04 first risks burning an hour:

- 04 pins `SS2_TORCH_SPEC=torch==2.0.1` / cu117 as a V100-era **guess**, and its own comment says the value is overridable "because ticket 05's inventory may find a driver that wants a different wheel". This ticket reads the driver's max CUDA, which is what settles it.
- If an audio-capable build already exists and is sound, 04's build step is minutes rather than an hour (this ticket's stated reason for existing).
- If GLIBC < 2.29 or disk is short, 04 fails an hour in on something readable in two minutes.

Cost of getting the order right is ~2 minutes. Cost of getting it wrong is ~1 hour.
