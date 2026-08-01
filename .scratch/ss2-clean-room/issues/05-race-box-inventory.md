# 05 — RACE box inventory

Type: task
Status: claimed
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
