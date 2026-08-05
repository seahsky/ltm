# RACE box runbook

How to operate the GPU box this project runs on.

Extracted 2026-08-02 from `scripts/` (44 files, ~8,200 LOC) before ticket 10's reset deletes that directory, plus the measured facts from `.scratch/ss2-clean-room/issues/{04,05,13}`.
Wayfinder ticket: `.scratch/ss2-clean-room/issues/14-extract-race-box-runbook.md`.

**This is a runbook, not a history.**
It records what a future session needs to *operate* the box.
What each driver measured is `PHASE2_ABLATION_REPORT.md`'s job, and the reset decision itself is ticket 10's.

**It is also not a setup script.**
The clean room writes its own thin bootstrap once its root exists.
This is the list of things that script will have to know.

---

## 1. The box

Hostname `riftvm`, a three-month reserved Tesla V100.
Measured 2026-08-01 by `.scratch/ss2-clean-room/probes/box_inventory.py` (read-only, stdlib-only, ~21s); full report `runs/ss2-box-inventory/inventory.{json,md}` on the box.

| | |
| --- | --- |
| GLIBC | **2.39** |
| CPU | **4 cores** |
| GPU | Tesla V100-SXM3-32GB, capability 7.0, 31.73 GB, driver **580.159.03** |
| Driver max CUDA | 13.0 (`nvcc` is **not** installed) |
| Disk | **680.1 G free of 773.9 G** |
| HM3D val mesh coverage | **20/20** |
| MP3D | not present anywhere on the box |
| `$HOME` | `/home/riftuser` (inferred from the conda env path, not directly measured) |

Two of these are load-bearing and were assumptions for a long time before anyone measured them:

- **GLIBC >= 2.29 is what makes the audio stack possible at all.**
  `libRLRAudioPropagation.so` is a prebuilt Linux-x64 binary; below 2.29 it fails to link with `undefined reference to pow@GLIBC_2.29` (habitat-sim issue #1810).
  2.39 clears it.
  This is also why the Mac cannot run the simulator: it is a structural exclusion, not a preference.
- **4 cores caps `threadCount`.**
  The audio sensor's `threadCount` defaults to 1 and was long described as "a free speed knob".
  The ceiling here is ~4x, not an order of magnitude.

### The loose end, closed 2026-08-03 (ticket 15)

Only **8.2 GB of 32 GB VRAM was free** at inventory time. It was not a leak: a live 13.7-day `nrun` job (`race-r1-objectnav.sh --tag r1v1`, the R1 S1+ arm in the `ltm-embodied` env) held 24,397 MiB and was still running. Torn down; the card now reads **0 MiB used, 32,495 MiB free**.

Two durable facts came out of it:

- **Usable ceiling is 32,495 MiB = 31.73 GiB**, not the 32,768 nameplate. The ~273 MiB gap is ECC reserve (ECC is Enabled).
- **The clean room's whole stack is 5.5 GiB co-resident** (1.3 GiB without the R2 captioner), so VRAM is not a constraint on this build. Live audio costs **exactly zero** VRAM — RLR propagation is CPU-side, so its budget is CPU (ticket 06), never memory.

Still run a check before anything that wants real VRAM, but run the one that attributes:

```bash
python3 .scratch/ss2-clean-room/probes/vram_probe.py --attribute   # stdlib-only, read-only, any env
bash .scratch/ss2-clean-room/probes/kill_nrun.sh                   # dry run; --yes to tear down
```

`nvidia-smi` alone is not enough: its process table is what ticket 05 captured and discarded, and a leaked habitat-sim holds an EGL *graphics* context that `--query-compute-apps` does not report.

### Large consumers (sizes as of 2026-08-01, for `du` context only)

`~/.cache/huggingface` 23.9 G · `data/` 19.9 G · `~/soundspaces-build` 5.5 G · `~/.cache/pip` 5.7 G · `runs/` 880.3 M.

`data/` reads as holding a **~9.3 G duplicate**: `val` and `versioned_data` both report 9.3 G at 100/36 files, and hardlinks would have made the parent `du` count them once.
That is an inference off `du` arithmetic, not a verified fact, and with 680 G free there is no urgency.

---

## 2. Getting a working shell after a pod restart

A RACE pod restart drops the `conda` CLI from `PATH`, but `~/miniconda3` survives.
Restore the hook:

```bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ss2
```

**Source it, never execute it.**
`conda activate` only sticks in the calling shell, so a bootstrap script that activates an env must be sourced.
`scripts/race-setup.sh` enforced this by refusing to run when executed directly:

```bash
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "ERROR: source this script, don't execute it."
  exit 1
fi
```

Whatever the clean room writes in its place should keep that guard.
The inverse guard is equally real: `setup-vm.sh`, `verify-setup.sh` and `setup-anomaly-vm.sh` all refused to be **sourced**, because their `exit` calls would kill an interactive session.

`race-setup.sh` did **not** export `PYTHONPATH`.
Every driver therefore set it itself, so that scripts invoked as `python <pkg>/scripts/<t>.py` from the repo root could import the package:

```bash
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
```

---

## 3. Environments on the box

| env | python | numpy | habitat_sim | torch |
| --- | --- | --- | --- | --- |
| **`ss2`** | 3.9.19 | 1.23.5 | 0.2.2, **audio=True** | 2.2.2+cu118 (was 2.0.1+cu117 until ticket 13) |
| `soundspaces-spike` | 3.9.19 | 1.23.5 | 0.2.2, audio=True | none |
| `ltm-embodied` | 3.9.23 | 1.26.4 | 0.3.3, audio=False | 2.8.0+cu128 |
| `miniconda3` (base) | 3.13.13 | absent | absent | none |

**`ss2` is the one that matters.**
It is the clean room's execution environment: it holds an audio-capable habitat-sim, torch on the GPU and the CLAP stack in one interpreter, which is the finding that killed the old two-env split (ticket 04).

`soundspaces-spike` is an earlier feasibility artifact with unknown drift.
Ticket 10 removes it as a footgun, precisely so nobody activates it by mistake and gets a torch-less env that still imports habitat-sim.

`ltm-embodied` belongs to the tree being deleted.
It is also the reason the `numpy<1.24` pin is one careless command away from breaking: it carries numpy 1.26.4 on the same box.

### The known-good version set

The first fully verified set, recorded by ticket 13 on 2026-08-02:

| | |
| --- | --- |
| python | 3.9.19 |
| numpy | 1.23.5 |
| torch | 2.2.2+cu118 (CUDA build 11.8) |
| transformers | 4.57.6 |
| scipy | 1.13.1 |
| habitat-sim | 0.2.2 @ `4f61e321` (`RLRAudioPropagationUpdate`, stock, no patches) |
| rlr-audio-propagation | `4fd446b4` |

**This table is a record; the pin lives in `earshot/tools/ss2-constraints.txt`.**
It used to say "a record of what worked once, not a lockfile", which was accurate until ticket 17's constraints file existed and false the moment it landed (ticket 20).
Nine exact versions are now passed as `-c` to every `pip install` in `earshot/tools/bootstrap_ss2.sh` — the numpy layer, habitat-sim's own `requirements.txt`, torch, and the CLAP stack — and habitat-sim is reset to the SHA rather than to whatever the branch serves.
Read this table for context and change versions in the constraints file, never here.

The drift it closed: `transformers` resolved to 4.57.6 against a frozen `torch==2.0.1` and silently disabled its own PyTorch backend, which made `ClapModel` a dummy object that imports fine and cannot instantiate.
Python 3.9 bounds one axis for free — transformers 5.x requires >= 3.10, so resolution is capped at the 4.x line — and the governing rule for the rest is **pin where failure is silent, leave ranges where failure is loud**, which is why the conda side (`python=3.9`, `cmake=3.14.0`, `gcc_linux-64=10.*`) is deliberately still ranged.

Two constraints worth knowing before touching pins:

- **numpy must stay < 1.24.** The 2022-era habitat-sim tree dies on numpy 2.x. Pass `-c <constraint-file>` containing `numpy<1.24` to *every* `pip install` in the env, and assert the pin afterwards rather than assuming it.
- **torch >= 2.1 or transformers turns itself off.** transformers gates `is_torch_available()` on 2.1.0. torch declares no numpy dependency, so moving it cannot disturb the numpy pin. cu118 is the last CUDA line where the V100's sm_70 is a first-class target.

---

## 4. How `ss2` and `~/ss2-build` were actually built

The real recipe is `earshot/tools/bootstrap_ss2.sh` (moved there from `.scratch/ss2-clean-room/probes/oneenv_gate.sh` by ticket 20 — two copies of a build recipe is a drift trap, and `.scratch` is not where an operator looks).
It was run once as `oneenv_gate.sh` on 2026-08-01 (exit 0, 24m50s including the habitat-sim build).
It is idempotent: a re-run skips the build when an audio-capable `habitat_sim` already imports **and** is built from the pinned SHA, which is why ticket 13's re-run took 1m49s.
The SHA half of that condition is new — the old skip fired on importability alone, which made the pin inert on every re-run against an existing env.

**`scripts/race-soundspaces-spike.sh` is the ancestor recipe, not this one.**
It builds a *different* env (`soundspaces-spike` in `~/soundspaces-build`) and never layers torch or transformers on top, which was the whole question.
Both it and the spike env are removed by the reset.

The steps, in order:

1. **Env.** `conda create -n ss2 python=3.9 cmake=3.14.0`.
   The cmake pin is SoundSpaces' own, and it also dodges CMake 4.x refusing the 2022 dependency tree.
2. **numpy first, before anything else can resolve it.**
   Write `~/ss2-build/np-constraint.txt` containing `numpy<1.24`, then `pip install "numpy>=1.16.1,<1.24" numpy-quaternion -c <that file>`.
   `numpy-quaternion` is needed because `import quaternion` must precede `import habitat_sim` (habitat-sim issue #1813).
3. **Toolchain.** `conda install -c conda-forge 'gcc_linux-64=10.*' 'gxx_linux-64=10.*' sysroot_linux-64`, then export `CC` / `CXX` to `$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-{cc,c++}`.
   The only attested toolchain for this tree is gcc 7 to 10; modern gcc 12/13 postdates the vendored magnum/corrade headers.
4. **System packages** (see section 7 for why this exact line).
5. **habitat-sim.**
   `git clone https://github.com/facebookresearch/habitat-sim.git ~/ss2-build/habitat-sim`, checkout `RLRAudioPropagationUpdate`, then `git submodule update --init --recursive`.
   It **must** be a git clone, not a tarball: `setup.py` only auto-inits submodules inside a git repo, and the closed-source audio engine *is* a submodule.
   Confirm `src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so` exists before building.
   Then `pip install -r requirements.txt -c <np-constraint>`.
6. **Build.** `python setup.py install --headless --audio`.
   `--headless` because the box has no display; `--audio` is what builds RLRAudioPropagation.
   `--with-cuda` is deliberately omitted: the audio engine is CPU and EGL needs no CUDA.
   Three environment knobs are required (section 7 explains each):
   ```bash
   export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
   export CMAKE_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib"
   export CMAKE_INCLUDE_PATH="/usr/include"
   export CPATH="$HOME/ss2-build/include-shim"   # symlinks to /usr/include/{EGL,KHR,GL,X11}
   ```
7. **torch.** `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118 -c <np-constraint>`.
8. **CLAP stack.** `pip install "transformers>=4.40,<5" scipy soundfile -c <np-constraint>`.
9. **Re-probe the audio build after every layer.** Layering is the actual question, so a probe that only runs at the end cannot tell you which layer broke it.

**The verdict is `env_check --strict`, and it is the same assertion the runtime makes** (ticket 24).
Stage 8 runs `python -m earshot.env_check --strict`; `task/`'s entry point calls `assert_env()`.
One implementation, two callers — an assertion that lived only in the gate could not run at episode time, which is exactly when a drifted env produces results instead of an error.

Run it standalone on the box any time, in seconds and with nothing to install:

```bash
conda activate ss2
cd <repo> && PYTHONPATH=. python -m earshot.env_check --strict          # + --clap for CLAP
PYTHONPATH=. python -m earshot.env_check --provenance \
    --constraints earshot/tools/ss2-constraints.txt --freeze <a pip freeze>
```

Every probe is **capability-shaped**: it allocates on the GPU and reads the result back, resolves the audio enum **member**, instantiates CLAP and reads a finite logit.
A version table would have printed green through the whole of ticket 13, and `--strict` treats a probe that *could not run* exactly as it treats one that failed.
The ticket-04 probe is kept as stage 9 for the one thing `env_check` does not do: open a scene and render.

**habitat-lab is deliberately not installed.**
Ticket 04 measured it rather than requiring it, and it is still not importable (`No module named 'habitat_sim.robots'`).
The clean room drives `habitat_sim` directly, so the new tree owns three small pieces habitat-lab used to supply: ObjectNav `.json.gz` episode loading, `sim.make_greedy_follower()` steering, and the SPL/SoftSPL arithmetic.

**The build is patch-capable and currently stock.**
`bootstrap_ss2.sh` applies any `*.patch` in `.scratch/ss2-clean-room/probes/patches/` before building and records the result in `~/ss2-build/applied-patches.txt`, so box state stays reproducible.
Nothing is applied today, and ticket 09 ruled no fork, so the directory is expected to stay empty.

### Verifying an audio build

Probe the enum **member**, not the class.
`AudioSensorSpec` is bound even in non-audio builds (habitat-sim issue #2340), so its presence proves nothing:

```python
import quaternion            # must precede habitat_sim (issue #1813)
import habitat_sim, habitat_sim.sensor
t = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType
assert t is not None and hasattr(t, "Binaural"), "built WITHOUT --audio"
```

Same class of trap on the CLAP side: `import ClapModel` succeeds against a disabled torch backend because transformers substitutes a DummyObject that only raises on *instantiation*.
Probe capability, not importability.

### The hermeticity gate (ticket 10 phase 2)

The gate the irreversible deletion commit hangs off, and the one box procedure that takes the repo apart while it runs:

```bash
source earshot/tools/notify/notify-run.sh
nrun bash earshot/tools/hermeticity_gate.sh --tag hermetic-1
```

It moves everything phase 3 deletes out of the repo, runs the box suite and one smoke episode with them gone, moves them back, and writes `hermeticity.json` into the run directory so `python -m earshot.task.smoke --run-dir <dir>` can answer criterion 9 from an artefact instead of from memory.

Three things worth knowing before running it:

- It **refuses a dirty tree**, because an uncommitted edit inside a moved path cannot be told apart from a restore failure afterwards.
- Restore is an EXIT trap, so a failure or a Ctrl-C repairs the tree. After a `kill -9` there is no trap: every moved path is tracked, and the recovery line is printed into the log *before* anything moves — `git checkout -- embodied_memory dialogue_memory scripts …`.
- `--dry-run` does the move and the restore and stops. Use it to rehearse; it needs no env and no GPU.

**A red gate is the gate working.** Fix the leak, restore, repeat.

---

## 5. Running things unattended

`nrun` wraps any command so it survives an SSH disconnect and emails a report when it ends, on success, crash, or Ctrl-C.

```bash
source earshot/tools/notify/notify-run.sh   # defines nrun; safe to source, never exits your shell
nrun bash <driver> --tag <t>                # self-detaches (nohup + background)
tail -f runs/nrun-*.out
```

The trio moved to `earshot/tools/notify/` when ticket 10 carried it out of `scripts/`, which phase 3 deletes.
Ticket 27 found all three of its self-references still written against the old location — `nrun` dispatched at `earshot/tools/scripts/notify-run.sh`, the emailer at `earshot/tools/scripts/notify_email.py`, and `.env` was read from `earshot/tools/` — so `nrun` printed a pid and ran nothing, and a foreground run emailed nothing while reporting success.
Every path is derived from the script's own location now, and `earshot/tests/mac/test_notify.py` runs the trio inside a skeleton holding only what survives the reset.

- **Do not prefix `nohup` yourself.** `nrun` is a shell function, and `nohup` cannot launch functions. `nrun` already does the `nohup` + background + `disown`.
- For a **foreground** run: `bash earshot/tools/notify/notify-run.sh <command...>`.
- The wrapper's exit code is **always** the wrapped command's. A notifier failure never changes it.
- Output is tee'd to `runs/notify-<tag>-<timestamp>.log`; the tag is the first `--tag` value in the args, else the first non-`bash`/`python` argument's basename.

`notify-run.sh` is the single source of truth for `nrun` and is safe to source: it defines the function and returns before any `set` or `exit` runs.
An earlier version hit `exit 2` on a no-arg source and closed the session.

### What it needs from `.env`

`.env` lives at the repo root, is gitignored, and is scaffolded from `.env.example`:

```
RESEND_API_KEY=re_xxxxxxxx
NOTIFY_EMAIL_TO=you@example.com
NOTIFY_EMAIL_FROM=onboarding@resend.dev   # Resend free tier: sends to your own account email, no domain needed
# NOTIFY_LOG_TAIL_LINES=400
MATTERPORT_TOKEN_ID=
MATTERPORT_TOKEN_SECRET=
```

Unconfigured is fine: the run works, there is just no email.
`NOTIFY_DISABLE=1` skips sending entirely.

`earshot/tools/notify/notify_email.py` is **stdlib-only on purpose**, so it adds no dependency to the env.
It posts to `https://api.resend.com/emails` with the gzipped log attached, caps the attachment at 35 MB (Resend's request limit is 40 MB) and truncates to the log tail beyond that, and **always exits 0**.

---

## 6. Data on the box

### HM3D layout

```
data/hm3d/scene_datasets/hm3d/{val,minival,versioned_data/...}
data/hm3d/datasets/objectnav/hm3d/v1/{train,val,val_mini,minival}/
```

Coverage: `val` holds 100 `.basis.glb` / 36 `.semantic.glb` at 9.3 G; `minival` 10 / 4 at 1.1 G.
ObjectNav val mesh coverage is **20/20**, which retires the long-standing "only 2 of 20 meshes" constraint that forced earlier work onto `val_mini` and grew a mesh preflight into the scale-up driver.

### Two symlink traps

1. **`val` -> `minival`.**
   ObjectNav v1 episode JSONs hardcode scene paths as `val/<scene>/...`, but the `hm3d_minival_full` download group lays meshes under `minival/<scene>/...`.
   The minival 10 are a strict subset of val, so `ln -sfn minival <scene_datasets>/hm3d/val` resolves both without re-downloading.
2. **The absolute symlink that does not survive an rsync.**
   An rsync from the laptop copies `data/hm3d/scene_datasets/hm3d` as an *absolute* symlink that does not resolve on the box.
   Repair by re-pointing it relatively, from inside the directory:
   ```bash
   cd data/hm3d/scene_datasets && ln -sfn ../versioned_data/hm3d-0.2/hm3d hm3d
   ```
   `race-setup.sh` did this on every source, detecting the dangling case as "is a symlink, but does not exist".

### Downloading more HM3D

Requires signing the Matterport academic agreement at <https://matterport.com/habitat-matterport-3d-research-dataset> and putting the token pair in `.env`.
Meshes come from habitat-sim's own downloader; the ObjectNav episodes do **not** (they ship as a single zip from habitat-lab's CDN):

```bash
python -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_minival_full \            # or hm3d_val_full for the 36-scene val split
  --no-replace --data-path data/hm3d

curl -fL --retry 3 -o objectnav.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
```

**`--no-replace` is not optional under `nrun`.**
Without it the downloader interactively asks "Replace versioned data?", reads stdin, and a detached process has none, producing `OSError [Errno 9] Bad file descriptor` about 21 seconds in.

### Model weights

The HF cache is `~/.cache/huggingface` (23.9 G).
Pulls should go through `huggingface_hub.snapshot_download`, which resumes a partial transfer and re-fetches only missing or corrupt files, wrapped in a retry loop for a flaky link.
Treat a repo as present only if a snapshot holds both a `*.json` and a real weight file: a lone `blobs`/`refs` skeleton from an aborted pull is a false positive.

The clean room's only measured model dependency is **CLAP** (`laion/clap-htsat-fused`, ~600 MB, peak VRAM 0.713 GB on this box).
The Qwen captioner/planner pair, CLIP and SBERT in that cache belong to the tree ticket 10 deletes.

ESC-50 anomaly and benign clips live under `data/{anomaly,benign}_audio/*.wav` and are gitignored: fetched, not committed.
RIR grids under `runs/**/ *_rir_grid.npz` are **rendered**, never downloaded, and the live-render decision retires them anyway.

---

## 7. Footguns that have cost real runs

Each of these was learned the hard way and is the reason for a specific line above.

**The driver self-update gotcha.**
Bash executes the body it loaded at launch.
A driver that `git pull`s itself updates the file on **disk**, not the running body, so an edit only takes effect on the **second** invocation.
This silently wasted a 10-hour run (`r1spin` executed the pre-anti-spin body at commit `32b3493`).
33 of the `race-*.sh` drivers `git pull` the repo they are running from; exactly one, `race-r1-objectnav.sh`, self-heals it, and that is the pattern to keep:

```bash
_self_before="$(md5sum "$0" | awk '{print $1}')"
git pull --ff-only || exit 1
_self_after="$(md5sum "$0" | awk '{print $1}')"
if [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
  export _REEXEC=1; exec bash "$0" "$@"
fi
```

**The conda `set -u` trap.**
conda's compiler-package hooks dereference `CONDA_BACKUP_*` variables that are unset on first install.
Observed on this box 2026-06-11: `conda install gcc_linux-64` triggers a reactivate that sources the brand-new `deactivate-gxx_linux-64.sh`, which raises `CONDA_BACKUP_CXX: unbound variable` and, under `set -u`, kills the whole script.
Run **every** conda state change with nounset off:

```bash
set +u
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
conda activate "$ENV_NAME"
set -u
```

The same applies to sourcing a bootstrap script from a `set -u` driver: `set +u; source ...; set -u`.

**SIGPIPE under `pipefail`.**
Two separate bugs, same cause: a downstream command exiting early SIGPIPEs the writer, and under `pipefail` the pipeline "fails" despite a good match.

- `conda env list | grep -q <name>` turned found-it into a failure. Check `[ -d "$MINICONDA/envs/$ENV_NAME" ]` instead.
- `ldd --version | head -1 | grep ...` flunked a healthy GLIBC 2.35, because the `|| echo 0.0` fallback then appended a bogus line that `sort -V` picked. Use `getconf GNU_LIBC_VERSION | awk '{print $2}'`: both consume their whole input, so nothing SIGPIPEs.

**The apt line, and why it is written this way.**

```bash
sudo apt-get install -y --no-install-recommends \
  libjpeg-dev libglm-dev libgl1 libglx-mesa0 libegl1-mesa-dev mesa-utils \
  xorg-dev freeglut3-dev libglvnd-dev
```

`libgl1-mesa-glx` was **dropped in Ubuntu 24.04 (Noble)**, split into `libgl1` + `libglx-mesa0`.
Those two also exist on 20.04 and 22.04, so this line resolves across releases.
The build is EGL-headless and never needs the old GLX metapackage.
Preflight for these in seconds rather than discovering them 40 minutes into a build.

**Three cmake/compiler knobs the conda toolchain makes necessary.**

- `CMAKE_LIBRARY_PATH` / `CMAKE_INCLUDE_PATH`: the conda cross-toolchain's triplet (`x86_64-conda-linux-gnu`) stops cmake searching Ubuntu's multiarch directory, so magnum's `find_package(OpenGL)` misses the GLVND libraries even with `libglvnd-dev` installed.
- `CPATH` pointing at a **shim** directory of symlinks to `/usr/include/{EGL,KHR,GL,X11}`: magnum's EGL object library includes `<EGL/egl.h>` via the compiler's default include path, which for the conda gcc is its sysroot (glibc and kernel headers only).
  Expose only the GL header trees; putting all of `/usr/include` on `CPATH` would shadow the sysroot's glibc headers with the host's newer ones.
- A build directory holding `CMakeCache.txt` but **no** `compile_commands.json` is a poisoned half-configure.
  `setup.py`'s argument cache then skips re-running cmake, cannot see env-only changes, and dies on the missing file.
  Delete `build/` and let cmake re-run.

**`AudioSensorSpec` silently swallows unknown keys.**
It is bound `py::dynamic_attr()`, so assigning a field that does not exist on this branch (say `irTime`, renamed to `maxIRLength`) attaches a new Python attribute that is never read, with no error.
`RLRAudioPropagationConfiguration` is **not** bound that way, so bad keys there raise.
Any wrapper must validate keys against the real field list, on the spec specifically.

**Do not read the branch's own docs.**
`docs/AUDIO.md` and `examples/tutorials/audio_agent.py` on `RLRAudioPropagationUpdate` are stale: the tutorial's line 39 raises `AttributeError`, and the doc lists fields the branch's own header does not have.
Read the constructor and the bindings, or the measured defaults dump in ticket 04.
