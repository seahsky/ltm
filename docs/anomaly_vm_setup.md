# Fresh GPU VM setup — anomaly-response arc

One-command bootstrap for a brand-new GPU box (RACE pod, cloud GPU, Lightning) to run the audio-cued anomaly-response evaluation.
The installer is `scripts/setup-anomaly-vm.sh`.
It reuses the base installer `scripts/setup-vm.sh` and adds only the audio layer, and every step is idempotent (checks for its artifact and skips if already present).

## TL;DR

```bash
git clone https://github.com/seahsky/ltm.git && cd ltm
bash scripts/setup-anomaly-vm.sh --yes
```

That installs Miniconda, the `ltm-embodied` conda env, HM3D, the ReMEmbR weights, the CLAP detector weights, and the ESC-50 clips, then runs a free sanity suite.
After every restart, re-bootstrap the shell with `source scripts/race-setup.sh` (fast — it just restores conda and env vars).

## Why a GPU VM is enough (the two-environment split)

Audio is handled in two separate environments so the main loop stays a plain navigation simulator:

- **Main loop** (`ltm-embodied`): headless `habitat-sim` + CLIP/CLAP/SBERT + faiss + transformers (ReMEmbR: a 2B captioner, a 7B planner).
  Runtime audio is a pre-rendered room-impulse-response (RIR) grid convolved in O(1) with `scipy` — **SoundSpaces is never imported at runtime.**
- **Render env** (`soundspaces-spike`): SoundSpaces 2.0, built once (~1 h) and used only to render a new RIR grid at an anomaly source.

The bulk of the compute needs only the main-loop env.
You need the render env **only** if you render new grids; pre-rendered grids run without it.
So the default `setup-anomaly-vm.sh` skips it — pass `--with-render` when you actually need to render.

## Prerequisites

- **GPU** (an ~24 GB card fits the 2B captioner + 7B planner + CLAP).
  Without a GPU the script still installs the env but skips the weight downloads (`--cpu`).
- **Matterport / HM3D token.**
  HM3D requires signing the agreement at <https://matterport.com/habitat-matterport-3d-research-dataset>, then putting `MATTERPORT_TOKEN_ID` / `MATTERPORT_TOKEN_SECRET` into `.env`.
  The base installer scaffolds `.env` from `.env.example` and pauses the HM3D step until the token is set.
- **Disk**: roughly 15–20 GB for the HM3D minival slice + Qwen weights + CLAP.

## What the installer does

| Step | Action | Idempotent skip-if |
|---|---|---|
| 1 | Checkout `main` (the primary branch) + `git pull --ff-only` | already on branch |
| 2 | `setup-vm.sh` — Miniconda, `ltm-embodied` env from `environment.yml`, HM3D, ReMEmbR weights | env/data/weights already present |
| 3 | Verify audio runtime deps: `scipy` (fftconvolve + wavfile), transformers CLAP classes | scipy importable |
| 4 | Download CLAP detector weights (`laion/clap-htsat-fused`) to the HF cache | cache dir present |
| 5 | Fetch real ESC-50 anomaly + benign clips (`fetch_anomaly_clips.py --include-benign`) | `data/{anomaly,benign}_audio/*.wav` present |
| 6 | (optional, `--with-render`) build the `soundspaces-spike` render env | audio-capable habitat imports |
| verify | Run the anomaly-response pure-logic + wiring tests (no GPU/data needed) | — |

Flags: `--with-render`, `--skip-data`, `--skip-models`, `--cpu`, `--yes`, `--env-name <n>`, `--branch <n>`.
Env overrides: `LTM_ENV_NAME`, `REMEMBR_CLAP_MODEL`, `CONDA_DIR`.

## After setup — first runs

Re-bootstrap the shell (every new session/restart):

```bash
source scripts/race-setup.sh
```

Cheap wiring smoke (real ReMEmbR, 1 scene — confirms the controller fires end-to-end):

```bash
nrun bash scripts/race-anomaly-response.sh --scene TEEsavR23oF --class alarm --category bed
```

The Phase-0 `$0` gates (run before any paid matrix — see `docs/anomaly_response_buildplan_2026-07-12.md`):

```bash
python embodied_memory/scripts/diagnose_energy_gradient.py --grid runs/<tag>/<grid>.npz   # G0.4 climbability
python embodied_memory/scripts/diagnose_room_clip_cosines.py --scene all                  # G0.1 room cosines
nrun bash scripts/race-convolved-anomaly-gate.sh --device cuda                             # G0.2/G0.3 gate calib
```

Powered matrix (the systems + memory headline) and the query-fix A/B:

```bash
nrun bash scripts/race-anomaly-response-matrix.sh --split val_mini --tag-prefix anomvm
nrun bash scripts/race-anomaly-response-matrix.sh --split val_mini --query-expansion prf --tag-prefix qfix
```

## If a step fails

The steps are independent and each is safe to re-run.

- **HM3D skipped** ("token not set"): fill `.env`, then `bash embodied_memory/scripts/download_hm3d.sh data/hm3d`.
- **Weights skipped** (no GPU at install time): on the GPU box run `python models/download_remembr_models.py` and re-run step 4's CLAP one-liner (printed by the script).
- **Clips missing**: `python embodied_memory/scripts/fetch_anomaly_clips.py --include-benign`.
- **Render env**: `bash scripts/race-soundspaces-spike.sh` (isolated; it never touches `ltm-embodied`).

## Notes

- The ESC-50 clips and HM3D data are gitignored — they are fetched, not committed.
- `run_hm3d_pol.py --backbone remembr` is the real backbone; omitting `--backbone remembr` silently uses the frontier stand-in.
- Keep the headline runs on the frozen local backbone; hosted (NVIDIA Build API) models are for offload/diagnostics only and break cross-quotability.
