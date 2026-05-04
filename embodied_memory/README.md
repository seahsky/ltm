# embodied_memory — HM3D proof-of-life

Minimum embodied loop for the LTM-Embodied Agent: Habitat ObjectNav on HM3D
feeds the existing `dialogue_memory/` LTM (STM → consolidation → hierarchical
LTM → re-ranking). Independent of the dialogue/MSC path.

This is **not** a paper-grade result. It exists to prove the loop runs and
that the existing LTM machinery transfers to embodied data.

## One-time setup

1. **Sign the Matterport HM3D academic agreement** at
   <https://matterport.com/habitat-matterport-3d-research-dataset>. Note the
   token ID and secret it gives you.

2. **Install conda** (Miniforge recommended on Apple Silicon).

3. **Create the env:**
   ```bash
   conda env create -f embodied_memory/environment.yml
   conda activate ltm-embodied
   ```

4. **Download a small HM3D slice** (uses your Matterport token):
   ```bash
   cp .env.example .env   # then fill in your Matterport token + secret
   bash embodied_memory/scripts/download_hm3d.sh data/hm3d
   ```

## Run proof-of-life

```bash
python -m embodied_memory.run_hm3d_pol \
    --scene <hm3d-val-scene-id> \
    --n-episodes 5 \
    --target chair \
    --out-dir runs/pol-001
```

Outputs land in `runs/pol-001/`:
- `episode_<i>.json` — per-step trajectory, captions, retrievals, reranker decisions
- `summary.json` — run-level metrics + pass/fail per criterion

### Pass conditions (asserted by the runner)

1. After episode 1, fine-layer count ≥ 1.
2. From episode 2 onward, every re-ranking call retrieves ≥ 1 LTM record.
3. At least one episode where the reranker's top-1 candidate is **not** the
   planner's raw top-1.
4. All 4 modules invoked at least once across the run.
5. No crashes on macOS CPU sim.

### Cached-trajectory escape hatch

If habitat-sim is uncooperative on Apple Silicon, swap to a pre-recorded bundle:

```bash
python -m embodied_memory.run_hm3d_pol \
    --mode cached \
    --cached-bundle path/to/bundle.npz \
    --n-episodes 5 \
    --out-dir runs/pol-cached
```

The pipeline downstream of `EpisodeSource` is unchanged.

## Module map

| File | Role |
|---|---|
| `episode_source.py` | `EpisodeSource` interface, dataclasses (`Step`, `Episode`) |
| `habitat_env.py` | Live HM3D ObjectNav env via Habitat-Lab |
| `cached_source.py` | Cached trajectory replay (escape hatch) |
| `perception.py` | CLIP keyframe encoder + semantic-sensor captioner |
| `frontier_planner.py` | 2-D occupancy grid + frontier candidate generation |
| `memory_bridge.py` | Adapter onto `dialogue_memory` LTM/Consolidator/PatternCluster/Reranker |
| `episode_runner.py` | env ↔ planner ↔ bridge loop, JSON logging |
| `run_hm3d_pol.py` | CLI entry point |

## What this slice does NOT do

- ReMEmbR backbone (CLIP + semantic-sensor captions stand in for now)
- Multi-scene lifelong eval
- Baselines / comparisons
- Training `dialogue_memory.train_predictor` / `train_scorer` on embodied data
- VLM planner (frontier heuristic instead)
- Coarse layer affordance learning (seeded from HM3D-Semantics labels)

See `Research Proposal_Embodied Agent.md` for the full method.
