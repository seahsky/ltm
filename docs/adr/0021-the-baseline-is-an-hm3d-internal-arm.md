# The paper's baseline is an internal HM3D arm; SAVN-CE leaves the critical path

**Status:** accepted (2026-09-01, on the direction "use HM3D for the baseline").

The row every earshot number is quoted against is the **`full` arm of `tools/ablation_sweep.sh`**: the complete system, on HM3D, on the ADR-0017 windowed task, at the sweep's own scenes and seed.
It is an internal baseline.
SAVN-CE remains a reproduced reference under [ADR-0015](0015-mp3d-on-the-box-for-a-reproduced-reference.md) and is **removed from the critical path**: nothing in the paper's results section waits on it, and the Matterport licence stops being a blocker for anything.

## Why the question was live

The paper (§III-A) commits to SAVN-CE's task formulation and cites Zeng et al., CVPR 2026.
That made "compare against SAVN-CE" look like a requirement rather than a choice, and the Matterport Terms of Use — signed 2026-08-31, not yet returned — sat in front of it.

It cannot be met by re-pointing a flag, and the reason is worth writing down once so it is not re-litigated.
SAVN-CE is MP3D-bound in three independent places, and each would have to be rebuilt separately:

| binding | where | what HM3D would cost |
|---|---|---|
| the config | `savnce_baselines/magnet/config/mp3d/rgbd_ddppo_clean.yaml` (`tools/savnce_eval.sh:47`) | a new config, and the scene-dataset plumbing under it |
| the episodes | the published SAVN-CE episode dataset, authored on MP3D scenes | a re-derived episode set, which is a second episode builder |
| the checkpoint | trained on MP3D; the authors report 4xA800 for 14 days | **a training run this project has no compute for** |

The third is decisive on its own.
The reserved compute is a 3-month V100-32GB.
There is no version of "run SAVN-CE on HM3D" that does not begin with retraining it, and retraining a baseline is not reproducing it.

## What this decision actually is

It is a decision about **which comparison the paper's results rest on**, and there were only two coherent answers.

*Quote SAVN-CE's published MP3D numbers beside earshot's HM3D numbers.*
ADR-0015 forbids this in as many words: the reproduced reference is "never paired with, subtracted from, or tabled beside any earshot number."
Different dataset, different episodes, different sensors, different success radius.
A table that puts 37.7 next to an earshot SR invites exactly the subtraction ADR-0015 exists to prevent, and no footnote survives a reader skimming a table.

*Make the baseline internal.*
Taken.
The ablation arms are already the right shape for it: each removes one component the paper claims is load-bearing, and every arm shares its episodes with the baseline, so `tools/episode_diff.py` runs an exact McNemar over the pairs rather than comparing two populations.
The measured 16.2% per-episode flip rate on byte-identical reruns is what makes that pairing not a refinement but the only test with power at this scale.

## What it costs, stated plainly

**There is no external number in the results section.** The paper reports what each component contributes to its own system and does not claim to beat a published method. That is a weaker headline than a cross-method win and it is the honest one available: the alternative was a comparison across two datasets that the project's own ADR already ruled inadmissible.

**The "weak baseline" risk does not disappear, it changes shape.** An internal baseline cannot be weak relative to the literature, because it makes no claim relative to the literature. What it can be is *uninformative* — if every ablation arm lands inside the 6.68-point paired MDE, the table says nothing about any component. That is a result the sweep can produce and the paper would have to report.

**§III-A needs rewriting either way.** It currently adopts SAVN-CE's task formulation. The task actually run is ADR-0017's windowed anomaly response on HM3D, which is not SAVN-CE's task, and the citation stays as related work rather than as the formulation.

## What it buys

The Matterport licence, `tools/savnce_licence_wizard.sh`, `savnce_bootstrap.sh`, `savnce_eval.sh` and the pinned submodule all stay exactly as they are and all move **off the critical path**.
Sending the signed form remains worth doing — SAVN-CE reproduced on its own benchmark is a real contribution to the related-work section, and SWS came from that literature — but no result now waits on a mailing list reply.

## Considered and rejected

- **Supersede ADR-0015.** Rejected. Its reasoning is untouched and its prohibition is what this ADR relies on. It is narrowed the same way ADR-0007 was: still accepted, still binding, no longer blocking.
- **Run earshot's controller on SAVN-CE's MP3D benchmark instead.** ADR-0015 already names this as "a defensible future move" that "needs its own ADR". It still does. It is not this ADR, and it does not fit before the submission.
- **Use AV-Nav or SAVi as the HM3D baseline.** Same three bindings, same missing checkpoint. Both are MP3D-trained arms *inside* the SAVN-CE paper.
- **A random-walk or frontier-only baseline.** Rejected as the headline row: it is not a baseline, it is a floor, and `scan-only` already measures something more informative for the same rendering cost. Nothing stops it being added later as a floor.

## Consequences

1. `earshot/tools/ablation_sweep.sh`'s `full` arm is the baseline of record. Its provenance file, scenes and seed are the baseline's definition.
2. No result in the paper depends on MP3D, on the Matterport licence, or on the `savnce` conda env.
3. ADR-0015's prohibition is unchanged and now load-bearing in both directions: SAVN-CE numbers are not tabled beside earshot numbers, and earshot is not measured on MP3D.
4. §III-A is rewritten to state the task this repo runs. SAVN-CE is cited as related work and as the origin of SWS.
