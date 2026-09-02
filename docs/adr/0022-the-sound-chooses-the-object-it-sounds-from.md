# The sound chooses the object it sounds from

**Status:** accepted (2026-09-02, on the direction "anchor the placement to the class and widen the class set").

`task.dataset.place_anomaly_source` now prefers the object category the anomaly class is anchored at.
The preference is rule 4's first sort key, above the existing decoupling and proximity keys.
It is a **preference, not a filter**: when no instance of the anchor qualifies under ADR-0010's separation and floor rules, the ranking falls through to exactly the pre-2026-09-02 behaviour and the placement records `at_class_anchor=False`.

**Every result measured before this date was produced under geometric placement, `abl-1` included, and none of them carries forward.**

## What was wrong

ADR-0018's heard axis claims that a semantic store which has heard a class on prior visits knows where to look for it.
The store's own module docstring states the claim: *"the unseen-and-heard cell's whole claim is that the semantic store LEARNED its room association by hearing the class on prior visits."*

The task did not have that structure.
`place_anomaly_source` ranked candidates by `(same_category, separation, category)` — geometry only.
Nothing under `task/` called `vocabulary.anchor_object`; only the analysis tools in `tools/` did.
So in every episode this repo has ever run, the alarm sat at whatever object cleared the separation rules.

A store that learned "an alarm is heard at a bed" was predicting a rule the world did not follow.
The `heard` column could not have measured memory, and the four cells could not have differed for the reason ADR-0018 says they differ.
This was found while wiring the memory prior, not by a run, which is why it is worth recording: the sweep would have produced four numbers and a table, and the table would have been about nothing.

## What changed, exactly

One sort key, and one recorded bool.

```
before:  (same_category, separation, category)
after:   (not at_class_anchor, same_category, separation, category)
```

`SourcePlacement.at_class_anchor` says which branch the episode took, and the runner publishes it as `source_at_class_anchor`.
That field is load-bearing rather than decorative: the memory prior recalls a *category*, so an episode whose source is not at that category is one the prior **could not** have got right.
A readout that pooled the two would charge the memory for episodes that did not follow the rule it learned, which is the same shape of error as counting an unroutable source as a miss.

The lookup from class to category lives in `task/prior_build.py`, not in `task/dataset.py`.
`test_task_dataset.TestAudibilityIsNotScreened` holds that the builder imports nothing from `audio/` — §2.5's rule is that audibility is not screened at build time, and the cheapest way to keep that true is to import none of it.
So the **decision** stays in `place_anomaly_source`'s ranking and only the table is elsewhere; `build_anomaly_episodes` is *told* the category by its caller.
A caller that does not know its class passes nothing and gets the old ordering exactly.

## What it costs

**`abl-1` is stale and must be re-run.** Its five arms, 282 paired episodes each, 9 h 30 m, are all under geometric placement.
The baseline of record (ADR-0021) is the `full` arm of that sweep, so the paper's baseline row and its whole ablation table are re-measured or they are describing a different task from the one the memory arm runs in.
This is the honest cost of having found the problem late rather than a reason not to fix it.

**Yield cannot drop.** The fallback is the old ranking, so any episode that could be built before can still be built.
What can change is *which object* a buildable episode's source sits at, and therefore every SR in the table.

**The decoupling preference is now second.** When a class's anchor is also the primary goal's category, the source and the goal share a category and `same_category` is recorded true.
The oracle detector's table is keyed by object name, so the visual confirm can fire at the wrong instance — the consequence `SourcePlacement`'s docstring already names. It is now reachable more often, it is recorded, and it is a disclosure rather than a surprise.

## The half of the direction that is NOT taken here

The direction was also to **widen the sounding class set beyond the three emergency names**, and that is a separate commit rather than a separate decision: ADR-0018 already made it.

`clips.ANOMALY_CLASSES` is `("baby_cry", "alarm", "glass_break")`.
`alarm` and `baby_cry` both anchor at `bed` and `glass_break` has no vocabulary row, so over HM3D's six goal categories exactly one category any sound is anchored at.
**A predictor with one answer is not a predictor**, so the anchoring above is necessary and not sufficient.

**The class set to widen to is already decided and already measured.**
ADR-0018's amendment of 2026-08-24 accepts a **bank of record** of 11 classes on `clapgate-2` (ESC-50 clips 0-7) and `clapheld-1` (clips 8-15), two disjoint recording sets, each class clearing the bar on both.
Mapped onto anchors it is four categories, which is a real discrimination:

| anchor | classes |
|---|---|
| `bed` | `breathing`, `clock_alarm`, `clock_tick`, `crying_baby`, `snoring` |
| `toilet` | `brushing_teeth`, `pouring_water`, `toilet_flush` |
| `tv_monitor` | `clapping`, `laughing` |
| `chair` | `keyboard_typing` |

So the widening needs no new gate run.
What it needs is the bank in code rather than only in an ADR and a gitignored `runs/bank_of_record.json`, the clips staged for eleven names instead of three, and the run-time classifier reading the bank instead of the three emergency prompts — which is what ADR-0018 means by the goal class being **inferred**.

**One caveat carries and it is ADR-0018's own.** `task/clap_gate.py`'s header records that the gate has never been re-measured on the ADR-0017 waveform: it scored `render_through_ir(ir, clip)`, and the runner now hands CLAP the accumulation buffer's read window, which is looped, rotated by phase, and can be partly full. The bank's *membership* rests on two agreeing runs and is not in doubt; its *separation numbers* describe the pre-ADR-0017 waveform. The runner already records `clap_window_fill` and `clap_after_offset` per episode so the confound is visible, and re-running the gate through `tail.steady_state_render` remains a box job worth doing before any accuracy number from it is quoted in the paper.

`tests/mac/test_prior_build.py::TestTheOneBlockerThatIsLeft` pins the three-class state so the day the sounding set widens, it fails and names what to re-read.

## Considered and rejected

- **Make the anchor a hard filter.** Rejected. Yield would fall by however many episodes have no qualifying instance of their class's anchor, and the drop would be silent and scene-dependent. A preference with a recorded flag measures the same thing and loses nothing.
- **Put the anchor above ADR-0010's separation and floor rules.** Rejected outright. The preference reorders *survivors*; it does not readmit a rejected candidate. A bed 0.5 m from the goal is still too near, and an anchor rule that overrode the geometry would place sources on top of the thing the agent is finding. Under test.
- **Keep geometric placement and let the store learn the empirical distribution.** Rejected. Placement is roughly uniform over categories, so the prior would be near-useless by construction. That is a designed-in null and not worth a night of rendering.
- **Re-anchor the three emergency classes onto three different categories instead of widening.** Rejected. It would make the experiment work by inventing semantics — a baby that cries in the living room because the matrix needs it to — and the vocabulary's affinities were reasoned rather than assigned.
- **Supersede ADR-0018.** Not needed. Its four cells and its two amendments are unchanged; this ADR supplies the world they were always described against.
