#!/usr/bin/env python3
"""Is the anomaly source on the agent's floor? V100 + ``ss2``, or any box with the dataset.

    python .scratch/ss2-clean-room/probes/check_episode_geometry.py runs/ss2-ep1

**The question ticket 26's first full run raised and could not answer from its own
record.** That episode walked 15.5 m cleanly, 0 of 62 forwards colliding, and still never
reached the source: ``min_d2source`` 1.88 m against a 1.0 m oracle radius,
``source_is_visible`` false at every one of 153 steps, and the measured RMS falling from
0.0407 at the onset to 0.0121 at the sub-budget abort. It got *quieter* for 120 steps.

The frame was the first suspect and the box **refuted** it: ``test_audio_box`` pins the
lateral cue agent-frame with no compensation term (ILD +0.081 facing / -0.065 turned), and
``test_agent_frame_box`` measures ``move_forward`` at 0.00 degrees of error on all four
yaws. The cue is right.

The second suspect is geometry, and it is this project's **known** structural break: the
`anommxv` matrix was invalidated in part because the source could be a floor away, which
fabricates the audio and inverts the feasibility of the task. The builder's floor rule
(``dataset._qualifying_sources``) measures ``max_dy_m`` against ``primary_anchor`` — the
primary goal view point nearest the start — and **not** against the agent's start pose. So
a source can satisfy the rule while sitting a storey below where the agent begins.

The run's own numbers are what raised it: ``source_xyz`` y is -0.536 while the episode's
start is y 2.064, a 2.6 m gap, and the audit nonetheless recorded ``source_dy_m`` 0.000.

This prints the three heights and says which reading they support. It renders nothing,
opens no simulator and needs no GPU — it reads the published dataset and the run's audit.

Two outcomes, and they lead to different work:

- **CROSS-FLOOR** — the start sits a storey above both the source and the primary goal.
  Criterion 5 was unreachable by construction, the climb was never given a winnable
  episode, and the floor rule needs to anchor on the start as well as the goal.
- **SAME FLOOR** — the geometry is sound and the cause is the non-line-of-sight gradient:
  a genuine limit of the realizable arm rather than a bug in the builder.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

from earshot.task.dataset import primary_anchor  # noqa: E402
from earshot.task.episodes import (  # noqa: E402
    find_scenes_dir,
    find_split_dir,
    load_scene,
    scene_label,
)

# A storey. HM3D navmesh heights are floor level, so a gap this size is a different level
# rather than a step or a threshold. Deliberately generous: the builder's own rule is
# 1.0 m, and anything between the two is reported as AMBIGUOUS rather than judged.
STOREY_M = 1.5
BUILDER_MAX_DY_M = 1.0


def _load_audit(run_dir, index):
    path = os.path.join(run_dir, "episodes", "ep{:04d}.audit.json".format(index))
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def main(argv):
    run_dir = argv[0] if argv else "runs/ss2-ep1"
    index = int(argv[1]) if len(argv) > 1 else 0

    audit = _load_audit(run_dir, index)
    source = audit.get("source_xyz")
    if source is None:
        print("no source_xyz in the audit — nothing to check")
        return 2
    source_y = float(source[1])

    label = scene_label(audit["scene_id"])
    dataset = load_scene(find_split_dir("val"), label, scenes_dir=find_scenes_dir())
    # The audit records the loop index, and `run` builds its episodes in dataset order,
    # so episode_index indexes the same list the runner walked.
    episode = dataset.episodes[int(audit.get("episode_index", index))]

    start_y = float(episode.start_position.y)
    anchor = primary_anchor(episode)
    anchor_y = float(anchor.y)
    # `view_points` is a method, not a property (`episodes.Episode.view_points`).
    view_ys = [float(vp.position.y) for vp in episode.view_points()]

    print("\n  run:      {}  episode {}".format(run_dir, index))
    print("  scene:    {}".format(label))
    print("  category: {}   source at: {}".format(
        episode.object_category, audit.get("localization_arm")))
    print()
    print("  agent start        y {:+.3f}".format(start_y))
    print("  primary anchor     y {:+.3f}   (the goal the builder measures from)".format(
        anchor_y))
    print("  anomaly source     y {:+.3f}".format(source_y))
    if view_ys:
        print("  primary view pts   y {:+.3f} .. {:+.3f}  ({} points)".format(
            min(view_ys), max(view_ys), len(view_ys)))
    print()

    start_to_source = abs(start_y - source_y)
    start_to_anchor = abs(start_y - anchor_y)
    anchor_to_source = abs(anchor_y - source_y)

    print("  |start  - source| {:.3f} m   <- NOT checked by the builder".format(
        start_to_source))
    print("  |start  - anchor| {:.3f} m   <- NOT checked by the builder".format(
        start_to_anchor))
    print("  |anchor - source| {:.3f} m   <- the rule, max_dy_m {:.1f}".format(
        anchor_to_source, BUILDER_MAX_DY_M))
    print("  audit's source_dy_m {:.3f}".format(
        float(audit.get("metrics", {}).get("source_dy_m", float("nan")))))
    print()

    # The measurements the run made, restated here so one paste carries the whole picture.
    metrics = audit.get("metrics", {})
    visible = audit.get("source_is_visible_history") or []
    print("  min_d2source {:.3f} m   source_is_visible true on {} of {} steps".format(
        float(metrics.get("min_d2source_m", float("nan"))),
        sum(1 for v in visible if v),
        len(visible),
    ))
    print("  funnel {}   forwards {} ({} collided)".format(
        audit.get("funnel_stage_name"),
        int(audit.get("forward_summary", {}).get("n_forward", 0)),
        int(audit.get("forward_summary", {}).get("n_collided", 0)),
    ))
    print()

    if anchor_to_source > BUILDER_MAX_DY_M:
        print("  BUILDER BUG — the source violates the rule the builder claims to")
        print("  enforce. |anchor - source| exceeds max_dy_m, so `_qualifying_sources`")
        print("  let through a candidate it should have counted as wrong-floor.")
        return 1

    if start_to_source >= STOREY_M:
        print("  CROSS-FLOOR. The agent starts about {:.1f} m above the source, which is a".format(
            start_to_source))
        print("  storey rather than a step. The floor rule holds between the source and the")
        print("  PRIMARY GOAL and says nothing about the START, so the episode is legal by")
        print("  the builder's own test and unwinnable in practice: a greedy energy climb")
        print("  cannot take stairs, and the loudest direction on the agent's own floor")
        print("  points at whatever aperture the sound leaks through.")
        print()
        print("  => criterion 5 was never reachable. The climb is not what failed.")
        print("  => the fix is in `dataset._qualifying_sources`: anchor the floor rule on")
        print("     the episode start as well as the primary goal.")
        return 1

    if start_to_source > BUILDER_MAX_DY_M:
        print("  AMBIGUOUS. {:.3f} m is above the builder's 1.0 m rule but below a storey.".format(
            start_to_source))
        print("  Look at the scene before concluding either way.")
        return 1

    print("  SAME FLOOR. Start, primary goal and source are within {:.1f} m in y, so the".format(
        BUILDER_MAX_DY_M))
    print("  episode is winnable geometry and the builder is not at fault.")
    print()
    print("  => the cause is the gradient itself. source_is_visible was false at every")
    print("     step, so the sound arrived entirely by reflection and diffraction, and an")
    print("     energy field with no line of sight has no reliable slope toward its")
    print("     source. That is a limit of the realizable arm, not a bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
