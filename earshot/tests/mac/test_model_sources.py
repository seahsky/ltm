"""The probe and the runtime must resolve the SAME checkpoint, and only for the model asked.

**Covers BOTH staged models.** It was CLAP only (`test_clap_source.py`) until `env_check`
grew `probe_clip_instantiable`, and every property below is a property of the resolver
rather than of CLAP -- so the file is parameterised over the pair rather than copied, and
a third staged model joins by adding one row to `MODELS`.

`env_check.py` may import nothing intra-package (ADR-0013: "ticket 17's assertion answers to
the environment, not to the tree"), so it cannot call `task.models.resolve_clap_source` and
carries its own `_clap_source` / `_clip_source` instead. A duplicated resolver is only safe
while the two agree, and "they agree" is the kind of claim this repo has repeatedly found to
have quietly stopped being true. So it is a test.

What the duplication is FOR: `laion/clap-htsat-unfused` ships only `pytorch_model.bin`, and
transformers >= 4.52 refuses `torch.load` on a `.bin` below torch 2.6 (CVE-2025-32434). The
box pins torch 2.2.2+cu118 for the V100's sm_70. Converting the checkpoint to safetensors
once is the fix; both sites have to prefer the converted copy or the probe passes on one
checkpoint and the run loads another.

**And agreement is not correctness.** The first version of both resolvers ignored `model_id`
entirely and returned the staged directory whenever it was complete. The two agreed perfectly
and the tests below passed, while `env_check`'s forced-failure arm asked for
`earshot/definitely-not-a-model`, received the real checkpoint, and reported a finite feature
vector. Only the box gate caught it. Every test in `TestTheStagedCopyMustNameItsModel` exists
because of that, and they check the RESOLVER rather than the pair.
"""

import json
import os
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.env_check import CLAP_LOCAL_DIR as PROBE_CLAP_LOCAL_DIR
from earshot.env_check import CLAP_MODEL_ID as PROBE_CLAP_MODEL_ID
from earshot.env_check import CLAP_STAGED_MARKER as PROBE_MARKER
from earshot.env_check import CLIP_LOCAL_DIR as PROBE_CLIP_LOCAL_DIR
from earshot.env_check import CLIP_MODEL_ID as PROBE_CLIP_MODEL_ID
from earshot.env_check import _clap_source, _clip_source
from earshot.task.models import (
    CLAP_LOCAL_DIR,
    CLAP_MODEL_ID,
    CLIP_LOCAL_DIR,
    CLIP_MODEL_ID,
    STAGED_MARKER,
    resolve_clap_source,
    resolve_clip_source,
)

# The files `from_pretrained` needs before a directory is a usable checkpoint, plus the marker
# that says WHICH model it holds.
COMPLETE = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    STAGED_MARKER,
)

# One row per staged model: its id, the two constants that must agree, and the two resolvers
# that must agree. Adding a third staged model is one row and no new test.
MODELS = (
    (
        "CLAP", CLAP_MODEL_ID, PROBE_CLAP_MODEL_ID, CLAP_LOCAL_DIR, PROBE_CLAP_LOCAL_DIR,
        resolve_clap_source, _clap_source, "laion/clap-htsat-fused",
    ),
    (
        "CLIP", CLIP_MODEL_ID, PROBE_CLIP_MODEL_ID, CLIP_LOCAL_DIR, PROBE_CLIP_LOCAL_DIR,
        resolve_clip_source, _clip_source, "openai/clip-vit-large-patch14",
    ),
)


def _resolvers(row):
    """The (label, resolver) pairs for one model row."""
    name, _model_id, _probe_id, _local, _probe_local, runtime, probe, _other = row
    return (("{} runtime".format(name), runtime), ("{} probe".format(name), probe))


def stage(directory, model_id, skip=None, marker_body=None):
    """Write a staged-looking directory. `skip` omits one file; `marker_body` corrupts it."""
    for name in COMPLETE:
        if name == skip:
            continue
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            if name == STAGED_MARKER:
                handle.write(
                    marker_body
                    if marker_body is not None
                    else json.dumps({"model_id": model_id})
                )
            else:
                handle.write("{}")
    return directory


class TestTheModelRowsAreRealAndDistinct(unittest.TestCase):
    """The control for every loop below. A `MODELS` that collapsed to one entry, or held
    two rows naming the same checkpoint, would make each test run twice over CLAP and
    report green while CLIP went unchecked."""

    def test_both_staged_models_are_covered(self):
        self.assertEqual([row[0] for row in MODELS], ["CLAP", "CLIP"])

    def test_the_two_models_are_not_the_same_checkpoint(self):
        self.assertNotEqual(MODELS[0][1], MODELS[1][1])
        self.assertNotEqual(MODELS[0][3], MODELS[1][3])

    def test_each_rows_two_resolvers_are_different_functions(self):
        """A row whose runtime and probe were the same object would make "they agree"
        trivially true."""
        for row in MODELS:
            self.assertIsNot(row[5], row[6], row[0])


class TestTheSourcesAgree(unittest.TestCase):
    def test_the_constants_match(self):
        for name, model_id, probe_id, local, probe_local, _r, _p, _o in MODELS:
            self.assertEqual(model_id, probe_id, name)
            self.assertEqual(local, probe_local, name)
        self.assertEqual(STAGED_MARKER, PROBE_MARKER)

    def test_both_fall_back_to_the_hub_when_nothing_is_staged(self):
        for row in MODELS:
            model_id = row[1]
            with tempfile.TemporaryDirectory() as empty:
                for label, resolve in _resolvers(row):
                    self.assertEqual(resolve(model_id, empty), model_id, label)

    def test_both_prefer_a_complete_local_copy_of_the_model_asked_for(self):
        for row in MODELS:
            model_id = row[1]
            with tempfile.TemporaryDirectory() as staged:
                stage(staged, model_id)
                for label, resolve in _resolvers(row):
                    self.assertEqual(resolve(model_id, staged), staged, label)

    def test_both_refuse_a_half_written_conversion(self):
        """A partial directory must lose to the Hub, not be preferred and then fail oddly.

        Every file is checked one at a time, so a resolver that only tests for
        `model.safetensors` fails here rather than in a run.
        """
        for row in MODELS:
            model_id = row[1]
            for missing in COMPLETE:
                with tempfile.TemporaryDirectory() as partial:
                    stage(partial, model_id, skip=missing)
                    for label, resolve in _resolvers(row):
                        self.assertEqual(
                            resolve(model_id, partial),
                            model_id,
                            "{} preferred a directory missing {}".format(label, missing),
                        )

    def test_a_staged_copy_of_one_model_is_not_handed_to_the_other(self):
        """**THE ARM TWO STAGED MODELS MAKE POSSIBLE.** Both checkpoints are staged side by
        side under `models/`, so a resolver that checked completeness without identity
        would hand CLIP's directory to a caller asking for CLAP. The single-model version
        of this file could not express that case at all."""
        for row in MODELS:
            model_id = row[1]
            other = [candidate[1] for candidate in MODELS if candidate[1] != model_id][0]
            with tempfile.TemporaryDirectory() as staged:
                stage(staged, other)
                for label, resolve in _resolvers(row):
                    self.assertEqual(
                        resolve(model_id, staged), model_id,
                        "{} handed a directory holding {} to a caller asking for "
                        "{}".format(label, other, model_id),
                    )


class TestTheStagedCopyMustNameItsModel(unittest.TestCase):
    """The bug the box gate caught: agreement without correctness.

    `env_check` ships a forced-failure arm that asks for a model id that cannot resolve and
    asserts the probe goes RED. While the resolvers ignored `model_id`, that arm loaded the
    real staged checkpoint and reported PASS -- a detector asserting nothing, which is the
    exact failure ADR-0014's both-arms rule exists to catch.
    """

    def test_a_different_model_id_does_not_get_the_staged_copy(self):
        for row in MODELS:
            with tempfile.TemporaryDirectory() as staged:
                stage(staged, row[1])
                for label, resolve in _resolvers(row):
                    self.assertEqual(
                        resolve("earshot/definitely-not-a-model", staged),
                        "earshot/definitely-not-a-model",
                        "{} handed the real checkpoint to a model that does not "
                        "exist".format(label),
                    )

    def test_a_directory_that_cannot_name_its_model_is_not_used(self):
        """No marker means no claim about what is inside, and no claim is not a match."""
        for row in MODELS:
            model_id = row[1]
            with tempfile.TemporaryDirectory() as staged:
                stage(staged, model_id, skip=STAGED_MARKER)
                for label, resolve in _resolvers(row):
                    self.assertEqual(resolve(model_id, staged), model_id, label)

    def test_an_unreadable_marker_falls_back_rather_than_raising(self):
        """A truncated write must send the caller to the Hub, not crash the probe."""
        for row in MODELS:
            model_id = row[1]
            with tempfile.TemporaryDirectory() as staged:
                stage(staged, model_id, marker_body="{not json")
                for label, resolve in _resolvers(row):
                    self.assertEqual(resolve(model_id, staged), model_id, label)

    def test_a_marker_with_no_model_id_falls_back(self):
        for row in MODELS:
            model_id = row[1]
            with tempfile.TemporaryDirectory() as staged:
                stage(staged, model_id, marker_body=json.dumps({"why": "converted"}))
                for label, resolve in _resolvers(row):
                    self.assertEqual(resolve(model_id, staged), model_id, label)

    def test_a_marker_naming_another_model_falls_back(self):
        """The dangerous case: a complete, readable directory holding the WRONG weights.

        The sibling id is a real near-miss for each family -- `clap-htsat-fused` beside
        `clap-htsat-unfused`, `clip-vit-large-patch14` beside `clip-vit-base-patch32` --
        because a resolver matching on a prefix rather than on equality would pass a test
        that used an obviously different name.
        """
        for row in MODELS:
            model_id, sibling = row[1], row[7]
            self.assertNotEqual(model_id, sibling)
            with tempfile.TemporaryDirectory() as staged:
                stage(staged, sibling)
                for label, resolve in _resolvers(row):
                    self.assertEqual(resolve(model_id, staged), model_id, label)


if __name__ == "__main__":
    unittest.main()
