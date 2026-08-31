"""The probe and the runtime must resolve the SAME CLAP checkpoint, and only for the model asked.

`env_check.py` may import nothing intra-package (ADR-0013: "ticket 17's assertion answers to
the environment, not to the tree"), so it cannot call `task.models.resolve_clap_source` and
carries its own `_clap_source` instead. A duplicated resolver is only safe while the two
agree, and "they agree" is the kind of claim this repo has repeatedly found to have quietly
stopped being true. So it is a test.

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

from earshot.env_check import CLAP_LOCAL_DIR as PROBE_LOCAL_DIR
from earshot.env_check import CLAP_MODEL_ID as PROBE_MODEL_ID
from earshot.env_check import CLAP_STAGED_MARKER as PROBE_MARKER
from earshot.env_check import _clap_source
from earshot.task.models import (
    CLAP_LOCAL_DIR,
    CLAP_MODEL_ID,
    STAGED_MARKER,
    resolve_clap_source,
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

RESOLVERS = (("runtime", resolve_clap_source), ("probe", _clap_source))


def stage(directory, model_id=CLAP_MODEL_ID, skip=None, marker_body=None):
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


class TestClapSourceAgrees(unittest.TestCase):
    def test_the_constants_match(self):
        self.assertEqual(CLAP_MODEL_ID, PROBE_MODEL_ID)
        self.assertEqual(CLAP_LOCAL_DIR, PROBE_LOCAL_DIR)
        self.assertEqual(STAGED_MARKER, PROBE_MARKER)

    def test_both_fall_back_to_the_hub_when_nothing_is_staged(self):
        with tempfile.TemporaryDirectory() as empty:
            for label, resolve in RESOLVERS:
                self.assertEqual(resolve(CLAP_MODEL_ID, empty), CLAP_MODEL_ID, label)

    def test_both_prefer_a_complete_local_copy_of_the_model_asked_for(self):
        with tempfile.TemporaryDirectory() as staged:
            stage(staged)
            for label, resolve in RESOLVERS:
                self.assertEqual(resolve(CLAP_MODEL_ID, staged), staged, label)

    def test_both_refuse_a_half_written_conversion(self):
        """A partial directory must lose to the Hub, not be preferred and then fail oddly.

        Every file is checked one at a time, so a resolver that only tests for
        `model.safetensors` fails here rather than in a run.
        """
        for missing in COMPLETE:
            with tempfile.TemporaryDirectory() as partial:
                stage(partial, skip=missing)
                for label, resolve in RESOLVERS:
                    self.assertEqual(
                        resolve(CLAP_MODEL_ID, partial),
                        CLAP_MODEL_ID,
                        "{} preferred a directory missing {}".format(label, missing),
                    )


class TestTheStagedCopyMustNameItsModel(unittest.TestCase):
    """The bug the box gate caught: agreement without correctness.

    `env_check` ships a forced-failure arm that asks for a model id that cannot resolve and
    asserts the probe goes RED. While the resolvers ignored `model_id`, that arm loaded the
    real staged checkpoint and reported PASS -- a detector asserting nothing, which is the
    exact failure ADR-0014's both-arms rule exists to catch.
    """

    def test_a_different_model_id_does_not_get_the_staged_copy(self):
        with tempfile.TemporaryDirectory() as staged:
            stage(staged, model_id=CLAP_MODEL_ID)
            for label, resolve in RESOLVERS:
                self.assertEqual(
                    resolve("earshot/definitely-not-a-model", staged),
                    "earshot/definitely-not-a-model",
                    "{} handed the real checkpoint to a model that does not exist".format(
                        label
                    ),
                )

    def test_a_directory_that_cannot_name_its_model_is_not_used(self):
        """No marker means no claim about what is inside, and no claim is not a match."""
        with tempfile.TemporaryDirectory() as staged:
            stage(staged, skip=STAGED_MARKER)
            for label, resolve in RESOLVERS:
                self.assertEqual(resolve(CLAP_MODEL_ID, staged), CLAP_MODEL_ID, label)

    def test_an_unreadable_marker_falls_back_rather_than_raising(self):
        """A truncated write must send the caller to the Hub, not crash the probe."""
        with tempfile.TemporaryDirectory() as staged:
            stage(staged, marker_body="{not json")
            for label, resolve in RESOLVERS:
                self.assertEqual(resolve(CLAP_MODEL_ID, staged), CLAP_MODEL_ID, label)

    def test_a_marker_with_no_model_id_falls_back(self):
        with tempfile.TemporaryDirectory() as staged:
            stage(staged, marker_body=json.dumps({"why": "converted"}))
            for label, resolve in RESOLVERS:
                self.assertEqual(resolve(CLAP_MODEL_ID, staged), CLAP_MODEL_ID, label)

    def test_a_marker_naming_another_model_falls_back(self):
        """The dangerous case: a complete, readable directory holding the WRONG weights."""
        with tempfile.TemporaryDirectory() as staged:
            stage(staged, model_id="laion/clap-htsat-fused")
            for label, resolve in RESOLVERS:
                self.assertEqual(resolve(CLAP_MODEL_ID, staged), CLAP_MODEL_ID, label)


if __name__ == "__main__":
    unittest.main()
