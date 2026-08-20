"""The probe and the runtime must resolve the SAME CLAP checkpoint.

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
"""

import os
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.env_check import CLAP_LOCAL_DIR as PROBE_LOCAL_DIR
from earshot.env_check import CLAP_MODEL_ID as PROBE_MODEL_ID
from earshot.env_check import _clap_source
from earshot.task.models import CLAP_LOCAL_DIR, CLAP_MODEL_ID, resolve_clap_source

# The four files `from_pretrained` needs before a directory is a usable checkpoint.
COMPLETE = ("model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json")


class TestClapSourceAgrees(unittest.TestCase):
    def test_the_two_constants_match(self):
        self.assertEqual(CLAP_MODEL_ID, PROBE_MODEL_ID)
        self.assertEqual(CLAP_LOCAL_DIR, PROBE_LOCAL_DIR)

    def test_both_fall_back_to_the_hub_when_nothing_is_staged(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(resolve_clap_source(CLAP_MODEL_ID, empty), CLAP_MODEL_ID)
            self.assertEqual(_clap_source(CLAP_MODEL_ID, empty), CLAP_MODEL_ID)

    def test_both_prefer_a_complete_local_copy(self):
        with tempfile.TemporaryDirectory() as staged:
            for name in COMPLETE:
                with open(os.path.join(staged, name), "w", encoding="utf-8") as handle:
                    handle.write("{}")
            self.assertEqual(resolve_clap_source(CLAP_MODEL_ID, staged), staged)
            self.assertEqual(_clap_source(CLAP_MODEL_ID, staged), staged)

    def test_both_refuse_a_half_written_conversion(self):
        """A partial directory must lose to the Hub, not be preferred and then fail oddly.

        Every one of the four files is checked, one at a time, so a resolver that only tests
        for `model.safetensors` fails here rather than in a run.
        """
        for missing in COMPLETE:
            with tempfile.TemporaryDirectory() as partial:
                for name in COMPLETE:
                    if name == missing:
                        continue
                    with open(os.path.join(partial, name), "w", encoding="utf-8") as handle:
                        handle.write("{}")
                self.assertEqual(
                    resolve_clap_source(CLAP_MODEL_ID, partial),
                    CLAP_MODEL_ID,
                    "runtime preferred a directory missing {}".format(missing),
                )
                self.assertEqual(
                    _clap_source(CLAP_MODEL_ID, partial),
                    CLAP_MODEL_ID,
                    "probe preferred a directory missing {}".format(missing),
                )


if __name__ == "__main__":
    unittest.main()
