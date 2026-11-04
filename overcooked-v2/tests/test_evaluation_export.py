"""Checkpoint evidence checks without optimizer updates or training."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checkpoint
from evaluation_core import source_records, policy_config_path


class CheckpointEvidenceTests(unittest.TestCase):
    def test_intermediate_exact_bytes_and_actual_progress_not_final_summary(self):
        config={"ALGORITHM":"ippo_rnn", "NUM_ENVS":2, "NUM_STEPS":8, "SEED":42}
        state=TrainState.create(apply_fn=lambda *args: None, params={"w":jnp.array([1.,2.])}, tx=optax.adam(1e-3))
        state=state.replace(step=jnp.int32(0))
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"config.json").write_text(json.dumps(config))
            (root/"metadata.json").write_text(json.dumps({"source_run_id":"run:stable", "device":{"backend":"cpu"}}))
            (root/"summary.json").write_text(json.dumps({"completed_environment_steps":999999}))
            selected=checkpoint.save_checkpoint(root,(state,jax.random.PRNGKey(0)),[{"x":np.zeros(3)}],3,config,Path(checkpoint.__file__))
            weights=(selected/"params.msgpack").read_bytes()
            record, policy=source_records(selected,config,weights)
            self.assertEqual(policy["progress"],48)
            self.assertEqual(policy["sha256"],hashlib.sha256(weights).hexdigest())
            self.assertEqual(record["source_run_id"],"run:stable")
            self.assertEqual(policy_config_path(selected),root/"config.json")
            restored,_,_=checkpoint.load_checkpoint(selected,(state,jax.random.PRNGKey(0)),config,Path(checkpoint.__file__))
            np.testing.assert_array_equal(restored[0].params["w"],state.params["w"])
            # A stale/mismatching sidecar is never trusted for another byte stream.
            _, unknown=source_records(selected,config,weights+b"different")
            self.assertIsNone(unknown["progress"])

    def test_initialization_zero_and_legacy_missing_progress_unknown(self):
        config={"ALGORITHM":"ippo_rnn","SEED":42}
        weights=serialization.to_bytes({"w":np.array([1.])})
        with TemporaryDirectory() as tmp:
            run,policy=source_records(Path(tmp),config,weights)
            self.assertIsNone(policy["progress"])
            self.assertEqual(run["device"],{})
            self.assertTrue(run["source_run_id"].startswith("run:"))
            self.assertEqual(source_records(Path(tmp),config,weights)[0],run)
            run,policy=source_records(None,config,weights,initialization=True,identity_directory=tmp)
            self.assertEqual(policy["progress"],0)
            self.assertEqual(run["algorithm"],"initialization_only")


if __name__ == "__main__": unittest.main()
