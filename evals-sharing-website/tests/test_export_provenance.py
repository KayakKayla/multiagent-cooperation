"""Stable local identities and selected-checkpoint evidence; no simulator needed."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from evals_shared.provenance import recorded_source_identity, source_records
from evals_shared.legacy import convert_folder


class ProvenanceTests(unittest.TestCase):
    def test_concurrent_identity_creation_and_copy_are_stable(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/"run";root.mkdir()
            with ThreadPoolExecutor(max_workers=8) as pool:
                identities=list(pool.map(lambda _:recorded_source_identity(root),range(16)))
            self.assertEqual(len(set(identities)),1)
            copied=Path(tmp)/"copied";shutil.copytree(root,copied)
            self.assertEqual(recorded_source_identity(copied),identities[0])
            self.assertEqual(list(root.glob("*.tmp-*")),[])

    def test_initialization_uuid_is_not_content_identity_and_converter_reuses_it(self):
        config={"ENV_NAME":"synthetic", "ENV_KWARGS":{"layout":"synthetic"}, "ALGORITHM":"ippo_rnn", "SEED":42}
        weights=b"identical fresh weights"
        digest=hashlib.sha256(weights).hexdigest()
        with TemporaryDirectory() as tmp:
            ids=[]
            for folder in ("first", "independent"):
                root=Path(tmp)/folder;root.mkdir()
                run,checkpoint=source_records(None,config,weights,initialization=True,identity_directory=root)
                ids.append(run["source_run_id"])
                self.assertNotIn(digest,run["source_run_id"])
                self.assertEqual(source_records(None,config,weights,initialization=True,identity_directory=root)[0],run)
                (root/"config.json").write_text(json.dumps(config))
                (root/"protocol.json").write_text(json.dumps({"policy":"untrained", "policy_sha256":digest, "episode_seeds":[10]}))
                (root/"summary.json").write_text(json.dumps({"episodes":1,"mean_team_reward":0}))
                (root/"episodes.json").write_text(json.dumps([{"episode":0,"seed":10,"steps":2,"team_reward":0,"correct_dishes":0,"wrong_dishes":0}]))
                converted=convert_folder(root)
                self.assertEqual(converted["training_runs"][0],run)
                self.assertEqual(converted["checkpoints"][0],checkpoint)
                self.assertEqual(converted,convert_folder(root))
            self.assertNotEqual(ids[0],ids[1])

    def test_root_sidecar_hash_mismatch_does_not_borrow_final_progress(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"metadata.json").write_text(json.dumps({"source_run_id":"run:stable"}))
            (root/"summary.json").write_text(json.dumps({"completed_environment_steps":999999}))
            (root/"checkpoint.json").write_text(json.dumps({"sha256":"0"*64,"progress":100,"progress_unit":"environment_steps"}))
            run,checkpoint=source_records(root,{"ALGORITHM":"ippo_rnn"},b"different bytes")
            self.assertIsNone(checkpoint["progress"])
            (root/"checkpoint.json").unlink()
            self.assertEqual(source_records(root,{"ALGORITHM":"ippo_rnn"},b"different bytes")[1]["progress"],999999)


if __name__ == "__main__": unittest.main()
