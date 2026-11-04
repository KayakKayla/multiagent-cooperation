"""Historical conversion never regenerates missing episode evidence."""
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from evals_shared.format import canonical_json, validate_document
from evals_shared.legacy import convert_folder


class LegacyTests(unittest.TestCase):
    def write_small_evaluation(self,root,config,protocol,partner_config=None):
        (root/"config.json").write_text(json.dumps(config))
        (root/"protocol.json").write_text(json.dumps(protocol))
        (root/"summary.json").write_text(json.dumps({"episodes":1,"mean_team_reward":0,"total_correct_dishes":0,"total_wrong_dishes":0}))
        (root/"episodes.json").write_text(json.dumps([{"episode":0,"seed":10,"steps":2,"team_reward":0,"correct_dishes":0,"wrong_dishes":0}]))
        if partner_config is not None:
            (root/"partner-config.json").write_text(json.dumps(partner_config))

    def test_moved_cross_policy_sources_use_saved_partner_config_seed_and_independent_progress(self):
        config={"ENV_NAME":"synthetic","ENV_KWARGS":{"layout":"synthetic"},"ALGORITHM":"ippo_rnn","SEED":842}
        partner=dict(config,SEED=373,LR=0.0001)
        protocol={"policy":"cross-policy","source_run":"/home/private/A","partner_run":"/home/private/B","episode_seeds":[10],
            "policies":[{"agent":"agent_0","source_run":"/home/private/A","training_seed":842,"policy_sha256":"a"*64},
                        {"agent":"agent_1","source_run":"/home/private/B","training_seed":373,"policy_sha256":"b"*64}]}
        with TemporaryDirectory() as tmp,patch("evals_shared.legacy._local_source",return_value=None):
            root=Path(tmp);self.write_small_evaluation(root,config,protocol,partner)
            document=validate_document(convert_folder(root,progress=100))
            self.assertEqual([r["seed"] for r in document["training_runs"]],[842,373])
            self.assertEqual(len(document["training_runs"]),2)
            self.assertEqual(document["training_runs"][1]["config"],partner)
            self.assertEqual([c["progress"] for c in document["checkpoints"]],[100,None])
            self.assertEqual(convert_folder(root,progress=100),document)
            explicit=validate_document(convert_folder(root,source_run_id="run:A",partner_source_run_id="run:B",progress=100,partner_progress=200))
            self.assertEqual([r["source_run_id"] for r in explicit["training_runs"]],["run:A","run:B"])
            self.assertEqual([c["progress"] for c in explicit["checkpoints"]],[100,200])

    def test_effective_override_does_not_become_unknown_original_training_configuration(self):
        config={"ENV_NAME":"oc2_sushi_1_3","ENV_KWARGS":{"layout":"oc2_sushi_1_3","rules":{"max_steps":4}},"TRAINING_METHOD":"behavior_cloning","ALGORITHM":"ippo_rnn","SEED":42}
        protocol={"policy_sha256":"a"*64,"episode_seeds":[10],"training_episode_steps":600,"episode_steps":4}
        with TemporaryDirectory() as tmp,patch("evals_shared.legacy._local_source",return_value=None):
            root=Path(tmp);self.write_small_evaluation(root,config,protocol)
            document=validate_document(convert_folder(root))
            self.assertEqual(document["training_runs"][0]["config"],{})
            self.assertEqual(document["training_runs"][0]["algorithm"],"behavior_cloning")
            self.assertEqual(document["training_runs"][0]["seed"],42)
            self.assertEqual(document["environment"]["config"]["rules"]["max_steps"],4)
            self.assertEqual(document["provenance"]["legacy_saved_configs"]["primary_effective_evaluation_config"],config)

    def test_two_checkpoints_of_one_run_have_independent_progress_and_shared_slots_reuse_it(self):
        config={"ENV_NAME":"synthetic","ENV_KWARGS":{"layout":"synthetic"},"ALGORITHM":"ippo_rnn","SEED":842}
        for available_source in (False,True):
            with self.subTest(available_source=available_source),TemporaryDirectory() as tmp:
                base=Path(tmp);root=base/"evaluation";root.mkdir()
                source=base/"training";source.mkdir()
                references=[]
                for update,digest in ((100,"a"*64),(200,"b"*64)):
                    selected=source/"checkpoints"/f"update-{update:06d}"
                    if available_source:
                        selected.mkdir(parents=True)
                        (selected/"meta.json").write_text("{}")
                        (selected/"checkpoint.json").write_text(json.dumps({"sha256":digest,"progress":update,"progress_unit":"environment_steps"}))
                        references.append(str(selected))
                    else:
                        references.append(f"/home/private/A/checkpoints/update-{update:06d}")
                if available_source:
                    (source/"config.json").write_text(json.dumps(config))
                    (source/"metadata.json").write_text(json.dumps({"source_run_id":"run:one-recorded-run"}))
                protocol={"policy":"cross-policy","source_run":references[0],"partner_run":references[1],"episode_seeds":[10],
                    "policies":[{"agent":"agent_0","source_run":references[0],"training_seed":842,"policy_sha256":"a"*64},
                                {"agent":"agent_1","source_run":references[1],"training_seed":842,"policy_sha256":"b"*64}]}
                self.write_small_evaluation(root,config,protocol,config)
                document=validate_document(convert_folder(root,progress=100,partner_progress=200))
                self.assertEqual(len(document["training_runs"]),1)
                self.assertEqual(len(document["checkpoints"]),2)
                self.assertEqual([c["progress"] for c in document["checkpoints"]],[100,200])
                automatic=validate_document(convert_folder(root,progress=100))
                self.assertEqual([c["progress"] for c in automatic["checkpoints"]],[100,200 if available_source else None])
                # A+A with one exact checkpoint remains one policy, used twice.
                protocol["partner_run"]=references[0]
                protocol["policies"][1]=dict(protocol["policies"][0],agent="agent_1")
                (root/"protocol.json").write_text(json.dumps(protocol))
                shared=validate_document(convert_folder(root,progress=100))
                self.assertEqual(len(shared["training_runs"]),1)
                self.assertEqual(len(shared["checkpoints"]),1)
                self.assertEqual(shared["checkpoints"][0]["progress"],100)
                self.assertEqual(shared["participants"][0]["source_checkpoint_id"],shared["participants"][1]["source_checkpoint_id"])
                with self.assertRaisesRegex(ValueError,"distinct partner checkpoint"):
                    convert_folder(root,progress=100,partner_progress=200)

    def test_inert_conversion_partial_recordings_unknown_training_device_and_progress(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/evaluation-v1.json").read_text(encoding="utf-8"))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"ENV_NAME": "synthetic", "ENV_KWARGS": {"layout": "synthetic"}, "SEED": 42, "ALGORITHM": "ippo_rnn"}
            protocol = {"source_run": "/home/private/absent-model", "policy_sha256": "a"*64, "episode_seeds": [10,11], "recorded_episode": 0, "backend": "gpu"}
            (root / "config.json").write_text(json.dumps(config))
            (root / "protocol.json").write_text(json.dumps(protocol))
            (root / "summary.json").write_text(json.dumps(fixture["summary"]))
            with (root / "episodes.csv").open("w", newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=["episode","seed","steps","team_reward","correct_dishes","wrong_dishes"])
                writer.writeheader()
                writer.writerows([dict(episode=e["index"],seed=e["seed"],steps=e["steps"],**e["metrics"]) for e in fixture["episodes"]])
            replay = fixture["episodes"][0]["recording"]
            replay["frames"]=["data:image/png;base64,unused"]*3
            (root / "replay.html").write_text('<script>throw new Error("must never execute")</script><script id="replay-data" type="application/json">'+json.dumps(replay)+'</script>',encoding="utf-8")
            document = validate_document(convert_folder(root, source_run_id="run:explicit-legacy"))
            self.assertEqual(len(document["participants"]), 3)
            self.assertEqual(len(document["checkpoints"]), 1)
            self.assertIsNotNone(document["episodes"][0]["recording"])
            self.assertIsNone(document["episodes"][1]["recording"])
            self.assertIsNone(document["checkpoints"][0]["progress"])
            self.assertEqual(document["training_runs"][0]["device"], {})
            self.assertEqual(document["device"], {"backend":"gpu"})
            raw=canonical_json(document)
            self.assertNotIn(b"private",raw)
            self.assertNotIn(b"data:image",raw)
            self.assertEqual(document, convert_folder(root, source_run_id="run:explicit-legacy"))


if __name__ == "__main__": unittest.main()
