"""Evidence validation, adversarial types, portable identity and atomic export."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from evals_shared.format import MAX_UPLOAD_BYTES, ValidationError, canonical_json, content_digest, parse_document, validate_document
from evals_shared.export import EvaluationWriter, write_document
from evals_shared.provenance import portable_metadata

FIXTURE = Path(__file__).parent / "fixtures/evaluation-v1.json"


def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class FormatTests(unittest.TestCase):
    def test_variable_participants_exact_checkpoint_and_all_states(self):
        document = parse_document(FIXTURE.read_bytes())
        self.assertEqual(len(document["participants"]), 3)
        self.assertEqual(len(document["checkpoints"]), 1)
        self.assertEqual([len(e["recording"]["steps"]) for e in document["episodes"]], [3, 3])
        self.assertEqual(content_digest(document), content_digest(json.loads(canonical_json(document))))

    def test_missing_recording_is_honest_valid_evidence(self):
        document = fixture()
        document["episodes"][1]["recording"] = None
        validate_document(document)

    def test_identity_references_totals_and_alignment_are_checked(self):
        mutations = [
            lambda d: d["checkpoints"][0].update(source_run_id="missing"),
            lambda d: d["participants"][0].update(source_checkpoint_id="missing"),
            lambda d: d["participants"][0].update(display_order=1),
            lambda d: d["episodes"][0].update(index=1),
            lambda d: d["episodes"][0]["recording"]["metadata"].update(episode=1),
            lambda d: d["episodes"][0]["recording"]["steps"][0]["agents"][0].update(action="stay"),
            lambda d: d["episodes"][0]["recording"]["steps"][1]["agents"][0].update(action=None),
            lambda d: d["episodes"][0]["recording"]["steps"] .pop(),
            lambda d: d["episodes"][0]["metrics"].update(team_reward=20),
            lambda d: d["summary"].update(total_correct_dishes=1),
            lambda d: d["summary"].update(mean_team_reward=1),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                document = fixture(); mutation(document)
                with self.assertRaises(ValidationError):
                    validate_document(document)

    def test_malformed_types_raise_validation_error(self):
        locations = [("created_at",), ("training_runs",), ("checkpoints", 0, "source_run_id"),
            ("participants", 0, "source_checkpoint_id"), ("episodes", 0, "metrics", "team_reward"),
            ("episodes", 0, "recording", "steps", 1, "totals", "team_reward"),
            ("episodes", 0, "recording", "steps", 1, "agents", 0, "position"),
            ("episodes", 0, "recording", "steps", 1, "reward"),
            ("episodes", 0, "recording", "steps", 1, "grid")]
        for location in locations:
            for value in (None, True, 123, [], {}, "bad"):
                with self.subTest(location=location, value=value):
                    document = fixture(); parent = document
                    for key in location[:-1]: parent = parent[key]
                    parent[location[-1]] = value
                    with self.assertRaises(ValidationError): validate_document(document)

    def test_limits_required_semantics_duplicates_and_nonfinite(self):
        raw = canonical_json(fixture())
        parse_document(raw + b" " * (MAX_UPLOAD_BYTES-len(raw)))
        with self.assertRaisesRegex(ValidationError, "20 MiB"):
            parse_document(raw + b" " * (MAX_UPLOAD_BYTES-len(raw)+1))
        for raw in (b'{"format":1,"format":2}', b'{"a":NaN}', b'{"a":Infinity}', b'['*1100+b']'*1100, b'\xff'):
            with self.assertRaises(ValidationError): parse_document(raw)
        for field, value in (("required_features", ["new_semantics"]), ("version", 2), ("frames", []), ("notes", "C:\\Users\\private\\secret"), ("notes", "data:text/html,hello")):
            document=fixture();document[field]=value
            with self.assertRaises(ValidationError): validate_document(document)

    def test_privacy_scrubbing(self):
        metadata = portable_metadata({"SOURCE_RUN": "/home/person/model", "dataset_path": "private", "settings": {"SEED": 42}, "notes": "file /mnt/c/Users/person/model", "backend": "cpu"})
        self.assertEqual(metadata, {"settings": {"SEED": 42}, "notes": "redacted-local-reference", "backend": "cpu"})

    def test_frozen_sushi_training_config_exports_without_losing_observation_mode(self):
        config_path = Path(__file__).resolve().parents[2] / "model/oc2_sushi_1_3/policy-spawn-v8/config.json"
        config = portable_metadata(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual(config["IMITATION"]["observations"], "full_map")
        document = fixture()
        document["training_runs"][0]["config"] = config
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation.json"
            write_document(path, document)
            self.assertEqual(parse_document(path.read_bytes()), document)

    def test_observation_labels_do_not_allow_raw_observations_or_other_excluded_fields(self):
        for value in ([], [[1, 2]], {"agent_0": [1, 2]}, 1, True, None):
            with self.subTest(value=value):
                document = fixture()
                document["training_runs"][0]["config"]["observations"] = value
                with self.assertRaisesRegex(ValidationError, "observations"):
                    validate_document(document)
        for field in ("weights", "params", "activations", "optimizer", "images", "frames"):
            with self.subTest(field=field):
                document = fixture()
                document["training_runs"][0]["config"].update(observations="full_map", **{field: "excluded"})
                with self.assertRaisesRegex(ValidationError, field):
                    validate_document(document)

    def test_database_scalars_are_bounded_and_never_coerced(self):
        for field, values in (("algorithm", ({}, [], True, 123, "", "x"*129)),
                              ("seed", ({}, [], True, 1.0, "42", -1, 2**80))):
            for value in values:
                with self.subTest(field=field,value=value):
                    document=fixture();document["training_runs"][0][field]=value
                    with self.assertRaises(ValidationError):validate_document(document)
        document=fixture();document["training_runs"][0].update(algorithm=None,seed=None)
        validate_document(document)

    def test_atomic_failure_preserves_previous_result(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation.json"
            path.write_text("previous complete export")
            document=fixture()
            with EvaluationWriter(path) as writer:
                episode=copy.deepcopy(document["episodes"][0]);episode["metrics"]["team_reward"]=9
                writer.add_episode(episode)
                writer.add_episode(document["episodes"][1])
                with self.assertRaises(ValidationError):
                    writer.finish({key:value for key,value in document.items() if key != "episodes"})
            self.assertEqual(path.read_text(), "previous complete export")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            result = write_document(path, document)
            self.assertTrue(result["within_upload_limit"])
            self.assertEqual(parse_document(path.read_bytes()), document)


if __name__ == "__main__":
    unittest.main()
