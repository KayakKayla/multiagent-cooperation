"""Standard-library checks; run python3 -m unittest discover -s <skill>/tests -v."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

SKILL = Path(__file__).resolve().parents[1]
HELPER = SKILL / "scripts/extract_replay.py"
spec = importlib.util.spec_from_file_location("extract_replay", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    states = []
    for t in range(3):
        states.append({
            "step": t,
            "agents": [{"id": f"agent_{i}", "position": [i, 0], "direction": 0,
                        "facing": "up", "inventory": 0, "holding": "nothing",
                        "action": None if t == 0 else "stay",
                        "action_code": None if t == 0 else 4} for i in range(2)],
            "grid": [[[0, 0, 0], [0, 0, 0]], [[5, 12, 3 - t], [1, 0, 0]]],
            "pots": [{"index": 1, "position": [0, 1], "contents": 12,
                      "ingredients": 3, "timer": 3 - t, "status": "cooking"}],
            "recipe_code": 12, "recipe": "three onions", "reward": 0,
            "correct_dishes": 0, "wrong_dishes": 0,
            "totals": {"team_reward": 0, "correct_dishes": 0, "wrong_dishes": 0},
        })
    return {"schema_version": 1, "metadata": {"layout": "synthetic", "steps": 2},
            "steps": states, "frames": ["image-placeholder"] * 3}


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "replay.html"
        self.data = fixture()
        self.write()

    def write(self, data=None):
        payload = json.dumps(self.data if data is None else data).replace("<", "\\u003c")
        self.path.write_text('<script>alert("must not execute")</script>'
                             '<script type="application/json" id="replay-data">'
                             + payload + '</script>', encoding="utf-8")

    def test_file_folder_and_read_only(self):
        before = self.path.read_bytes()
        evidence = module.extract(self.path)
        self.assertEqual(evidence, module.extract(self.path.parent))
        self.assertEqual(evidence["source_replay"], str(self.path.resolve()))
        self.assertEqual(len(evidence["steps"]), 3)
        self.assertIsNone(evidence["context_state"])
        self.assertNotIn("frames", evidence)
        self.assertEqual(before, self.path.read_bytes())

    def test_inclusive_range_preserves_action_alignment(self):
        self.data["steps"][1]["agents"][0].update(action="right", action_code=0,
                                                  position=[1, 0], facing="right", direction=2)
        self.write()
        evidence = module.extract(self.path, 1, 1)
        self.assertEqual([s["step"] for s in evidence["steps"]], [1])
        self.assertEqual(evidence["context_state"]["step"], 0)
        self.assertIsNone(evidence["context_state"]["agents"][0]["action"])
        self.assertEqual(evidence["steps"][0]["agents"][0]["action"], "right")
        self.assertEqual(module.extract(self.path, end=0)["steps"], [self.data["steps"][0]])
        self.assertEqual(module.extract(self.path, start=2)["range"], {"start": 2, "end": 2})

    def test_simultaneous_events_and_full_episode_totals_in_range(self):
        self.data["steps"][1].update(reward=40, correct_dishes=2, wrong_dishes=1)
        for state in self.data["steps"][1:]:
            state["totals"] = {"team_reward": 40, "correct_dishes": 2, "wrong_dishes": 1}
        self.write()
        self.assertEqual(module.extract(self.path, 0, 0)["episode_totals"],
                         {"team_reward": 40, "correct_dishes": 2, "wrong_dishes": 1})

    def test_failed_movement_and_useful_wait_remain_raw_evidence(self):
        # A wall-directed move and a cooking wait both keep position unchanged.
        # Preserve their differing actions and pot progress; do not classify intent.
        self.data["steps"][1]["agents"][1].update(action="down", action_code=1,
                                                  facing="down", direction=1)
        self.write()
        evidence = module.extract(self.path)
        before, after = evidence["steps"][:2]
        for i in range(2):
            self.assertEqual(before["agents"][i]["position"], after["agents"][i]["position"])
        self.assertEqual(after["agents"][0]["action"], "stay")
        self.assertEqual(after["agents"][1]["action"], "down")
        self.assertEqual(after["grid"][1][1][0], 1)  # wall, not teammate
        self.assertLess(after["pots"][0]["timer"], before["pots"][0]["timer"])
        self.assertNotIn("diagnosis", evidence)

    def test_bad_payloads(self):
        cases = [
            (lambda d: d.update(schema_version=2), "schema_version"),
            (lambda d: d["steps"][1].update(step=2), "numbering"),
            (lambda d: d["metadata"].update(steps=3), "initial state"),
            (lambda d: d["steps"][1]["totals"].update(team_reward=1), "cumulative"),
            (lambda d: d["steps"][1]["totals"].update(correct_dishes=1), "cumulative"),
            (lambda d: d["steps"][0]["agents"][0].update(action="stay"), "initial actions"),
            (lambda d: d["steps"][1].update(agents=None), "agents"),
            (lambda d: d["steps"][1].update(grid=[None]), "grid"),
            (lambda d: d["steps"][1].update(pots=[None]), "pots"),
            (lambda d: d["steps"][1].update(reward=float("nan")), "invalid reward"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                data = deepcopy(self.data)
                mutate(data)
                self.write(data)
                with self.assertRaisesRegex(module.ReplayError, message):
                    module.extract(self.path)

    def test_missing_duplicate_unclosed_and_malformed_scripts(self):
        for html in ("<html></html>", self.path.read_text() * 2,
                     '<script id="replay-data" type="application/json">{}',
                     '<script id="replay-data" type="application/json">{oops}</script>'):
            with self.subTest(html=html[:60]):
                self.path.write_text(html)
                with self.assertRaises(module.ReplayError):
                    module.extract(self.path)

    def test_invalid_ranges_and_missing_file(self):
        for start, end in ((-1, 1), (2, 1), (0, 3)):
            with self.assertRaisesRegex(module.ReplayError, "range"):
                module.extract(self.path, start, end)
        with self.assertRaises(OSError):
            module.extract(self.path.parent / "missing.html")

    def test_cli_clean_json_and_error(self):
        result = subprocess.run([sys.executable, "-S", str(HELPER), str(self.path),
                                 "--start", "1", "--end", "1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["context_state"]["step"], 0)
        self.assertEqual(result.stderr, "")
        bad = subprocess.run([sys.executable, "-S", str(HELPER), str(self.path), "--end", "9"],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, "")
        self.assertIn("Replay error:", bad.stderr)
        self.assertNotIn("Traceback", bad.stderr)

    def test_repository_scripted_reference(self):
        # Installed location: repository/.agents/skills/<name>/tests/this_file.
        repository = SKILL.parents[2]
        replay = repository / "docs/theoretical-maximum/media/18-soups/replay.html"
        if not replay.exists():
            self.skipTest("scripted reference is not present alongside this skill")
        before = hashlib.sha256(replay.read_bytes()).digest()
        evidence = module.extract(replay)
        self.assertEqual(evidence["metadata"]["steps"], 400)
        self.assertEqual(len(evidence["steps"]), 401)
        self.assertEqual(evidence["episode_totals"],
                         {"team_reward": 360, "correct_dishes": 18, "wrong_dishes": 0})
        self.assertEqual(before, hashlib.sha256(replay.read_bytes()).digest())


if __name__ == "__main__":
    unittest.main()
