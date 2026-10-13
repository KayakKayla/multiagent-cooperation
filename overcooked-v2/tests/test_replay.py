"""Run with the simulator dependencies: python -m unittest discover -s tests."""

import ast
import csv
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview import build_replay_data
from replay import save_replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        # Two transitions, with simultaneous correct/wrong deliveries on the first.
        self.states = SimpleNamespace(
            grid=np.zeros((3, 2, 2, 3), dtype=int),
            recipe=np.array([12, 12, 12]),
            agents=SimpleNamespace(
                pos=SimpleNamespace(x=np.array([[0, 1], [1, 0], [0, 1]]),
                                    y=np.array([[0, 1], [0, 1], [1, 0]])),
                dir=np.array([[0, 0], [2, 3], [1, 0]]),
                inventory=np.array([[0, 1], [4, 0], [0, 15]]),
            ),
        )
        self.actions = np.array([[2, 3], [4, 5]])
        self.config = {"ENV_KWARGS": {"layout": "coord_ring", "agent_view_size": 2}}

    def build(self):
        return build_replay_data(self.states, self.actions,
                                 np.array([10., 20.]), np.array([1, 2]), np.array([1, 0]),
                                 self.config, "untrained", 1000)

    def test_state_action_and_event_alignment(self):
        data = self.build()
        initial, first, last = data["steps"]
        self.assertIsNone(initial["agents"][0]["action"])
        self.assertEqual(initial["totals"], {"team_reward": 0., "correct_dishes": 0, "wrong_dishes": 0})
        self.assertEqual(first["agents"][0]["action_code"], self.actions[0, 0])
        self.assertEqual(first["agents"][0]["position"], [1, 0])
        self.assertEqual(first["agents"][0]["facing"], "right")
        self.assertEqual(first["agents"][0]["holding"], "onion")
        self.assertEqual(last["agents"][1]["holding"], "plated soup")
        self.assertEqual(first["totals"], {"team_reward": 10., "correct_dishes": 1, "wrong_dishes": 1})
        self.assertEqual(last["totals"], {"team_reward": 30., "correct_dishes": 3, "wrong_dishes": 1})
        self.assertEqual(last["agents"][1]["action_code"], self.actions[1, 1])
        self.assertEqual(last["step"], 2)

    def test_pot_contents_and_cooking_status(self):
        self.states.grid[:, 0, 1, 0] = 5  # POT
        self.states.grid[1, 0, 1, 1:] = [12, 19]
        self.states.grid[2, 0, 1, 1:] = [14, 0]
        data = self.build()
        self.assertEqual(data["steps"][0]["pots"][0]["status"], "waiting for ingredients")
        cooking = data["steps"][1]["pots"][0]
        self.assertEqual(cooking["position"], [1, 0])
        self.assertEqual(cooking["ingredients"], 3)
        self.assertEqual(cooking["timer"], 19)
        self.assertEqual(data["steps"][2]["pots"][0]["status"], "ready to plate")

    def test_missing_state_or_events_rejected(self):
        with self.assertRaisesRegex(ValueError, "event counts"):
            build_replay_data(self.states, self.actions, [0], [0, 0], [0, 0],
                              self.config, "untrained", 1000)
        self.states.grid = self.states.grid[:-1]
        with self.assertRaisesRegex(ValueError, "initial state"):
            self.build()

    def test_self_contained_payload_and_images(self):
        data = self.build()
        data["metadata"]["layout"] = '</script><script>alert("example")</script>'
        frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_replay(data, frames, output)
            html = (output / "replay.html").read_text()
            embedded = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html).group(1)
            payload = json.loads(embedded)
            self.assertEqual(payload["metadata"], data["metadata"])
            self.assertEqual(payload["steps"], data["steps"])
            self.assertNotIn("</script>", embedded)
            self.assertEqual(len(payload["frames"]), 3)
            from base64 import b64decode
            from io import BytesIO
            for frame in payload["frames"]:
                self.assertTrue(frame.startswith("data:image/png;base64,"))
                with Image.open(BytesIO(b64decode(frame.split(",", 1)[1]))) as image:
                    self.assertEqual(image.size, (2, 2))
            self.assertEqual(list(output.iterdir()), [output / "replay.html"])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                save_replay(data, frames[:-1], output)

    def test_saved_scripted_reference_matches_full_audit(self):
        reference = Path(__file__).resolve().parents[2] / "docs/theoretical-maximum"
        html_path = reference / "media/18-soups/replay.html"
        if not html_path.exists():
            self.skipTest("Generate the reference with run.sh scripted first.")
        html = html_path.read_text()
        data = json.loads(re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html).group(1))
        with (reference / "theoretical-maximum-schedule.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(data["steps"]), len(rows) + 1)
        self.assertEqual(len(data["frames"]), len(data["steps"]))
        self.assertEqual(data["metadata"]["mode"], "scripted")
        inventory_codes = [0, 4, 1, 15]
        for row, state in zip(rows, data["steps"][1:]):
            with self.subTest(step=row["step"]):
                self.assertEqual(state["step"], int(row["step"]))
                for agent in state["agents"]:
                    prefix = f"chef{agent['index']}"
                    self.assertEqual(agent["position"], list(ast.literal_eval(row[prefix + "_position"])))
                    self.assertEqual(agent["direction"], int(row[prefix + "_facing"]))
                    self.assertEqual(agent["inventory"], inventory_codes[int(row[prefix + "_inventory"])])
                    self.assertEqual(agent["action"], row[prefix + "_action"])
                for pot, column in zip(state["pots"], ["north_pot", "east_pot"]):
                    code = int(row[column])
                    contents = (code - 10) * 4 if code < 13 else 14 if code == 13 else 12
                    self.assertEqual(pot["contents"], contents)
                    self.assertEqual(pot["timer"], code - 13 if code >= 14 else 0)
                self.assertEqual(state["grid"][2][2][1], inventory_codes[int(row["central_counter"])])
                self.assertEqual(state["totals"]["correct_dishes"], int(row["deliveries"]))
                self.assertEqual(state["totals"]["team_reward"], int(row["deliveries"]) * 20)
        self.assertEqual(data["steps"][-1]["totals"],
                         {"team_reward": 360., "correct_dishes": 18, "wrong_dishes": 0})


if __name__ == "__main__":
    unittest.main()
