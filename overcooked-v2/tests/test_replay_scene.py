"""Check that illustration preserves the meaning of both recorded state schemas."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import read_replay, write_replay
from replay_scene import build_presentation, ingredients


ROOT = Path(__file__).resolve().parents[2]


def soup_record():
    return dict(schema_version=1, metadata=dict(layout="test_layout", mode="untrained"),
        frames=["data:image/png;base64,test"], steps=[dict(step=0,
            agents=[dict(index=0, id="agent_0", color="red", facing="right",
                         position=[0, 0], inventory=16, holding="ingredient", action=None)],
            grid=[[[0, 0, 0], [11, 0, 0], [7, 12, 0]],
                  [[5, 20, 0], [1, 4, 0], [6, 12, 0]]],
            pots=[dict(index=1, position=[0, 1], contents=20, ingredients=2,
                       timer=0, status="waiting for ingredients")],
            recipe_code=20, recipe="code 20",
            totals=dict(team_reward=0, correct_dishes=0, wrong_dishes=0))])


def sushi_record():
    data = soup_record()
    data["metadata"].update(layout="oc2_sushi_1_3",
        renderer=dict(kind="sushi", cook_ticks=96, chop_ticks=8, wash_ticks=24))
    state = data["steps"][0]
    state.pop("recipe_code")
    state.update(recipe="cucumber + cooked rice + nori", grid=[[0, 4, 2], [3, 10, 9]],
        pots=[dict(index=5, position=[0, 0], ingredients=1, capacity=1,
                   timer=72, status="cooking: 72 steps remaining")], items=[
            dict(slot=5, kind="POT", location=-1, contents=2, progress=24,
                 description="pot: rice 24/96", return_due=0),
            dict(slot=2, kind="PLATE", location=2, contents=5, progress=0,
                 description="plate: cucumber, nori", return_due=0),
            dict(slot=3, kind="DIRTY_PLATE", location=-3, contents=0, progress=0,
                 description="dirty plate (wash 0/24)", return_due=100),
            dict(slot=4, kind="RAW_CUCUMBER", location=3, contents=0, progress=4,
                 description="raw cucumber (chop 4/8)", return_due=0)])
    return data


class ReplaySceneTests(unittest.TestCase):
    def test_mixed_ingredients_keep_identity_and_recipe(self):
        data = soup_record()
        before = deepcopy(data)
        result = build_presentation(data)
        self.assertTrue(result["supported"])
        self.assertEqual((result["width"], result["height"]), (3, 2))
        scene = result["scenes"][0]
        self.assertEqual(scene["recipe"], "1 × onion + 1 × ingredient 1")
        self.assertEqual(scene["held"]["0"]["ingredients"], [1])
        self.assertEqual(scene["held"]["0"]["sprite"], "ingredient")
        self.assertEqual(scene["pots"][0]["ingredient_types"], [0, 1])
        self.assertEqual(data, before)

    def test_recipe_switch_is_recorded_per_state(self):
        data = soup_record()
        second = deepcopy(data["steps"][0])
        second["grid"][0][2] = [7, 48, 1]
        second["recipe_code"] = 48
        data["steps"].append(second)
        result = build_presentation(data)
        first, last = result["scenes"]
        switch = lambda scene: next(i for i in scene["items"] if i["sprite"] == "recipe_id" and i["position"] == [2, 0])
        self.assertFalse(switch(first)["active"])
        indicator = next(i for i in first["items"] if i["sprite"] == "recipe_id" and i["position"] == [2, 1])
        self.assertEqual(indicator["ingredients"], [0, 1])
        self.assertTrue(switch(last)["active"])
        self.assertEqual(switch(last)["ingredients"], [1, 1, 1])
        self.assertEqual(last["recipe"], "3 × ingredient 1")

    def test_movable_pot_partial_plate_queue_and_custom_timings(self):
        data = sushi_record()
        result = build_presentation(data)
        self.assertTrue(result["supported"])
        scene = result["scenes"][0]
        self.assertEqual(scene["held"]["0"]["sprite"], "rice_pot")
        self.assertEqual(scene["held"]["0"]["progress"], .25)
        plate = next(i for i in scene["items"] if i["slot"] == 2)
        self.assertEqual(plate["sprite"], "plate")
        self.assertEqual(plate["toppings"], ["chopped_cucumber", "nori"])
        self.assertNotIn(3, [i["slot"] for i in scene["items"]])
        cucumber = next(i for i in scene["items"] if i["slot"] == 4)
        self.assertEqual(cucumber["progress"], .5)
        self.assertEqual(scene["pots"][0]["progress"], .25)
        self.assertIn("Off hob", scene["pots"][0]["status"])
        self.assertEqual(scene["pots"][0]["position"], [0, 0])

    def test_legacy_sushi_does_not_invent_cooking_duration(self):
        data = sushi_record()
        data["metadata"].pop("renderer")
        result = build_presentation(data)
        self.assertTrue(result["supported"])
        scene = result["scenes"][0]
        self.assertIsNone(scene["pots"][0]["progress"])
        self.assertIsNone(scene["held"]["0"]["progress"])
        self.assertEqual(next(i for i in scene["items"] if i["slot"] == 4)["progress"], .5)

    def test_unknown_tiles_and_chef_art_fall_back_without_changing_data(self):
        for change in ("tile", "chef"):
            data = soup_record()
            if change == "tile":
                data["steps"][0]["grid"][0][0][0] = 8
            else:
                data["steps"][0]["agents"][0]["color"] = "green"
            before = deepcopy(data)
            result = build_presentation(data)
            self.assertFalse(result["supported"])
            self.assertTrue(result["reason"])
            self.assertEqual(result["scenes"], [])
            self.assertEqual(data, before)
        with self.assertRaises(ValueError):
            ingredients(-1)

    def test_verified_reference_survives_restyling_and_roundtrip(self):
        from tempfile import TemporaryDirectory
        original = read_replay(ROOT / "docs/theoretical-maximum/media/18-soups/replay.html")
        before = deepcopy(original)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "replay.html"
            write_replay(original, target)
            upgraded = read_replay(target)
            self.assertEqual(upgraded["steps"], original["steps"])
            self.assertEqual(upgraded["metadata"], original["metadata"])
            self.assertEqual(upgraded["frames"], original["frames"])
            self.assertTrue(upgraded["presentation"]["supported"])
            write_replay(upgraded, target)
            self.assertEqual(read_replay(target), upgraded)
            self.assertNotIn("__KITCHEN_ATLAS__", target.read_text())
            self.assertEqual(upgraded["steps"][-1]["totals"],
                             dict(team_reward=360., correct_dishes=18, wrong_dishes=0))
        self.assertEqual(original, before)

    def test_empty_and_mismatched_payloads_rejected(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            data = soup_record()
            data["frames"] = []
            with self.assertRaisesRegex(ValueError, "incomplete"):
                write_replay(data, Path(directory) / "replay.html")


if __name__ == "__main__":
    unittest.main()
