"""Milestone and GIF checks using tiny fixtures without saved training output."""

from collections import namedtuple
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jax
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preview
from jaxmarl.environments.overcooked_v2.common import (
    Direction, DynamicObject, OvercookedActionsEnum as Action, StaticObject,
)


EMPTY = int(DynamicObject.EMPTY)
ONION = int(DynamicObject.ingredient(0))
PLATE = int(DynamicObject.PLATE)
SOUP = 3 * ONION | PLATE | int(DynamicObject.COOKED)


def snapshot(step, inventories, actions, stations, items=(0, 0), timers=(0, 0)):
    """Chefs face separate stations, so simultaneous events have clear owners."""
    grid = np.zeros((3, 5, 3), dtype=int)
    agents, pots = [], []
    for chef, (x, direction, target_x) in enumerate(
        ((1, Direction.LEFT, 0), (3, Direction.RIGHT, 4))
    ):
        grid[1, target_x] = [stations[chef], items[chef], timers[chef]]
        agents.append({
            "index": chef, "position": [x, 1], "direction": int(direction),
            "inventory": inventories[chef], "action_code": actions[chef],
        })
        if stations[chef] == StaticObject.POT:
            pots.append({"position": [target_x, 1], "contents": items[chef],
                         "timer": timers[chef]})
    return {"step": step, "grid": grid.tolist(), "agents": agents, "pots": pots}


class MilestoneTests(unittest.TestCase):
    def test_successful_station_interactions_have_completion_steps(self):
        # kind, station, old inventory, new inventory, old/new station contents
        examples = [
            ("pickup_onion", StaticObject.ingredient_pile(0), EMPTY, ONION, 0, 0),
            ("pickup_plate", StaticObject.PLATE_PILE, EMPTY, PLATE, 0, 0),
            ("place_onion_in_pot", StaticObject.POT, ONION, EMPTY, 0, ONION),
            ("collect_soup", StaticObject.POT, PLATE, SOUP, SOUP & ~PLATE, 0),
            ("deliver_soup", StaticObject.GOAL, SOUP, EMPTY, 0, 0),
            ("pickup_from_counter", StaticObject.WALL, EMPTY, ONION, ONION, 0),
            ("place_on_counter", StaticObject.WALL, ONION, EMPTY, 0, ONION),
        ]
        for kind, station, old, new, old_item, new_item in examples:
            with self.subTest(kind=kind):
                stations = (station, StaticObject.WALL)
                before = snapshot(8, (old, EMPTY), (None, None), stations, (old_item, 0))
                after = snapshot(9, (new, EMPTY), (Action.interact, Action.stay),
                                 stations, (new_item, 0))
                event, = preview.detect_milestones({"steps": [before, after]})
                self.assertEqual(event["kind"], kind)
                self.assertEqual(event["end_step"], 9)
                self.assertEqual(event["agent"], 0)
                self.assertEqual(event["target_position"], [0, 1])

    def test_failed_interactions_and_other_actions_do_not_create_milestones(self):
        stations = (StaticObject.ingredient_pile(0), StaticObject.WALL)
        before = snapshot(0, (EMPTY, EMPTY), (None, None), stations)
        unsuccessful = snapshot(1, (EMPTY, EMPTY), (Action.interact, Action.stay), stations)
        changed_without_interacting = snapshot(
            1, (ONION, EMPTY), (Action.stay, Action.stay), stations)
        for after in (unsuccessful, changed_without_interacting):
            with self.subTest(actions=[a["action_code"] for a in after["agents"]]):
                self.assertEqual(preview.detect_milestones({"steps": [before, after]}), [])

    def test_simultaneous_events_keep_their_agent_assignment(self):
        stations = (StaticObject.ingredient_pile(0), StaticObject.PLATE_PILE)
        before = snapshot(0, (EMPTY, EMPTY), (None, None), stations)
        after = snapshot(1, (ONION, PLATE), (Action.interact, Action.interact), stations)
        events = preview.detect_milestones({"steps": [before, after]})
        self.assertEqual([(e["agent"], e["kind"]) for e in events],
                         [(0, "pickup_onion"), (1, "pickup_plate")])

    def test_counter_transfer_is_detected_even_when_final_counter_is_empty(self):
        before = snapshot(0, (ONION, EMPTY), (None, None),
                          (StaticObject.WALL, StaticObject.WALL))
        after = snapshot(1, (EMPTY, ONION), (Action.interact, Action.interact),
                         (StaticObject.WALL, StaticObject.WALL))
        # Chef 0 drops first, then chef 1 picks it up in the same simulator step.
        for state in (before, after):
            state["grid"][1][2][0] = int(StaticObject.WALL)
            for agent, direction in zip(state["agents"], (Direction.RIGHT, Direction.LEFT)):
                agent["direction"] = int(direction)
        events = preview.detect_milestones({"steps": [before, after]})
        self.assertEqual([(e["agent"], e["kind"], e["target_position"]) for e in events],
                         [(0, "place_on_counter", [2, 1]),
                          (1, "pickup_from_counter", [2, 1])])

    def test_cooking_events_are_owned_by_pot_and_ready_is_reported_once(self):
        stations = (StaticObject.POT, StaticObject.WALL)
        states = [
            snapshot(0, (EMPTY, EMPTY), (None, None), stations, (3 * ONION, 0)),
            snapshot(1, (EMPTY, EMPTY), (Action.stay, Action.stay), stations,
                     (3 * ONION, 0), (2, 0)),
            snapshot(2, (EMPTY, EMPTY), (Action.stay, Action.stay), stations,
                     (3 * ONION, 0), (1, 0)),
            snapshot(3, (EMPTY, EMPTY), (Action.stay, Action.stay), stations,
                     (SOUP & ~PLATE, 0)),
            snapshot(4, (EMPTY, EMPTY), (Action.stay, Action.stay), stations,
                     (SOUP & ~PLATE, 0)),
        ]
        events = preview.detect_milestones({"steps": states})
        self.assertEqual([(e["agent"], e["kind"], e["end_step"]) for e in events],
                         [(None, "cooking_started", 1), (None, "soup_ready", 3)])

    def test_segment_advances_exactly_on_completion_and_keeps_unfinished_tail(self):
        events = [{"end_step": 3}, {"end_step": 8}]
        expected = {0: (0, 3, events[0]), 2: (0, 3, events[0]),
                    3: (3, 8, events[1]), 7: (3, 8, events[1]),
                    8: (8, 8, None), 11: (8, 11, None)}
        for step, segment in expected.items():
            with self.subTest(step=step):
                self.assertEqual(preview.milestone_segment(events, step), segment)
        self.assertEqual(preview.milestone_segment([], 4), (0, 4, None))
        # Another chef's earlier milestone must not change this chef's start.
        self.assertEqual(preview.milestone_segment([{"end_step": 6}], 4),
                         (0, 6, {"end_step": 6}))


class TrajectoryDrawingTests(unittest.TestCase):
    def test_past_is_solid_and_future_has_gaps(self):
        image = Image.new("RGB", (180, 60), "black")
        positions = np.array([[[0, 0]], [[1, 0]], [[2, 0]]])
        milestone = {"target_position": [2, 0]}
        preview.draw_milestone_trajectories(
            image, positions, {0: (0, 2, milestone)}, 1,
            grid_shape=(1, 3), origin=(0, 0), kitchen_size=(180, 60))
        # Tile centers are x=30, 90, 150. These probes avoid endpoint markers.
        self.assertGreater(image.getpixel((45, 30))[0], 0)  # Past remains continuous.
        self.assertEqual(image.getpixel((105, 30)), (0, 0, 0))  # Gap in future path.
        self.assertGreater(image.getpixel((115, 30))[0], 0)  # A future dash exists.

    def test_shared_corridor_keeps_both_chef_colors_visible(self):
        image = Image.new("RGB", (180, 60), "black")
        positions = np.array([[[0, 0], [0, 0]], [[1, 0], [1, 0]]])
        preview.draw_milestone_trajectories(
            image, positions, {0: (0, 1, None), 1: (0, 1, None)}, 1,
            grid_shape=(1, 3), origin=(0, 0), kitchen_size=(180, 60))
        pixels = np.asarray(image).astype(int)
        self.assertTrue((pixels[:, :, 0] > pixels[:, :, 2] + 30).any())
        self.assertTrue((pixels[:, :, 2] > pixels[:, :, 0] + 30).any())

    def test_stationary_tail_does_not_invent_a_route_or_divide_by_zero(self):
        image = Image.new("RGB", (60, 60), "black")
        preview.draw_milestone_trajectories(
            image, np.zeros((3, 1, 2), dtype=int), {0: (0, 2, None)}, 2,
            grid_shape=(1, 1), origin=(0, 0), kitchen_size=(60, 60))
        self.assertFalse(np.asarray(image).any())


class PreviewGifTests(unittest.TestCase):
    def setUp(self):
        Position = namedtuple("Position", "x y")
        Agents = namedtuple("Agents", "pos dir inventory")
        State = namedtuple("State", "grid agents recipe")
        grid = np.zeros((4, 4, 5, 3), dtype=int)
        grid[:, 1, 0, 0] = StaticObject.ingredient_pile(0)
        grid[:, 1, 4, 0] = StaticObject.PLATE_PILE
        grid[:, 2, 2, 0] = StaticObject.POT
        # A pot completes independently, alongside the chefs' pickups.
        grid[0, 2, 2, 1:] = [3 * ONION, 1]
        grid[1:, 2, 2, 1] = SOUP & ~PLATE
        states = State(
            grid,
            Agents(Position(np.array([[1, 3], [1, 3], [2, 3], [2, 3]]),
                            np.ones((4, 2), dtype=int)),
                   np.array([[Direction.LEFT, Direction.RIGHT]] * 2
                            + [[Direction.RIGHT, Direction.RIGHT]] * 2),
                   np.array([[EMPTY, EMPTY], [ONION, PLATE], [ONION, PLATE], [ONION, PLATE]])),
            np.full(4, 3 * ONION),
        )
        self.initial = jax.tree.map(lambda value: value[0], states)
        self.actions = np.array([[Action.interact, Action.interact],
                                 [Action.right, Action.stay], [Action.stay, Action.stay]])
        self.trace = (jax.tree.map(lambda value: value[1:], states), self.actions,
                      np.zeros(3), np.zeros(3, dtype=int), np.zeros(3, dtype=int),
                      np.ones(3, dtype=bool))
        self.frames = np.stack([np.full((128, 160, 3), 40 + 20 * step, dtype=np.uint8)
                                for step in range(4)])
        self.probabilities = np.full((3, 2, 6), 0.1)
        for step in range(3):
            for chef in range(2):
                self.probabilities[step, chef, (step + chef) % 6] = 0.5
        self.config = {"ENV_KWARGS": {"layout": "coord_ring", "agent_view_size": 2}}

    def save(self, output, probabilities):
        preview.save_preview(self.initial, self.trace, output, self.config,
                             "trained", 1000, action_probs=probabilities)

    def test_gif_keeps_every_step_and_both_panels_use_previous_decision(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(preview, "render_replay_frames", return_value=self.frames),
            patch.object(preview, "save_replay") as save_replay,
            patch.object(preview, "draw_action_probability_panel",
                         wraps=preview.draw_action_probability_panel) as panels,
            patch.object(preview, "draw_milestone_trajectories",
                         wraps=preview.draw_milestone_trajectories) as trajectories,
        ):
            output = Path(directory)
            self.save(output, self.probabilities)
            with Image.open(output / "rollout.gif") as gif:
                self.assertEqual(gif.n_frames, 4)
                for frame in range(4):
                    gif.seek(frame)
                    self.assertEqual(gif.info["duration"], 1200 if frame == 3 else 100)
                gif_width = gif.width
            with Image.open(output / "initial.png") as first, Image.open(output / "final.png") as last:
                self.assertEqual(first.size, last.size)
                self.assertEqual(first.width, gif_width)
            self.assertEqual(panels.call_count, 8)
            for step in range(4):
                for chef in range(2):
                    call = panels.call_args_list[step * 2 + chef].kwargs
                    if step == 0:
                        self.assertIsNone(call["probabilities"])
                        self.assertIsNone(call["selected_action"])
                    else:
                        np.testing.assert_array_equal(
                            call["probabilities"], self.probabilities[step - 1, chef])
                        self.assertEqual(call["selected_action"], self.actions[step - 1, chef])
            self.assertEqual(trajectories.call_count, 4)
            for step, call in enumerate(trajectories.call_args_list):
                segments = call.args[2]
                self.assertEqual(set(segments), {0, 1})  # Pot events never become chef paths.
                self.assertEqual(call.args[3], step)
                for segment in segments.values():
                    if step == 0:
                        self.assertEqual(segment[1], 1)
                    else:
                        self.assertEqual(segment, (1, step, None))
            save_replay.assert_called_once()
            self.assertEqual(len(save_replay.call_args.args[0]["steps"]), 4)

    def test_gif_without_probabilities_still_renders(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(preview, "render_replay_frames", return_value=self.frames),
            patch.object(preview, "save_replay"),
            patch.object(preview, "draw_action_probability_panel") as panels,
        ):
            self.save(Path(directory), None)
            panels.assert_not_called()
            with Image.open(Path(directory) / "rollout.gif") as gif:
                self.assertEqual(gif.n_frames, 4)

    def test_invalid_probability_arrays_are_rejected(self):
        nan = self.probabilities.copy()
        nan[0, 0, 0] = np.nan
        negative = self.probabilities.copy()
        negative[0, 0, 0] = -0.1
        over_one = self.probabilities.copy()
        over_one[0, 0, 0] = 1.1
        cases = [
            (self.probabilities[:-1], "Expected action probabilities"),
            (self.probabilities[:, :, :-1], "Expected action probabilities"),
            (nan, "finite and between"), (negative, "finite and between"),
            (over_one, "finite and between"),
            (self.probabilities * 0.5, "sum to 1"),
        ]
        with (
            TemporaryDirectory() as directory,
            patch.object(preview, "render_replay_frames", return_value=self.frames),
            patch.object(preview, "save_replay"),
        ):
            for probabilities, message in cases:
                with self.subTest(message=message, shape=probabilities.shape):
                    with self.assertRaisesRegex(ValueError, message):
                        self.save(Path(directory), probabilities)
            self.assertFalse((Path(directory) / "rollout.gif").exists())


if __name__ == "__main__":
    unittest.main()
