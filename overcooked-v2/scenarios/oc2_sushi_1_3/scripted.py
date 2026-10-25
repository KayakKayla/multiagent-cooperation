"""A two-chef handoff script that exercises real actions, never edits state."""
from collections import deque
import json
from pathlib import Path

import jax
import numpy as np

from .environment import Action, Item, RECIPE, RICE_BIT
from .layout import DRYING_CELL, RETURN_CELL, Tile


def audit(env, state):
    """Check ownership, finite resources, and timer bounds after every action."""
    s = jax.device_get(state)
    active = s.kind != Item.NONE
    assert np.all((s.kind >= Item.NONE) & (s.kind <= Item.DIRTY_PLATE)), "Unknown item kind"
    assert np.all(s.location[active] >= -3), "Active item has no owner"
    assert np.sum(s.kind == Item.POT) == 3, "Lost or duplicated pot"
    assert np.sum((s.kind == Item.PLATE) | (s.kind == Item.DIRTY_PLATE)) == 4, "Lost or duplicated plate"
    assert np.all(s.location[~active] == -4), "Inactive item still has an owner"
    assert np.all(s.contents[~active] == 0) and np.all(s.progress[~active] == 0)
    for holder in (-1, -2):
        assert np.sum(active & (s.location == holder)) <= 1, "Chef holds duplicate items"
    cells = s.location[active & (s.location >= 0)]
    assert len(cells) == len(set(cells.tolist())), "Two items occupy one station"
    assert np.all((cells >= 0) & (cells < env.width * env.height))
    assert np.all(np.asarray(env.flat_tiles)[cells] != Tile.FLOOR), "Item escaped onto floor"
    assert np.all(s.progress >= 0)
    for kind, maximum in ((Item.RAW_CUCUMBER, env.chop_ticks), (Item.POT, env.cook_ticks),
                           (Item.DIRTY_PLATE, env.wash_ticks)):
        assert np.all(s.progress[s.kind == kind] <= maximum)
    assert np.all(np.isin(s.contents[s.kind == Item.POT], [0, RICE_BIT]))
    assert np.all((s.contents[s.kind == Item.PLATE] >= 0) & (s.contents[s.kind == Item.PLATE] <= RECIPE))
    containers = (s.kind == Item.POT) | (s.kind == Item.PLATE)
    assert np.all(s.contents[~containers] == 0), "Contents attached to a non-container"
    assert np.all(s.kind[s.location == -3] == Item.DIRTY_PLATE)
    xy = s.positions
    assert len(set(map(tuple, xy.tolist()))) == 2, "Chefs overlap"
    assert all(np.asarray(env.tiles)[y, x] == Tile.FLOOR for x, y in xy)


def food_counts(state):
    s = jax.device_get(state)
    return np.array([
        np.sum(np.isin(s.kind, [Item.RAW_CUCUMBER, Item.CHOPPED_CUCUMBER])) + np.sum((s.contents & 1) != 0),
        np.sum(s.kind == Item.RICE) + np.sum((s.contents & 2) != 0),
        np.sum(s.kind == Item.NORI) + np.sum((s.contents & 4) != 0),
    ], dtype=int)


class Script:
    def __init__(self, env):
        self.env = env
        self.key = jax.random.PRNGKey(0)
        _, self.state = env.reset(self.key)
        self.states = [self.state]
        self.actions = [None]
        self.labels = ["Initial kitchen"]
        self.milestones = []
        self.events = []
        self.spawn_counts = np.zeros(3, dtype=int)
        audit(env, self.state)

    def advance(self, agent, action, label, require_success=False):
        if bool(self.state.done):
            raise RuntimeError("Script exceeded episode limit")
        old = self.state
        actions = {name: int(Action.STAY) for name in self.env.agents}
        actions[self.env.agents[agent]] = int(action)
        _, self.state, _, _, info = self.env.step_env(self.key, self.state, actions)
        audit(self.env, self.state)
        if require_success and not bool(info["interaction_succeeded"][agent]):
            raise RuntimeError(f"Failed scripted interaction: {label}")
        # Independent food ledger: only successful dispenser pickups introduce food.
        if action == Action.PICKUP_PLACE and not np.any(np.asarray(old.location) == -agent - 1):
            directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
            dx, dy = directions[int(old.directions[agent])]
            x, y = map(int, old.positions[agent])
            tile = int(self.env.tiles[y + dy, x + dx])
            if tile in (Tile.CUCUMBER, Tile.RICE, Tile.NORI) and bool(info["interaction_succeeded"][agent]):
                self.spawn_counts[(Tile.CUCUMBER, Tile.RICE, Tile.NORI).index(tile)] += 1
        assert np.array_equal(food_counts(self.state) + int(self.state.deliveries), self.spawn_counts), "Food ledger mismatch"
        self.states.append(self.state)
        self.actions.append([actions[a] for a in self.env.agents])
        self.labels.append(label)
        self.events.append({k: np.asarray(v).tolist() for k, v in info.items()})

    def wait(self, steps, label):
        for _ in range(steps):
            self.advance(0, Action.STAY, label)

    def approach(self, agent, station, label):
        env = self.env
        start = tuple(map(int, self.state.positions[agent]))
        other = tuple(map(int, self.state.positions[1-agent]))
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        goals = {(station[0] - dx, station[1] - dy): action for action, (dx, dy) in enumerate(directions)}
        queue, paths = deque([start]), {start: []}
        chosen = None
        while queue:
            point = queue.popleft()
            if point in goals:
                chosen = point
                break
            for action, (dx, dy) in enumerate(directions):
                nxt = point[0] + dx, point[1] + dy
                x, y = nxt
                if (nxt not in paths and nxt != other and 0 <= x < env.width and 0 <= y < env.height
                        and int(env.tiles[y, x]) == Tile.FLOOR):
                    paths[nxt] = paths[point] + [action]
                    queue.append(nxt)
        if chosen is None:
            raise RuntimeError(f"No route for chef {agent} to {station}")
        for action in paths[chosen]:
            self.advance(agent, action, label)
        facing = goals[chosen]
        if int(self.state.directions[agent]) != facing:
            self.advance(agent, facing, label)

    def interact(self, agent, station, label):
        self.approach(agent, station, label)
        self.advance(agent, Action.PICKUP_PLACE, label, require_success=True)

    def mark(self, name, **details):
        self.milestones.append(dict(name=name, step=int(self.state.time), **details))
        self.labels[-1] = name

    def run(self):
        env = self.env
        shared, plate, hob, board = (6, 4), (3, 4), (10, 0), (5, 1)
        for cycle in (1, 2):
            tag = f"Dish {cycle}"
            self.interact(0, (0, 3), tag + ": collect rice")
            self.interact(0, shared, tag + ": pass rice to chef 1")
            self.interact(1, shared, tag + ": receive rice")
            self.interact(1, hob, tag + ": start rice cooking")
            self.mark(tag + ": rice cooking")
            self.interact(1, (13, 3), tag + ": collect cucumber")
            self.interact(1, shared, tag + ": pass cucumber to chef 0")
            self.interact(0, shared, tag + ": receive cucumber")
            self.interact(0, board, tag + ": place cucumber on board")
            self.approach(0, board, tag + ": chop cucumber")
            self.advance(0, Action.WORK, tag + ": partial chopping", require_success=True)
            self.mark(tag + ": partial chopping")
            for _ in range(env.chop_ticks - 1):
                self.advance(0, Action.WORK, tag + ": chop cucumber", require_success=True)
            self.interact(0, board, tag + ": collect chopped cucumber")
            self.interact(0, plate, tag + ": plate cucumber")
            self.interact(0, (0, 5), tag + ": collect nori")
            self.interact(0, plate, tag + ": add nori")
            pot_mask = np.asarray(self.state.kind) == Item.POT
            pot = int(np.flatnonzero(pot_mask & (np.asarray(self.state.location) == env.cell(hob)))[0])
            self.wait(max(0, env.cook_ticks - int(self.state.progress[pot])), tag + ": wait for cooked rice")
            self.mark(tag + ": rice cooked", pot_slot=pot)
            self.interact(1, hob, tag + ": carry cooked pot")
            self.interact(1, shared, tag + ": pass cooked pot")
            self.mark(tag + ": movable cooked pot")
            self.interact(0, plate, tag + ": collect plate")
            self.interact(0, shared, tag + ": add cooked rice")
            self.interact(0, plate, tag + ": put assembled sushi on counter")
            plated = np.flatnonzero((np.asarray(self.state.kind) == Item.PLATE) &
                                    (np.asarray(self.state.contents) == RECIPE))
            assert len(plated) == 1 and int(plated[0]) == 3, "Washed plate was not reused"
            self.mark(tag + ": assembled sushi", plate_slot=3)
            self.interact(1, shared, tag + ": retrieve empty pot")
            self.interact(1, hob, tag + ": replace empty pot")
            self.interact(0, plate, tag + ": collect sushi")
            self.interact(0, (0, 6), tag + ": serve sushi")
            self.mark(tag + ": served", plate_slot=3)
            # Chef 0 steps away from the serving corner while chef 1 washes.
            self.wait(env.return_ticks, tag + ": wait for dirty plate")
            self.mark(tag + ": dirty plate returned")
            self.interact(1, RETURN_CELL, tag + ": collect dirty plate")
            self.interact(1, (10, 7), tag + ": put dirty plate in sink")
            self.approach(1, (10, 7), tag + ": wash plate")
            for _ in range(env.wash_ticks):
                self.advance(1, Action.WORK, tag + ": wash plate", require_success=True)
            self.mark(tag + ": washed plate", plate_slot=3)
            self.interact(1, DRYING_CELL, tag + ": collect washed plate")
            self.interact(1, shared, tag + ": return washed plate to chef 0")
            self.interact(0, shared, tag + ": receive washed plate")
            self.interact(0, plate, tag + ": restore plate to assembly counter")
            self.mark(tag + ": plate ready for reuse", plate_slot=3)
        assert int(self.state.deliveries) == 2
        return self


def compare_reference(script):
    """Compare semantic checkpoints, not exact time, IDs, geometry, or score."""
    reference = json.loads(Path(__file__).with_name("reference_workflow.json").read_text())
    env = script.env
    checkpoints = {m["name"]: script.states[m["step"]] for m in script.milestones}
    observed = reference["observed"]
    assert env.rules.cook_seconds == observed["required_rice_cooking_units"]
    names = [name for bit, name in ((1, "cucumber"), (2, "cooked rice"), (4, "nori")) if RECIPE & bit]
    assert names == observed["plate_ingredients"]
    for cycle in (1, 2):
        partial = checkpoints[f"Dish {cycle}: rice cooking"]
        pots = np.flatnonzero((np.asarray(partial.kind) == Item.POT) & (np.asarray(partial.contents) == RICE_BIT))
        assert len(pots) == 1 and 0 < int(partial.progress[pots[0]]) < env.cook_ticks
        moved = checkpoints[f"Dish {cycle}: movable cooked pot"]
        pot = int(pots[0])
        assert int(moved.progress[pot]) == env.cook_ticks and int(moved.contents[pot]) == RICE_BIT
        assert int(env.flat_tiles[int(moved.location[pot])]) == Tile.COUNTER
        assembled = checkpoints[f"Dish {cycle}: assembled sushi"]
        assert int(assembled.contents[3]) == RECIPE
        assert int(assembled.contents[pot]) == 0 and int(assembled.progress[pot]) == 0
        returned = checkpoints[f"Dish {cycle}: dirty plate returned"]
        assert int(returned.kind[3]) == Item.DIRTY_PLATE and int(returned.location[3]) == env.return_cell
        washed = checkpoints[f"Dish {cycle}: washed plate"]
        assert int(washed.kind[3]) == Item.PLATE and int(washed.contents[3]) == 0
        assert int(washed.location[3]) == env.drying_cell
    return {"semantic_checkpoints_passed": True, "reference": "reference_workflow.json",
            "scope": reference["scope"], "source_assembly_sha256": reference["assembly_sha256"]}
