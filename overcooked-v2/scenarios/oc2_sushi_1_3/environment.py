"""JAX-native reference rules for one sushi recipe, not commercial physics.

State is a fixed-size pytree. Item slots have exactly one location: a cell,
a chef's hands (-1/-2), the return queue (-3), or no location (-4).
Transfers consume ingredient slots but preserve the four plates/three pots.
"""
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .layout import DRYING_CELL, PLATES, POTS, RETURN_CELL, SPAWNS, Tile, grid


class Action(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    STAY = 4
    PICKUP_PLACE = 5
    WORK = 6


class Item(IntEnum):
    NONE = 0
    RAW_CUCUMBER = 1
    CHOPPED_CUCUMBER = 2
    RICE = 3
    NORI = 4
    POT = 5
    PLATE = 6
    DIRTY_PLATE = 7


CUCUMBER_BIT, RICE_BIT, NORI_BIT = 1, 2, 4
RECIPE = CUCUMBER_BIT | RICE_BIT | NORI_BIT
QUEUED, ABSENT = -3, -4
DIRECTIONS = jnp.array(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=jnp.int32)


@dataclass(frozen=True)
class Rules:
    # See calibration_reference.json for measurements and remaining uncertainty.
    step_seconds: float = 0.25
    cook_seconds: float = 12.0
    chop_seconds: float = 1.4
    wash_seconds: float = 3.0
    return_seconds: float = 10.0
    delivery_reward: float = 20.0
    max_steps: int = 600
    capacity: int = 32

    def __post_init__(self):
        for name in ("step_seconds", "cook_seconds", "chop_seconds", "wash_seconds", "return_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(self.delivery_reward) or self.delivery_reward <= 0:
            raise ValueError("delivery_reward must be finite and positive")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(self.capacity, int) or self.capacity < len(POTS) + len(PLATES) + 1:
            raise ValueError("capacity must leave at least one ingredient slot")

    def ticks(self, seconds):
        return max(1, math.ceil(seconds / self.step_seconds))


class State(NamedTuple):
    positions: jax.Array
    directions: jax.Array
    kind: jax.Array
    location: jax.Array
    contents: jax.Array
    progress: jax.Array
    return_due: jax.Array
    time: jax.Array
    deliveries: jax.Array
    score: jax.Array
    done: jax.Array


@dataclass(frozen=True)
class Discrete:
    n: int

    def sample(self, key):
        return jax.random.randint(key, (), 0, self.n)


@dataclass(frozen=True)
class Box:
    shape: tuple
    low: float = 0.0
    high: float = 1.0


class SushiEnvironment:
    name = "oc2_sushi_1_3"
    schema_version = 1
    num_agents, num_actions = 2, len(Action)
    agents = ["agent_0", "agent_1"]
    agent_classes = {"chef": agents}

    def __init__(self, rules=None, agent_view_size=None):
        self.rules = rules or Rules()
        if agent_view_size is not None and (not isinstance(agent_view_size, int) or agent_view_size < 0):
            raise ValueError("agent_view_size must be None or a nonnegative integer")
        self.agent_view_size = agent_view_size
        self.tiles = jnp.array(grid(), dtype=jnp.int32)
        self.height, self.width = self.tiles.shape
        self.flat_tiles = self.tiles.reshape(-1)
        self.cook_ticks = self.rules.ticks(self.rules.cook_seconds)
        self.chop_ticks = self.rules.ticks(self.rules.chop_seconds)
        self.wash_ticks = self.rules.ticks(self.rules.wash_seconds)
        self.return_ticks = self.rules.ticks(self.rules.return_seconds)
        self.return_cell = self.cell(RETURN_CELL)
        self.drying_cell = self.cell(DRYING_CELL)
        # terrain14 + item12 + self/other position2 + directions8 + held12 = 48
        size = (self.height, self.width) if agent_view_size is None else (2 * agent_view_size + 1,) * 2
        self.obs_shape = (*size, 48)

    def cell(self, xy):
        return xy[1] * self.width + xy[0]

    def action_space(self, agent):
        return Discrete(self.num_actions)

    def observation_space(self, agent):
        return Box(self.obs_shape)

    def get_avail_actions(self, state):
        return {agent: jnp.ones(self.num_actions, dtype=jnp.bool_) for agent in self.agents}

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        del key  # Fixed starts make the first scripted comparison reproducible.
        n = self.rules.capacity
        kind = jnp.zeros(n, dtype=jnp.int32).at[:3].set(int(Item.POT)).at[3:7].set(int(Item.PLATE))
        locations = jnp.full(n, ABSENT, dtype=jnp.int32)
        locations = locations.at[:7].set(jnp.array([self.cell(p) for p in (*POTS, *PLATES)]))
        s = State(jnp.array(SPAWNS, dtype=jnp.int32), jnp.array([3, 3], dtype=jnp.int32),
                  kind, locations, jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
                  jnp.full(n, -1, dtype=jnp.int32), jnp.int32(0), jnp.int32(0), jnp.float32(0), jnp.bool_(False))
        return self.get_obs(s), s

    def _write(self, s, slot, enabled, **values):
        updates = {}
        for field, value in values.items():
            array = getattr(s, field)
            updates[field] = array.at[slot].set(jnp.where(enabled, value, array[slot]))
        return s._replace(**updates)

    def _empty(self, s, slot, enabled):
        return self._write(s, slot, enabled, kind=int(Item.NONE), location=ABSENT,
                           contents=0, progress=0, return_due=-1)

    def _interact(self, s, agent, action, worked):
        held_mask = (s.kind != Item.NONE) & (s.location == -agent - 1)
        has_hand, hand = held_mask.any(), jnp.argmax(held_mask)
        front = s.positions[agent] + DIRECTIONS[s.directions[agent]]
        inside = ((front >= 0) & (front < jnp.array([self.width, self.height]))).all()
        cell = jnp.clip(self.cell(front), 0, self.width * self.height - 1)
        tile = jnp.where(inside, self.flat_tiles[cell], int(Tile.WALL))
        at_front = (s.kind != Item.NONE) & (s.location == cell) & inside
        has_target, target = at_front.any(), jnp.argmax(at_front)
        hk, tk = s.kind[hand], s.kind[target]
        hc, tc = s.contents[hand], s.contents[target]
        pickup = action == Action.PICKUP_PLACE

        spawn_kind = jnp.where(tile == Tile.CUCUMBER, int(Item.RAW_CUCUMBER),
                     jnp.where(tile == Tile.RICE, int(Item.RICE),
                     jnp.where(tile == Tile.NORI, int(Item.NORI), int(Item.NONE))))
        free = s.kind == Item.NONE
        slot = jnp.argmax(free)
        spawn = pickup & ~has_hand & (spawn_kind != Item.NONE) & free.any()
        lift = pickup & ~has_hand & has_target
        s = self._write(s, slot, spawn, kind=spawn_kind, location=-agent - 1,
                        contents=0, progress=0, return_due=-1)
        s = self._write(s, target, lift, location=-agent - 1)

        # Surfaces have one slot. Dirty plates only go to a counter or sink.
        surface = ((tile == Tile.COUNTER) | (tile == Tile.BOARD)) & (hk != Item.DIRTY_PLATE)
        surface |= (tile == Tile.COUNTER) & (hk == Item.DIRTY_PLATE)
        surface |= (tile == Tile.HOB) & (hk == Item.POT)
        surface |= (tile == Tile.SINK) & (hk == Item.DIRTY_PLATE)
        place = pickup & has_hand & ~has_target & surface
        s = self._write(s, hand, place, location=cell)

        # A cooked pot pours rice without destroying the pot. Ingredient slots
        # are consumed only after a valid transfer to an empty recipe component.
        hand_bit = jnp.where(hk == Item.CHOPPED_CUCUMBER, CUCUMBER_BIT,
                   jnp.where(hk == Item.NORI, NORI_BIT,
                   jnp.where((hk == Item.POT) & (hc == RICE_BIT) &
                             (s.progress[hand] >= self.cook_ticks), RICE_BIT, 0)))
        target_bit = jnp.where(tk == Item.CHOPPED_CUCUMBER, CUCUMBER_BIT,
                     jnp.where(tk == Item.NORI, NORI_BIT,
                     jnp.where((tk == Item.POT) & (tc == RICE_BIT) &
                               (s.progress[target] >= self.cook_ticks), RICE_BIT, 0)))
        base = pickup & has_hand & has_target
        onto_plate = base & (tk == Item.PLATE) & (hand_bit != 0) & ((tc & hand_bit) == 0)
        into_plate = base & (hk == Item.PLATE) & (target_bit != 0) & ((hc & target_bit) == 0)
        into_pot = base & (hk == Item.RICE) & (tk == Item.POT) & (tc == 0)
        s = self._write(s, target, onto_plate, contents=tc | hand_bit)
        s = self._write(s, hand, into_plate, contents=hc | target_bit)
        s = self._write(s, target, into_pot, contents=RICE_BIT, progress=0)
        s = self._empty(s, hand, (onto_plate & (hk != Item.POT)) | into_pot)
        s = self._empty(s, target, into_plate & (tk != Item.POT))
        s = self._write(s, hand, onto_plate & (hk == Item.POT), contents=0, progress=0)
        s = self._write(s, target, into_plate & (tk == Item.POT), contents=0, progress=0)

        delivered = pickup & has_hand & (tile == Tile.SERVE) & (hk == Item.PLATE) & (hc == RECIPE)
        s = self._write(s, hand, delivered, kind=int(Item.DIRTY_PLATE), location=QUEUED,
                        contents=0, progress=0, return_due=s.time + self.return_ticks + 1)
        discard = pickup & has_hand & (tile == Tile.BIN) & (hk != Item.DIRTY_PLATE)
        keep = (hk == Item.POT) | (hk == Item.PLATE)
        s = self._empty(s, hand, discard & ~keep)
        s = self._write(s, hand, discard & keep, contents=0, progress=0)

        work = (action == Action.WORK) & ~has_hand & has_target & ~worked[target]
        chop = work & (tile == Tile.BOARD) & (tk == Item.RAW_CUCUMBER)
        wash = work & (tile == Tile.SINK) & (tk == Item.DIRTY_PLATE)
        new_chop = jnp.minimum(s.progress[target] + 1, self.chop_ticks)
        s = self._write(s, target, chop, progress=new_chop,
                        kind=jnp.where(new_chop >= self.chop_ticks, int(Item.CHOPPED_CUCUMBER), int(Item.RAW_CUCUMBER)))
        new_wash = jnp.minimum(s.progress[target] + 1, self.wash_ticks)
        dry_free = ~((s.kind != Item.NONE) & (s.location == self.drying_cell)).any()
        wash_done = wash & (new_wash >= self.wash_ticks) & dry_free
        s = self._write(s, target, wash, progress=new_wash)
        s = self._write(s, target, wash_done, kind=int(Item.PLATE), location=self.drying_cell,
                        contents=0, progress=0, return_due=-1)
        worked = worked.at[target].set(worked[target] | chop | wash)
        success = spawn | lift | place | onto_plate | into_plate | into_pot | delivered | discard | chop | wash
        invalid = ((action == Action.PICKUP_PLACE) | (action == Action.WORK)) & ~success
        return s, jnp.array([delivered, success, invalid]), worked

    def _advance(self, s, actions):
        valid_actions = (actions >= 0) & (actions < self.num_actions)
        actions = jnp.where(valid_actions, actions, int(Action.STAY))
        direction = jnp.minimum(actions, 3)
        moving = actions < 4
        target = s.positions + DIRECTIONS[direction] * moving[:, None]
        inside = ((target >= 0) & (target < jnp.array([self.width, self.height]))).all(axis=1)
        cells = jnp.clip(target[:, 1] * self.width + target[:, 0], 0, self.width * self.height - 1)
        can_move = inside & (self.flat_tiles[cells] == Tile.FLOOR)
        target = jnp.where(can_move[:, None], target, s.positions)
        same = (target[0] == target[1]).all()
        swap = (target[0] == s.positions[1]).all() & (target[1] == s.positions[0]).all()
        collision = same | swap
        positions = jnp.where(collision, s.positions, target)
        blocked = moving & (positions == s.positions).all(axis=1)
        s = s._replace(positions=positions, directions=jnp.where(moving, direction, s.directions))
        events = jnp.zeros((2, 3), dtype=jnp.bool_)
        worked = jnp.zeros(self.rules.capacity, dtype=jnp.bool_)
        # Alternate interaction priority; simultaneous movement is resolved first.
        for offset in range(2):
            agent = (s.time + offset) % 2
            s, event, worked = self._interact(s, agent, actions[agent], worked)
            events = events.at[agent].set(event)
        on_grid = s.location >= 0
        tile = self.flat_tiles[jnp.clip(s.location, 0, self.width * self.height - 1)]
        cooking = (s.kind == Item.POT) & (s.contents == RICE_BIT) & on_grid & (tile == Tile.HOB)
        s = s._replace(progress=jnp.where(cooking, jnp.minimum(s.progress + 1, self.cook_ticks), s.progress), time=s.time + 1)

        def return_plate(slot, state):
            free = ~((state.kind != Item.NONE) & (state.location == self.return_cell)).any()
            due = (state.kind[slot] == Item.DIRTY_PLATE) & (state.location[slot] == QUEUED)
            due &= (state.return_due[slot] <= state.time) & free
            return self._write(state, slot, due, location=self.return_cell, return_due=-1)

        s = jax.lax.fori_loop(0, self.rules.capacity, return_plate, s)
        deliveries = events[:, 0].sum(dtype=jnp.int32)
        reward = deliveries.astype(jnp.float32) * self.rules.delivery_reward
        s = s._replace(deliveries=s.deliveries + deliveries, score=s.score + reward,
                       done=s.time >= self.rules.max_steps)
        info = {"delivered": events[:, 0], "interaction_succeeded": events[:, 1],
                "invalid_action": events[:, 2] | ~valid_actions, "blocked": blocked,
                "deliveries": s.deliveries}
        return s, reward, info

    @partial(jax.jit, static_argnums=0)
    def step_env(self, key, state, actions):
        del key
        values = jnp.stack([jnp.asarray(actions[a], dtype=jnp.int32) for a in self.agents])
        empty_info = {"delivered": jnp.zeros(2, dtype=jnp.bool_),
                      "interaction_succeeded": jnp.zeros(2, dtype=jnp.bool_),
                      "invalid_action": jnp.zeros(2, dtype=jnp.bool_),
                      "blocked": jnp.zeros(2, dtype=jnp.bool_), "deliveries": state.deliveries}
        s, reward, info = jax.lax.cond(state.done, lambda: (state, jnp.float32(0), empty_info),
                                     lambda: self._advance(state, values))
        return (self.get_obs(s), s, {a: reward for a in self.agents},
                {**{a: s.done for a in self.agents}, "__all__": s.done}, info)

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, actions, reset_state=None):
        key, reset_key = jax.random.split(key)
        obs, s, rewards, dones, info = self.step_env(key, state, actions)
        _, initial = self.reset(reset_key) if reset_state is None else (None, reset_state)
        s = jax.tree.map(lambda old, new: jnp.where(dones["__all__"], new, old), s, initial)
        return self.get_obs(s), s, rewards, dones, info

    def item_features(self, s):
        kinds = jax.nn.one_hot(s.kind, len(Item)) * (s.kind != Item.NONE)[:, None]
        contents = ((s.contents[:, None] & jnp.array([1, 2, 4])) != 0).astype(jnp.float32)
        duration = jnp.where(s.kind == Item.POT, self.cook_ticks,
                   jnp.where(s.kind == Item.DIRTY_PLATE, self.wash_ticks, self.chop_ticks))
        progress = jnp.clip(s.progress / duration, 0, 1)[:, None]
        return jnp.concatenate([kinds, contents, progress], axis=1)

    def get_obs(self, s):
        features = self.item_features(s)
        on_grid = (s.kind != Item.NONE) & (s.location >= 0)
        held = (s.kind != Item.NONE) & ((s.location == -1) | (s.location == -2))
        carriers = s.positions[jnp.clip(-s.location - 1, 0, 1)]
        carried_cell = carriers[:, 1] * self.width + carriers[:, 0]
        cell = jnp.clip(jnp.where(held, carried_cell, s.location), 0, self.width * self.height - 1)
        items = jnp.zeros((self.width * self.height, 12), dtype=jnp.float32).at[cell].max(features * (on_grid | held)[:, None])
        items = items.reshape(self.height, self.width, 12)
        terrain = jax.nn.one_hot(self.tiles, len(Tile))
        yy, xx = jnp.indices((self.height, self.width))
        observations = {}
        for agent, name in enumerate(self.agents):
            own = (xx == s.positions[agent, 0]) & (yy == s.positions[agent, 1])
            other = (xx == s.positions[1-agent, 0]) & (yy == s.positions[1-agent, 1])
            facing = jnp.concatenate([own[:, :, None] * jax.nn.one_hot(s.directions[agent], 4),
                                      other[:, :, None] * jax.nn.one_hot(s.directions[1-agent], 4)], axis=2)
            held = (s.kind != Item.NONE) & (s.location == -agent - 1)
            inventory = (features * held[:, None]).sum(axis=0)
            inventory = jnp.broadcast_to(inventory, (self.height, self.width, 12))
            obs = jnp.concatenate([terrain, items, own[:, :, None], other[:, :, None], facing, inventory], axis=2)
            if self.agent_view_size is not None:
                r = self.agent_view_size
                obs = jnp.pad(obs, ((r, r), (r, r), (0, 0)))
                obs = jax.lax.dynamic_slice(obs, (s.positions[agent, 1], s.positions[agent, 0], 0), self.obs_shape)
            observations[name] = obs
        return observations
