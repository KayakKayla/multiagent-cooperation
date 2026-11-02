"""Compact an audited serial teacher into simultaneous real chef actions.

This is a training-only planner. It never belongs in neural evaluation. A longer
reference supplies per-chef task queues; the recorded episode keeps the original
600-step rules, resources and empty initial kitchen. No food/state is synthesized.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scenarios import make_env
from .demonstrations import save_demo
from .environment import Action, Item
from .layout import Tile
from .scripted import audit, food_counts
from .throughput_expert import ThroughputExpert


DIRECTIONS = np.asarray(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=np.int32)


@dataclass
class QueuedAction:
    serial_step: int
    action: int
    before: object
    after: object
    success: bool
    delivered: bool
    label: str


def _item_at(state, mask):
    slots = np.flatnonzero(np.asarray(mask))
    if not len(slots):
        return (int(Item.NONE), 0, 0)
    if len(slots) != 1:
        raise AssertionError("A semantic holder has more than one item")
    slot = int(slots[0])
    return (int(state.kind[slot]), int(state.contents[slot]), int(state.progress[slot]))


def _hand(state, agent):
    return _item_at(state, (np.asarray(state.kind) != Item.NONE)
                    & (np.asarray(state.location) == -agent - 1))


def _front(env, state, agent):
    point = np.asarray(state.positions[agent]) + DIRECTIONS[int(state.directions[agent])]
    x, y = map(int, point)
    if not (0 <= x < env.width and 0 <= y < env.height):
        return (x, y, int(Tile.WALL), int(Item.NONE), 0, 0)
    cell = env.cell((x, y))
    item = _item_at(state, (np.asarray(state.kind) != Item.NONE)
                    & (np.asarray(state.location) == cell))
    return (x, y, int(env.tiles[y, x]), *item)


def _same_item(actual, expected, cook_ticks):
    if actual[:2] != expected[:2]:
        return False
    # Passive cooking can advance during compaction; ready rice must truly be ready.
    if expected[0] == Item.POT and expected[1]:
        threshold = min(expected[2], cook_ticks)
        return actual[2] >= threshold
    return actual[2] == expected[2]


def _same_front(actual, expected, cook_ticks):
    return actual[:3] == expected[:3] and _same_item(actual[3:], expected[3:], cook_ticks)


class CompactedSchedule:
    def __init__(self, env, reference, *, initial_state=None, delay_probability=0., delay_seed=0):
        self.env = env
        self.reference = reference
        self.key = jax.random.PRNGKey(0)
        _, reset = env.reset(self.key)
        self.state = reset if initial_state is None else initial_state
        if not 0 <= int(self.state.time) < env.max_steps:
            raise ValueError("Initial state must precede the episode limit")
        if not np.isfinite(delay_probability) or not 0 <= delay_probability <= 1:
            raise ValueError("Delay probability must be finite and between zero and one")
        audit(env, self.state)
        self.delay_probability = float(delay_probability)
        self.delay_rng = np.random.default_rng(delay_seed)
        self.supervision = []
        self.imposed_delays = []
        self.partner_delay_masked_decisions = 0
        self.states = [self.state]
        self.actions = []
        self.labels = []
        self.events = []
        self.queue_indices = [0, 0]
        self.queues = [[], []]
        self.spawn_counts = np.zeros(3, dtype=np.int32)
        self.deadlock = None
        self.delivery_steps = []
        self.parallel_ticks = 0
        self.wait_ticks = 0
        for serial_step, pair in enumerate(reference.actions[1:]):
            for agent, action in enumerate(pair):
                if int(action) == Action.STAY:
                    continue
                info = reference.events[serial_step]
                self.queues[agent].append(QueuedAction(
                    serial_step + 1, int(action), reference.states[serial_step],
                    reference.states[serial_step + 1], bool(info['interaction_succeeded'][agent]),
                    bool(info['delivered'][agent]), reference.labels[serial_step + 1]))

        @jax.jit
        def step(state, pair):
            _, after, rewards, _, info = env.step_env(self.key, state, dict(zip(env.agents, pair)))
            return after, rewards[env.agents[0]], info
        self.step = step

    def _head(self, agent):
        index = self.queue_indices[agent]
        return self.queues[agent][index] if index < len(self.queues[agent]) else None

    def _ready(self, state, agent, head):
        if head is None:
            return False
        if not np.array_equal(state.positions[agent], head.before.positions[agent]):
            return False
        if int(state.directions[agent]) != int(head.before.directions[agent]):
            return False
        if not _same_item(_hand(state, agent), _hand(head.before, agent), self.env.cook_ticks):
            return False
        return (head.action < Action.PICKUP_PLACE or
                _same_front(_front(self.env, state, agent), _front(self.env, head.before, agent),
                            self.env.cook_ticks))

    def _post_matches(self, after, info, agent, head, both_interact_same_front):
        if not np.array_equal(after.positions[agent], head.after.positions[agent]):
            return False
        if int(after.directions[agent]) != int(head.after.directions[agent]):
            return False
        if not _same_item(_hand(after, agent), _hand(head.after, agent), self.env.cook_ticks):
            return False
        if bool(info['interaction_succeeded'][agent]) != head.success:
            return False
        if bool(info['delivered'][agent]) != head.delivered:
            return False
        # A valid same-tick handoff can clear a counter just filled by the first
        # actor. Both actor hand postconditions and actual success still must match.
        if head.action >= Action.PICKUP_PLACE and not both_interact_same_front:
            if not _same_front(_front(self.env, after, agent), _front(self.env, head.after, agent),
                               self.env.cook_ticks):
                return False
        return True

    def _select(self, available=(True, True)):
        heads = [self._head(agent) for agent in range(2)]
        priority = int(self.state.time) % 2
        cache = {}

        def predict(pair):
            pair = tuple(map(int, pair))
            if pair not in cache:
                cache[pair] = jax.device_get(self.step(self.state, jnp.asarray(pair, dtype=jnp.int32)))
            return cache[pair]

        for selected in ((True, True), tuple(i == priority for i in range(2)),
                         tuple(i != priority for i in range(2))):
            if any(selected[a] and (not available[a] or heads[a] is None) for a in range(2)):
                continue
            pair = [heads[a].action if selected[a] else int(Action.STAY) for a in range(2)]
            ready = [self._ready(self.state, a, heads[a]) if selected[a] else True for a in range(2)]
            if all(selected) and ready[priority] and not ready[1 - priority]:
                intermediate_pair = [int(Action.STAY), int(Action.STAY)]
                intermediate_pair[priority] = pair[priority]
                intermediate, _, _ = predict(intermediate_pair)
                ready[1 - priority] = self._ready(intermediate, 1 - priority, heads[1 - priority])
            if not all(ready):
                continue
            after, reward, info = predict(pair)
            same_front = (all(selected) and all(h.action >= Action.PICKUP_PLACE for h in heads)
                          and _front(self.env, self.state, 0)[:2] == _front(self.env, self.state, 1)[:2])
            if all(not selected[a] or self._post_matches(after, info, a, heads[a], same_front)
                   for a in range(2)):
                return pair, selected, (after, reward, info)
        pair = [int(Action.STAY), int(Action.STAY)]
        return pair, (False, False), predict(pair)

    def run(self):
        idle = 0
        for _ in range(self.env.max_steps - int(self.state.time)):
            before = self.state
            # Draw independent imposed delays; ordinary state-based partner waits remain labels.
            available = (self.delay_rng.random(2) >= self.delay_probability
                         if self.delay_probability else np.ones(2, dtype=bool))
            if self.deadlock:
                pair = [int(Action.STAY), int(Action.STAY)]
                selected = (False, False)
                after, reward, info = jax.device_get(self.step(before, jnp.asarray(pair, dtype=jnp.int32)))
            else:
                pair, selected, (after, reward, info) = self._select(available)
            # Compare both choices before applying the transition or advancing queues.
            proposed = self._select()[0] if self.delay_probability else pair
            for agent, action in enumerate(pair):
                if action == Action.PICKUP_PLACE and _hand(before, agent)[0] == Item.NONE:
                    tile = _front(self.env, before, agent)[2]
                    if tile in (Tile.CUCUMBER, Tile.RICE, Tile.NORI) and info['interaction_succeeded'][agent]:
                        self.spawn_counts[(Tile.CUCUMBER, Tile.RICE, Tile.NORI).index(tile)] += 1
            delta = int(after.deliveries) - int(before.deliveries)
            if float(reward) != delta * self.env.rules.delivery_reward:
                raise AssertionError("Raw reward disagrees with actual deliveries")
            if delta:
                self.delivery_steps.extend([int(after.time)] * delta)
            self.state = after
            audit(self.env, after)
            if not np.array_equal(food_counts(after) + int(after.deliveries), self.spawn_counts):
                raise AssertionError("Compacted actions violate the food ledger")
            self.states.append(after)
            self.actions.append(pair)
            self.imposed_delays.append(~available)
            supervision = available.copy()
            if self.delay_probability:
                # A chef cannot predict its partner's independent current delay.
                # Keep genuine state-based waits, but mask labels changed by that coin flip.
                changed = available & (np.asarray(pair) != np.asarray(proposed))
                supervision &= ~changed
                self.partner_delay_masked_decisions += int(changed.sum())
            self.supervision.append(supervision.astype(np.float32))
            self.events.append({name: np.asarray(value).tolist() for name, value in info.items()})
            self.labels.append(' | '.join(self._head(a).label for a in range(2) if selected[a]))
            for agent in range(2):
                self.queue_indices[agent] += int(selected[agent])
            self.parallel_ticks += int(all(selected))
            self.wait_ticks += int(not any(selected))
            idle = 0 if any(selected) else idle + 1
            if idle >= max(self.env.cook_ticks, self.env.return_ticks) + 16 and not self.deadlock:
                self.deadlock = {'step': int(after.time), 'queue_indices': self.queue_indices.copy(),
                    'heads': [None if self._head(a) is None else {
                        'serial_step': self._head(a).serial_step, 'action': self._head(a).action,
                        'label': self._head(a).label, 'actual_hand': _hand(after, a),
                        'expected_hand': _hand(self._head(a).before, a),
                        'actual_front': _front(self.env, after, a),
                        'expected_front': _front(self.env, self._head(a).before, a)} for a in range(2)]}
        if int(self.state.time) != self.env.max_steps or not bool(self.state.done):
            raise AssertionError("Compacted rollout did not preserve the actual episode horizon")
        return self


def generate_compacted_demo(seed=0, agent_view_size=None, routing='short', reference_steps=1200, nori_first=False,
                            prefetch_rice=False):
    if not 0 <= seed < 2**32 or reference_steps < 600:
        raise ValueError('Use an unsigned seed and at least600 reference steps')
    with jax.default_device(jax.devices('cpu')[0]):
        actual_env = make_env('oc2_sushi_1_3', agent_view_size=agent_view_size)
        reference_env = make_env('oc2_sushi_1_3', agent_view_size=agent_view_size,
                                 rules={'max_steps': reference_steps})
        options = {} if routing == 'baseline' else {'routing': routing}
        if nori_first:
            options['nori_first'] = True
        if prefetch_rice:
            options['prefetch_rice'] = True
        reference = ThroughputExpert(reference_env, **options).run()
        actual_rules, reference_rules = asdict(actual_env.rules), asdict(reference_env.rules)
        assert {k: v for k, v in actual_rules.items() if k != 'max_steps'} == {
            k: v for k, v in reference_rules.items() if k != 'max_steps'}
        plan = CompactedSchedule(actual_env, reference).run()
        before = jax.tree.map(lambda *parts: np.stack(parts), *plan.states[:-1])
        observations = jax.jit(jax.vmap(actual_env.get_obs))(before)
        observations = np.stack([np.asarray(observations[a]) for a in actual_env.agents], axis=1).astype(np.float32)
        states = tuple(jax.device_get(plan.states))
    actions = np.asarray(plan.actions, dtype=np.int32)
    metadata = {
        'schema_version': 1, 'layout': actual_env.name, 'seed': int(seed),
        'seed_scope': 'deterministic compacted semantic queues; seed labels the artifact only',
        'teacher_training_only': True, 'expert_used_at_neural_evaluation': False,
        'agent_view_size': agent_view_size, 'agents': list(actual_env.agents),
        'steps': len(actions), 'deliveries': int(plan.state.deliveries),
        'rules': actual_rules, 'reference_episode_steps': reference_steps, 'reference_routing': routing,
        'nori_first': bool(nori_first), 'prefetch_rice': bool(prefetch_rice),
        'reference_deliveries': int(reference.state.deliveries),
        'delivery_steps': plan.delivery_steps, 'parallel_action_ticks': plan.parallel_ticks,
        'joint_stay_ticks': plan.wait_ticks, 'deadlock': plan.deadlock,
        'queue_indices': plan.queue_indices, 'queue_lengths': list(map(len, plan.queues)),
        'alignment': 'observations[t]=get_obs(states[t]); actions[t] produces states[t+1]',
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ('environment.py', 'layout.py', 'throughput_expert.py', 'throughput_schedule.py')},
    }
    return {'observations': observations, 'actions': actions,
            'supervision_mask': np.ones_like(actions, dtype=np.float32),
            'lengths': np.asarray(len(actions), dtype=np.int32), 'states': states, 'metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--routing', choices=('baseline', 'short'), default='short')
    parser.add_argument('--reference-steps', type=int, default=1200)
    parser.add_argument('--nori-first', action='store_true')
    parser.add_argument('--prefetch-rice', action='store_true')
    args = parser.parse_args()
    demo = generate_compacted_demo(routing=args.routing, reference_steps=args.reference_steps,
                                   nori_first=args.nori_first, prefetch_rice=args.prefetch_rice)
    save_demo(demo, args.output)
    print(json.dumps({k: demo['metadata'][k] for k in ('deliveries', 'delivery_steps',
                      'parallel_action_ticks', 'joint_stay_ticks', 'deadlock', 'queue_indices')}, indent=2))


if __name__ == '__main__':
    main()
