"""Action-derived throughput demonstrations; this teacher is for training only.

The neural evaluator must never call this planner. Kitchen rules, episode length,
resource quantities, and delivery reward are the original simulator's values.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from .environment import Action, Item, QUEUED, RECIPE, RICE_BIT
from .layout import DRYING_CELL, RETURN_CELL
from .scripted import Script
from .demonstrations import _environment, save_demo


class _EpisodeFinished(Exception):
    pass


class ThroughputExpert(Script):
    """Serial two-chef plan using clean plates before washing returned ones.

    Rice stays on the hob. The left chef passes a cucumber+nori plate to the
    right chef, who adds cooked rice in place and passes the completed sushi
    back. This avoids carrying the rice pot out and returning it every dish.
    """
    shared, assembly, hob, board = (6, 4), (3, 4), (10, 0), (5, 1)

    def __init__(self, env, routing="baseline", nori_first=False, prefetch_rice=False):
        if routing not in ("baseline", "short"):
            raise ValueError("routing must be baseline or short")
        if not isinstance(nori_first, bool):
            raise ValueError("nori_first must be a boolean")
        if not isinstance(prefetch_rice, bool):
            raise ValueError("prefetch_rice must be a boolean")
        super().__init__(env)
        self.routing = routing
        self.nori_first = nori_first
        self.prefetch_rice = prefetch_rice
        self.rice_buffer = (6, 2)
        self._rice_prefetched = False
        self.rice_load_events = []
        if routing == "short":
            self.shared, self.board, self.assembly = (6, 3), (3, 1), (3, 4)
        self.delivery_events = []
        self.wash_events = []
        self.plate_selection = []
        self.return_wait_steps = 0
        self.cook_wait_steps = 0
        self.completed_cycles = 0

    def advance(self, agent, action, label, require_success=False):
        if bool(self.state.done):
            raise _EpisodeFinished()
        before = self.state
        super().advance(agent, action, label, require_success)
        after = self.state
        delivered = np.flatnonzero((np.asarray(before.kind) == Item.PLATE)
            & (np.asarray(before.contents) == RECIPE)
            & (np.asarray(after.kind) == Item.DIRTY_PLATE)
            & (np.asarray(after.location) == QUEUED))
        washed = np.flatnonzero((np.asarray(before.kind) == Item.DIRTY_PLATE)
                              & (np.asarray(after.kind) == Item.PLATE))
        for slot in delivered:
            self.delivery_events.append({"step": int(after.time), "plate_slot": int(slot)})
        for slot in washed:
            self.wash_events.append({"step": int(after.time), "plate_slot": int(slot)})

    def _xy(self, slot):
        cell = int(self.state.location[slot])
        return cell % self.env.width, cell // self.env.width

    def _prepare_plate(self, tag):
        clean = np.flatnonzero((np.asarray(self.state.kind) == Item.PLATE)
                              & (np.asarray(self.state.contents) == 0)
                              & (np.asarray(self.state.location) >= 0))
        if len(clean):
            # Initial slots are the two left plates followed by two right plates.
            slot = int(clean[0])
            self.plate_selection.append({"step": int(self.state.time), "plate_slot": slot, "source": "clean"})
            source = self._xy(slot)
            if source == self.assembly:
                return slot
            if source[0] < self.shared[0]:
                self.interact(0, source, tag + ": collect left clean plate")
                self.interact(0, self.assembly, tag + ": place clean assembly plate")
            else:
                self.interact(1, source, tag + ": collect right clean plate")
                self.interact(1, self.shared, tag + ": hand off clean plate")
                self.interact(0, self.shared, tag + ": receive clean plate")
                self.interact(0, self.assembly, tag + ": place clean assembly plate")
            return slot

        # No clean plate remains. Earlier dishes have been returning while the
        # chefs prepared subsequent ones, so only genuinely necessary waits occur.
        while True:
            dirty = np.flatnonzero((np.asarray(self.state.kind) == Item.DIRTY_PLATE)
                                  & (np.asarray(self.state.location) == self.env.return_cell))
            if len(dirty):
                slot = int(dirty[0])
                break
            self.advance(0, Action.STAY, tag + ": await an actually pending plate")
            self.return_wait_steps += 1
        self.plate_selection.append({"step": int(self.state.time), "plate_slot": slot, "source": "washed"})
        self.interact(1, RETURN_CELL, tag + ": collect dirty plate")
        self.interact(1, (10, 7), tag + ": put dirty plate in sink")
        self.approach(1, (10, 7), tag + ": wash plate")
        for _ in range(self.env.wash_ticks):
            self.advance(1, Action.WORK, tag + ": wash plate", require_success=True)
        self.mark(tag + ": washed plate", plate_slot=slot)
        self.interact(1, DRYING_CELL, tag + ": collect washed plate")
        self.interact(1, self.shared, tag + ": pass washed plate")
        self.interact(0, self.shared, tag + ": receive washed plate")
        self.interact(0, self.assembly, tag + ": place washed assembly plate")
        return slot

    def _dish(self, cycle):
        tag = f"Dish {cycle}"
        # Start passive cooking before any plate transport or washing work.
        if self._rice_prefetched:
            # The previous cycle loaded this portion through the second counter.
            pot = np.flatnonzero((np.asarray(self.state.kind) == Item.POT)
                                & (np.asarray(self.state.location) == self.env.cell(self.hob)))
            if len(pot) != 1 or int(self.state.contents[int(pot[0])]) != RICE_BIT:
                raise AssertionError("The prefetched rice portion is not on its hob")
            self._rice_prefetched = False
        else:
            self.interact(0, (0, 3), tag + ": collect rice")
            self.interact(0, self.shared, tag + ": pass rice")
            self.interact(1, self.shared, tag + ": receive rice")
            self.interact(1, self.hob, tag + ": start rice cooking")
            self.rice_load_events.append({"step": int(self.state.time), "for_dish": int(cycle),
                                          "prefetched": False})
        self.mark(tag + ": rice cooking")
        plate = self._prepare_plate(tag)
        if self.nori_first:
            # Independent per-chef queues can overlap this left-side work with
            # the right chef's rice/cucumber courier during later compaction.
            self.interact(0, (0, 5), tag + ": collect nori")
            self.interact(0, self.assembly, tag + ": add nori")

        self.interact(1, (13, 3), tag + ": collect cucumber")
        self.interact(1, self.shared, tag + ": pass cucumber")
        self.interact(0, self.shared, tag + ": receive cucumber")
        self.interact(0, self.board, tag + ": place cucumber on board")
        self.approach(0, self.board, tag + ": chop cucumber")
        for _ in range(self.env.chop_ticks):
            self.advance(0, Action.WORK, tag + ": chop cucumber", require_success=True)
        self.interact(0, self.board, tag + ": collect chopped cucumber")
        self.interact(0, self.assembly, tag + ": plate cucumber")
        if not self.nori_first:
            self.interact(0, (0, 5), tag + ": collect nori")
            self.interact(0, self.assembly, tag + ": add nori")
        self.interact(0, self.assembly, tag + ": collect partial plate")
        self.interact(0, self.shared, tag + ": pass partial plate")
        if self.prefetch_rice:
            # A separate native counter keeps this next-dish supply independent
            # from the current plate's handoff. Compaction can overlap the trips.
            self.interact(0, (0, 3), tag + ": collect next rice")
            self.interact(0, self.rice_buffer, tag + ": buffer next rice")
        self.interact(1, self.shared, tag + ": receive partial plate")

        self.approach(1, self.hob, tag + ": finish plate at stove")
        pot = np.flatnonzero((np.asarray(self.state.kind) == Item.POT)
                            & (np.asarray(self.state.location) == self.env.cell(self.hob)))
        if len(pot) != 1:
            raise AssertionError("The stationary rice pot was lost")
        while int(self.state.progress[int(pot[0])]) < self.env.cook_ticks:
            self.advance(0, Action.STAY, tag + ": wait for remaining rice cooking")
            self.cook_wait_steps += 1
        self.advance(1, Action.PICKUP_PLACE, tag + ": add cooked rice", require_success=True)
        if int(self.state.contents[plate]) != RECIPE:
            raise AssertionError("The held plate is not a complete sushi dish")
        self.mark(tag + ": assembled sushi", plate_slot=plate)
        self.interact(1, self.shared, tag + ": hand off complete sushi")
        if self.prefetch_rice:
            self.interact(1, self.rice_buffer, tag + ": receive next rice")
            self.interact(1, self.hob, tag + ": start next rice cooking")
            self._rice_prefetched = True
            self.rice_load_events.append({"step": int(self.state.time), "for_dish": int(cycle + 1),
                                          "prefetched": True})
        self.interact(0, self.shared, tag + ": receive complete sushi")
        self.interact(0, (0, 6), tag + ": serve sushi")
        self.mark(tag + ": served", plate_slot=plate)
        self.completed_cycles += 1

    def run(self):
        try:
            cycle = 1
            while True:
                self._dish(cycle)
                cycle += 1
        except _EpisodeFinished:
            pass
        if int(self.state.time) != self.env.max_steps or not bool(self.state.done):
            raise AssertionError("Throughput demonstration did not finish the full episode")
        if len(self.delivery_events) != int(self.state.deliveries):
            raise AssertionError("Delivery log disagrees with actual kitchen state")
        return self


def generate_throughput_demo(seed=0, agent_view_size=None, routing="baseline", nori_first=False,
                             prefetch_rice=False):
    """Return a full original-rules episode in the imitation NPZ schema.

    Each routing variant is deterministic; seed identifies the artifact only.
    Baseline preserves the original5-dish trace. Short routing uses adjacent
    native stations to finish its fifth dish earlier, without changing rules.
    Initial state is the unchanged empty kitchen, and all actions are supervised.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    with jax.default_device(jax.devices("cpu")[0]):
        env = _environment(agent_view_size)
        expert = ThroughputExpert(env, routing=routing, nori_first=nori_first,
                                  prefetch_rice=prefetch_rice).run()
        before = jax.tree.map(lambda *parts: np.stack(parts), *expert.states[:-1])
        obs = jax.jit(jax.vmap(env.get_obs))(before)
        obs = np.stack([np.asarray(obs[agent]) for agent in env.agents], axis=1).astype(np.float32)
        actions = np.asarray(expert.actions[1:], dtype=np.int32)
        states = tuple(jax.device_get(expert.states))
    delivery_steps = [event["step"] for event in expert.delivery_events]
    metadata = {
        "schema_version": 1, "layout": env.name, "seed": int(seed),
        "seed_scope": "fixed deterministic serial expert; no starting-state or trajectory variation",
        "teacher_training_only": True, "expert_used_at_neural_evaluation": False,
        "routing": expert.routing, "nori_first": expert.nori_first,
        "prefetch_rice": expert.prefetch_rice,
        "rice_buffer": list(expert.rice_buffer) if expert.prefetch_rice else None,
        "rice_load_events": expert.rice_load_events,
        "stations": {"shared": list(expert.shared), "board": list(expert.board),
                     "assembly": list(expert.assembly), "hob": list(expert.hob)},
        "agent_view_size": agent_view_size, "agents": list(env.agents),
        "steps": len(actions), "deliveries": int(expert.state.deliveries),
        "rules": asdict(env.rules), "delivery_events": expert.delivery_events,
        "delivery_steps": delivery_steps,
        "first_delivery_step": delivery_steps[0] if delivery_steps else None,
        "last_delivery_step": delivery_steps[-1] if delivery_steps else None,
        "first_delivery_seconds": delivery_steps[0] * env.rules.step_seconds if delivery_steps else None,
        "last_delivery_seconds": delivery_steps[-1] * env.rules.step_seconds if delivery_steps else None,
        "wash_events": expert.wash_events, "plate_selection": expert.plate_selection,
        "return_wait_steps": expert.return_wait_steps, "cook_wait_steps": expert.cook_wait_steps,
        "completed_cycles": expert.completed_cycles, "milestones": expert.milestones,
        "alignment": "observations[t] = get_obs(states[t]); actions[t] produces states[t+1]",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("environment.py", "layout.py", "scripted.py", "throughput_expert.py")
        },
    }
    return {"observations": obs, "actions": actions, "supervision_mask": np.ones_like(actions, dtype=np.float32),
            "lengths": np.asarray(len(actions), dtype=np.int32), "states": states, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-view-size", type=int, default=None)
    parser.add_argument("--routing", choices=("baseline", "short"), default="baseline")
    parser.add_argument("--nori-first", action="store_true")
    parser.add_argument("--prefetch-rice", action="store_true")
    args = parser.parse_args()
    demo = generate_throughput_demo(args.seed, args.agent_view_size, args.routing,
                                    args.nori_first, args.prefetch_rice)
    path = save_demo(demo, args.output)
    print(json.dumps({"output": str(path), "steps": len(demo["actions"]),
                      "deliveries": demo["metadata"]["deliveries"],
                      "routing": demo["metadata"]["routing"],
                      "nori_first": demo["metadata"]["nori_first"],
                      "prefetch_rice": demo["metadata"]["prefetch_rice"],
                      "delivery_steps": demo["metadata"]["delivery_steps"],
                      "return_wait_steps": demo["metadata"]["return_wait_steps"],
                      "cook_wait_steps": demo["metadata"]["cook_wait_steps"]}, indent=2))


if __name__ == "__main__":
    main()
