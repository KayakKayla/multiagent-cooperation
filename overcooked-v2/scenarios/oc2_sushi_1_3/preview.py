"""Render the extended state and adapt it to the existing offline replay UI."""
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .environment import Action, Item
from .layout import Tile


COLORS = {
    Tile.FLOOR: "#e6dfce", Tile.WALL: "#35495c", Tile.COUNTER: "#bb906b",
    Tile.BOARD: "#92ac79", Tile.HOB: "#64717c", Tile.CUCUMBER: "#75b9b0",
    Tile.RICE: "#75b9b0", Tile.NORI: "#75b9b0", Tile.PRAWN: "#7d999c",
    Tile.SERVE: "#72a387", Tile.RETURN: "#bba38a", Tile.SINK: "#7eb5c9",
    Tile.DRYING: "#9bb8bc", Tile.BIN: "#798387",
}
LABELS = {Tile.BOARD: "CHOP", Tile.HOB: "HOB", Tile.CUCUMBER: "CUCUMBER",
          Tile.RICE: "RICE", Tile.NORI: "NORI", Tile.PRAWN: "PRAWN*",
          Tile.SERVE: "SERVE", Tile.RETURN: "RETURN", Tile.SINK: "WASH",
          Tile.DRYING: "CLEAN", Tile.BIN: "BIN"}
FOOD_NAMES = ("cucumber", "cooked rice", "nori")


def describe(state, slot, env):
    kind = int(state.kind[slot])
    bits = int(state.contents[slot])
    progress = int(state.progress[slot])
    if kind == Item.POT:
        status = "empty" if bits == 0 else "cooked rice" if progress >= env.cook_ticks else f"rice {progress}/{env.cook_ticks}"
        return "pot: " + status
    if kind == Item.PLATE:
        food = [name for bit, name in zip((1, 2, 4), FOOD_NAMES) if bits & bit]
        return "plate: " + (", ".join(food) if food else "empty")
    if kind == Item.DIRTY_PLATE:
        return f"dirty plate (wash {progress}/{env.wash_ticks})"
    if kind == Item.RAW_CUCUMBER:
        return f"raw cucumber (chop {progress}/{env.chop_ticks})"
    return Item(kind).name.lower().replace("_", " ")


def item_position(state, slot, env):
    location = int(state.location[slot])
    if location >= 0:
        return location % env.width, location // env.width
    if location in (-1, -2):
        return tuple(map(int, state.positions[-location - 1]))
    return None


def render(env, state, tile_size=64):
    state = jax.device_get(state)
    image = Image.new("RGB", (env.width * tile_size, env.height * tile_size))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=10)
    small = ImageFont.load_default(size=9)
    large = ImageFont.load_default(size=19)
    for y, row in enumerate(np.asarray(env.tiles)):
        for x, value in enumerate(row):
            tile = Tile(int(value))
            left, top = x * tile_size, y * tile_size
            draw.rectangle((left, top, left + tile_size - 1, top + tile_size - 1), fill=COLORS[tile], outline="#ffffff42")
            if tile in LABELS:
                draw.text((left + tile_size / 2, top + 3), LABELS[tile], fill="#142c35", font=small, anchor="mt")
    # Show floor chefs before their held objects so the hand badge stays visible.
    for agent, (x, y) in enumerate(state.positions):
        cx, cy = (int(x) + .5) * tile_size, (int(y) + .5) * tile_size
        color = ("#c64445", "#356eac")[agent]
        draw.ellipse((cx-21, cy-21, cx+21, cy+21), fill=color, outline="white", width=3)
        draw.text((cx, cy), str(agent), font=large, anchor="mm", fill="white")
        dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[int(state.directions[agent])]
        draw.line((cx+dx*19, cy+dy*19, cx+dx*29, cy+dy*29), fill="#1a2731", width=4)
    for slot, kind in enumerate(state.kind):
        if int(kind) == Item.NONE:
            continue
        point = item_position(state, slot, env)
        if point is None:
            continue
        x, y = point
        held = int(state.location[slot]) < 0
        cx, cy = (x + .5) * tile_size, (y + .56) * tile_size
        radius = 18
        if held:
            cx += 17
            cy += 17
            radius = 11
        fill = "#f7f1dd" if kind == Item.PLATE else "#a98762" if kind == Item.DIRTY_PLATE else "#3f4b53" if kind == Item.POT else "#f8f0cf"
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=fill, outline="#243844", width=2)
        text = {Item.RAW_CUCUMBER: "Cu", Item.CHOPPED_CUCUMBER: "C", Item.RICE: "R",
                Item.NORI: "N", Item.POT: "P", Item.PLATE: "", Item.DIRTY_PLATE: "D"}[Item(int(kind))]
        if kind == Item.PLATE:
            text = "+".join(c for bit, c in ((1, "C"), (2, "R"), (4, "N")) if int(state.contents[slot]) & bit)
        draw.text((cx, cy), text, fill="white" if kind == Item.POT else "#172d37", font=font, anchor="mm")
        if kind in (Item.POT, Item.RAW_CUCUMBER, Item.DIRTY_PLATE) and int(state.progress[slot]) > 0:
            maximum = env.cook_ticks if kind == Item.POT else env.chop_ticks if kind == Item.RAW_CUCUMBER else env.wash_ticks
            fraction = int(state.progress[slot]) / maximum
            draw.rectangle((cx-radius, cy+radius+2, cx+radius, cy+radius+6), fill="#35495c")
            draw.rectangle((cx-radius, cy+radius+2, cx-radius+2*radius*fraction, cy+radius+6), fill="#8ad19c")
    return np.asarray(image)


def replay_data(script):
    env = script.env
    steps = []
    for index, raw in enumerate(script.states):
        s = jax.device_get(raw)
        agents = []
        for agent in range(2):
            held = np.flatnonzero((s.kind != Item.NONE) & (s.location == -agent - 1))
            action = script.actions[index][agent] if index else None
            agents.append(dict(index=agent, id=env.agents[agent], color=("red", "blue")[agent],
                               position=s.positions[agent].tolist(), facing=("right", "down", "left", "up")[int(s.directions[agent])],
                               holding=describe(s, int(held[0]), env) if len(held) else "empty hands",
                               action=Action(action).name.lower().replace("_", " ") if action is not None else None))
        pots, items = [], []
        for slot, kind in enumerate(s.kind):
            if kind == Item.NONE:
                continue
            point = item_position(s, slot, env)
            items.append(dict(slot=slot, kind=Item(int(kind)).name, description=describe(s, slot, env),
                              location=int(s.location[slot]), contents=int(s.contents[slot]),
                              progress=int(s.progress[slot]), return_due=int(s.return_due[slot])))
            if kind == Item.POT and point is not None:
                ready = int(s.contents[slot]) != 0 and int(s.progress[slot]) >= env.cook_ticks
                timer = max(0, env.cook_ticks - int(s.progress[slot])) if int(s.contents[slot]) else 0
                pots.append(dict(index=slot, position=list(point), ingredients=int(s.contents[slot] != 0),
                                 capacity=1, status="ready to plate" if ready else f"cooking: {timer} steps remaining" if timer else "empty pot",
                                 timer=0 if ready else timer))
        steps.append(dict(step=index, recipe="cucumber + cooked rice + nori", agents=agents, pots=pots,
                          grid=np.asarray(env.tiles).tolist(), items=items, milestone=script.labels[index],
                          totals=dict(team_reward=float(s.score), correct_dishes=int(s.deliveries), wrong_dishes=0)))
    return dict(schema_version=1, metadata=dict(layout=env.name, mode="scripted", starts="Fixed grid starts · provisional geometry/timing",
                seed=0, episode=0, frame_ms=env.rules.step_seconds*1000, agent_view_size=env.agent_view_size,
                perspective="Two-chef script; no learned policy. Prototype geometry and timing; score uses a fixed delivery reward.",
                legend="C = chopped cucumber · R = rice · N = nori · P = pot · D = dirty plate. Only cooked rice can enter a plate. Prawn supply is inactive.",
                image_max_width=720), steps=steps)
