"""Add a readable spectator dashboard to upstream simulator frames."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jaxmarl.environments.overcooked_v2.common import (
    DynamicObject,
    OvercookedActionsEnum,
    StaticObject,
)
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer


def ingredient_count(value):
    """Ingredients occupy pairs of bits in the simulator's object encoding."""
    value = int(value) >> 2
    count = 0
    while value:
        count += value & 3
        value >>= 2
    return count


def inventory_name(value):
    value = int(value)
    if value == DynamicObject.EMPTY:
        return "nothing"
    if value == DynamicObject.PLATE:
        return "empty plate"
    if value & DynamicObject.COOKED:
        return "plated soup"
    return "onion" if value == DynamicObject.ingredient(0) else "ingredient"


def save_preview(initial, trace, output: Path, config, mode, seed):
    states, actions, rewards, correct, wrong, _ = trace
    # Include the starting kitchen and terminal kitchen; sample every second step.
    states = jax.tree.map(
        lambda first, rest: np.concatenate(([first], rest)), initial, states
    )
    indices = np.unique(np.append(np.arange(0, len(actions) + 1, 2), len(actions)))
    sampled = jax.tree.map(lambda value: jnp.asarray(value[indices]), states)
    radius = config["ENV_KWARGS"]["agent_view_size"]
    frames = np.asarray(
        OvercookedV2Visualizer().render_sequence(sampled, agent_view_size=radius)
    )
    view = f"local {2 * radius + 1} x {2 * radius + 1}" if radius is not None else "full-kitchen"
    totals = [
        np.concatenate(([0], np.cumsum(values)))
        for values in (rewards, correct, wrong)
    ]
    pot_positions = np.argwhere(np.asarray(initial.grid)[:, :, 0] == StaticObject.POT)
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 20)
    title_font = ImageFont.truetype(str(font_path), 26)
    small_font = ImageFont.truetype(str(font_path), 16)
    width, height = frames.shape[2] * 3, frames.shape[1] * 3
    canvas_width = max(880, width + 390)
    canvas_height = max(690, height + 205)
    right = width + 45
    annotated = []
    for index, raw in zip(indices, frames):
        image = Image.new("RGB", (canvas_width, canvas_height), "#111827")
        draw = ImageDraw.Draw(image)

        def text(x, y, value, color="white", face=font):
            draw.text((x, y), value, fill=color, font=face)

        text(20, 16, f"{config['ENV_KWARGS']['layout']} | {mode.upper()} policy", face=title_font)
        text(20, 56, f"Episode 1 | test seed {seed} | step {index} / {len(actions)}", "#cbd5e1")
        counters = (
            f"Correct dishes: {int(totals[1][index])}     "
            f"Wrong dishes: {int(totals[2][index])}     "
            f"Team score: {totals[0][index]:g}"
        )
        text(20, 91, counters)
        kitchen = Image.fromarray(raw).resize((width, height), Image.Resampling.NEAREST)
        image.paste(kitchen, (20, 138))
        # Label fixed pot locations to connect the graphics to the status panel.
        for pot, (y, x) in enumerate(pot_positions, start=1):
            xy = (20 + int(x) * 96 + 4, 138 + int(y) * 96 + 3)
            draw.rectangle((*xy, xy[0] + 31, xy[1] + 23), fill="#111827")
            text(*xy, f"P{pot}", face=small_font)

        for agent in range(len(initial.agents.inventory)):
            y = 139 + agent * 104
            color = ("#ff6b6b", "#60a5fa")[agent % 2]
            text(right, y, f"Chef {agent + 1} ({'red' if agent == 0 else 'blue'})", color)
            holding = inventory_name(states.agents.inventory[index, agent])
            text(right, y + 29, f"Holding: {holding}")
            action = "start" if index == 0 else (
                OvercookedActionsEnum(int(actions[index - 1, agent])).name
            )
            text(right, y + 58, f"Action: {action}", "#cbd5e1")

        for pot, (y, x) in enumerate(pot_positions, start=1):
            contents, timer = states.grid[index, y, x, 1:]
            panel_y = 362 + (pot - 1) * 75
            text(right, panel_y, f"Pot {pot}: {ingredient_count(contents)} / 3 ingredients")
            status = "ready to plate" if int(contents) & DynamicObject.COOKED else (
                f"cooking: {int(timer)} steps left" if timer > 0 else "waiting for ingredients"
            )
            text(right, panel_y + 29, status, "#cbd5e1")

        recipe = int(states.recipe[index])
        recipe_name = "three onions" if recipe == 3 * DynamicObject.ingredient(0) else f"code {recipe}"
        text(right, 525, f"Recipe: {recipe_name}", "#fde68a")
        text(right, 557, "Yellow dots: onion supply", face=small_font)
        text(right, 581, "White circles: plate supply", face=small_font)
        text(right, 605, "Green square: deliver here", face=small_font)
        text(
            20, canvas_height - 53,
            f"Full kitchen shown for viewers. Chefs still receive only their {view} observations.",
            "#cbd5e1", small_font,
        )
        text(
            20, canvas_height - 28,
            "Counters are display only. One frame per two steps; actions shown for the latest step.",
            "#94a3b8", small_font,
        )
        annotated.append(image)

    annotated[0].save(output / "initial.png")
    annotated[-1].save(output / "final.png")
    annotated[0].save(
        output / "rollout.gif", save_all=True, append_images=annotated[1:],
        duration=[200] * (len(annotated) - 1) + [1200], loop=0,
    )
