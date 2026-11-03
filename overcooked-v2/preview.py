"""Add a readable spectator dashboard to upstream simulator frames."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jaxmarl.environments.overcooked_v2.common import (
    Direction,
    DynamicObject,
    OvercookedActionsEnum,
    StaticObject,
)
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from jaxmarl.environments.overcooked_v2.settings import POT_COOK_TIME
from replay import save_replay


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


def build_replay_data(states, actions, rewards, correct, wrong, config, mode, seed):
    """Describe each post-action state, including the action-free initial state."""
    count = len(actions)
    if any(len(values) != count for values in (rewards, correct, wrong)):
        raise ValueError("Replay action and event counts differ.")
    if len(states.grid) != count + 1:
        raise ValueError("Replay needs the initial state plus every resulting state.")
    totals = [np.concatenate(([0], np.cumsum(values)))
              for values in (rewards, correct, wrong)]
    pot_positions = np.argwhere(states.grid[0, :, :, 0] == StaticObject.POT)
    steps = []
    for index in range(count + 1):
        agents = []
        for agent in range(states.agents.inventory.shape[1]):
            inventory = int(states.agents.inventory[index, agent])
            direction = int(states.agents.dir[index, agent])
            action = None if index == 0 else int(actions[index - 1, agent])
            agents.append({
                "id": f"agent_{agent}", "index": agent,
                "color": ("red", "blue", "green", "purple", "yellow", "orange")[agent % 6],
                "position": [int(states.agents.pos.x[index, agent]),
                             int(states.agents.pos.y[index, agent])],
                "direction": direction, "facing": Direction(direction).name.lower(),
                "inventory": inventory, "holding": inventory_name(inventory),
                "action_code": action,
                "action": None if action is None else OvercookedActionsEnum(action).name,
            })
        pots = []
        for pot, (y, x) in enumerate(pot_positions, 1):
            contents, timer = map(int, states.grid[index, y, x, 1:])
            pots.append({
                "index": pot, "position": [int(x), int(y)],
                "contents": contents, "ingredients": ingredient_count(contents),
                "timer": timer,
                "status": "ready to plate" if contents & DynamicObject.COOKED else (
                    f"cooking: {timer} steps left" if timer > 0 else "waiting for ingredients"
                ),
            })
        recipe = int(states.recipe[index])
        steps.append({
            "step": index, "agents": agents, "pots": pots,
            "grid": states.grid[index].tolist(), "recipe_code": recipe,
            "recipe": "three onions" if recipe == 3 * DynamicObject.ingredient(0)
                      else f"code {recipe}",
            "reward": 0.0 if index == 0 else float(rewards[index - 1]),
            "correct_dishes": 0 if index == 0 else int(correct[index - 1]),
            "wrong_dishes": 0 if index == 0 else int(wrong[index - 1]),
            "totals": {"team_reward": float(totals[0][index]),
                       "correct_dishes": int(totals[1][index]),
                       "wrong_dishes": int(totals[2][index])},
        })
    return {
        "schema_version": 1,
        "metadata": {
            "layout": config["ENV_KWARGS"]["layout"], "mode": mode,
            "episode": 0, "seed": int(seed), "steps": count,
            "starts": "original fixed starts" if mode == "scripted" else "randomized starts",
            "agent_view_size": config["ENV_KWARGS"]["agent_view_size"],
            "upstream_revision": (Path(__file__).parent / "UPSTREAM_REVISION").read_text().strip(),
            "frame_stride": 1, "frame_ms": 100,
            "reward": "shared raw environment reward counted once; no shaping",
            "coordinates": "(column, row), zero-based",
            "renderer": {"kind": "overcooked", "cook_ticks": int(POT_COOK_TIME)},
        },
        "steps": steps,
    }


def render_replay_frames(states, radius):
    """Render all states in bounded batches to limit renderer GPU memory use."""
    visualizer = OvercookedV2Visualizer()
    batches = []
    for start in range(0, len(states.grid), 32):
        batch = jax.tree.map(lambda value: jnp.asarray(value[start:start + 32]), states)
        batches.append(np.asarray(visualizer.render_sequence(batch, agent_view_size=radius)))
    return np.concatenate(batches)


def draw_action_probability_panel(draw, *, x, y, title, probabilities, selected_action, font, small_font, color):
    """Draw one chef's probabilities and highlight its selected action."""
    draw.text((x, y), title, fill=color, font=font)
    if probabilities is None:
        draw.text(
            (x, y + 32),
            "No previous decision",
            fill="#cbd5e1",
            font=small_font,
        )
        return
    action_names = [action.name for action in OvercookedActionsEnum]
    bar_width = 130
    for action_index, (name, probability) in enumerate(
        zip(action_names, probabilities)
    ):
        row_y = y + 34 + action_index * 26
        selected = action_index == selected_action
        label = f"> {name}" if selected else f"  {name}"
        row_color = color if selected else "#cbd5e1"
        draw.text(
            (x, row_y),
            label,
            fill=row_color,
            font=small_font,
        )
        bar_x = x + 100
        draw.rectangle(
            (bar_x, row_y + 5, bar_x + bar_width, row_y + 17),
            fill="#334155",
        )
        filled_width = int(round(bar_width * float(probability)))
        if filled_width > 0:
            draw.rectangle(
                (bar_x, row_y + 5, bar_x + filled_width, row_y + 17),
                fill=color if selected else "#94a3b8",
            )
        draw.text(
            (x + 242, row_y),
            f"{float(probability):.1%}",
            fill=row_color,
            font=small_font,
        )


def save_preview(initial, trace, output: Path, config, mode, seed, policy_seeds=None, action_probs=None):
    scripted = mode == "scripted"
    states, actions, rewards, correct, wrong, _ = trace
    # HTML and GIF retain the initial state and every resulting state.
    states = jax.tree.map(
        lambda first, rest: np.concatenate(([first], rest)), initial, states
    )
    indices = np.arange(len(actions) + 1)
    radius = config["ENV_KWARGS"]["agent_view_size"]
    all_frames = render_replay_frames(states, radius)
    save_replay(build_replay_data(states, actions, rewards, correct, wrong,
                                 config, mode, seed), all_frames, output)
    frames = all_frames[indices]
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
    probability_x = None
    if action_probs is not None:
        action_probs = np.asarray(action_probs)
        expected_shape = (
            len(actions),
            len(initial.agents.inventory),
            len(OvercookedActionsEnum),
        )
        if action_probs.shape != expected_shape:
            raise ValueError(
                f"Expected action probabilities shaped {expected_shape}, "
                f"got {action_probs.shape}"
            )
        if (
            not np.isfinite(action_probs).all()
            or np.any(action_probs < 0)
            or np.any(action_probs > 1)
        ):
            raise ValueError("Action probabilities must be finite and between 0 and 1.")
        if not np.allclose(action_probs.sum(axis=-1), 1.0, atol=1e-5):
            raise ValueError("Each agent's action probabilities must sum to 1.")
        probability_x = canvas_width + 20
        canvas_width += 350
    annotated = []
    for index, raw in zip(indices, frames):
        image = Image.new("RGB", (canvas_width, canvas_height), "#111827")
        draw = ImageDraw.Draw(image)

        def text(x, y, value, color="white", face=font):
            draw.text((x, y), value, fill=color, font=face)

        label = "SCRIPTED reference" if scripted else f"{mode.upper()} policy"
        text(20, 16, f"{config['ENV_KWARGS']['layout']} | {label}", face=title_font)
        start_label = "original fixed starts" if scripted else f"test seed {seed}"
        text(20, 56, f"Episode 1 | {start_label} | step {index} / {len(actions)}", "#cbd5e1")
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
            number = agent if scripted else agent + 1
            label = f"Chef {number} ({'red' if agent == 0 else 'blue'})"
            if policy_seeds is not None:
                label += f" | seed {policy_seeds[agent]}"
            text(right, y, label, color)
            holding = inventory_name(states.agents.inventory[index, agent])
            text(right, y + 29, f"Holding: {holding}")
            action = "start" if index == 0 else (
                OvercookedActionsEnum(int(actions[index - 1, agent])).name
            )
            text(right, y + 58, f"Action: {action}", "#cbd5e1")
            if action_probs is not None:
                if index == 0:
                    probabilities = None
                    selected_action = None
                else:
                    probabilities = action_probs[index - 1, agent]
                    selected_action = int(actions[index - 1, agent])
                draw_action_probability_panel(
                    draw,
                    x=probability_x,
                    y=139 + agent * 240,
                    title=f"Chef {number}: probabilities",
                    probabilities=probabilities,
                    selected_action=selected_action,
                    font=font,
                    small_font=small_font,
                    color=color,
                )

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
        description = (
            "Deterministic joint actions from a saved schedule; no learned policy."
            if scripted else
            f"Full kitchen shown for viewers. Chefs still receive only their {view} observations."
        )
        text(
            20, canvas_height - 53,
            description,
            "#cbd5e1", small_font,
        )
        text(
            20, canvas_height - 28,
            "One frame per step; actions and probabilities describe the decision that produced this state.",
            "#94a3b8", small_font,
        )
        annotated.append(image)

    annotated[0].save(output / "initial.png")
    annotated[-1].save(output / "final.png")
    annotated[0].save(
        output / "rollout.gif", 
        save_all=True, 
        append_images=annotated[1:],
        duration=[100] * (len(annotated) - 1) + [1200], 
        loop=0,
    )
