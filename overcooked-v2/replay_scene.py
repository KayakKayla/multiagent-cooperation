"""Translate recorded state to spectator sprites; never infer policy intentions.

This module deliberately has no JAX imports, so saved replays can be restyled
without loading a simulator, a checkpoint, or a GPU runtime. Raw states remain
unchanged alongside this derived presentation layer.
"""
import re


SOUP_TILES = {0: "floor", 1: "counter", 4: "serve", 5: "hob",
              6: "recipe", 7: "switch", 9: "plates"}
SUSHI_TILES = dict(enumerate(("floor", "wall", "counter", "board", "hob",
                            "cucumber_supply", "rice_supply", "nori_supply",
                            "serve", "return", "sink", "drying", "bin", "prawn")))


def ingredients(code):
    """Decode upstream's pairs of ingredient bits, including mixed recipes."""
    code = int(code)
    if code < 0:
        raise ValueError("Ingredient encoding must be nonnegative")
    code >>= 2
    result, index = [], 0
    while code:
        result.extend([index] * (code & 3))
        code >>= 2
        index += 1
    return result


def ingredient_name(index):
    return "onion" if index == 0 else f"ingredient {index}"


def recipe_name(state):
    if "recipe_code" not in state:
        return state.get("recipe", "Recorded recipe")
    values = ingredients(state["recipe_code"])
    return " + ".join(f"{values.count(index)} × {ingredient_name(index)}"
                      for index in sorted(set(values))) or "No recipe"


def soup_item(code):
    if not code:
        return None
    values = ingredients(code)
    if code & 1:
        # A cooked plated item is a soup; an empty plate stays visibly empty.
        return dict(sprite="soup" if code & 2 else "plate",
                    ingredients=values, description="Plated soup" if code & 2 else "Empty plate")
    return dict(sprite="onion" if values and all(v == 0 for v in values) else "ingredient",
                ingredients=values,
                description=" + ".join(ingredient_name(v) for v in values) or "Unknown item")


def sushi_limits(data):
    rules = data["metadata"].get("renderer", {})
    limits = {key: rules.get(key) for key in ("cook_ticks", "chop_ticks", "wash_ticks")}
    # Older recorded sushi payloads included limits in descriptions, or a full
    # cooking progress value once ready. Otherwise leave the fraction unknown.
    for state in data["steps"]:
        for item in state.get("items", []):
            description = item.get("description", "")
            for key, operation in (("chop_ticks", "chop"), ("wash_ticks", "wash")):
                match = re.search(rf"{operation} \d+/(\d+)", description)
                if limits[key] is None and match:
                    limits[key] = int(match[1])
            if (limits["cook_ticks"] is None and item["kind"] == "POT"
                    and description == "pot: cooked rice" and item["progress"] > 0):
                limits["cook_ticks"] = item["progress"]
    return limits


def sushi_item(item, limits):
    kind, bits = item["kind"], item["contents"]
    result = dict(description=item["description"], ingredients=[], slot=item["slot"])
    simple = {"RAW_CUCUMBER": "cucumber", "CHOPPED_CUCUMBER": "chopped_cucumber",
              "RICE": "raw_rice", "NORI": "nori", "DIRTY_PLATE": "dirty_plate"}
    if kind in simple:
        result["sprite"] = simple[kind]
    elif kind == "POT":
        result["sprite"] = "rice_pot" if bits else "empty_pot"
        result["filled"] = bool(bits)
    elif kind == "PLATE":
        result["sprite"] = "sushi" if bits == 7 else "plate"
        if bits != 7:
            result["toppings"] = [sprite for bit, sprite in
                                  ((1, "chopped_cucumber"), (2, "raw_rice"), (4, "nori")) if bits & bit]
    else:
        raise ValueError(f"Unsupported sushi item {kind}")
    if kind in ("POT", "RAW_CUCUMBER", "DIRTY_PLATE") and item["progress"] > 0:
        key = {"POT": "cook_ticks", "RAW_CUCUMBER": "chop_ticks", "DIRTY_PLATE": "wash_ticks"}[kind]
        maximum = limits[key]
        result["progress"] = min(1, item["progress"] / maximum) if maximum else None
        result["progress_text"] = f"{item['progress']} / {maximum}" if maximum else f"Progress {item['progress']}"
    return result


def build_presentation(data):
    """Build layout-independent display data, or explain an original-view fallback."""
    steps, meta = data["steps"], data["metadata"]
    sushi = meta.get("renderer", {}).get("kind") == "sushi" or meta["layout"] == "oc2_sushi_1_3"
    result = dict(version=1, supported=True, kind="sushi" if sushi else "overcooked",
                  ingredient_ids=[], scenes=[])
    try:
        limits = sushi_limits(data) if sushi else {"cook_ticks": meta.get("renderer", {}).get("cook_ticks", 20)}
        first_grid = steps[0]["grid"]
        height, width = len(first_grid), len(first_grid[0])
        if not width or any(len(row) != width for row in first_grid):
            raise ValueError("Kitchen grid is not rectangular")
        result.update(width=width, height=height)
        for state in steps:
            if len(state["grid"]) != height or any(len(row) != width for row in state["grid"]):
                raise ValueError("Kitchen dimensions change during this replay")
            if any(a["color"] not in ("red", "blue") for a in state["agents"]):
                raise ValueError("Illustrated chef artwork currently supports red and blue chefs")
            if any(a["facing"] not in ("up", "down", "left", "right") for a in state["agents"]):
                raise ValueError("Unknown chef facing direction")
            scene = dict(tiles=[], items=[], held={}, pots=[], recipe=recipe_name(state))
            for y, row in enumerate(state["grid"]):
                tiles = []
                for x, cell in enumerate(row):
                    kind = cell if sushi else cell[0]
                    if sushi:
                        tile = SUSHI_TILES.get(kind)
                    else:
                        tile = "onion_supply" if kind == 10 else "ingredient_supply" if kind >= 11 else SOUP_TILES.get(kind)
                    if tile is None:
                        raise ValueError(f"No illustrated artwork for tile type {kind}")
                    tiles.append(tile)
                    if not sushi and kind in (0, 1) and cell[1]:
                        scene["items"].append(dict(soup_item(cell[1]), position=[x, y]))
                    if not sushi and kind >= 11:
                        if kind-10 not in result["ingredient_ids"]:
                            result["ingredient_ids"].append(kind-10)
                        scene["items"].append(dict(sprite="ingredient_id", ingredient_id=kind-10, position=[x, y]))
                    if not sushi and kind in (6, 7):
                        # Upstream overlays the current recipe at indicators, and
                        # at recipe buttons only while their timer is active.
                        code = state["recipe_code"] if kind == 6 or cell[2] > 0 else cell[1]
                        values = ingredients(code)
                        scene["items"].append(dict(sprite="recipe_id", ingredients=values,
                                                  active=bool(cell[2]) if kind == 7 else None, position=[x, y]))
                scene["tiles"].append(tiles)
            if sushi:
                for item in state.get("items", []):
                    location = item["location"]
                    if location >= 0:
                        scene["items"].append(dict(sushi_item(item, limits), position=[location % width, location // width]))
                    elif location in (-1, -2):
                        scene["held"][str(-location-1)] = sushi_item(item, limits)
            else:
                for agent in state["agents"]:
                    held = soup_item(agent["inventory"])
                    if held:
                        scene["held"][str(agent["index"])] = held
            for pot in state["pots"]:
                ready, cooking = pot["status"] == "ready to plate", pot["timer"] > 0
                x, y = pot["position"]
                off_hob = sushi and cooking and scene["tiles"][y][x] != "hob"
                maximum = limits["cook_ticks"]
                fraction = 1 if ready else max(0, min(1, (maximum-pot["timer"])/maximum)) if cooking and maximum else None
                capacity = pot.get("capacity", 3)
                detail = f"{pot['ingredients']} / {capacity} {'rice' if sushi else 'ingredients'}"
                scene["pots"].append(dict(index=pot["index"], position=pot["position"],
                    ready=ready, cooking=cooking, progress=fraction, detail=detail,
                    status="Ready to plate" if ready else f"Off hob · {pot['timer']} steps remaining" if off_hob else f"Cooking · {pot['timer']} steps" if cooking else pot["status"],
                    badge="✓ READY" if ready else "OFF HOB" if off_hob else f"{pot['timer']} steps" if cooking else f"{pot['ingredients']} / {capacity}",
                    ingredient_types=[] if sushi else ingredients(pot.get("contents", 0)),
                    ingredients=pot["ingredients"]))
            result["scenes"].append(scene)
    except (KeyError, TypeError, IndexError, ValueError) as error:
        # Keep unfamiliar schemas readable; never invent geometry or item state.
        result.update(supported=False, reason=str(error), scenes=[])
    return result
