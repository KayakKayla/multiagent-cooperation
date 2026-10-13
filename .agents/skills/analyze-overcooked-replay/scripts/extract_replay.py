#!/usr/bin/env python3
"""Read schema-v1 replay evidence without executing HTML or loading JAX."""

import argparse
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import sys


class ReplayError(ValueError):
    """An input cannot support reliable replay analysis."""


class PayloadParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.payloads = []
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id") == "replay-data":
            if attrs.get("type") != "application/json":
                raise ReplayError("replay-data must have type application/json")
            self.payloads.append([])
            self.active = True

    def handle_data(self, data):
        if self.active:
            self.payloads[-1].append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = False


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def integer(value):
    return type(value) is int


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(data):
    require(isinstance(data, dict), "payload must be a JSON object")
    require(type(data.get("schema_version")) is int and data["schema_version"] == 1,
            "unsupported schema_version; expected 1")
    metadata, steps = data.get("metadata"), data.get("steps")
    require(isinstance(metadata, dict), "missing or malformed metadata")
    count = metadata.get("steps")
    require(integer(count) and count >= 0, "metadata.steps must be a nonnegative integer")
    require(isinstance(steps, list) and len(steps) == count + 1,
            "steps must contain the initial state plus every resulting state")
    cumulative = {"team_reward": 0.0, "correct_dishes": 0, "wrong_dishes": 0}
    agent_ids = None
    for index, state in enumerate(steps):
        prefix = f"step {index}: "
        require(isinstance(state, dict), prefix + "state must be an object")
        require(integer(state.get("step")) and state["step"] == index,
                prefix + "inconsistent step numbering")
        agents = state.get("agents")
        require(isinstance(agents, list) and bool(agents), prefix + "missing agents")
        ids = []
        for agent in agents:
            require(isinstance(agent, dict), prefix + "malformed agent")
            require(isinstance(agent.get("id"), str), prefix + "missing agent ID")
            ids.append(agent["id"])
            pos = agent.get("position")
            require(isinstance(pos, list) and len(pos) == 2 and all(integer(v) for v in pos),
                    prefix + "malformed agent position")
            require(integer(agent.get("inventory")) and isinstance(agent.get("holding"), str)
                    and integer(agent.get("direction")) and isinstance(agent.get("facing"), str),
                    prefix + "missing agent inventory or facing")
            if index == 0:
                require("action" in agent and "action_code" in agent
                        and agent["action"] is None and agent["action_code"] is None,
                        prefix + "initial actions must be null")
            else:
                require(isinstance(agent.get("action"), str) and integer(agent.get("action_code")),
                        prefix + "missing resulting-state action")
        require(len(set(ids)) == len(ids), prefix + "duplicate agent IDs")
        if agent_ids is None:
            agent_ids = ids
        require(ids == agent_ids, prefix + "agent IDs/order changed")
        grid = state.get("grid")
        require(isinstance(grid, list) and bool(grid) and isinstance(grid[0], list)
                and bool(grid[0]), prefix + "missing grid")
        width = len(grid[0])
        require(all(isinstance(row, list) and len(row) == width and all(
            isinstance(cell, list) and len(cell) == 3 and all(integer(v) for v in cell)
            for cell in row) for row in grid), prefix + "malformed grid cells")
        require(all(0 <= a["position"][0] < width and 0 <= a["position"][1] < len(grid)
                    for a in agents), prefix + "agent position outside grid")
        pots = state.get("pots")
        require(isinstance(pots, list) and all(isinstance(p, dict) and all(
            integer(p.get(k)) for k in ("index", "contents", "ingredients", "timer"))
            and isinstance(p.get("status"), str) and isinstance(p.get("position"), list)
            and len(p["position"]) == 2 and all(integer(v) for v in p["position"])
            for p in pots), prefix + "missing or malformed pots")
        require(integer(state.get("recipe_code")) and isinstance(state.get("recipe"), str),
                prefix + "missing recipe")
        totals = state.get("totals")
        require(isinstance(totals, dict), prefix + "missing totals")
        for event, total in (("reward", "team_reward"), ("correct_dishes", "correct_dishes"),
                             ("wrong_dishes", "wrong_dishes")):
            value, recorded = state.get(event), totals.get(total)
            valid = number(value) and number(recorded) if event == "reward" else (
                integer(value) and value >= 0 and integer(recorded) and recorded >= 0)
            require(valid, prefix + f"invalid {event} event or total")
            require(index != 0 or value == 0, prefix + "initial events must be zero")
            cumulative[total] += value
            matches = math.isclose(recorded, cumulative[total], rel_tol=1e-9, abs_tol=1e-9) \
                if event == "reward" else recorded == cumulative[total]
            require(matches, prefix + f"inconsistent cumulative {total}")


def load_replay(source):
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / "replay.html"
    parser = PayloadParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    require(len(parser.payloads) == 1 and not parser.active,
            "expected one complete replay-data JSON script")
    try:
        data = json.loads("".join(parser.payloads[0]))
    except (ValueError, RecursionError) as exc:
        raise ReplayError(f"malformed replay JSON: {exc}") from exc
    validate(data)
    return path.resolve(), data


def extract(source, start=None, end=None):
    path, data = load_replay(source)
    last = data["metadata"]["steps"]
    start = 0 if start is None else start
    end = last if end is None else end
    require(integer(start) and integer(end) and 0 <= start <= end <= last,
            f"range must satisfy 0 <= start <= end <= {last}")
    return {
        "source_replay": str(path), "schema_version": 1,
        "metadata": data["metadata"], "episode_totals": data["steps"][-1]["totals"],
        "range": {"start": start, "end": end},
        "context_state": data["steps"][start - 1] if start > 0 else None,
        "steps": data["steps"][start:end + 1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="replay.html or its containing folder")
    parser.add_argument("--start", type=int, help="first included timestep (default: 0)")
    parser.add_argument("--end", type=int, help="last included timestep (default: episode end)")
    args = parser.parse_args()
    try:
        evidence = extract(args.source, args.start, args.end)
        print(json.dumps(evidence, separators=(",", ":"), allow_nan=False))
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        parser.exit(2, f"Replay error: {exc}\n")


if __name__ == "__main__":
    main()
