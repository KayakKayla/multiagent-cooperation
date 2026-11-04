"""Version 1 evaluation contract, resource bounds, and scientific consistency."""
import hashlib
import json
import math
import re
from datetime import datetime

FORMAT_NAME = "se390-evaluation"
FORMAT_VERSION = 1
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DEPTH = 32
MAX_EPISODES = 10000
MAX_PARTICIPANTS = 64
MAX_STEPS = 100000
MAX_NODES = 4000000
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN = {"png", "frames", "images", "html", "javascript", "weights", "params", "optimizer", "observations", "activations", "environment_variables"}


class ValidationError(ValueError):
    pass


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_digest(value):
    """Digest every field of the source document, including annotations."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _integer(value, name, low=0, high=2**63-1):
    _require(type(value) is int and low <= value <= high, f"{name} must be an integer from {low} to {high}")


def _text(value, name, limit=512):
    _require(isinstance(value, str) and 0 < len(value) <= limit, f"{name} must be nonempty text (maximum {limit} characters)")


def _walk(value):
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _require(depth <= MAX_DEPTH and count <= MAX_NODES, "JSON nesting or node limit exceeded")
        if isinstance(item, dict):
            _require(len(item) <= 4096, "Object has too many fields")
            for key, child in item.items():
                _require(isinstance(key, str) and len(key) <= 256, "Invalid object key")
                # Training config can describe its observation mode ("full_map").
                # Raw observation arrays/objects remain excluded.
                observation_label = key.lower() == "observations" and isinstance(child, str)
                _require(key.lower() not in _FORBIDDEN or observation_label, f"Excluded executable/media/model field: {key}")
                pending.append((child, depth + 1))
        elif isinstance(item, list):
            _require(len(item) <= MAX_STEPS + 1, "Array exceeds sanity limit")
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            _require(len(item) <= 65536, "Text exceeds sanity limit")
            _require(not item.lower().startswith(("data:", "javascript:")), "Embedded executable/media URI is excluded")
            _require(not re.search(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|tmp)/|\\\\[^\\]+\\)", item), "Machine-local paths are excluded")
        elif isinstance(item, float):
            _require(math.isfinite(item), "Non-finite numbers are forbidden")
        else:
            _require(item is None or type(item) in (bool, int), "Unsupported JSON value")


def _unique(records, key, label):
    _require(isinstance(records, list) and 0 < len(records) <= MAX_EPISODES, f"Invalid {label} list")
    result = {}
    for record in records:
        _require(isinstance(record, dict), f"Invalid {label} object")
        value = record.get(key)
        _text(value, key)
        _require(value not in result, f"Duplicate {label} identity: {value}")
        result[value] = record
    return result


def _validate_document(document):
    """Validate without coercion; unknown optional data belongs in extensions.

    Version 1 permits additive fields. Nonempty required_features is rejected
    until those semantics are implemented by every consumer.
    """
    _require(isinstance(document, dict), "Evaluation must be a JSON object")
    _walk(document)
    _require(document.get("format") == FORMAT_NAME and type(document.get("version")) is int and document["version"] == FORMAT_VERSION, "Unsupported evaluation format/version")
    _require(not document.get("required_features"), "Unsupported required format features")
    _text(document.get("source_evaluation_id"), "source_evaluation_id")
    try:
        created = datetime.fromisoformat(document["created_at"].replace("Z", "+00:00"))
        _require(created.utcoffset() is not None, "created_at must include a timezone")
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError("created_at must be a timezone-qualified ISO timestamp") from error
    for key in ("environment", "protocol", "summary", "provenance", "device"):
        _require(isinstance(document.get(key), dict), f"{key} must be an object")
    for key in ("name", "layout"):
        _text(document["environment"].get(key), f"environment.{key}")
    runs = _unique(document.get("training_runs"), "source_run_id", "training run")
    for run in runs.values():
        _text(run.get("name"), "training run name")
        if run.get("algorithm") is not None:
            _text(run["algorithm"], "training run algorithm", 128)
        if run.get("seed") is not None:
            _integer(run["seed"], "training run seed", 0, 2**32-1)
        for key in ("config", "provenance", "device"):
            _require(isinstance(run.get(key), dict), f"training run {key} must be an object")
        _text(run.get("progress_unit"), "progress_unit", 64)
    checkpoints = _unique(document.get("checkpoints"), "source_checkpoint_id", "checkpoint")
    for checkpoint in checkpoints.values():
        _require(checkpoint.get("source_run_id") in runs, "Checkpoint references an unknown training run")
        _require(isinstance(checkpoint.get("sha256"), str) and _SHA.fullmatch(checkpoint["sha256"]), "Checkpoint sha256 must be lowercase hexadecimal")
        if checkpoint.get("progress") is not None:
            _integer(checkpoint["progress"], "checkpoint.progress")
            _require(runs[checkpoint["source_run_id"]]["progress_unit"] != "unknown", "Known checkpoint progress requires a known unit")
    participants = _unique(document.get("participants"), "agent_id", "participant")
    _require(len(participants) <= MAX_PARTICIPANTS, "Too many participants")
    orders = set()
    for participant in participants.values():
        _require(participant.get("source_checkpoint_id") in checkpoints, "Participant references an unknown checkpoint")
        _integer(participant.get("display_order"), "display_order", 0, MAX_PARTICIPANTS-1)
        _require(participant["display_order"] not in orders, "Duplicate participant display order")
        orders.add(participant["display_order"])
        for key in ("observation_settings", "action_settings"):
            _require(isinstance(participant.get(key), dict), f"participant {key} must be an object")
    episodes = document.get("episodes")
    _require(isinstance(episodes, list) and 1 <= len(episodes) <= MAX_EPISODES, "Invalid episode count")
    seeds = document["protocol"].get("episode_seeds")
    _require(isinstance(seeds, list) and len(seeds) == len(episodes), "Protocol seed count differs from episodes")
    for index, episode in enumerate(episodes):
        _require(isinstance(episode, dict) and type(episode.get("index")) is int and episode["index"] == index, "Episodes must be ordered with contiguous zero-based indices")
        _integer(episode.get("steps"), "episode.steps", 0, MAX_STEPS)
        if episode.get("seed") is not None:
            _integer(episode["seed"], "episode.seed", 0, 2**32-1)
        _require(episode.get("seed") == seeds[index], "Episode and protocol seeds disagree")
        metrics = episode.get("metrics")
        _require(isinstance(metrics, dict), "Episode metrics must be an object")
        _require(type(metrics.get("team_reward")) in (int, float), "Episode team_reward is required")
        for key in ("correct_dishes", "wrong_dishes"):
            if metrics.get(key) is not None:
                _integer(metrics[key], f"metrics.{key}")
        recording = episode.get("recording")
        if recording is None:
            continue
        _require(isinstance(recording, dict) and recording.get("schema_version") == 1, "Unsupported recording schema")
        metadata = recording.get("metadata", {})
        _require(isinstance(metadata, dict), "Recording metadata must be an object")
        _require(metadata.get("episode") == index and metadata.get("seed") == episode.get("seed"), "Recording episode/seed metadata disagrees")
        if "steps" in metadata:
            _require(metadata["steps"] == episode["steps"], "Recording metadata step count disagrees")
        if "frame_ms" in metadata:
            _require(type(metadata["frame_ms"]) in (int, float) and 0 < metadata["frame_ms"] <= 60000, "Invalid display frame timing")
        _require(metadata.get("frame_stride", 1) == 1, "Full-resolution recordings require frame_stride 1")
        steps = recording.get("steps")
        _require(isinstance(steps, list) and len(steps) == episode["steps"]+1, "Recording needs initial state plus one state per action")
        for number, step in enumerate(steps):
            _require(isinstance(step, dict) and type(step.get("step")) is int and step["step"] == number, "Recording step indices disagree")
            agents = step.get("agents")
            _require(isinstance(agents, list) and len(agents) == len(participants), "Recording participant count disagrees")
            _require({a.get("id") for a in agents if isinstance(a, dict)} == set(participants), "Recording agent identities disagree")
            for agent in agents:
                _require((agent.get("action") is None) == (number == 0), "Initial state must be action-free; later states must carry preceding actions")
                if number:
                    _text(agent["action"], "recording agent action", 128)
                if "action_code" in agent:
                    _require((agent["action_code"] is None) == (number == 0), "Action code/state alignment disagrees")
                    if number:
                        _integer(agent["action_code"], "action_code", 0, 4096)
                position = agent.get("position")
                _require(isinstance(position, list) and len(position) == 2, "Agent position must be [column,row]")
                for coordinate in position:
                    _integer(coordinate, "agent position", 0, 255)
                if "facing" in agent:
                    _require(agent["facing"] in ("up", "down", "left", "right"), "Invalid agent facing")
                if "holding" in agent:
                    _require(isinstance(agent["holding"], str) and len(agent["holding"]) <= 1024, "Invalid holding description")
            grid = step.get("grid")
            if "grid" in step:
                _require(isinstance(grid, list) and 0 < len(grid) <= 256 and all(isinstance(row, list) and 0 < len(row) <= 256 for row in grid), "Invalid recording grid dimensions")
                _require(all(len(row) == len(grid[0]) for row in grid), "Grid rows must have the same width")
                for row in grid:
                    for cell in row:
                        values = cell if isinstance(cell, list) else [cell]
                        _require(1 <= len(values) <= 32 and all(type(v) is int and 0 <= v < 2**31 for v in values), "Invalid grid cell")
                _require(all(a["position"][0] < len(grid[0]) and a["position"][1] < len(grid) for a in agents), "Agent position is outside the grid")
            totals = step.get("totals")
            _require(isinstance(totals, dict) and type(totals.get("team_reward")) in (int, float), "Recording totals need numeric team_reward")
            for key in ("correct_dishes", "wrong_dishes"):
                if key in totals:
                    _integer(totals[key], f"recording totals.{key}")
                if key in step:
                    _integer(step[key], f"recording {key}")
            if "reward" in step:
                _require(type(step["reward"]) in (int, float) and (number != 0 or step["reward"] == 0), "Invalid per-step reward")
                previous = steps[number-1]["totals"]["team_reward"] if number else 0
                _require(math.isclose(totals["team_reward"] - previous, step["reward"], rel_tol=1e-7, abs_tol=1e-6), "Per-step and cumulative rewards disagree")
            for key in ("correct_dishes", "wrong_dishes"):
                if key in totals:
                    previous = steps[number-1]["totals"].get(key, 0) if number else 0
                    _require(totals[key] >= previous, "Cumulative delivery totals decrease")
                    if key in step:
                        _require(totals[key] - previous == step[key], "Per-step and cumulative deliveries disagree")
            for key in ("pots", "items"):
                if key in step:
                    _require(isinstance(step[key], list) and len(step[key]) <= 4096 and all(isinstance(item, dict) for item in step[key]), f"Invalid {key} list")
        final = steps[-1].get("totals", {})
        for key in ("team_reward", "correct_dishes", "wrong_dishes"):
            if key in final and metrics.get(key) is not None:
                _require(math.isclose(final[key], metrics[key], rel_tol=1e-7, abs_tol=1e-6), f"Recording total {key} disagrees with episode metrics")
        if all("reward" in step for step in steps):
            _require(math.isclose(sum(step["reward"] for step in steps), metrics["team_reward"], rel_tol=1e-7, abs_tol=1e-6), "Recording rewards disagree with shared team reward")
    summary = document["summary"]
    if "episodes" in summary:
        _require(summary["episodes"] == len(episodes), "Summary episode count disagrees")
    for key in ("correct_dishes", "wrong_dishes"):
        total_key = "total_" + key
        if total_key in summary and all(e["metrics"].get(key) is not None for e in episodes):
            _require(summary[total_key] == sum(e["metrics"][key] for e in episodes), f"Summary {total_key} disagrees")
    for key in ("mean_team_reward", "mean_raw_team_reward"):
        if key in summary:
            _require(math.isclose(summary[key], sum(e["metrics"]["team_reward"] for e in episodes)/len(episodes), rel_tol=1e-7, abs_tol=1e-6), "Summary mean team reward disagrees")
    return document


def validate_document(document):
    """Every malformed JSON shape raises the public validation exception."""
    try:
        return _validate_document(document)
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError) as error:
        raise ValidationError("Malformed evaluation field type or value") from error


def parse_document(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    _require(len(data) <= MAX_UPLOAD_BYTES, "Evaluation exceeds 20 MiB file limit")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        document = json.loads(data, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValidationError("Non-finite JSON number")))
    except ValidationError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError) as error:
        raise ValidationError("Malformed JSON evaluation") from error
    return validate_document(document)
