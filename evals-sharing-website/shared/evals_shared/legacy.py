"""Convert historical result folders as inert evidence, without simulation."""
import argparse
import csv
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .export import write_document
from .format import FORMAT_NAME, FORMAT_VERSION, content_digest
from .provenance import recorded_source_identity, portable_metadata, stable_run_provenance


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _local_source(value, directory):
    """Use only explicitly local existing directories; never fetch URLs."""
    if not value or "://" in value:
        return None
    path = Path(value)
    candidates = [path, directory.parent.parent / "train" / path.name]
    # WSL metadata copied into a Windows checkout: match only the run basename.
    basename = str(value).replace("\\", "/").rstrip("/").split("/")[-1]
    candidates.append(directory.parent.parent / "train" / basename)
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def convert_folder(directory, *, source_run_id=None, partner_source_run_id=None, progress=None, partner_progress=None):
    directory = Path(directory)
    config = _json(directory / "config.json")
    protocol = _json(directory / "protocol.json")
    partner_config = _json(directory / "partner-config.json")
    summary = _json(directory / "summary.json")
    rows = _json(directory / "episodes.json")
    if not rows:
        with (directory / "episodes.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key, value in list(row.items()):
                try:
                    row[key] = json.loads(value)
                except (ValueError, TypeError):
                    pass
    recording = None
    replay = directory / "replay.html"
    if replay.is_file():
        match = re.search(r'<script\s+id=["\']replay-data["\']\s+type=["\']application/json["\']>(.*?)</script>', replay.read_text(encoding="utf-8"), re.S)
        if match:
            raw = json.loads(match[1])
            recording = {key: value for key, value in raw.items() if key not in ("frames", "presentation")}
    agents = [agent["id"] for agent in recording["steps"][0]["agents"]] if recording else [p["agent"] for p in protocol.get("policies", [])]
    if not agents:
        # Historical local simulator configuration explicitly has two players.
        agents = ["agent_0", "agent_1"]
    policies = protocol.get("policies") or [{"agent": agent, "source_run": protocol.get("source_run"), "policy_sha256": protocol.get("policy_sha256")} for agent in agents]
    runs, checkpoints, participants = {}, {}, []
    primary_reference = policies[0].get("source_run") or protocol.get("source_run")
    primary_digest = policies[0].get("policy_sha256", protocol.get("policy_sha256"))

    def reference_root(value):
        value = str(value or "").replace("\\", "/").rstrip("/")
        return re.sub(r"/checkpoints/update-\d+$", "", value)

    primary_reference = reference_root(primary_reference)
    for index, agent in enumerate(agents):
        policy = next((p for p in policies if p.get("agent") == agent), policies[min(index, len(policies)-1)])
        digest = policy.get("policy_sha256", protocol.get("policy_sha256"))
        if not digest:
            raise ValueError("Historical checkpoint hash is missing; cannot identify exact weights")
        reference = reference_root(policy.get("source_run"))
        same_primary = index == 0 or (bool(reference) and reference == primary_reference)
        if not reference and not protocol.get("partner_run") and protocol.get("policy") != "cross-policy":
            same_primary = True
        source = _local_source(policy.get("source_run"), directory)
        root = source.parent.parent if source and source.parent.name == "checkpoints" and (source / "meta.json").is_file() else source
        metadata = _json(root / "metadata.json") if root else {}
        initialization = protocol.get("policy") == "untrained"
        saved_config = config if same_primary else partner_config
        source_config = _json(root / "config.json") if root else {}
        # The general evaluator saved the training config verbatim. Specialized
        # evaluation saved effective overrides; retain those only as evaluation
        # evidence when the original training config is unavailable.
        unmodified_saved = bool(saved_config) and (not same_primary or "training_episode_steps" not in protocol)
        training_config = copy.deepcopy(source_config or (saved_config if unmodified_saved else {}))
        if training_config:
            training_config.setdefault("ALGORITHM", "ippo_rnn")
        seed = policy.get("training_seed", training_config.get("SEED", saved_config.get("SEED")))
        if seed is not None and training_config:
            training_config["SEED"] = seed
        algorithm_config = training_config or saved_config
        method = "initialization_only" if initialization else algorithm_config.get("TRAINING_METHOD", algorithm_config.get("ALGORITHM", "ippo_rnn" if algorithm_config else "unknown"))
        unit = "optimizer_updates" if method == "behavior_cloning" else ("environment_steps" if method in ("ippo_rnn", "initialization_only") else "unknown")
        override = source_run_id if same_primary else partner_source_run_id
        identity_scope = "initialization" if initialization else ("run" if root else "legacy-reference:" + (reference or ("primary" if same_primary else "partner-" + str(index))))
        source_id = recorded_source_identity(root or directory, scope=identity_scope, recorded=metadata.get("source_run_id"), supplied=override)
        # Run association and exact checkpoint selection are different axes:
        # two player slots may select different checkpoints of the SAME run.
        same_primary_checkpoint = same_primary and digest == primary_digest
        if index and same_primary_checkpoint and partner_progress is not None:
            raise ValueError("--partner-progress requires a distinct partner checkpoint; shared checkpoint slots use --progress")
        explicit_progress = progress if same_primary_checkpoint else partner_progress
        selected_progress = 0 if initialization else explicit_progress
        if source and explicit_progress is None:
            sidecar = _json(source / "checkpoint.json")
            if sidecar.get("sha256") == digest:
                selected_progress = sidecar.get("progress")
                unit = sidecar.get("progress_unit", unit)
        if selected_progress is not None and unit == "unknown":
            raise ValueError("Checkpoint progress requires a known training progress unit")
        run_record = {"source_run_id": source_id, "name": root.name if root else ("Untrained initialization" if initialization else "Historical training run"),
            "notes": "", "algorithm": method, "seed": seed, "progress_unit": unit,
            "config": portable_metadata(training_config), "provenance": stable_run_provenance(metadata),
            "device": portable_metadata(metadata.get("device", {})), "started_at": metadata.get("started_at")}
        if source_id in runs and {k:v for k,v in runs[source_id].items() if k not in ("name", "notes")} != {k:v for k,v in run_record.items() if k not in ("name", "notes")}:
            raise ValueError("Participants assigned the same source-run identity have conflicting training evidence")
        runs[source_id] = run_record
        checkpoint_id = source_id + ":sha256:" + digest
        checkpoint_record = {"source_checkpoint_id": checkpoint_id, "source_run_id": source_id, "sha256": digest, "progress": selected_progress,
                             "provenance": {"training_resumes": portable_metadata(metadata.get("resumes", []))}}
        if checkpoint_id in checkpoints and checkpoints[checkpoint_id] != checkpoint_record:
            raise ValueError("Conflicting checkpoint progress for the same exact source checkpoint")
        checkpoints[checkpoint_id] = checkpoint_record
        participants.append({"agent_id": agent, "display_order": index, "source_checkpoint_id": checkpoint_id,
            "observation_settings": {"agent_view_size": config.get("ENV_KWARGS", {}).get("agent_view_size")},
            "action_settings": {"action_selection": protocol.get("action_selection", "unknown")}})
    episodes = []
    recorded_index = protocol.get("recorded_episode", protocol.get("replay_episode", 0))
    for index, row in enumerate(rows):
        metrics = dict(row)
        for key in ("episode", "seed", "steps"):
            metrics.pop(key, None)
        metrics["team_reward"] = metrics.get("team_reward", metrics.get("raw_team_reward"))
        metrics.setdefault("wrong_dishes", 0)
        seed = row.get("seed", protocol.get("episode_seeds", [None] * len(rows))[index])
        selected = recording if index == recorded_index else None
        if selected:
            selected["metadata"].update(episode=index, seed=seed, steps=row["steps"])
        episodes.append({"index": index, "seed": seed, "steps": row["steps"], "termination": {"reason": "historical_not_recorded"}, "metrics": metrics, "recording": selected})
    public_protocol = portable_metadata(protocol)
    public_protocol.update(episode_seeds=[e["seed"] for e in episodes], recording_coverage="historical_partial" if recording else "unavailable",
                           legacy_conversion="No trajectories regenerated; unknown progress/device retained")
    evidence = {"config": portable_metadata(config), "protocol": public_protocol, "summary": summary, "episodes": episodes}
    timestamp = re.search(r"(\d{8}T\d{12}Z)$", directory.name)
    if timestamp:
        created_at = datetime.strptime(timestamp[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc).isoformat()
        time_source = "historical_output_directory_timestamp"
    else:
        created_at = datetime.fromtimestamp((directory / "protocol.json").stat().st_mtime, timezone.utc).isoformat()
        time_source = "protocol_file_mtime_not_original_evaluation_time"
    evaluation_id = recorded_source_identity(directory, scope="legacy-evaluation", namespace="evaluation")
    return {"format": FORMAT_NAME, "version": FORMAT_VERSION, "source_evaluation_id": evaluation_id,
        "created_at": created_at, "title": "Historical " + config.get("ENV_KWARGS", {}).get("layout", "evaluation"), "notes": "",
        "training_runs": list(runs.values()), "checkpoints": list(checkpoints.values()), "participants": participants,
        "environment": {"name": config.get("ENV_NAME", "unknown"), "layout": config.get("ENV_KWARGS", {}).get("layout", "unknown"),
                        "config": portable_metadata(config.get("ENV_KWARGS", {})), "renderer": recording.get("metadata", {}).get("renderer", {}) if recording else {}},
        "protocol": public_protocol, "summary": summary,
        "provenance": {"converted_from": "legacy_local_evaluation", "creation_time_source": time_source, "upstream_revision": protocol.get("upstream_revision"),
                       "legacy_saved_configs": {"primary_effective_evaluation_config": portable_metadata(config), "partner_saved_training_config": portable_metadata(partner_config)},
                       "training_checkpoint_provenance": {key: value.get("provenance", {}) for key, value in checkpoints.items()}},
        "device": {"backend": protocol["backend"]} if protocol.get("backend") else {}, "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-run-id", help="Explicit portable identity if historical metadata is ambiguous")
    parser.add_argument("--partner-source-run-id")
    parser.add_argument("--progress", type=int, help="Primary checkpoint progress only, when independently known")
    parser.add_argument("--partner-progress", type=int, help="Distinct partner checkpoint progress, independently known")
    args = parser.parse_args()
    document = convert_folder(args.directory, source_run_id=args.source_run_id, partner_source_run_id=args.partner_source_run_id, progress=args.progress, partner_progress=args.partner_progress)
    result = write_document(args.output or args.directory / "evaluation.json", document)
    print(f"Wrote {result['bytes']:,} bytes; within 20 MiB: {result['within_upload_limit']}")


if __name__ == "__main__":
    main()
