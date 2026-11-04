"""Best-effort portable provenance, with no host/user/network identifiers."""
import hashlib
from importlib import metadata
import os
import platform
import re
from pathlib import Path
import json
from uuid import uuid4


def device_info(*, backend=None, devices=None, label=None):
    result = {"os": platform.system(), "os_release": platform.release(),
              "architecture": platform.machine(), "logical_cpu_count": os.cpu_count(),
              "python": platform.python_version(), "versions": {}}
    for package in ("jax", "jaxlib", "flax", "optax", "numpy"):
        try:
            result["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    if backend is not None:
        result["backend"] = str(backend)
    if devices is not None:
        result["devices"] = [{"kind": str(getattr(device, "device_kind", type(device).__name__)),
                              "platform": str(getattr(device, "platform", backend or "unknown"))}
                             for device in devices]
    if label:
        result["label"] = label
    return result


_PRIVATE_KEYS = {"source_run", "partner_run", "run", "path", "output", "resume_from", "init_weights", "dataset", "datasets", "demonstrations", "xla_flags", "entity", "project", "hostname", "username", "home", "environment_variables"}
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|tmp)/|\\\\[^\\]+\\)")


def portable_metadata(value):
    """Remove local operational references while retaining scientific settings.

    Source paths never become identities. Path-valued fields are removed or
    redacted so copied exports do not expose account names.
    """
    if isinstance(value, dict):
        return {key: portable_metadata(child) for key, child in value.items()
                if key.lower() not in _PRIVATE_KEYS and not key.lower().endswith(("_path", "_file", "_dir"))}
    if isinstance(value, (list, tuple)):
        return [portable_metadata(child) for child in value]
    if isinstance(value, str) and _PATH.search(value):
        return "redacted-local-reference"
    return value


def recorded_source_identity(directory, *, scope="run", recorded=None, supplied=None, namespace="run"):
    """Persist an independent UUID, never infer run sameness from weights/seed.

    Recorded trainer IDs are authoritative. Historical/initialization identities
    live beside the local evidence, so copying the sidecar preserves identity.
    A supplied ID is an explicit correction and is persisted for later exports.
    """
    if recorded:
        if supplied and supplied != recorded:
            raise ValueError("Supplied source identity conflicts with the recorded trainer identity; use an explicit website association for an alias")
        return recorded
    if directory is None:
        raise ValueError("A local evidence directory is required to persist source-run identity")
    directory = Path(directory)
    suffix = "" if scope == "run" else "-" + hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24]
    path = directory / ("source-identity" + suffix + ".json")
    if supplied:
        if not isinstance(supplied, str) or not supplied or len(supplied) > 512:
            raise ValueError("Supplied source-run identity must be nonempty bounded text")
        temporary = path.with_name(path.name + ".tmp-" + str(uuid4()))
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump({"version": 1, "source_id": supplied}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return supplied
    temporary = path.with_name(path.name + ".tmp-" + str(uuid4()))
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            identity = namespace + ":" + str(uuid4())
            json.dump({"version": 1, "source_id": identity}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Publish the already complete file atomically without replacing a
            # concurrent export's identity. Both names are on the same filesystem.
            os.link(temporary, path)
            return identity
        except FileExistsError:
            pass
        saved = json.loads(path.read_text(encoding="utf-8"))
        identity = saved.get("source_id", saved.get("source_run_id"))
        if saved.get("version") != 1 or not isinstance(identity, str) or not 0 < len(identity) <= 512:
            raise ValueError("Malformed local source identity sidecar; recover it rather than inventing another run")
        return identity
    finally:
        temporary.unlink(missing_ok=True)


def stable_run_provenance(metadata):
    """Original run evidence stays immutable; resume lineage belongs to checkpoints."""
    return portable_metadata({key: value for key, value in metadata.items()
                              if key not in {"resumes", "source_run_id"}})


def policy_run_root(run):
    run = Path(run)
    return run.parent.parent if run.parent.name == "checkpoints" and (run / "meta.json").is_file() else run


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def source_records(run, config, weights, *, initialization=False, identity_directory=None):
    """Identify exact bytes and progress; never borrow final progress for intermediates."""
    run = Path(run) if run else None
    root = policy_run_root(run) if run else None
    metadata = _read_json(root / "metadata.json") if run else {}
    checkpoint = _read_json(run / "checkpoint.json") if run else {}
    digest = hashlib.sha256(weights).hexdigest()
    method = "initialization_only" if initialization else config.get("TRAINING_METHOD", config.get("ALGORITHM", "ippo_rnn"))
    unit = "optimizer_updates" if method == "behavior_cloning" else "environment_steps"
    source_id = recorded_source_identity(identity_directory if initialization else root,
        scope="initialization" if initialization else "run", recorded=metadata.get("source_run_id"))
    progress = 0 if initialization else None
    if checkpoint.get("sha256") == digest:
        progress = checkpoint.get("progress")
        unit = checkpoint.get("progress_unit", unit)
    elif run and not (run / "checkpoint.json").exists() and root == run and (run / "summary.json").is_file():
        # Root params are the final/most recent weights. Intermediate selections
        # must have their own sidecar; never search the parent for a final summary.
        summary = _read_json(run / "summary.json")
        progress = summary.get("updates") if method == "behavior_cloning" else summary.get("completed_environment_steps")
    record = {"source_run_id": source_id, "name": "Untrained initialization" if initialization else (root.name if root else "Training run"),
              "notes": "", "algorithm": method, "seed": config.get("SEED"), "progress_unit": unit,
              "config": portable_metadata(config), "provenance": stable_run_provenance(metadata),
              "device": portable_metadata(metadata.get("device", {})),
              "started_at": metadata.get("started_at")}
    checkpoint_record = {"source_checkpoint_id": source_id + ":sha256:" + digest,
                         "source_run_id": source_id, "sha256": digest, "progress": progress,
                         "provenance": {"training_resumes": portable_metadata(metadata.get("resumes", []))}}
    return record, checkpoint_record
