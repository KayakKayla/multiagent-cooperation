"""Transactional evidence import. Names/notes are annotations; identity never is."""
import shutil
import time
from sqlalchemy import select, func, delete
from fastapi import HTTPException
from evals_shared.format import content_digest, canonical_json
from .models import TrainingRun, Policy, Evaluation, Participant, Episode, UploadPreview, LoginSession, RunIdentity, CheckpointIdentity
from .storage import pack_recording
from .views import checkpoint_dict


def evidence(run):
    return {k: v for k, v in run.items() if k not in {"name", "notes", "source_run_id"}}


def find_run(db, source):
    alias = db.get(RunIdentity, source)
    return db.get(TrainingRun, alias.run_id) if alias else db.scalar(select(TrainingRun).where(TrainingRun.source_run_id == source))


def find_checkpoint(db, source):
    alias = db.get(CheckpointIdentity, source)
    return db.get(Policy, alias.policy_id) if alias else db.scalar(select(Policy).where(Policy.source_checkpoint_id == source))


def conflict(message):
    raise HTTPException(409, message)


def annotation(value, field, default="", maximum=20000):
    if value is None:
        value = default
    if not isinstance(value, str) or len(value) > maximum or (field in {"name", "title"} and not value.strip()):
        raise HTTPException(422, f"{field} must be plain text of 1–{maximum} characters" if field != "notes" else f"notes must be plain text of at most {maximum} characters")
    return value


def cleanup(db):
    db.execute(delete(UploadPreview).where(UploadPreview.expires_at <= int(time.time())))
    db.execute(delete(LoginSession).where(LoginSession.expires_at <= int(time.time())))


def check_capacity(db, settings, added=0):
    # SQLite allocated pages include reusable deleted space. Include all runtime files.
    used = sum(p.stat().st_size for p in settings.data_dir.iterdir() if p.is_file())
    if used + added > settings.max_storage_bytes:
        raise HTTPException(507, "Archive storage quota reached; contact the administrator")
    if shutil.disk_usage(settings.data_dir).free < settings.reserve_bytes + added:
        raise HTTPException(507, "Insufficient free disk reserve for this upload")


def preview_metadata(db, doc):
    runs = db.scalars(select(TrainingRun).order_by(TrainingRun.name).limit(1000)).all()
    policy_rows = db.execute(select(Policy, TrainingRun).join(TrainingRun, Policy.run_id == TrainingRun.id).limit(5000)).all()
    policies = [p for p, _ in policy_rows]
    policy_runs = {r.id: r for _, r in policy_rows}
    uploaded_runs = {r["source_run_id"]: r for r in doc["training_runs"]}
    by_run = {r["source_run_id"]: find_run(db, r["source_run_id"]) for r in doc["training_runs"]}
    by_run = {k: v for k, v in by_run.items() if v is not None}
    by_policy = {p["source_checkpoint_id"]: find_checkpoint(db, p["source_checkpoint_id"]) for p in doc["checkpoints"]}
    by_policy = {k: v for k, v in by_policy.items() if v is not None}
    existing = db.scalar(select(Evaluation).where(Evaluation.source_evaluation_id == doc["source_evaluation_id"]))
    errors = []
    if existing and existing.digest != content_digest(doc):
        errors.append("This source evaluation identity already exists with different contents")
    for r in doc["training_runs"]:
        if r["source_run_id"] in by_run and evidence(r) != by_run[r["source_run_id"]].evidence:
            errors.append(f"Training run metadata conflicts: {r['source_run_id']}")
    for p in doc["checkpoints"]:
        old = by_policy.get(p["source_checkpoint_id"])
        if old:
            established_run = by_run.get(p["source_run_id"])
            old_run = policy_runs.get(old.run_id) or db.get(TrainingRun, old.run_id)
            if (old.sha256 != p["sha256"] or old.progress != p.get("progress")
                or (established_run and old.run_id != established_run.id)
                or old_run.evidence != evidence(uploaded_runs[p["source_run_id"]])):
                errors.append(f"Checkpoint identity conflicts: {p['source_checkpoint_id']}")
    suggestions = {}
    for checkpoint in doc["checkpoints"]:
        source_run = checkpoint["source_run_id"]
        established_run = by_run.get(source_run)
        suggestions[checkpoint["source_checkpoint_id"]] = [p.id for p in policies
            if p.sha256 == checkpoint["sha256"] and p.progress == checkpoint.get("progress")
            and policy_runs[p.run_id].evidence == evidence(uploaded_runs[source_run])
            and (established_run is None or p.run_id == established_run.id)]
    return {
        "document": {k: v for k, v in doc.items() if k not in {"episodes"}},
        "training_runs": doc["training_runs"], "checkpoints": doc["checkpoints"],
        "episode_count": len(doc["episodes"]), "recording_count": sum(e["recording"] is not None for e in doc["episodes"]),
        "matches": {"training_runs": {r["source_run_id"]: by_run[r["source_run_id"]].id for r in doc["training_runs"] if r["source_run_id"] in by_run},
                    "checkpoints": {p["source_checkpoint_id"]: by_policy[p["source_checkpoint_id"]].id for p in doc["checkpoints"] if p["source_checkpoint_id"] in by_policy}},
        "existing_runs": [{"id": r.id, "name": r.name, "source_run_id": r.source_run_id} for r in runs],
        "existing_checkpoints": [checkpoint_dict(p, r) for p, r in policy_rows],
        "checkpoint_suggestions": suggestions,
        "duplicate_id": existing.id if existing and existing.digest == content_digest(doc) else None,
        "conflicts": errors,
    }


def import_document(db, doc, user_id, choices=None):
    """Caller holds BEGIN IMMEDIATE; constraints also protect identities."""
    choices = choices or {}
    old = db.scalar(select(Evaluation).where(Evaluation.source_evaluation_id == doc["source_evaluation_id"]))
    if old:
        if old.digest != content_digest(doc):
            conflict("Source evaluation identity already exists with different contents")
        return old, True
    runs = {}
    for item in doc["training_runs"]:
        source = item["source_run_id"]
        old = find_run(db, source)
        choice = choices.get("run_associations", {}).get(source)
        if choice and choice != "new":
            selected = db.get(TrainingRun, choice)
            if selected is None or (old and old.id != selected.id):
                conflict("Selected run conflicts with the established source identity")
            old = selected
        if old:
            if old.evidence != evidence(item):
                conflict(f"Training run metadata conflict for {source}; correct the source record before import")
            runs[source] = old
            if not db.get(RunIdentity, source):
                db.add(RunIdentity(source_id=source, run_id=old.id))
        else:
            annotations = choices.get("run_annotations", {}).get(source, {})
            row = TrainingRun(source_run_id=source, name=annotation(annotations.get("name", item["name"]), "name", maximum=300), notes=annotation(annotations.get("notes", item.get("notes", "")), "notes"), algorithm=item.get("algorithm"), seed=item.get("seed"), progress_unit=item.get("progress_unit"), evidence=evidence(item), created_by=user_id, updated_by=user_id)
            db.add(row)
            db.flush()
            runs[source] = row
            db.add(RunIdentity(source_id=source, run_id=row.id))
    policies = {}
    for item in doc["checkpoints"]:
        source = item["source_checkpoint_id"]
        run = runs[item["source_run_id"]]
        old = find_checkpoint(db, source)
        choice = choices.get("checkpoint_associations", {}).get(source)
        if choice and choice != "new":
            selected = db.get(Policy, choice)
            if selected is None or (old and old.id != selected.id):
                conflict("Selected checkpoint conflicts with the established source identity")
            old = selected
        if old:
            if old.run_id != run.id or old.sha256 != item["sha256"] or old.progress != item.get("progress"):
                conflict("Checkpoint identity, weights hash, run or progress conflicts")
            policies[source] = old
            if not db.get(CheckpointIdentity, source):
                db.add(CheckpointIdentity(source_id=source, policy_id=old.id))
        else:
            row = Policy(source_checkpoint_id=source, run_id=run.id, sha256=item["sha256"], progress=item.get("progress"))
            db.add(row)
            db.flush()
            policies[source] = row
            db.add(CheckpointIdentity(source_id=source, policy_id=row.id))
    metadata = {k: v for k, v in doc.items() if k not in {"episodes", "training_runs", "checkpoints", "participants", "title", "notes"}}
    env = doc["environment"]
    row = Evaluation(source_evaluation_id=doc["source_evaluation_id"], digest=content_digest(doc), title=annotation(choices.get("title", doc.get("title", env.get("layout", "Evaluation"))), "title", maximum=300), notes=annotation(choices.get("notes", doc.get("notes", "")), "notes"), layout=env["layout"], environment_name=env["name"], episode_count=len(doc["episodes"]), metadata_json=metadata, created_by=user_id, updated_by=user_id)
    db.add(row)
    db.flush()
    for p in doc["participants"]:
        db.add(Participant(evaluation_id=row.id, policy_id=policies[p["source_checkpoint_id"]].id, agent_id=p["agent_id"], display_order=p["display_order"], observation_settings=p.get("observation_settings", {}), action_settings=p.get("action_settings", {})))
    for ep in doc["episodes"]:
        db.add(Episode(evaluation_id=row.id, index=ep["index"], seed=ep["seed"], steps=ep["steps"], termination=ep.get("termination", {}), metrics=ep["metrics"], **pack_recording(ep["recording"])))
    db.flush()
    return row, False
