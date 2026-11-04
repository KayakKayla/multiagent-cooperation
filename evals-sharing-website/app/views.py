from sqlalchemy import select, func, or_, exists
from .presentation import format_progress
from .models import Evaluation, TrainingRun, Policy, Participant, Episode


def run_dict(run):
    return {**run.evidence, "id": run.id, "source_run_id": run.source_run_id, "name": run.name, "notes": run.notes, "algorithm": run.algorithm, "version": run.version, "created_at": run.created_at, "updated_at": run.updated_at}


def evaluation_dict(evaluation):
    return {**evaluation.metadata_json, "id": evaluation.id, "title": evaluation.title, "notes": evaluation.notes, "layout": evaluation.layout, "episode_count": evaluation.episode_count, "version": evaluation.version, "evaluated_at": evaluation.metadata_json.get("created_at"), "uploaded_at": evaluation.created_at, "created_at": evaluation.created_at, "updated_at": evaluation.updated_at}


def checkpoint_dict(policy, run):
    progress = format_progress(policy.progress, run.progress_unit)
    return {"id": policy.id, "checkpoint_id": policy.id, "source_checkpoint_id": policy.source_checkpoint_id, "run_id": run.id, "run_name": run.name, "training_run_name": run.name, "progress": policy.progress, "progress_unit": run.progress_unit, "sha256": policy.sha256, "label": f"{run.name} · {progress}"}


def participants(db, evaluation_id):
    rows = db.execute(select(Participant, Policy, TrainingRun).join(Policy, Participant.policy_id == Policy.id).join(TrainingRun, Policy.run_id == TrainingRun.id).where(Participant.evaluation_id == evaluation_id).order_by(Participant.display_order)).all()
    return [{**checkpoint_dict(p, r), "agent_id": a.agent_id, "display_order": a.display_order, "observation_settings": a.observation_settings, "action_settings": a.action_settings} for a, p, r in rows]


def episode_dict(e):
    return {"index": e.index, "seed": e.seed, "steps": e.steps, "termination": e.termination, "metrics": e.metrics, "recording_available": e.recording_available}


def lists(db, q="", layout="", algorithm="", page=1, tab="evaluations"):
    ev = select(Evaluation)
    runs = select(TrainingRun)
    if q:
        # LIKE wildcard escaping makes search literal rather than an accidental pattern.
        pattern = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        ev = ev.where(or_(Evaluation.title.ilike(pattern, escape="\\"), Evaluation.notes.ilike(pattern, escape="\\"), Evaluation.layout.ilike(pattern, escape="\\")))
        runs = runs.where(or_(TrainingRun.name.ilike(pattern, escape="\\"), TrainingRun.notes.ilike(pattern, escape="\\")))
    if layout:
        ev = ev.where(Evaluation.layout == layout)
        related = select(Policy.run_id).join(Participant, Participant.policy_id == Policy.id).join(Evaluation, Participant.evaluation_id == Evaluation.id).where(Evaluation.layout == layout)
        runs = runs.where(TrainingRun.id.in_(related))
    if algorithm:
        runs = runs.where(TrainingRun.algorithm == algorithm)
        ev = ev.where(Evaluation.id.in_(select(Participant.evaluation_id).join(Policy, Participant.policy_id == Policy.id).join(TrainingRun, Policy.run_id == TrainingRun.id).where(TrainingRun.algorithm == algorithm)))
    count = db.scalar(select(func.count()).select_from((runs if tab == "training-runs" else ev).subquery()))
    pages = max(1, (count + 19) // 20)
    page = min(max(1, page), pages)
    return {"evaluations": [evaluation_dict(e) for e in db.scalars(ev.order_by(Evaluation.created_at.desc()).offset((page - 1) * 20).limit(20))], "training_runs": [run_dict(r) for r in db.scalars(runs.order_by(TrainingRun.created_at.desc()).offset((page - 1) * 20).limit(20))], "page": page, "pages": pages, "total": count, "tab": tab, "filters": {"q": q, "layout": layout, "algorithm": algorithm}}
