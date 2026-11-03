from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func, text, event
from sqlalchemy.orm import undefer
from app.config import Settings
from app.main import create_app
from app.models import Base, User, Evaluation, TrainingRun, Policy, Participant, Episode, LoginSession, UploadPreview
from app.auth import new_session, COOKIE
from app.imports import import_document
from app.cli import backup, restore, migrate


@pytest.fixture
def doc():
    return json.loads((Path(__file__).parent / "fixtures" / "evaluation-v1.json").read_text())


@pytest.fixture
def site(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, base_url="http://testserver", allowed_github_ids=frozenset({123}), reserve_bytes=0))
    Base.metadata.create_all(app.state.engine)
    with app.state.db() as db:
        user = User(github_id=123, login="teammate", display_name="Teammate")
        db.add(user); db.flush()
        token, session = new_session(db, app.state.settings, user.id)
    with TestClient(app) as client:
        yield app, client, token, session.csrf_token
    app.state.engine.dispose()


def login(site):
    app, client, token, csrf = site
    client.cookies.set(COOKIE, token)
    return {"x-csrf-token": csrf, "content-type": "application/json"}


def upload(site, doc):
    _, client, *_ = site
    response = client.post("/api/uploads", content=json.dumps(doc), headers=login(site))
    assert response.status_code == 200, response.text
    return response.json()


def publish(site, doc, **choices):
    preview = upload(site, doc)
    response = site[1].post("/api/imports", json={"preview_id": preview["id"], **choices}, headers=login(site))
    assert response.status_code == 200, response.text
    return response.json()


def test_public_read_private_write_and_selected_blob(site, doc):
    result = publish(site, doc)
    app, client, *_ = site
    client.cookies.clear()
    statements = []
    def capture(conn, cursor, statement, params, context, many):
        statements.append(statement)
    event.listen(app.state.engine, "before_cursor_execute", capture)
    assert client.get("/").status_code == 200
    assert client.get(result["url"]).status_code == 200
    assert not any("episodes.recording " in s or "episodes.recording," in s for s in statements)
    data = client.get(f"/api/evaluations/{result['id']}/episodes/0")
    assert data.status_code == 200, data.text
    assert data.json()["recording"]["steps"] == doc["episodes"][0]["recording"]["steps"]
    assert client.patch(f"/api/evaluations/{result['id']}", json={"version": 1, "title": "x"}).status_code == 401
    with app.state.db() as db:
        episode = db.scalar(select(Episode).where(Episode.evaluation_id == result["id"]))
        with pytest.raises(Exception, match="raiseload"):
            _ = episode.recording


def test_preview_atomic_dedup_and_optimistic_edit(site, doc):
    preview = upload(site, doc)
    app, client, *_ = site
    with app.state.db() as db:
        assert db.scalar(select(func.count()).select_from(Evaluation)) == 0
    first = client.post("/api/imports", json={"preview_id": preview["id"]}, headers=login(site)).json()
    second = publish(site, doc)
    assert first["id"] == second["id"] and second["duplicate"]
    route = f"/api/evaluations/{first['id']}"
    assert client.patch(route, json={"version": 1, "title": "<script>alert(1)</script>", "notes": "text"}, headers=login(site)).status_code == 200
    assert client.patch(route, json={"version": 1, "title": "stale"}, headers=login(site)).status_code == 409
    assert "&lt;script&gt;" in client.get(first["url"]).text
    conflicting = copy.deepcopy(doc); conflicting["device"] = {"label": "changed"}
    p = upload(site, conflicting)
    assert p["conflicts"]
    assert client.post("/api/imports", json={"preview_id": p["id"]}, headers=login(site)).status_code == 409
    with app.state.db() as db:
        assert db.scalar(select(func.count()).select_from(Evaluation)) == 1


def test_reviewed_aliases_later_checkpoint_and_reference_deletion(site, doc):
    first = publish(site, doc)
    app, client, *_ = site
    with app.state.db() as db:
        run = db.scalar(select(TrainingRun)); policy = db.scalar(select(Policy))
        run_id, policy_id = run.id, policy.id
        assert db.scalar(select(func.count()).select_from(Policy)) == 1  # A+A
    alias = copy.deepcopy(doc)
    alias["source_evaluation_id"] += "-alias"
    for r in alias["training_runs"]:
        r["source_run_id"] += "-portable"
    for p in alias["checkpoints"]:
        p["source_run_id"] += "-portable"; p["source_checkpoint_id"] += "-portable"
    for p in alias["participants"]:
        p["source_checkpoint_id"] += "-portable"
    choices = {"run_associations": {alias["training_runs"][0]["source_run_id"]: run_id}, "checkpoint_associations": {alias["checkpoints"][0]["source_checkpoint_id"]: policy_id}}
    second = publish(site, alias, **choices)
    later = copy.deepcopy(alias); later["source_evaluation_id"] += "-later"
    later["checkpoints"][0]["source_checkpoint_id"] += "-later"
    later["checkpoints"][0]["sha256"] = "b" * 64
    later["checkpoints"][0]["progress"] = 200
    for p in later["participants"]:
        p["source_checkpoint_id"] += "-later"
    third = publish(site, later)
    with app.state.db() as db:
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 1
        assert db.scalar(select(func.count()).select_from(Policy)) == 2
    assert client.delete(f"/api/checkpoints/{policy_id}", headers=login(site)).status_code == 409
    assert client.delete(f"/api/training-runs/{run_id}", headers=login(site)).status_code == 409
    for item in (first, second, third):
        assert client.delete(f"/api/evaluations/{item['id']}", headers=login(site)).status_code == 200
    with app.state.db() as db:
        assert db.scalar(select(func.count()).select_from(Episode)) == 0
        assert db.scalar(select(func.count()).select_from(Participant)) == 0
        assert db.scalar(select(func.count()).select_from(Policy)) == 2


def test_wrong_checkpoint_rejected_without_partial_records(site, doc):
    publish(site, doc)
    app, client, *_ = site
    with app.state.db() as db:
        policy_id = db.scalar(select(Policy.id))
    new = copy.deepcopy(doc); new["source_evaluation_id"] += "-bad"
    new["checkpoints"][0]["source_checkpoint_id"] += "-other"
    new["checkpoints"][0]["sha256"] = "c" * 64
    for p in new["participants"]:
        p["source_checkpoint_id"] += "-other"
    preview = upload(site, new)
    result = client.post("/api/imports", json={"preview_id": preview["id"], "checkpoint_associations": {new["checkpoints"][0]["source_checkpoint_id"]: policy_id}}, headers=login(site))
    assert result.status_code == 409
    with app.state.db() as db:
        assert db.scalar(select(func.count()).select_from(Evaluation)) == 1
        assert db.scalar(select(func.count()).select_from(Policy)) == 1


def test_csrf_origin_allowlist_expiry_and_logout(site):
    app, client, token, csrf = site
    route = "/api/evaluations/missing"
    client.cookies.set(COOKIE, token)
    assert client.delete(route).status_code == 403
    assert client.delete(route, headers={"x-csrf-token": csrf, "origin": "https://evil.example"}).status_code == 403
    with app.state.db() as db:
        user = db.scalar(select(User)); user.active = False; db.commit()
    assert client.delete(route, headers={"x-csrf-token": csrf}).status_code == 401
    with app.state.db() as db:
        user = db.scalar(select(User)); user.active = True; db.commit()
    assert client.post("/auth/logout", headers={"x-csrf-token": csrf}).status_code == 200
    client.cookies.set(COOKIE, token)
    assert client.delete(route, headers={"x-csrf-token": csrf}).status_code == 401


def test_exact_actual_size_cap_malformed_and_missing_recording(site, doc):
    _, client, *_ = site
    headers = login(site)
    raw = json.dumps(doc).encode()
    exact = raw + b" " * (20 * 1024**2 - len(raw))
    assert client.post("/api/uploads", content=exact, headers=headers).status_code == 200
    # Deliberately no trustworthy Content-Length: actual streamed bytes control limit.
    assert client.post("/api/uploads", content=iter([exact, b" "]), headers=headers).status_code == 413
    for raw_bad in (b'{"x":NaN}', b"{" + b'"x":[' * 1000):
        assert client.post("/api/uploads", content=raw_bad, headers=headers).status_code == 422
    doc["source_evaluation_id"] += "-missing"
    doc["episodes"][0]["recording"] = None
    published = publish(site, doc)
    assert client.get(f"/api/evaluations/{published['id']}/episodes/0").json()["recording"] is None


def test_concurrent_imports_and_backup_restore(site, doc, tmp_path):
    app, *_ = site
    def worker():
        with app.state.db() as db:
            db.execute(text("BEGIN IMMEDIATE"))
            user = db.scalar(select(User))
            result, duplicate = import_document(db, doc, user.id)
            db.commit()
            return result.id
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: worker(), range(2)))
    assert len(set(results)) == 1
    saved = tmp_path / "backup.db"; restored = tmp_path / "restored.db"
    backup(app.state.settings.database_path, saved); restore(saved, restored)
    import sqlite3
    with sqlite3.connect(restored) as conn:
        assert conn.execute("SELECT count(*) FROM evaluations").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migrations_clean_database(tmp_path):
    settings = Settings(data_dir=tmp_path, base_url="http://testserver")
    settings.prepare(); migrate(settings)
    import sqlite3
    with sqlite3.connect(settings.database_path) as db:
        assert db.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0001"
        assert "episodes" in {r[0] for r in db.execute("SELECT name FROM sqlite_master")}


def test_preview_suggests_only_exact_compatible_checkpoints_with_readable_labels(site, doc):
    publish(site, doc)
    with site[0].state.db() as db:
        policy = db.scalar(select(Policy))
        policy_id, run_id = policy.id, policy.run_id
    portable = copy.deepcopy(doc)
    portable["source_evaluation_id"] += "-portable-review"
    for run in portable["training_runs"]:
        run["source_run_id"] += "-portable-review"
    for checkpoint in portable["checkpoints"]:
        checkpoint["source_run_id"] += "-portable-review"
        checkpoint["source_checkpoint_id"] += "-portable-review"
    for participant in portable["participants"]:
        participant["source_checkpoint_id"] += "-portable-review"
    preview = upload(site, portable)
    source_checkpoint = portable["checkpoints"][0]["source_checkpoint_id"]
    source_run = portable["training_runs"][0]["source_run_id"]
    assert preview["checkpoint_suggestions"][source_checkpoint] == [policy_id]
    assert preview["matches"] == {"training_runs": {}, "checkpoints": {}}
    candidate = next(c for c in preview["existing_checkpoints"] if c["id"] == policy_id)
    assert candidate["label"].startswith(doc["training_runs"][0]["name"] + " · ")
    assert candidate["run_name"] == doc["training_runs"][0]["name"]
    assert candidate["run_id"] == run_id
    assert candidate["progress"] == doc["checkpoints"][0]["progress"]
    assert candidate["progress_unit"] == doc["training_runs"][0]["progress_unit"]
    changed_progress = copy.deepcopy(portable)
    changed_progress["checkpoints"][0]["progress"] += 1
    assert upload(site, changed_progress)["checkpoint_suggestions"][source_checkpoint] == []
    changed_model = copy.deepcopy(portable)
    changed_model["training_runs"][0]["config"]["model"] = {"hidden_size": 999}
    assert upload(site, changed_model)["checkpoint_suggestions"][source_checkpoint] == []
    # A suggestion does not change associations until explicitly reviewed.
    with site[0].state.db() as db:
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 1
        assert db.scalar(select(func.count()).select_from(Policy)) == 1
    response = site[1].post("/api/imports", json={"preview_id": preview["id"], "run_associations": {source_run: run_id}, "checkpoint_associations": {source_checkpoint: policy_id}}, headers=login(site))
    assert response.status_code == 200, response.text
    with site[0].state.db() as db:
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 1
        assert db.scalar(select(func.count()).select_from(Policy)) == 1


@pytest.mark.parametrize(("field", "value"), [("algorithm", {}), ("seed", {}), ("seed", 2**80)])
def test_upload_rejects_invalid_database_scalar_types_before_preview(site, doc, field, value):
    doc["training_runs"][0][field] = value
    response = site[1].post("/api/uploads", content=json.dumps(doc), headers=login(site))
    assert response.status_code == 422, response.text
    with site[0].state.db() as db:
        assert db.scalar(select(func.count()).select_from(UploadPreview)) == 0
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 0
