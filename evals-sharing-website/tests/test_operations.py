"""Operational security boundaries, independent of live OAuth credentials."""
from pathlib import Path
import sqlite3
import time
from fastapi.testclient import TestClient
from sqlalchemy import delete
from app.auth import new_session, COOKIE, token_hash
from app.cli import backup, restore
from app.config import Settings
from app.main import create_app
from app.models import Base, User, LoginSession, UploadPreview


def test_restore_does_not_resurrect_revoked_access_or_pending_uploads(tmp_path):
    original_dir = tmp_path / "original"
    settings = Settings(data_dir=original_dir, base_url="http://testserver", allowed_github_ids=frozenset({123}))
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    with app.state.db() as db:
        user = User(github_id=123, login="teammate", display_name="Teammate")
        db.add(user); db.flush()
        token, session = new_session(db, settings, user.id)
        # Include nonexpired OAuth state and preview data in the backup.
        db.get(LoginSession, session.token_hash).data = {"oauth_state": "private-verifier"}
        preview = UploadPreview(user_id=user.id, expires_at=int(time.time()) + 1800, size=2, document=b"{}")
        db.add(preview); db.commit()
    saved = tmp_path / "before-revocation.sqlite3"
    backup(settings.database_path, saved)
    with app.state.db() as db:
        db.execute(delete(LoginSession).where(LoginSession.token_hash == token_hash(token)))
        db.execute(delete(UploadPreview)); db.commit()
    restored_dir = tmp_path / "restored"
    restored_dir.mkdir()
    restored_db = restored_dir / "evaluations.sqlite3"
    restore(saved, restored_db)
    with sqlite3.connect(restored_db) as db:
        assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM upload_previews").fetchone()[0] == 0
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    restored_app = create_app(Settings(data_dir=restored_dir, base_url="http://testserver", allowed_github_ids=frozenset({123})))
    with TestClient(restored_app) as client:
        client.cookies.set(COOKIE, token)
        response = client.delete("/api/evaluations/missing", headers={"x-csrf-token": session.csrf_token})
        assert response.status_code == 401
    restored_app.state.engine.dispose()
    app.state.engine.dispose()


def test_documented_startups_use_prefix_aware_safe_launcher(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parent.parent
    starts = []
    for path in [root / "README.md", root / "deploy" / "evals-site.service"]:
        starts.extend(line for line in path.read_text().splitlines() if "-m app.cli serve" in line)
    assert len(starts) == 3
    from app.cli import main
    from unittest.mock import Mock
    import uvicorn
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setenv("EVALS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EVALS_BASE_URL", "https://testserver/sims")
    monkeypatch.setattr("sys.argv", ["evals-site", "serve"])
    main()
    run.assert_called_once_with("app.main:app", host="127.0.0.1", port=8000,
        workers=1, root_path="/sims", proxy_headers=True,
        forwarded_allow_ips="127.0.0.1", access_log=False, timeout_keep_alive=5)
