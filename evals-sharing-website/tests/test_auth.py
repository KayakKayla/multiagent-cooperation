from urllib.parse import urlparse, parse_qs
import time
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy import select
from app.main import create_app
from app.config import Settings
from app.models import Base, LoginSession, User
from app.auth import COOKIE, token_hash


def oauth_site(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, base_url="https://testserver", github_client_id="client", github_client_secret="secret", allowed_github_ids=frozenset({123})))
    Base.metadata.create_all(app.state.engine)
    return app, TestClient(app, base_url="https://testserver", follow_redirects=False)


def test_oauth_state_pkce_server_side_and_callback_validation(tmp_path):
    app, client = oauth_site(tmp_path)
    response = client.get("/auth/login")
    assert response.status_code == 302
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["code_challenge_method"] == ["S256"]
    assert query["scope"] == ["read:user"]
    assert query["redirect_uri"] == ["https://testserver/auth/callback"]
    cookie = response.headers["set-cookie"]
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
    raw = client.cookies.get(COOKIE)
    assert query["state"][0] not in raw and "code_verifier" not in raw
    with app.state.db() as db:
        session = db.get(LoginSession, token_hash(raw))
        assert session and "code_verifier" in str(session.data)
        assert query["state"][0] in str(session.data)
    assert client.get("/auth/callback?state=wrong&code=fake").status_code == 400
    client.cookies.clear()
    assert client.get("/auth/callback?state=anything&code=fake").status_code == 400
    app.state.engine.dispose()


def test_identity_allowlist_rotation_no_token_retention_and_disabled(tmp_path):
    app, client = oauth_site(tmp_path)
    github = app.state.oauth.github
    github.authorize_access_token = AsyncMock(return_value={"access_token": "NEVER-PERSIST-ME", "token_type": "bearer"})
    class Identity:
        def __init__(self, id): self.id = id
        def raise_for_status(self): pass
        def json(self): return {"id": self.id, "login": "teammate", "name": "Teammate"}
    github.get = AsyncMock(return_value=Identity(999))
    client.get("/auth/login")
    assert client.get("/auth/callback?code=fake&state=test").status_code == 403
    github.get = AsyncMock(return_value=Identity(123))
    client.get("/auth/login")
    old_token = client.cookies.get(COOKIE)
    response = client.get("/auth/callback?code=fake&state=test")
    assert response.status_code == 303
    new_token = client.cookies.get(COOKIE)
    assert new_token != old_token
    with app.state.db() as db:
        assert db.get(LoginSession, token_hash(old_token)) is None
        sessions = db.scalars(select(LoginSession)).all()
        assert "NEVER-PERSIST-ME" not in str([s.data for s in sessions])
        user = db.scalar(select(User)); user.active = False; db.commit()
    client.get("/auth/login")
    assert client.get("/auth/callback?code=fake&state=test").status_code == 403
    app.state.engine.dispose()


def test_expired_session_and_untrusted_host(tmp_path):
    app, client = oauth_site(tmp_path)
    client.get("/auth/login")
    with app.state.db() as db:
        session = db.scalar(select(LoginSession)); session.expires_at = int(time.time()) - 1; db.commit()
    assert client.get("/auth/callback?state=x&code=y").status_code == 400
    assert client.get("/", headers={"host": "evil.example"}).status_code == 400
    app.state.engine.dispose()
