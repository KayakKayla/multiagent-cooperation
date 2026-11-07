"""Exercise the public /sims URL through stripped-prefix proxy/server semantics."""
import json
from pathlib import Path
import secrets
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlparse, parse_qs
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from app.auth import COOKIE, new_session, token_hash
from app.config import Settings
from app.main import create_app
from app.models import Base, User, LoginSession


class ProxyAndServer:
    """Apache strips /sims; Uvicorn(root_path=/sims) reconstructs ASGI path.

    Retain the real external client URL for cookie matching while presenting
    the application the HTTP/loopback scope and sanitized forwarded headers.
    """
    def __init__(self, app):
        self.settings = app.state.settings
        self.app = ProxyHeadersMiddleware(app, trusted_hosts=["127.0.0.1"])

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            prefix = self.settings.root_path
            external = scope["path"]
            assert external.startswith(prefix + "/")
            stripped = external[len(prefix):]
            path = prefix + stripped  # Uvicorn's root_path + upstream request path.
            headers = [(k, v) for k, v in scope["headers"] if k.lower() not in {b"x-forwarded-proto", b"x-forwarded-for"}]
            headers += [(b"x-forwarded-proto", b"https"), (b"x-forwarded-for", b"198.51.100.3")]
            scope = {**scope, "path": path, "raw_path": path.encode(), "root_path": prefix,
                     "scheme": "http", "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 43210), "headers": headers}
        await self.app(scope, receive, send)


@pytest.fixture
def subsite(tmp_path):
    settings = Settings(data_dir=tmp_path, base_url="https://testserver/sims", github_client_id="client", github_client_secret="secret", allowed_github_ids=frozenset({123}), reserve_bytes=0)
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    with app.state.db() as db:
        user = User(github_id=123, login="teammate", display_name="Teammate")
        db.add(user); db.flush()
        token, session = new_session(db, settings, user.id)
    with TestClient(ProxyAndServer(app), base_url="https://testserver", follow_redirects=False) as client:
        yield app, client, token, session.csrf_token
    app.state.engine.dispose()


def test_subpath_public_upload_edit_delete_reference_and_cookie_scope(subsite):
    app, client, token, csrf = subsite
    home = client.get("/sims/")
    assert home.status_code == 200
    assert 'href="/sims/' in home.text
    assert 'src="/sims/static/' in home.text
    assert 'content="/sims"' in home.text
    stylesheet = client.get("/sims/static/site.css")
    assert stylesheet.status_code == 200
    assert stylesheet.headers["cache-control"] == "public, max-age=86400"
    assert client.get("/sims/api/evaluations").json()["evaluations"] == []
    assert client.get("/sims/upload").status_code == 401
    client.cookies.set(COOKIE, token, path="/sims/")
    headers = {"x-csrf-token": csrf, "origin": "https://testserver", "content-type": "application/json"}
    assert client.get("/sims/upload").status_code == 200
    doc = json.loads((Path(__file__).parent / "fixtures" / "evaluation-v1.json").read_text())
    preview = client.post("/sims/api/uploads", content=json.dumps(doc), headers=headers)
    assert preview.status_code == 200, preview.text
    result = client.post("/sims/api/imports", json={"preview_id": preview.json()["id"]}, headers=headers)
    assert result.status_code == 200, result.text
    record = result.json()
    assert record["url"] == f"/sims/evaluations/{record['id']}"
    detail = client.get(record["url"])
    assert detail.status_code == 200 and '/sims/api/evaluations/' in detail.text
    recording = client.get(f"/sims/api/evaluations/{record['id']}/episodes/0")
    assert recording.status_code == 200 and recording.json()["recording"] is not None
    run_id = recording.json()["participants"][0]["run_id"]
    checkpoint_id = recording.json()["participants"][0]["checkpoint_id"]
    assert client.get(f"/sims/training-runs/{run_id}").status_code == 200
    blocked = client.delete(f"/sims/api/checkpoints/{checkpoint_id}", headers=headers)
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["references"][0]["url"] == record["url"]
    blocked_run = client.delete(f"/sims/api/training-runs/{run_id}", headers=headers)
    assert blocked_run.status_code == 409
    assert blocked_run.json()["detail"]["references"][0]["url"].startswith(f"/sims/training-runs/{run_id}")
    route = f"/sims/api/evaluations/{record['id']}"
    # Origin is the origin alone, not the configured site URL containing /sims.
    assert client.patch(route, json={"version": 1, "title": "Edited"}, headers=headers).status_code == 200
    assert client.patch(route, json={"version": 2, "notes": "bad origin"}, headers={**headers, "origin": "https://evil.example"}).status_code == 403
    assert client.patch(route, json={"version": 2, "notes": "no CSRF"}, headers={"origin": "https://testserver"}).status_code == 403
    redirected = client.get(record["url"] + "/")
    assert redirected.status_code == 307
    assert redirected.headers["location"] == "https://testserver" + record["url"]
    assert client.delete(route, headers=headers).status_code == 200
    missing = client.get(f"/sims/api/evaluations/{record['id']}/episodes/0")
    assert missing.status_code == 404 and missing.json()["detail"] == "Record not found"


def test_subpath_oauth_callback_pkce_rotation_logout_and_origin(subsite):
    app, client, _, _ = subsite
    response = client.get("/sims/auth/login")
    assert response.status_code == 302
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["redirect_uri"] == ["https://testserver/sims/auth/callback"]
    assert query["code_challenge_method"] == ["S256"]
    assert "Path=/sims/" in response.headers["set-cookie"]
    assert "Secure" in response.headers["set-cookie"]
    old = client.cookies.get(COOKIE)
    with app.state.db() as db:
        assert query["state"][0] in str(db.get(LoginSession, token_hash(old)).data)
    assert client.get("/sims/auth/callback?state=wrong&code=fake").status_code == 400
    # Separate provider identity test follows real state/PKCE rejection above.
    client.get("/sims/auth/login")
    old = client.cookies.get(COOKIE)
    github = app.state.oauth.github
    github.authorize_access_token = AsyncMock(return_value={"access_token": "discard", "token_type": "bearer"})
    class Identity:
        def raise_for_status(self): pass
        def json(self): return {"id": 123, "login": "teammate", "name": "Teammate"}
    github.get = AsyncMock(return_value=Identity())
    callback = client.get("/sims/auth/callback?code=fake&state=test")
    assert callback.status_code == 303 and callback.headers["location"] == "/sims/"
    assert "Path=/sims/" in callback.headers["set-cookie"]
    token = client.cookies.get(COOKIE)
    assert token != old
    with app.state.db() as db:
        assert db.get(LoginSession, token_hash(old)) is None
        session = db.get(LoginSession, token_hash(token))
        csrf = session.csrf_token
        assert "discard" not in str(session.data)
    logout = client.post("/sims/auth/logout", headers={"origin": "https://testserver", "x-csrf-token": csrf})
    assert logout.status_code == 200 and "Path=/sims/" in logout.headers["set-cookie"]
    assert client.cookies.get(COOKIE) is None
    client.cookies.set(COOKIE, token, path="/sims/")
    assert client.delete("/sims/api/evaluations/missing", headers={"x-csrf-token": csrf}).status_code == 401


@pytest.mark.parametrize("value", ["https://testserver/sims?x=1", "https://testserver/sims#fragment", "https://name:secret@testserver/sims", "https://testserver/sims/../outside", "https://testserver//sims", "https://testserver/%2fsims"])
def test_invalid_site_prefixes_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        Settings(data_dir=tmp_path, base_url=value).prepare()


def test_terminal_login_uses_browser_approval_and_limits_access(subsite):
    app, browser, _, _ = subsite
    token = secrets.token_urlsafe(32)
    challenge = token_hash(token)
    approval = f"/sims/auth/cli?challenge={challenge}"
    login = browser.get(approval)
    assert login.headers["location"] == f"/sims/auth/login?cli_challenge={challenge}"
    assert browser.get(login.headers["location"]).status_code == 302
    github = app.state.oauth.github
    github.authorize_access_token = AsyncMock(return_value={"access_token": "discard"})
    identity = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"id": 123, "login": "teammate"})
    github.get = AsyncMock(return_value=identity)
    callback = browser.get("/sims/auth/callback?code=fake&state=test")
    assert callback.headers["location"] == approval
    page = browser.get(approval)
    assert page.status_code == 200 and challenge[:12] in page.text
    with app.state.db() as db:
        csrf = db.get(LoginSession, token_hash(browser.cookies.get(COOKIE))).csrf_token

    with TestClient(ProxyAndServer(app), base_url="https://testserver") as terminal:
        terminal.cookies.set(COOKIE, token, path="/sims/")
        assert terminal.get("/sims/api/cli/session").status_code == 401
        form = {"challenge": challenge, "csrf_token": csrf}
        assert browser.post("/sims/auth/cli", data={"challenge": challenge}).status_code == 403
        assert browser.post("/sims/auth/cli", data=form, headers={"origin": "https://evil.example"}).status_code == 403
        assert browser.post("/sims/auth/cli", data=form).status_code == 200
        session = terminal.get("/sims/api/cli/session")
        assert session.status_code == 200 and session.json()["csrf_token"] != csrf
        headers = {"x-csrf-token": session.json()["csrf_token"], "content-type": "application/json"}
        raw = (Path(__file__).parent / "fixtures" / "evaluation-v1.json").read_bytes()
        preview = terminal.post("/sims/api/uploads", content=raw, headers=headers)
        assert preview.status_code == 200, preview.text
        result = terminal.post("/sims/api/imports", json={"preview_id": preview.json()["id"]}, headers=headers)
        assert result.status_code == 200, result.text
        assert terminal.delete(f"/sims/api/evaluations/{result.json()['id']}", headers=headers).status_code == 401
        assert terminal.get("/sims/upload").status_code == 401
        with app.state.db() as db:
            session = db.get(LoginSession, challenge)
            assert time.time() < session.expires_at <= time.time() + 1800
            session.expires_at = int(time.time()) - 1
            db.commit()
        assert terminal.get("/sims/api/cli/session").status_code == 401
    assert browser.get("/sims/upload").status_code == 200
