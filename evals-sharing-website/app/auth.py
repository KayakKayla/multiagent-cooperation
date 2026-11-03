import hashlib
import secrets
import time
from collections import defaultdict, deque
from fastapi import HTTPException
from sqlalchemy import select, delete
from starlette.middleware.base import BaseHTTPMiddleware
from .models import LoginSession, User

COOKIE = "evals_session"


def token_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Sessions(BaseHTTPMiddleware):
    """Only an opaque token crosses the browser boundary; OAuth state stays in DB."""
    async def dispatch(self, request, call_next):
        request.state.user = None
        request.state.session = None
        token = request.cookies.get(COOKIE)
        if token and len(token) < 256:
            with request.app.state.db() as db:
                session = db.get(LoginSession, token_hash(token))
                if session and session.expires_at > time.time():
                    request.state.session = session
                    request.scope["session"] = dict(session.data)
                    if session.user_id:
                        user = db.get(User, session.user_id)
                        if user and user.active and user.github_id in request.app.state.settings.allowed_github_ids:
                            request.state.user = user
        request.scope.setdefault("session", {})
        response = await call_next(request)
        # Authlib changes request.session during OAuth authorization/callback.
        session = request.state.session
        if session:
            with request.app.state.db() as db:
                current = db.get(LoginSession, session.token_hash)
                if current:
                    current.data = request.scope["session"]
                    db.commit()
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith(("/api/", "/auth/", "/upload")) or request.state.user:
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=86400"
        return response


def new_session(db, settings, user_id=None):
    token = secrets.token_urlsafe(48)
    session = LoginSession(token_hash=token_hash(token), user_id=user_id, csrf_token=secrets.token_urlsafe(32), expires_at=int(time.time()) + settings.session_seconds, data={})
    db.add(session)
    db.commit()
    return token, session


def set_cookie(response, token, settings):
    response.set_cookie(COOKIE, token, max_age=settings.session_seconds, secure=settings.secure_cookie, httponly=True, samesite="lax", path="/")


def require_user(request):
    user = request.state.user
    if not user:
        raise HTTPException(401, "Sign in with an approved GitHub account")
    session = request.state.session
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != request.app.state.settings.base_url:
            raise HTTPException(403, "Untrusted request origin")
        csrf = request.headers.get("x-csrf-token", "")
        if not session or not secrets.compare_digest(csrf, session.csrf_token):
            raise HTTPException(403, "CSRF token is missing or invalid; refresh the page")
    return user


class Throttle:
    def __init__(self):
        self.events = defaultdict(deque)

    def check(self, key, limit, seconds=60):
        now = time.monotonic()
        # Bounded key count; a single worker is the documented deployment.
        if len(self.events) > 10000:
            self.events = defaultdict(deque, {k: v for k, v in self.events.items() if v and v[-1] > now - seconds})
        events = self.events[key]
        while events and events[0] <= now - seconds:
            events.popleft()
        if len(events) >= limit:
            raise HTTPException(429, "Too many requests; try again shortly", headers={"Retry-After": str(seconds)})
        events.append(now)
