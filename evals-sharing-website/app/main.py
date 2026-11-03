from pathlib import Path
import json
import time
import httpx
from authlib.integrations.starlette_client import OAuth
from authlib.integrations.base_client.errors import OAuthError
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, delete, update, text, func
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import undefer
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette._utils import get_route_path
from evals_shared.format import parse_document, canonical_json, ValidationError, MAX_UPLOAD_BYTES
from evals_shared.replay import build_web_payload
from .config import Settings
from .db import database
from .models import Base, User, TrainingRun, Policy, Evaluation, Participant, Episode, LoginSession, UploadPreview, now
from .auth import Sessions, Throttle, require_user, new_session, set_cookie, token_hash, COOKIE
from .imports import cleanup, check_capacity, preview_metadata, import_document, annotation
from .storage import unpack_recording
from .views import lists, evaluation_dict, run_dict, participants, episode_dict, checkpoint_dict

ROOT = Path(__file__).resolve().parent.parent


async def bounded_body(request, limit):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise HTTPException(413, f"File exceeds the {limit // 1024 // 1024} MiB limit")
        body.extend(chunk)
    return bytes(body)


async def body_json(request, limit=65536):
    try:
        value = json.loads(await bounded_body(request, limit))
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPException(422, "Expected a JSON object")


def create_app(settings=None):
    settings = settings or Settings()
    settings.prepare()
    app = FastAPI(title="SE 390 evaluation archive", docs_url=None, redoc_url=None, root_path=settings.root_path)
    app.state.settings = settings
    app.state.engine, app.state.db = database(settings)
    app.state.throttle = Throttle()
    app.add_middleware(Sessions)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[settings.host])
    templates = Jinja2Templates(directory=str(ROOT / "templates"))
    app.mount("/static", StaticFiles(directory=ROOT / "static", check_dir=False), name="static")
    oauth = OAuth()
    oauth.register("github", client_id=settings.github_client_id, client_secret=settings.github_client_secret, access_token_url="https://github.com/login/oauth/access_token", authorize_url="https://github.com/login/oauth/authorize", api_base_url="https://api.github.com/", client_kwargs={"scope": "read:user", "code_challenge_method": "S256", "timeout": 15})
    app.state.oauth = oauth

    def render(request, filename, context=None, status_code=200):
        session = request.state.session
        return templates.TemplateResponse(request=request, name=filename, context={"user": request.state.user, "csrf_token": session.csrf_token if session else "", "base_path": settings.root_path, "root_path": settings.root_path, **(context or {})}, status_code=status_code)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        if get_route_path(request.scope).startswith("/api/") or request.method != "GET":
            return JSONResponse({"detail": error.detail}, status_code=error.status_code, headers=error.headers)
        return render(request, "error.html", {"status_code": error.status_code, "message": error.detail}, error.status_code)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request, error):
        return JSONResponse({"detail": "Invalid request parameters"}, status_code=422)

    @app.exception_handler(OperationalError)
    async def database_busy(request, error):
        return JSONResponse({"detail": "Database temporarily unavailable; retry shortly"}, status_code=503)

    def get_row(db, model, id):
        row = db.get(model, id)
        if row is None:
            raise HTTPException(404, "Record not found")
        return row

    @app.get("/")
    def home(request: Request, q: str = "", layout: str = "", algorithm: str = "", tab: str = "evaluations", page: int = 1):
        with app.state.db() as db:
            return render(request, "home.html", lists(db, q[:300], layout[:300], algorithm[:300], page, "training-runs" if tab == "training-runs" else "evaluations"))

    @app.get("/api/evaluations")
    def evaluation_list(q: str = "", layout: str = "", algorithm: str = "", page: int = 1):
        with app.state.db() as db:
            return lists(db, q[:300], layout[:300], algorithm[:300], page)

    @app.get("/api/training-runs")
    def training_list(q: str = "", layout: str = "", algorithm: str = "", page: int = 1):
        with app.state.db() as db:
            return lists(db, q[:300], layout[:300], algorithm[:300], page, "training-runs")

    @app.get("/evaluations/{id}")
    def evaluation_page(request: Request, id: str):
        with app.state.db() as db:
            ev = evaluation_dict(get_row(db, Evaluation, id))
            ev["participants"] = participants(db, id)
            ev["episodes"] = [episode_dict(e) for e in db.scalars(select(Episode).where(Episode.evaluation_id == id).order_by(Episode.index))]
            row = db.get(Evaluation, id)
            for field, user_id in (("uploader", row.created_by), ("editor", row.updated_by)):
                person = db.get(User, user_id)
                ev[field] = {"login": person.login, "display_name": person.display_name} if person else None
            return render(request, "evaluation.html", {"evaluation": ev})

    @app.get("/training-runs/{id}")
    def run_page(request: Request, id: str):
        with app.state.db() as db:
            run = get_row(db, TrainingRun, id)
            checkpoints = [checkpoint_dict(p, run) for p in db.scalars(select(Policy).where(Policy.run_id == id).order_by(Policy.progress))]
            evaluations = [evaluation_dict(e) for e in db.scalars(select(Evaluation).where(Evaluation.id.in_(select(Participant.evaluation_id).join(Policy, Participant.policy_id == Policy.id).where(Policy.run_id == id))).order_by(Evaluation.created_at.desc()))]
            detail = run_dict(run)
            for field, user_id in (("uploader", run.created_by), ("editor", run.updated_by)):
                person = db.get(User, user_id)
                detail[field] = {"login": person.login, "display_name": person.display_name} if person else None
            return render(request, "training_run.html", {"run": detail, "checkpoints": checkpoints, "evaluations": evaluations})

    @app.get("/api/evaluations/{id}/episodes/{index}")
    def episode_data(id: str, index: int):
        with app.state.db() as db:
            ev = get_row(db, Evaluation, id)
            episode = db.scalar(select(Episode).where(Episode.evaluation_id == id, Episode.index == index).options(undefer(Episode.recording)))
            if episode is None:
                raise HTTPException(404, "Episode not found")
            people = participants(db, id)
            try:
                recording = unpack_recording(episode)
            except (ValueError, OSError):
                raise HTTPException(500, "Recording integrity check failed")
            payload = build_web_payload(recording, people) if recording is not None else {"recording": None}
            return {"episode": episode_dict(episode), "participants": people, "environment": ev.metadata_json["environment"], **payload}

    @app.get("/upload")
    def upload_page(request: Request):
        require_user(request)
        return render(request, "upload.html")

    @app.post("/api/uploads")
    async def preview(request: Request):
        user = require_user(request)
        app.state.throttle.check(("upload", user.id), 10)
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise HTTPException(415, "Upload evaluation.json as application/json")
        raw = await bounded_body(request, MAX_UPLOAD_BYTES)
        try:
            doc = parse_document(raw)
        except (ValidationError, ValueError, RecursionError) as error:
            raise HTTPException(422, str(error))
        with app.state.db() as db:
            db.execute(text("BEGIN IMMEDIATE"))
            cleanup(db)
            if db.scalar(select(func.count()).select_from(UploadPreview)) >= settings.max_previews:
                raise HTTPException(429, "Too many pending uploads; import a preview or wait for expiry")
            check_capacity(db, settings, len(raw) * 2)
            metadata = preview_metadata(db, doc)
            row = UploadPreview(user_id=user.id, expires_at=int(time.time()) + settings.preview_seconds, size=len(raw), document=raw)
            db.add(row)
            db.flush()
            result = {"id": row.id, "expires_at": row.expires_at, **metadata}
            db.commit()
            return result

    @app.post("/api/imports")
    async def commit_import(request: Request):
        user = require_user(request)
        choices = await body_json(request)
        if not isinstance(choices.get("preview_id"), str):
            raise HTTPException(422, "preview_id is required")
        for key in {"run_associations", "checkpoint_associations", "run_annotations"}:
            if key in choices and not isinstance(choices[key], dict):
                raise HTTPException(422, f"{key} must be an object")
        for key in {"run_associations", "checkpoint_associations"}:
            if any(not isinstance(v, str) for v in choices.get(key, {}).values()):
                raise HTTPException(422, f"{key} choices must be record identifiers")
        if any(not isinstance(v, dict) or set(v) - {"name", "notes"} for v in choices.get("run_annotations", {}).values()):
            raise HTTPException(422, "Run annotations must contain name/notes objects only")
        try:
            with app.state.db() as db:
                db.execute(text("BEGIN IMMEDIATE"))
                row = db.scalar(select(UploadPreview).where(UploadPreview.id == choices["preview_id"]).options(undefer(UploadPreview.document)))
                if row is None or row.user_id != user.id or row.expires_at <= time.time():
                    raise HTTPException(404, "Upload preview missing or expired; upload the file again")
                check_capacity(db, settings, row.size * 2)
                doc = parse_document(row.document)
                result, duplicate = import_document(db, doc, user.id, choices)
                db.delete(row)
                db.commit()
                return {"id": result.id, "url": settings.app_path(f"/evaluations/{result.id}"), "duplicate": duplicate}
        except IntegrityError:
            raise HTTPException(409, "Import conflicts with an existing record; review the upload again")

    async def patch_record(request, model, id, name):
        user = require_user(request)
        data = await body_json(request)
        if set(data) - {"version", name, "notes"} or type(data.get("version")) is not int:
            raise HTTPException(422, "Supply the current version and editable annotations only")
        values = {k: annotation(v, k, maximum=300 if k == name else 20000) for k, v in data.items() if k != "version"}
        with app.state.db() as db:
            get_row(db, model, id)
            result = db.execute(update(model).where(model.id == id, model.version == data["version"]).values(**values, version=data["version"] + 1, updated_by=user.id, updated_at=now()))
            if result.rowcount != 1:
                raise HTTPException(409, "This record changed; refresh before editing again")
            db.commit()
            row = db.get(model, id)
            return evaluation_dict(row) if model == Evaluation else run_dict(row)

    @app.patch("/api/evaluations/{id}")
    async def edit_evaluation(request: Request, id: str):
        return await patch_record(request, Evaluation, id, "title")

    @app.patch("/api/training-runs/{id}")
    async def edit_run(request: Request, id: str):
        return await patch_record(request, TrainingRun, id, "name")

    async def delete_record(request, model, id):
        require_user(request)
        with app.state.db() as db:
            db.execute(text("BEGIN IMMEDIATE"))
            row = get_row(db, model, id)
            if model == TrainingRun and db.scalar(select(Policy.id).where(Policy.run_id == id).limit(1)):
                policies = db.scalars(select(Policy).where(Policy.run_id == id)).all()
                raise HTTPException(409, {"message": "This training run still has checkpoints; remove unreferenced checkpoints first", "references": [{"id": p.id, "url": settings.app_path(f"/training-runs/{id}#checkpoints"), "name": f"Checkpoint {p.sha256[:12]} · {p.progress if p.progress is not None else 'unknown'} {row.progress_unit or 'units'}"} for p in policies]})
            if model == Policy and db.scalar(select(Participant.id).where(Participant.policy_id == id).limit(1)):
                refs = db.scalars(select(Participant.evaluation_id).where(Participant.policy_id == id).distinct()).all()
                raise HTTPException(409, {"message": "This checkpoint is referenced by evaluations", "references": [{"id": ref, "url": settings.app_path(f"/evaluations/{ref}")} for ref in refs]})
            db.delete(row)
            db.commit()
            return {"ok": True}

    @app.delete("/api/evaluations/{id}")
    async def remove_evaluation(request: Request, id: str):
        return await delete_record(request, Evaluation, id)

    @app.delete("/api/training-runs/{id}")
    async def remove_run(request: Request, id: str):
        return await delete_record(request, TrainingRun, id)

    @app.delete("/api/checkpoints/{id}")
    async def remove_checkpoint(request: Request, id: str):
        return await delete_record(request, Policy, id)

    @app.get("/auth/login")
    async def login(request: Request):
        app.state.throttle.check(("login", request.client.host), 10)
        if not settings.github_client_id or not settings.github_client_secret:
            raise HTTPException(503, "GitHub sign-in is not configured on this local instance")
        with app.state.db() as db:
            cleanup(db)
            token, session = new_session(db, settings)
        request.state.session = session
        request.scope["session"] = {}
        response = await oauth.github.authorize_redirect(request, settings.base_url + "/auth/callback")
        set_cookie(response, token, settings)
        return response

    @app.get("/auth/callback")
    async def callback(request: Request):
        app.state.throttle.check(("callback", request.client.host), 20)
        if not request.state.session:
            raise HTTPException(400, "Login session expired or missing; sign in again")
        try:
            token = await oauth.github.authorize_access_token(request)
            response = await oauth.github.get("user", token=token)
            response.raise_for_status()
            identity = response.json()
            github_id = identity["id"]
        except (OAuthError, ValueError, KeyError):
            raise HTTPException(400, "GitHub sign-in state or response was invalid; sign in again")
        except httpx.HTTPError:
            raise HTTPException(502, "GitHub sign-in is temporarily unavailable; try again")
        finally:
            # Never retain an access token after the identity lookup.
            request.scope["session"] = {}
        if type(github_id) is not int or github_id not in settings.allowed_github_ids:
            raise HTTPException(403, "This GitHub account is not approved for editing")
        with app.state.db() as db:
            user = db.scalar(select(User).where(User.github_id == github_id))
            if user and not user.active:
                raise HTTPException(403, "This account has been disabled")
            if user is None:
                user = User(github_id=github_id, login=identity["login"], display_name=identity.get("name") or identity["login"])
                db.add(user)
                db.flush()
            else:
                user.login = identity["login"]
                user.display_name = identity.get("name") or identity["login"]
                user.updated_at = now()
            db.execute(delete(LoginSession).where(LoginSession.token_hash == request.state.session.token_hash))
            token, session = new_session(db, settings, user.id)
        request.state.session = session
        response = RedirectResponse(settings.app_path("/"), status_code=303)
        set_cookie(response, token, settings)
        return response

    @app.post("/auth/logout")
    def logout(request: Request):
        require_user(request)
        with app.state.db() as db:
            db.execute(delete(LoginSession).where(LoginSession.token_hash == request.state.session.token_hash))
            db.commit()
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path=settings.cookie_path)
        return response

    return app


app = create_app()
