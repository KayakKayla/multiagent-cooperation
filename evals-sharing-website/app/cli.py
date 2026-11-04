"""Local maintenance; no web-accessible administration or authentication bypass."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.orm import undefer
from evals_shared.format import parse_document
from .config import Settings
from .db import database
from .models import User, Evaluation, Episode, LoginSession, now
from .imports import import_document, cleanup, check_capacity
from .storage import unpack_recording

ROOT = Path(__file__).resolve().parent.parent


def migrate(settings):
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["settings"] = settings
    command.upgrade(config, "head")


def backup(source, destination):
    """SQLite online backup includes committed journal/WAL state consistently."""
    destination = Path(destination).resolve()
    if not Path(source).is_file():
        raise ValueError("Source database does not exist")
    if destination.exists():
        raise ValueError("Backup destination already exists; choose a new file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Backup integrity check failed")
    if os.name != "nt":
        destination.chmod(0o600)


def restore(source, destination):
    """Restore evidence without resurrecting sessions, OAuth state or previews.

    Destination must not exist; stop service first. Never overwrite live DB.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file() or destination.exists():
        raise ValueError("Restore needs an existing backup and a NEW destination")
    with sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True) as src:
        if src.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Source backup failed integrity_check")
        with sqlite3.connect(destination) as dst:
            src.backup(dst)
            # Backups can predate logout/revocation. Restoring credentials would
            # undo those decisions, so every restored instance starts signed out.
            dst.execute("DELETE FROM sessions")
            dst.execute("DELETE FROM upload_previews")
            if dst.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Restored backup failed foreign_key_check")
    if os.name != "nt":
        destination.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate")
    sub.add_parser("cleanup")
    p = sub.add_parser("serve", help="Serve loopback HTTP with the configured public path prefix")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("backup"); p.add_argument("destination")
    p = sub.add_parser("restore"); p.add_argument("source"); p.add_argument("destination")
    p = sub.add_parser("inspect-episode"); p.add_argument("evaluation_id"); p.add_argument("index", type=int)
    p = sub.add_parser("set-user-active"); p.add_argument("github_id", type=int); p.add_argument("active", choices=["yes", "no"])
    p = sub.add_parser("import-local", help="Offline admin import; does not create a login session")
    p.add_argument("file"); p.add_argument("--github-id", type=int, required=True); p.add_argument("--login", required=True)
    args = parser.parse_args()
    settings = Settings(); settings.prepare()
    if args.command == "serve":
        import uvicorn
        # The ASGI server must supply root_path, not just the framework: mounted
        # static routes need the external prefix in scope.path as Uvicorn builds it.
        uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, workers=1,
                    root_path=settings.root_path, proxy_headers=True,
                    forwarded_allow_ips="127.0.0.1", access_log=False,
                    timeout_keep_alive=5)
        return
    if args.command == "migrate":
        migrate(settings); return
    if args.command == "backup":
        backup(settings.database_path, args.destination); return
    if args.command == "restore":
        restore(args.source, args.destination); return
    engine, db_factory = database(settings)
    with db_factory() as db:
        if args.command == "cleanup":
            cleanup(db); db.commit()
        elif args.command == "set-user-active":
            user = db.scalar(select(User).where(User.github_id == args.github_id))
            if not user:
                raise ValueError("Unknown account")
            user.active = args.active == "yes"; user.updated_at = now()
            if not user.active:
                db.query(LoginSession).filter_by(user_id=user.id).delete()
            db.commit()
        elif args.command == "inspect-episode":
            episode = db.scalar(select(Episode).where(Episode.evaluation_id == args.evaluation_id, Episode.index == args.index).options(undefer(Episode.recording)))
            if not episode:
                raise ValueError("Unknown episode")
            print(json.dumps(unpack_recording(episode), ensure_ascii=False, indent=2))
        elif args.command == "import-local":
            # This administrative CLI requires filesystem/server account access.
            # It cannot grant browser access or bypass the configured allowlist.
            doc = parse_document(Path(args.file).read_bytes())
            db.execute(text("BEGIN IMMEDIATE"))
            check_capacity(db, settings, Path(args.file).stat().st_size * 2)
            user = db.scalar(select(User).where(User.github_id == args.github_id))
            if not user:
                user = User(github_id=args.github_id, login=args.login, display_name=args.login)
                db.add(user); db.flush()
            result, duplicate = import_document(db, doc, user.id)
            db.commit()
            print(json.dumps({"id": result.id, "url": settings.app_path(f"/evaluations/{result.id}"), "duplicate": duplicate}))
    engine.dispose()


if __name__ == "__main__":
    main()
