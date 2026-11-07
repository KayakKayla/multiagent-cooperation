"""Publish a saved evaluation.json to Sims Archive using GitHub browser sign-in."""

import argparse
import hashlib
import json
import os
import secrets
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_ARCHIVE_URL = "https://rowan.gaertner.ca/sims"
MAX_UPLOAD_BYTES = 20 * 1024**2


class UploadError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Keep the upload credential and evaluation on the configured site.


def archive_request(opener, url, headers, body=None):
    request = Request(url, data=body, headers=headers)
    try:
        with opener.open(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        with error:
            try:
                detail = json.load(error).get("detail")
                if isinstance(detail, dict):
                    detail = detail.get("message")
            except (ValueError, AttributeError):
                detail = None
        raise UploadError(detail or f"Archive returned HTTP {error.code}.", error.code) from None


def authorize(opener, base_url):
    token = secrets.token_urlsafe(32)
    challenge = hashlib.sha256(token.encode()).hexdigest()
    headers = {"Cookie": f"evals_session={token}"}

    def check_session():
        try:
            return archive_request(opener, base_url + "/api/cli/session", headers)
        except UploadError as error:
            if error.status == 401:
                return None
            if error.status == 404:
                raise UploadError("This archive does not have terminal upload enabled yet.") from None
            raise

    check_session()
    approval_url = f"{base_url}/auth/cli?challenge={challenge}"
    print(f"Authorize Sims Archive upload in your browser. Terminal code: {challenge[:12]}", flush=True)
    print(f"Approval link: {approval_url}", flush=True)
    try:
        webbrowser.open(approval_url)
    except (OSError, webbrowser.Error):
        pass  # The link above also works from a headless machine or WSL.
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        session = check_session()
        if session is not None:
            return {**headers, "X-CSRF-Token": session["csrf_token"], "Content-Type": "application/json"}
        time.sleep(2)
    raise UploadError("Browser approval timed out.")


def associations(preview):
    runs = dict(preview["matches"]["training_runs"])
    checkpoints = dict(preview["matches"]["checkpoints"])
    existing = {cp["id"]: cp["run_id"] for cp in preview["existing_checkpoints"]}
    suggestions = preview["checkpoint_suggestions"]
    for run in preview["training_runs"]:
        source = run["source_run_id"]
        candidates = {}
        for cp in preview["checkpoints"]:
            if cp["source_run_id"] != source:
                continue
            key = cp["source_checkpoint_id"]
            if key in checkpoints and source in runs:
                continue  # Exact source matches win, even beyond the preview's catalog limit.
            ids = [checkpoints[key]] if key in checkpoints else suggestions.get(key, [])
            if ids:
                candidates[key] = ids
        if source not in runs and candidates:
            run_options = [{existing[id] for id in ids if id in existing} for ids in candidates.values()]
            common = set.intersection(*run_options)
            if len(common) != 1:
                raise UploadError(f"Training run '{run['name']}' has ambiguous matches. Review it in the browser.")
            runs[source] = common.pop()
        for key, ids in candidates.items():
            matches = [id for id in ids if existing.get(id) == runs[source]]
            if len(matches) != 1:
                raise UploadError(f"Training run '{run['name']}' has ambiguous checkpoint matches. Review it in the browser.")
            checkpoints[key] = matches[0]
    return runs, checkpoints


def prompt(label, maximum, default=""):
    suffix = f" [{default}]" if default else " [none]"
    while True:
        value = input(f"{label}{suffix}: ").strip() or default
        if len(value) <= maximum:
            return value
        print(f"Use at most {maximum} characters.")


def import_choices(preview):
    choices = {"preview_id": preview["id"]}
    if preview["duplicate_id"]:
        return choices  # The existing import path returns the record and consumes the preview.
    if preview["conflicts"]:
        raise UploadError("; ".join(preview["conflicts"]))
    runs, checkpoints = associations(preview)
    doc = preview["document"]
    title = prompt("Evaluation name", 300, doc.get("title") or doc["environment"]["layout"])
    notes = prompt("Evaluation notes", 20000)
    annotations = {}
    for run in preview["training_runs"]:
        source = run["source_run_id"]
        if source not in runs:
            print(f"New training run: {run['name']}")
            annotations[source] = {"name": prompt("Model name", 300, run["name"]),
                                   "notes": prompt("Model notes", 20000)}
    return dict(choices, title=title, notes=notes, run_associations=runs,
                checkpoint_associations=checkpoints, run_annotations=annotations)


def upload_evaluation(path, *, base_url=None):
    """Try to publish a saved export. Return the archive URL, or None on failure."""
    path = Path(path)
    base_url = (base_url or os.getenv("SIMS_ARCHIVE_URL") or DEFAULT_ARCHIVE_URL).rstrip("/")
    try:
        parsed = urlsplit(base_url)
        local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if (not parsed.hostname or parsed.username is not None or parsed.query or parsed.fragment
                or (parsed.scheme != "https" and not local_http)):
            raise UploadError("Use an HTTPS archive URL (HTTP is allowed on localhost).")
        with path.open("rb") as handle:
            raw = handle.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise UploadError("The export exceeds the archive's 20 MiB limit.")

        opener = build_opener(NoRedirect())
        headers = authorize(opener, base_url)
        preview = archive_request(opener, base_url + "/api/uploads", headers, raw)
        choices = json.dumps(import_choices(preview), ensure_ascii=False).encode("utf-8")
        result = archive_request(opener, base_url + "/api/imports", headers, choices)
        url = f"{parsed.scheme}://{parsed.netloc}{result['url']}"
        status = "Already published" if result["duplicate"] else "Published to Sims Archive"
        print(f"{status}: {url}", flush=True)
        return url
    except (EOFError, KeyboardInterrupt):
        message = "Upload cancelled."
    except (KeyError, TypeError, ValueError):
        message = "The archive URL or response was invalid."
    except (OSError, UploadError) as error:
        message = str(error)
    print(f"Sims Archive upload did not finish: {message} Local export: {path}", file=sys.stderr, flush=True)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path, help="Path to a saved evaluation.json.")
    args = parser.parse_args()
    return 0 if upload_evaluation(args.evaluation) else 1


if __name__ == "__main__":
    sys.exit(main())
