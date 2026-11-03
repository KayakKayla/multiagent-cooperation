# Evaluation archive

A public, lightweight archive of locally produced SE 390 evaluations. Approved
GitHub teammates can review/import JSON, annotate evidence, and permanently delete
records. The server never loads policy weights, JAX, or JaxMARL. Synthetic demo
data is labelled as synthetic; it is not research evidence.

## Launch an existing local setup

From `evals-sharing-website/`, use the same data directory as during setup. No
installation or migration is needed just to restart an unchanged local version.

Windows PowerShell:

```powershell
$env:EVALS_DATA_DIR = "$PWD/.data"
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Linux/macOS:

```bash
export EVALS_DATA_DIR="$PWD/.data"
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open [the local archive](http://127.0.0.1:8000). Stop it with Ctrl+C. If your setup
uses a different data directory or OAuth configuration, retain those settings.

## Local setup

Use Python 3.12. From this directory, on Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
export EVALS_DATA_DIR="$PWD/.data"
.venv/bin/python -m app.cli migrate
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock
.venv/Scripts/python -m pip install --no-deps --no-build-isolation -e .
$env:EVALS_DATA_DIR = "$PWD/.data"
.venv/Scripts/python -m app.cli migrate
```

Then use the launch commands above. Public reads work without OAuth
credentials. The development directory `.data` is ignored; production state must
live outside the checkout. Install editable from the source checkout so templates,
static files, and migrations remain alongside `app/`; standalone wheel deployment
is not supported. Keep `--no-access-log` when starting Uvicorn: its default request
log includes OAuth callback query parameters containing authorization code/state.

To seed a public synthetic preview without granting browser access:

```bash
.venv/bin/python -m app.cli import-local tests/fixtures/evaluation-v1.json \
  --github-id 123 --login synthetic-demo
```

The offline maintenance command requires server filesystem access. It creates no
login session. There is no development login route, password, or production
authentication bypass. Tests create private database sessions in isolated temp
directories; browser write QA requires an approved GitHub sign-in.

## Sign-in configuration

Register a GitHub OAuth App with callback `<EVALS_BASE_URL>/auth/callback`.
Configure `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, and a comma-separated
`GITHUB_ALLOWED_IDS` containing the five **numeric stable account IDs**.
Set `EVALS_BASE_URL` to the exact HTTPS origin in production (HTTP is accepted
only on loopback). Username changes do not grant/revoke access. An account must
also be locally active. Restart after changing configuration.

Authlib manages OAuth state and S256 PKCE. Scope `read:user` accesses identity,
not repositories. OAuth state and verifier live in SQLite behind an opaque
HttpOnly/SameSite session cookie; production cookies also have Secure. The provider
token is discarded after identity lookup. Successful sign-in rotates the cookie.
Every mutation rechecks the ID allowlist and active account, validates its CSRF
header and supplied Origin, and uses local session expiry/revocation.

## Data and API

See [architecture](docs/architecture.md), [the format](docs/format-and-export.md), and
[operations](docs/operations.md). The [implementation plan](docs/implementation-plan.md)
records scope and research limitations.

- Public pages: `/`, `/evaluations/<id>`, `/training-runs/<id>`.
- Metadata lists: `GET /api/evaluations`, `GET /api/training-runs`, with `q`,
  `layout`, `algorithm`, and `page` filters.
- Selected recording: `GET /api/evaluations/<id>/episodes/<index>`.
- Upload: authenticated raw `application/json` body to `POST /api/uploads`, then
  review metadata/associations and `POST /api/imports` using its `preview_id`.
- Annotation edits: `PATCH /api/evaluations/<id>` (`version,title,notes`) or
  `PATCH /api/training-runs/<id>` (`version,name,notes`). Measurements are immutable.
- Permanent deletion: `DELETE` those resources or `/api/checkpoints/<id>`.
  Referenced runs/checkpoints return 409 and reference links.
- Authentication: `GET /auth/login`, `GET /auth/callback`, `POST /auth/logout`.

Cookie-authenticated mutations require `X-CSRF-Token` from the rendered page.
Errors use `{detail: message}` or `{detail: {message,references}}`; 401/403 deny
access, 409 identifies stale/conflicting evidence, 413 rejects oversized files,
422 rejects malformed data, 429 throttles, 507 reports storage reserve/quota.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

Backend tests exercise actual streamed 20 MiB limits, rollback/concurrent imports,
identity aliases, reference protection, selected-only compression, OAuth state and
PKCE, allowlisting, CSRF/logout/expiry, migrations, and consistent backup/restore.
OAuth provider identity/error tests use mocks; real GitHub registration and VPS
deployment remain operational checks. No training is part of website verification.
