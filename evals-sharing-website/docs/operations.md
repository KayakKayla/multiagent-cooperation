# Preparatory deployment and operations

These files prepare a deployment; no VPS deployment has occurred. Still needed:
VPS OS/RAM/disk capacity, domain and existing proxy configuration, the five numeric
GitHub IDs/OAuth App credentials, and off-VPS backup destination. Obtain deployment
authorization before changing the VPS. Local tests exercise migrations, consistent
backup and restore; real HTTPS/GitHub sign-in, Linux service/proxy setup, partial
clone support and off-VPS restoration remain deployment acceptance checks.

## Minimal checkout and installation

Use Python 3.12, a supported SQLite build, Git, systemd and Caddy on the target.
Choose an install path readable by the unprivileged `evals-site` service account.

```bash
git clone --filter=blob:none --sparse --depth=1 \
  --single-branch --no-tags --no-recurse-submodules <repo-url> results-site
cd results-site
git sparse-checkout set evals-sharing-website
git config submodule.recurse false
git config fetch.recurseSubmodules false
git config pull.ff only
git config --get remote.origin.promisor
git sparse-checkout list
test ! -d overcooked-v2
cd evals-sharing-website
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
```

The remote must actually support partial clone: inspect clone output, the
promisor/filter config, checkout contents and `.git/objects` size. Sparse checkout
is not an access boundary: root files and Git commit/tree metadata are present,
and explicit excluded-content/history queries can fetch blobs. Ordinary
fast-forward pulls preserve the sparse/filter configuration. Never recurse
submodules. The website needs no simulator checkout, ML dependencies, or weights.

Place actual runtime data in `/var/lib/evals-site`, owned by `evals-site`, mode
0700. Store environment credentials in `/etc/evals-site/site.env`, owned by root,
mode 0600, using `deploy/site.env.example` as the field reference. Do not put actual
secrets into the checkout. The code checkout must not be writable by the runtime
service account. Run the migration as that unprivileged account with its exact
environment before starting the service:

```bash
# With EVALS_DATA_DIR=/var/lib/evals-site in the administrative process environment:
.venv/bin/python -m app.cli migrate
```

Adapt/install `deploy/evals-site.service`, cleanup service/timer, and Caddy config.
Bind Uvicorn only to loopback; only Caddy serves public HTTPS. Trust forwarded
headers from loopback only. The service deliberately runs one worker: the app's
throttle is process-local and SQLite serializes writers. Select systemd MemoryMax,
CPUQuota and any concurrency/rate limits from measured target capacity; do not
copy a high-worker template. Validate Caddy configuration before reloading it.
Use journald retention/rotation appropriate to disk capacity. The service disables
Uvicorn access logging because the default log includes OAuth code/state query
parameters. Keep `--no-access-log` in local startup commands too; do not enable
query/response-body logging on the proxy or log cookie/token values.

Smoke checks: public home/filter/detail/selected-episode requests, correct static
MIME types, real GitHub callback/state/PKCE and allowlist rejection, CSRF rejection,
approved upload/review/import/edit/delete and failed-delete messages, 20 MiB/over
limit uploads, Secure/HttpOnly/SameSite cookie, HTTPS redirects, and cleanup timer.
Confirm the service account cannot write code or read unrelated host files.

## Updates and rollback

Before update, record the current Git revision and make a consistent backup to a
separate directory. Stop the app for migration. Pull with `git pull --ff-only`,
install the committed lock, reinstall editable, run `app.cli migrate`, restart,
and perform the smoke checks. Do not overwrite runtime files with a checkout.

If rollback is required, stop the service, select the recorded known-good code
revision and dependency lock, and restore its matching backup into a **new**
database path. Verify it before switching `EVALS_DATA_DIR`/starting the service.
Do not run an old executable against an incompatible newer schema. The initial
migration supports downgrade for development; operational rollback should use
the tested matching backup rather than a destructive downgrade on live evidence.

## Backups and tested restore

Use SQLite's online backup API, not a standalone copy of a live database:

```bash
.venv/bin/python -m app.cli backup /var/backups/evals-site/archive-YYYYMMDD.sqlite3
```

The destination must not exist. Restrict backups to the administrator (they
include private sessions and OAuth state), encrypt/protect any off-VPS copy,
transfer to the chosen off-VPS destination, and configure retention there. Do not
place large backup archives in the live quota directory. Check free space for
live data, temporary previews, journals, logs, backups, and reserve separately.

Exercise a restore without overwriting live storage:

```bash
.venv/bin/python -m app.cli restore /var/backups/evals-site/archive-YYYYMMDD.sqlite3 \
  /var/lib/evals-site-restored/evaluations.sqlite3
```

Create the destination directory first. The source undergoes integrity_check;
restore uses SQLite backup, deletes **all** restored sessions/OAuth state and
pending upload previews, and verifies foreign_key_check. This prevents a backup
from resurrecting access that was subsequently revoked; every user must sign in
again, and pending uploads must be uploaded again. The destination must not exist.
Point an isolated local server at that restored directory, verify record counts
and selected replay integrity, and record the result. For an actual switch, stop
the service first, reconcile account activation/allowlist changes made since the
backup, and start against the verified directory. Never restore over an open
database or copy only a WAL main file. `tests/test_backend.py` exercises a local
backup/restore after concurrent import and verifies records/foreign keys plus a
backup→revoke→restore check proving that old sessions/previews are removed.

## Storage and account maintenance

`app.cli cleanup` deletes expired previews/sessions. Run it hourly using the
provided timer. The default 2 GiB runtime quota and 512 MiB reserve are deployment
inputs, not a claim about available VPS space. `EVALS_STORAGE_BYTES` includes the
allocated database/journal and all regular files directly in the runtime
directory. Deleted SQLite pages can be reused while the file stays the same
size. Inspect allocated size/free disk before changing quotas. If shrinking is
necessary, stop the app, take a backup and use SQLite VACUUM only with enough
extra disk space; this is deliberate offline maintenance.

Disable an already-known account and revoke its sessions:

```bash
.venv/bin/python -m app.cli set-user-active <numeric-github-id> no
```

Reenable with `yes`; the configured allowlist must still contain the ID. Removing
an ID from configuration and restarting also revokes effective mutation access
on every request. There is no public account-management endpoint.
