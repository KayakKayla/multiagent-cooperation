# Preparatory deployment and operations

These files prepare a deployment; no VPS deployment has occurred. Still needed:
VPS OS/RAM/disk capacity, domain and existing proxy configuration, the five numeric
GitHub IDs/OAuth App credentials, and off-VPS backup destination. Obtain deployment
authorization before changing the VPS. Local tests exercise migrations, consistent
backup and restore; real HTTPS/GitHub sign-in, Linux service/proxy setup, partial
clone support and off-VPS restoration remain deployment acceptance checks.

## Minimal checkout and installation

Use Python 3.12, a supported SQLite build, Git, systemd and an HTTPS reverse proxy
on the target. Reuse Apache when it already serves the VPS; the optional Caddy
example is for hosts without an existing proxy. Do not replace a working proxy.
Choose an install path readable by the unprivileged `evals-site` service account.
Choose the deployment branch explicitly and obtain its reviewed full commit SHA
before cloning. The current implementation branch is `rowan/evals-sharing-website`;
the remote's default branch is not a substitute. Replace the repository URL and
`<reviewed-full-commit-sha>` below with the intended deployment inputs.

```bash
DEPLOY_BRANCH='rowan/evals-sharing-website'
EXPECTED_COMMIT='<reviewed-full-commit-sha>'
git clone --filter=blob:none --no-checkout --depth=1 \
  --branch "$DEPLOY_BRANCH" --single-branch --no-tags --no-recurse-submodules \
  <repo-url> results-site
cd results-site
git rev-parse HEAD
if [ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]; then
  echo "Deployment stopped: checkout does not match the reviewed commit." >&2
  exit 1
fi
git config core.sparseCheckout true
git config core.sparseCheckoutCone true
mkdir -p .git/info
printf '%s\n' '/*' '!/*/' '/evals-sharing-website/' > .git/info/sparse-checkout
git read-tree -mu HEAD
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

Stop before installation if the commit comparison fails. Resolve the branch/SHA
difference and deploy only the reviewed revision; do not silently accept a newer
branch tip. Record that full SHA with the deployment's smoke-check results.

The direct cone configuration works with Git 2.25.1 on Ubuntu 20.04. That version's
`clone --sparse` can mistake the source URL for the sparse initialization directory;
its `sparse-checkout init` also requires a clean worktree, which a fresh
`--no-checkout` clone does not provide. Set the patterns before `read-tree` so only
the selected content is checked out, after the reviewed commit is verified.

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

Adapt/install `deploy/evals-site.service`, cleanup service/timer, and the selected
proxy configuration. Bind Uvicorn only to loopback; the proxy serves public HTTPS.
Trust forwarded
headers from loopback only. The service deliberately runs one worker: the app's
throttle is process-local and SQLite serializes writers. Select systemd MemoryMax,
CPUQuota and any concurrency/rate limits from measured target capacity; do not
copy a high-worker template. Validate proxy configuration before reloading it.
Use journald retention/rotation appropriate to disk capacity. The service disables
Uvicorn access logging because the default log includes OAuth code/state query
parameters. Use `app.cli serve` for safe prefix/logging defaults; do not enable
query/response-body logging on the proxy or log cookie/token values.

Smoke checks: public home/filter/detail/selected-episode requests, correct static
MIME types, real GitHub callback/state/PKCE and allowlist rejection, CSRF rejection,
approved upload/review/import/edit/delete and failed-delete messages, 20 MiB/over
limit uploads, Secure/HttpOnly/SameSite cookie, HTTPS redirects, and cleanup timer.
Confirm the service account cannot write code or read unrelated host files.

## Existing Apache HTTPS host: `/sims`

The selected public URL is `https://rowan.gaertner.ca/sims/`, reusing the existing
HTTPS hostname and certificate. Set `EVALS_BASE_URL=https://rowan.gaertner.ca/sims`
and GitHub's callback to `https://rowan.gaertner.ca/sims/auth/callback`. No DNS
record, new virtual host or replacement proxy is needed.

Inspect `sudo apache2ctl -S`, enabled modules and the existing HTTPS site file
first. `deploy/apache-evals-site.conf.example` is now an **additive snippet inside
that existing `*:443` virtual host**, not a standalone site. Preserve its root
application, TLS directives, certificate renewal, enabled sites and default-host
ordering. Check any existing ProxyPass/RewriteRule rules: the `/sims/` mapping must
precede a broader mapping that would otherwise consume it. Back up the original
site file before editing, and include the snippet only after its contents and
logging exclusions have been adapted to the actual file.

Required modules are `proxy`, `proxy_http`, `headers`, `setenvif`, `alias`,
`log_config`, plus the site's already-active `ssl`. Verify with `sudo apache2ctl
-M`; Ubuntu's `sudo a2enmod proxy proxy_http headers setenvif alias` can enable
missing proxy modules without disabling existing sites.

The `/sims` redirect adds the trailing slash. The ProxyPass mappings have trailing
slashes on both sides, so `/sims/api/evaluations` becomes `/api/evaluations` on
the dedicated loopback nginx cap proxy at port 8001, which forwards unchanged to
Uvicorn at port 8000. The `serve` CLI derives Uvicorn's ASGI root_path from `EVALS_BASE_URL`,
so generated page/API/OAuth/static URLs and cookie paths retain `/sims`. The proxy
preserves Host, removes client forwarding assertions and sets HTTPS forwarding
only for this location. Uvicorn trusts loopback only. A root deployment still uses
a base URL without a prefix and a root proxy mapping instead.

The application rejects uploads over exactly 20 MiB by counting actual streamed
JSON bytes, independently of Content-Length. The snippet also sets
`LimitRequestBody 20971520`, but **upstream Apache 2.4.41 bypasses this limit for
reverse-proxy requests**. Do not claim an independent proxy-layer cap without
verifying the target's patched behavior. The dedicated nginx configuration below
provides that independent boundary without replacing Apache. The application's
cap remains required too. Test exact-limit and one-extra-byte requests through
the real proxy, including chunked/unknown-length requests.

The snippet identifies `/sims` requests with `sims_request` and logs only `%U`
(path without query), method/status and basic metadata. **Every existing log that
would otherwise record those requests needs an exclusion**: for example change
`CustomLog ${APACHE_LOG_DIR}/access.log combined` to
`CustomLog ${APACHE_LOG_DIR}/access.log combined env=!sims_request`. Keep ordinary
root-site requests in their previous logs. Do not duplicate or override an
existing condition blindly; combine conditions after inspecting the actual config.
`TransferLog` does **not** accept `env=` or an explicit format argument. If it
would record `/sims`, identify its effective preceding `LogFormat` first, then
convert that directive to `CustomLog` using the same destination and format plus
the exclusion. Confirm ordinary root-site entries retain their previous fields;
do not blindly append a condition to a TransferLog line.
Also inspect server-wide `GlobalLog`, `TransferLog`, `ErrorLogFormat request`,
security-module audit logs and trace settings: adding a new CustomLog alone does
not stop duplicate credential-bearing logs. The supported error format retains
Apache's required `%M` field and omits Referer interpolation; module messages can
still contain URLs. Probe with dummy callback values and confirm no active logger
records query credentials before real sign-in. Do not enable raw request dumps or
proxy trace logging for OAuth troubleshooting.

Run `sudo apache2ctl configtest` and `sudo apache2ctl -S` after the edit. Stop on
any error or unintended routing change. Only after successful validation, run
`sudo systemctl reload apache2`; do not enable another site or stop Apache.
Check the existing root application and other URLs still work, then test
`/sims` → `/sims/`, public pages/static/replay data, GitHub callback, CSRF, cookie
scope, upload limits and logging. These remain target-side acceptance checks until
actually performed. The optional Caddy example is only for a separate new host;
do not start it alongside this VPS's Apache ports.

## Loopback nginx body-limit proxy

Install `deploy/nginx-evals-site.conf` as the root-owned
`/etc/nginx/evals-site.conf` and `deploy/evals-site-proxy.service` as its own systemd
unit. Leave the distribution `nginx.service` inactive. This standalone config has
no includes and listens **only on 127.0.0.1:8001**; it cannot load package sites on
public port 80. Apache remains the TLS server, routing only `/sims/` to this proxy.
The normal nginx root master starts a single `www-data` worker; this unit does
not apply an untested filesystem sandbox to nginx's package temporary paths.

`client_max_body_size 20971520` applies to Content-Length and parsed chunked
bodies. `proxy_request_buffering off` together with `proxy_http_version 1.1`
avoids nginx's chunked buffering fallback; responses are also unbuffered. Thus
large unauthenticated bodies are not intentionally spooled to disk. The default
package temp directories may still be initialized at startup; validate them and
the installed nginx version/config with `sudo /usr/sbin/nginx -t -c
/etc/nginx/evals-site.conf`. Do not use the newer `nginx -e` option on nginx 1.18.

The proxy preserves the public Host and Apache's sanitized client forwarding
address, sets HTTPS for the application, and clears alternative forwarding
assertions. It does not add `/sims` again. Nginx access logging is disabled, and
its request error log is directed to `/dev/null` because those messages can carry
OAuth query strings; a marked plain-text 413 response identifies its size limit.
Diagnose syntax/startup with `nginx -t`/systemd status and application failures with
the service journal and safe Apache access log. Do not turn on request logging
for live OAuth troubleshooting.

Validate the service with `systemd-analyze verify`, start only the dedicated
`evals-site-proxy.service`, and check listeners before adding the public Apache
mapping. Confirm port 8001 is loopback only and the existing Apache 80/443
listeners remain unchanged. Test the boundary first with a temporary consuming
ASGI upstream on loopback port 8000, while the real app is stopped and `/sims/`
is not publicly mapped. In one SSH session from the website directory:

```bash
.venv/bin/python - <<'PY'
import uvicorn

async def app(scope, receive, send):
    total = 0
    while True:
        message = await receive()
        if message['type'] == 'http.disconnect':
            return
        total += len(message.get('body', b''))
        if not message.get('more_body', False):
            break
    body = f'received={total}\n'.encode()
    await send({'type': 'http.response.start', 'status': 200,
                'headers': [(b'content-type', b'text/plain')]})
    await send({'type': 'http.response.body', 'body': body})

uvicorn.run(app, host='127.0.0.1', port=8000, lifespan='off',
            access_log=False, log_level='warning')
PY
```

In a second session, send actual bytes with both encodings:

```bash
(
set -eu
BODY=$(mktemp /tmp/evals-cap.XXXXXX)
trap 'rm -f "$BODY"' EXIT
for SIZE in 20971520 20971521; do
  truncate -s "$SIZE" "$BODY"
  printf '\nContent-Length body: %s bytes\n' "$SIZE"
  curl --http1.1 --max-time 60 -sS -D - -H 'Expect:' \
    --data-binary @"$BODY" http://127.0.0.1:8001/cap-test
  printf '\nChunked body: %s bytes\n' "$SIZE"
  curl --http1.1 --max-time 60 -sS -D - -H 'Expect:' \
    -H 'Transfer-Encoding: chunked' -H 'Content-Length:' \
    --data-binary @"$BODY" http://127.0.0.1:8001/cap-test
done
)
```

Both exact-limit requests must return 200 with `received=20971520`. Both
one-extra-byte requests must return 413 with `X-Evals-Body-Limit: 20971520`.
Stop the temporary upstream with Ctrl+C, start the real app, then add/test Apache's
public mapping. An anonymous real-app request may return 401 before consuming a
chunked body; that alone does not test the proxy's chunked cap. Record all four
private boundary results and repeat public upload checks with authentication
before calling deployment complete. The probe must never be exposed publicly.

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
