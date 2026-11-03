# Storage and trust boundaries

The server uses synchronous SQLAlchemy/SQLite with a single Uvicorn worker. Foreign
keys are enabled for every connection; a 15-second busy timeout permits short
write contention. DELETE journaling avoids persistent WAL growth for this small
archive. Reads fetch metadata independently of per-episode recordings. SQLAlchemy
maps each recording column as deferred with raiseload; only the selected-episode
route explicitly requests that blob. A storage helper checks gzip encoding,
uncompressed size, and SHA-256 before returning ordinary JSON.

`EVALS_BASE_URL` records the full public URL, optionally including a path such as
`/sims`. The proxy strips that prefix before forwarding. The `serve` launcher
passes it to Uvicorn's root_path so ASGI routing, mounted static files and generated
URLs retain the external prefix. The framework uses the same configured prefix.
The HTTP origin is derived separately for CSRF checks; session cookies are scoped
to the site's path (for example `/sims/`). Root deployments keep an empty prefix.

The research tables are users, training_runs, policies, evaluations,
evaluation_participants, and episodes. Internal tables hold server-side sessions,
expiring upload previews, and run/checkpoint source aliases. Alembic's frozen
initial migration creates constraints/indexes; schema changes require a new
migration. Application startup does not silently create or alter tables.

Run names/notes and evaluation titles/notes are editable annotations. Immutable
run evidence contains algorithm, seed, training configuration, progress unit,
dates/budget, device and provenance. Evaluation source digests cover the complete
canonical input document, including its original annotations. Changing a site's
annotations does not rewrite original evidence or its digest.

## Reviewed association and import

Uploads are JSON data only. The streamed application/json body is capped at
20 × 1024 × 1024 bytes, regardless of Content-Length. The shared stdlib format
validator handles finite numbers, depth/node/text/episode/participant limits,
references, state/action alignment, and metric consistency before preview storage.
No uploaded code, model, image sequence, external URL or local file is executed or
loaded. Text is escaped by Jinja and updated in the browser using textContent.

A preview is private to its uploader, expires after 30 minutes, and publishes no
research rows. At most 20 previews exist globally (at most 400 MiB source data).
An import starts BEGIN IMMEDIATE and commits new runs, exact checkpoints,
participants, episode metadata/recordings and the evaluation together. Concurrent
imports therefore serialize; unique source constraints provide a second defense.
The same source evaluation/digest reuses the existing evaluation. Changed contents
under that source identity return a conflict, with no partial research rows.

Stable source IDs auto-match established identities. A reviewer may explicitly
associate a new portable source-run ID with an existing run only when immutable
evidence agrees exactly. This creates an identity alias so later imports reuse the
decision. It never automatically merges independent runs just because their
configurations happen to match. To reuse a checkpoint under a new source ID, the
selected run, SHA-256, and progress must all match; checkpoint source aliases are
also persisted. Selecting a mismatched checkpoint cannot attach results to other
weights. A new checkpoint of an existing run is a new policy row. Unknown legacy
progress remains null. One checkpoint may occupy any number of participant slots.

Review suggestions require the exact checkpoint hash/progress and compatible
immutable training-run evidence. When a source run is already associated, the
candidate must belong to that run. Candidate labels show run name and checkpoint
progress/unit; unfamiliar source identities are never automatically merged.

Known source IDs cannot be redirected. Conflicting metadata requires correcting
the source/export or making an honestly distinct source run. Existing site run
annotations are never overwritten by imported source annotations. Review-time
name/notes edits apply only to newly created runs.

## Authorization and edits

All approved active teammates have equal permissions; public pages/recordings
remain readable anonymously. Every mutation verifies a server-side session, the
current configured stable-ID allowlist, active status, a CSRF header, and any
Origin. OAuth uses Authlib with S256 and state stored server-side. Tokens are
hashed at rest; only a random opaque session token is in the cookie. Provider
access tokens are used once for identity lookup and discarded.

Edits use an atomic `UPDATE ... WHERE version = submitted_version`, returning 409
on a stale version. Deleting an evaluation cascades only to its episode and
participant rows. Run/checkpoint deletion is restricted while referenced, with
409 reference links. There is no trash/restore route.

Page CSP permits only local scripts/styles/resources and blocks framing. Static
renderer/artwork is shared with offline viewers and cached for one day; uploaded
renderer metadata cannot select arbitrary scripts. Unsupported and missing
recordings remain visibly distinguished.

## Resource limits

The default archive directory budget is 2 GiB with a separate 512 MiB free-disk
reserve. Configure `EVALS_STORAGE_BYTES` and `EVALS_RESERVE_BYTES` for the actual
VPS. Checks account for current allocated SQLite/journal files and conservatively
reserve two source-file sizes for preview/import work. SQLite can reuse deleted
pages without reducing file size; exceeding allocation quota requires maintenance
or a deliberately increased budget, even after deletions.

Upload throttle is 10 attempts per user per minute; login is 10 attempts per
client IP per minute. It is process-local and intentionally paired with a single
worker deployment and reverse-proxy request limits. Public list results have
20-record pagination, review choices are bounded to 1,000 runs/5,000 checkpoints,
and import requests have a 64 KiB body cap. A source ID outside those choice lists
still resolves correctly on import. Sessions expire after 12 hours; periodic CLI
cleanup removes expired sessions/previews. Backups, logs and other files outside
the runtime directory require their own retention/free-space budget.
