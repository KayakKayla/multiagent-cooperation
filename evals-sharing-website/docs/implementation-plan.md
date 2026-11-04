# Evaluation sharing website — implementation plan

Agreed design from the planning conversation, 2026-11-02. This document is the implementation handoff. No website implementation or deployment has been performed as part of writing it.

## 1. Purpose and scope

Build a small website for the five SE 390 team members to share locally produced policy evaluations and inspect project progress. Anyone can view published research records. Approved teammates can upload, edit, and delete records, with the same permissions for all teammates.

Sharing is optional. Training and evaluation continue locally. Users upload selected evaluations, not a file for every command or training run. An evaluation upload also registers any training runs and checkpoints it needs. Standalone registration of a run or checkpoint before an evaluation is not required.

The website supplements the current evaluation and preview system. Existing commands, local HTML replays, GIFs, images, and result files remain usable. Nobody needs the website to inspect their own evaluation results.

The VPS has limited disk space, approximately less than 10 GB. Upload data and render recorded states; do not upload model weights or an image for every state.

## 2. Technology and repository organization

- Python 3.12, FastAPI, Uvicorn, Jinja2 templates, CSS, and vanilla JavaScript.
- No React or other heavy client-side framework. Use small JavaScript modules for replay, inline forms, fetch requests, dialogs, and toasts.
- SQLite for application data, including episode recordings.
- SQLAlchemy for database access and Alembic for migrations.
- Authlib for GitHub OAuth, with local users and server-side login sessions.
- Caddy for HTTPS/reverse proxying, with Uvicorn managed by systemd on the Linux VPS.
- Pin dependencies in the implementation. Exact compatible versions, package layout, and synchronous versus asynchronous database plumbing are implementation decisions.

Keep all website-specific source, dependencies, assets, schema helpers, tests, deployment material, and documentation under one repository parent directory:

```text
evals-sharing-website/
  app/
  shared/
  templates/
  static/
  migrations/
  tests/
  deploy/
  docs/
    implementation-plan.md
  pyproject.toml
```

This is a suggested internal arrangement, not a requirement to create every directory immediately. Do not introduce unrelated root-level `packages/` or `deploy/` folders.

Simulation/evaluator changes remain under `overcooked-v2/`. Put lightweight shared result-format and replay helpers where both the local tools and website can use them, inside the website parent directory. Keep their imports independent of FastAPI and JAX as appropriate: local evaluation must not need the web server stack, and the VPS must not need the simulation/training stack.

The website must install and run from a checkout containing only this parent directory and root-level repository files. Shared artwork and renderer code needed by the server must be available there. Existing local scripts can become wrappers/importers of the shared implementation; avoid maintaining two divergent copies of the renderer.

Read the repository `AGENTS.md` and `overcooked-v2/README.md` before implementation. Keep the pinned JaxMARL submodule unchanged. Follow local instructions about short verification checks and do not launch training or research experiments without user direction.

## 3. Data model

The application has six research/user tables: `users`, `training_runs`, `policies`, `evaluations`, `evaluation_participants`, and `episodes`. Add internal session/OAuth/migration tables as needed. A separate audit-event history is not required initially; keep creator/editor attribution and timestamps.

Use stable identifiers, UTC timestamps, proper foreign keys, and migrations. Enable SQLite foreign-key enforcement on connections. Commonly filtered fields should be columns; variable configurations, provenance, and additional metrics can be JSON. The field groups below specify required meaning; exact SQL names and how to split flexible metadata into columns are implementation decisions.

### Users

- Local UID and unique, stable GitHub account ID.
- Current GitHub login and display name.
- Active/allowed status and account timestamps.
- No locally managed passwords.

GitHub usernames can change; the stable GitHub account ID is the identity used for authorization.

### Training runs

This is the named, annotated entity describing training. Store:

- UID and stable source-run identity for recognizing repeated imports.
- Name and notes.
- Training method/algorithm, seed, available training dates, and budget information.
- Training configuration, including environment, model architecture, and observation settings.
- Code/dependency provenance and training-device information where known.
- Progress unit, such as environment steps for PPO or optimizer updates for behavior cloning.
- Created/updated timestamps and user attribution.

Keep shared training information here instead of duplicating it across checkpoints. The highest uploaded checkpoint does not necessarily represent the final progress of the local training run.

The existing untrained-policy evaluation is represented by an initialization-only run and a checkpoint at progress zero. Label it accurately; do not imply that training occurred.

### Policies / checkpoints

A policy row represents one specific checkpoint, not another independently named research entity:

- Checkpoint UID.
- **Required** foreign key to its training run.
- Training progress/step count, interpreted using the run's progress unit.
- Checkpoint SHA-256 hash for matching and integrity checks.

No separate checkpoint name or notes. Derive display labels from the run name and progress, for example `Coordination baseline · 5M steps` or `Sushi BC · 6000 updates`.

Every policy reference points to a specific checkpoint UID. Never substitute a run reference, a seed, or an implicit "latest checkpoint."

Do not identify checkpoints by training seed/name alone or assume that step count alone uniquely identifies weights after a resume/fork. Do not make a content hash alone the global training-run identity. Define a consistent import identity strategy using source identities, checkpoint contents, and run association. Unknown progress in legacy data must be reported honestly, not fabricated as zero or taken from the final checkpoint's summary.

### Evaluations

Store:

- UID and stable source-evaluation identity/content digest for duplicate detection.
- Editable title and notes.
- Evaluation time, upload time, uploader, and editor attribution.
- Environment/layout identity and effective environment configuration.
- Episode count, evaluation protocol, reward/metric definitions, and aggregate statistics.
- Export format version, code/dependency provenance, and evaluation-device information.

An evaluation records one declared protocol and participant assignment across its episodes. Different pairings are separate evaluations. Avoid silently pooling incompatible protocols.

### Evaluation participants

One row per player slot in an evaluation:

- Evaluation foreign key.
- Stable player/agent identifier and display order.
- **Required checkpoint UID foreign key.**
- Effective per-player observation settings and action-selection/perturbation settings where applicable.

Enforce one assignment per player slot within an evaluation. The same checkpoint can appear in several slots. Support a variable-length participant list rather than fixed `policy_1`/`policy_2` columns. Record training settings on the run and effective evaluation overrides here or in the evaluation protocol; do not confuse the two.

### Episodes

One row per episode, with:

- Evaluation foreign key and unique episode index within that evaluation.
- Seed, actual step count, available ending/termination information, and recording availability.
- Team reward, delivery statistics, and other episode metrics.
- Recorded replay data, stored independently for that episode in SQLite.
- Encoding/version, sizes, and integrity information needed to read the recording reliably.

Use gzip compression for the recording field internally. Uploads and API responses remain ordinary JSON. Compress/decompress only the selected episode, behind a small storage helper. Searchable metadata and summary statistics stay outside the compressed recording. Provide a simple developer inspection/export path so recordings can still be read as JSON.

Do not load recording fields for list pages. Loading a replay must not fetch/decompress every episode in the evaluation.

### Editing and deletion

- All approved teammates can edit names, titles, notes, and other explicitly editable annotations on all records.
- Imported measurements, checkpoint identities, configurations, and trajectories are recorded evidence, not ordinary editable fields. Correct those by importing corrected data, not silently rewriting past results.
- Use a record version or equivalent check to prevent an edit silently overwriting a teammate's intervening edit.
- Deletion is permanent after a confirmation dialog. No trash, restore route, or retention period.
- Deleting an evaluation deletes its episode and participant rows in the same transaction. It does not delete checkpoints or training runs.
- Reject deletion of a checkpoint while any participant row references it.
- Reject deletion of a training run while any checkpoint references it. Do not cascade from runs/checkpoints into evaluations.
- Explain a blocked deletion and link/show the references. Evaluation deletion is the important deletion workflow; standalone run/checkpoint deletion is lower priority.

## 4. Evaluation JSON and upload workflow

### One uploadable file

Each successful evaluation additionally writes `evaluation.json`, containing the information needed to import it without collecting several local files. Continue producing existing local outputs.

The new document is a versioned outer container, not a gratuitous replacement of the existing state format. Include:

- Format identity/version, source evaluation identity, and creation time.
- Source training-run records and exact checkpoint identities/progress/hashes.
- Participant assignments referencing those exact checkpoints.
- Environment/layout information, rules, action meanings, and renderer information.
- Evaluation protocol: actual seeds and their meaning, action-selection mode, reset conditions, perturbations, reward definition, and recording coverage.
- Aggregate statistics and every episode's metadata, statistics, and recorded states.
- Relevant source/dependency/device provenance.

Generate this from existing local configs, metadata, checkpoint information, and evaluation results. Users should not have to type information already available locally. Record progress for the selected checkpoint, not the final run's progress when evaluating an intermediate checkpoint.

Exclude model weights, optimizer checkpoints, per-state PNGs, GIFs/videos, and high-dimensional observation/recurrent-activation dumps by default. Do not embed uploaded HTML/JavaScript. Exclude full machine-local paths and unnecessary account/network identifiers; retain meaningful scientific configuration and useful device metadata.

Use portable source identities independent of website URLs, usernames, and absolute paths. Define version handling and identity generation during implementation. Additive optional fields and an extension mechanism should allow future metrics/settings without changing the relational model. Reject unsupported required semantics or incompatible versions clearly rather than silently misinterpreting them.

### Preserve the existing step convention

The current replay already has an explicit action/state relationship:

```text
steps[0] = initial state; no preceding action
steps[t] = state after action t, with that action attached, for t >= 1
```

Keep it. A 400-action episode has 401 states. Do not replace it with a separate `initial_state`/`transitions` structure merely for this website.

Reuse existing state serializers for agents, inventories/items, grid, recipes, pots/timers, and scores. Existing replay state is sufficient for visualization; it is not a promise of an exact resumable simulator checkpoint. Add fields only where a concrete requirement calls for them, preserving compatible readers or providing adapters.

Serialize all episodes at full recorded step resolution, not just the first. Correct currently first-episode-specific metadata such as episode index. Process/release trajectories episode by episode rather than accumulating all native JAX trajectories in memory. Finish the JSON atomically so an interrupted export is not presented as a complete uploadable result.

The sushi evaluator already records proposed actions, executed actions, and intervention masks. Preserve those diagnostics. A proposed action is the policy's choice; an executed action is the input actually passed to the environment after an optional action-dropout/random-action intervention. Being blocked by the environment is a different event. Ordinary evaluation does not enable these interventions automatically.

### Upload limits and validation

- Accept plain `.json` only; compressed uploads are not required or supported initially.
- **Maximum uploaded file size: 20 MiB = 20,971,520 bytes.** This is the file size, not a multipart request's framing overhead.
- Enforce the limit server-side while receiving the upload, with a compatible reverse-proxy body cap. Do not rely on the browser or the declared content length alone.
- Add generous sanity bounds for nesting, counts, dimensions, text lengths, and processing resources. Do not accidentally encode "exactly two players" or today's episode length as a format restriction.
- Validate structure, finite numeric values, participant/checkpoint references, episode numbering, step/action alignment, and known metric totals. Reject duplicate/conflicting identities and misleading inconsistencies.
- Shared team reward is counted once, not summed over players. Preserve metric definitions and scenario-specific meanings.
- Limit temporary storage and total persistent usage with reserved free space. Final quota and memory limits depend on the VPS.
- Report the local export size. If it exceeds the upload limit, fail clearly at upload; never silently discard episodes or downsample states to fit.

At 20 MiB, representative exports should be measured during implementation. The earlier 401-state reference was approximately 373 KB of uncompressed recorded JSON and 9 KB after gzip; these are one sample, not universal size estimates.

### Upload and policy association UI

1. The user selects one evaluation JSON file.
2. Validate it and show a review form for evaluation title/notes and its distinct training runs/checkpoints.
3. Automatically suggest existing exact checkpoint matches.
4. For unmatched checkpoints, let the user create a named training run or select an existing training run and add a checkpoint to it.
5. Save the evaluation and any new run/checkpoint/participant/episode records together.

For A+A, resolve the checkpoint once and assign it to both player slots. For A+B, resolve two checkpoints. A new checkpoint of an existing run creates a new policy row; it never replaces the previous checkpoint. Shared run name/notes belong to the run, not each checkpoint.

An existing checkpoint choice must agree with the uploaded checkpoint contents and relevant model metadata. Do not let a convenient dropdown silently associate results with different weights. Conflicting run metadata needs an explicit error/review path, not silent overwriting.

Repeated upload of the same source evaluation must not create duplicates. Reuse an existing record or report that it already exists. The same source identity with different contents is a conflict. Database constraints must also protect against concurrent duplicate imports.

Do not publish partial records before the review/import completes. Temporary preview storage, exact preview endpoint names, and whether parsing is streamed are implementation choices; bound memory/disk use and clean up abandoned temporary data.

### Existing results

Provide a local conversion path for existing evaluation folders where practical, reusing recorded files without rerunning evaluation. Extract only inert JSON from existing replay HTML; never execute the HTML.

Historical evaluations generally have statistics for all episodes but replay states only for the first. Import that honestly: retain all available statistics and show "recording unavailable" for missing trajectories. Do not regenerate trajectories from seeds and claim they were recorded originally. Missing provenance/progress remains unknown or is explicitly supplied; do not invent it.

## 5. Consolidate evaluation code and preserve local tools

### Current implementation

- `overcooked-v2/evaluate.py` supports upstream OvercookedV2 and the sushi environment, untrained policies, and compatible cross-policy evaluation.
- `overcooked-v2/evaluate_sushi_policy.py` adds sampled/greedy selection, action/reset perturbations, richer diagnostics and checks, and CPU-only enforcement.
- Both generate trajectories for each episode but retain the first for local replay.
- `overcooked-v2/preview.py` and `scenarios/oc2_sushi_1_3/preview.py` serialize states and generate existing previews.
- `overcooked-v2/replay_scene.py` is a pure-Python presentation adapter without JAX imports.
- `overcooked-v2/replay_template.html` already contains canvas rendering and playback controls.
- `overcooked-v2/replay.py` packages the offline viewer and currently requires PNG frames.

### Refactor

Create one shared evaluation implementation and one common JSON export path. Keep environment-specific components for state serialization, delivery/event diagnostics, supported reset perturbations, and preview rendering. The common implementation owns policy loading/execution, episode iteration, protocol recording, and export.

Keep existing command entry points as compatible wrappers where needed, preserving defaults and supported options. In particular, do not silently change episode counts/lengths, action sampling, seed splitting, recurrent-memory resets, reward accounting, cross-policy assignments, or the specialized CPU restriction.

The general evaluator and sushi evaluator must not each acquire a separately maintained implementation of the new upload format. Consolidate shared behavior rather than only wrapping two divergent exporters.

Retain existing local GIF/PNG/HTML output behavior, including the default first-episode previews and existing no-replay controls. The new all-episode JSON output is additional. Existing local HTML can retain original PNG frames and its Illustrated/Original comparison. Existing restyling and scripted-preview use cases should continue working.

Keep the current policy compatibility checks until the corresponding capabilities are implemented. This work does not include arbitrary heterogeneous architectures, different per-player observation pipelines, or changing the simulator's player count. The database, format, and viewer must support variable participants and per-player metadata without implying the simulator already supports every combination.

Small changes to training/checkpoint metadata are in scope to provide stable source-run/checkpoint identity, actual progress, and device provenance. They must not change training algorithms, random-number streams, or learning behavior. Legacy missing metadata requires a documented import strategy.

## 6. Pages, replay, and HTTP interface

### Pages

- `/`: a short project header/introduction and two tabs, **Evaluations** and **Training runs**, each searchable and filterable. No separate list pages for those collections.
- `/training-runs/<id>`: name/notes, training configuration/provenance/device information, checkpoint list, and related evaluations.
- `/evaluations/<id>`: title/notes, aggregate statistics, exact checkpoint assignments and effective settings, episode table, and selected episode replay.
- An optional `/about` can contain longer project context.
- An authenticated upload/review page and GitHub login/logout controls.
- Evaluation download is useful but lower priority; it is not required to delay the initial useful website.

Anonymous visitors can read research pages and their underlying replay data. Authenticated approved teammates see inline edit controls and delete controls. Server authorization must enforce permissions independently of button visibility.

Use a consistent responsive style. Toasts for success/failure stack centered at the top. Keep validation messages useful at the relevant form fields as well. Use accessible dialogs, keyboard controls, and status announcements. A failed deletion/import must not show a success toast.

### Replay

Reuse the existing canvas renderer and playback features: episode selection, play/pause, step/seek, speed, event navigation, agent information, pot/item information, and state inspection. Restyle to match the website.

- Load one selected episode at a time.
- Clearly distinguish selected-episode statistics from evaluation-wide averages.
- Use exact checkpoint labels for participants; support more than two player panels and a generic numbered/tinted fallback if artwork only covers two colors.
- Keep frame timing as recorded display timing; do not imply it measures simulator execution speed.
- Serve shared artwork as cached static assets. Do not upload per-step images or persist redundant derived scene data when it can be generated from the recording.
- Unsupported artwork/state types must produce an honest labelled fallback or unsupported-view message, not a misleading replay. Historical missing recordings remain distinguishable from failed/zero-reward episodes.
- Support a shareable selected episode/step in the URL if straightforward.

### Routes and methods

Use GET for reads, POST for creation/import, PATCH for partial field edits, PUT only for whole-resource replacement where appropriate, and DELETE for deletion. Do not implement deletion as POST `/delete/`. Vanilla JavaScript `fetch()` can submit PATCH/DELETE requests from the forms/dialogs.

The detail-page paths above are settled. Exact API prefixes, preview-upload paths, query parameters, and response DTO names can be chosen during implementation. The required endpoint capabilities are:

- Home-page filtered/paginated research lists, without fetching trajectories.
- Individual run/evaluation details and a selected episode's JSON/replay data.
- Upload validation/review and final transactional import.
- Allowed annotation edits on runs/evaluations.
- Permanent evaluation deletion, and restricted run/checkpoint deletion if exposed.
- GitHub login/callback/logout.
- Optional later evaluation JSON download.

Use appropriate failure responses: unauthorized/forbidden, validation errors, missing records, upload too large, referenced-record deletion conflicts, duplicate-content conflicts, and stale-edit conflicts. Exact error shapes are an implementation decision, but they must be usable by forms and toasts.

## 7. GitHub authentication and security

Use GitHub sign-in restricted to an explicit allowlist of the five stable GitHub account IDs. No unrestricted public registration and no local-password workflow. All allowed teammates have the same research-record editing permissions. A separate account-management UI is not necessary initially; allowlist/activation can be managed through deployment configuration or a small administrative command.

Use Authlib's OAuth integration rather than hand-writing the authorization-code protocol. Configure the callback URL, validate OAuth state, and use PKCE. Request only the identity access needed for sign-in; do not request repository permissions. Use the GitHub token to establish identity and avoid retaining it when no subsequent provider access is required.

After login, create a local server-side session. Use an unpredictable session identifier in a Secure, HttpOnly, SameSite cookie. Support expiry, logout/revocation, and disabling a user's access. Keep session secrets/token material out of public responses and logs. Do not put GitHub access tokens or login credentials in browser localStorage.

Use CSRF protection for cookie-authenticated state-changing requests, including PATCH and DELETE. Validate redirects and trusted origins/hosts. Apply reasonable login/upload throttling and request limits. Recheck authorization for every mutation; being on a public page or knowing a UID grants no edit access.

Escape titles/notes and other uploaded text. Plain text notes are sufficient initially; rich HTML is not required. Render embedded JSON safely or fetch it as JSON. Use an appropriate content security policy and avoid arbitrary remote resources supplied by uploads.

Treat uploads as data only. Never execute uploaded code, deserialize executable formats, load uploaded model checkpoints, import modules selected by the file, or follow arbitrary local paths/URLs from metadata. Validate both JSON structure and scientific consistency.

Store production OAuth secrets, session secrets, the SQLite database, backups, and runtime configuration outside committed source. Keep private operational data out of the public research pages. Public source in this repository is the chosen arrangement.

## 8. Device and provenance metadata

Distinguish the training machine, evaluation machine, and uploader. A model may be trained, evaluated, and uploaded on three different devices.

Capture best-effort, useful fields from the producing Python process:

- OS and release, CPU architecture, logical CPU count.
- Python and relevant package versions.
- Compute backend and available device description; memory information if easily available.
- Optional human-readable device label, such as `Rowan desktop` or `Modal GPU`.

Store training-device information with the run and evaluation-device information with the evaluation. Uploader identity/time is separate. Coarse browser/OS information at upload is optional and lower priority; it must not be presented as the training device.

Do not replace missing historical training-device metadata with the export/upload machine. Unknown fields stay unknown. WSL/container data describes the execution environment unless host information is separately and reliably identified. Avoid unnecessary identifiers such as full local paths, account names, MAC addresses, or wholesale environment-variable dumps.

This is useful provenance, not a hardware-inventory project; missing fields must not prevent evaluation or upload.

## 9. VPS deployment and storage operations

Deploy only the website and lightweight shared code. No JAX, training environment, model weights, JaxMARL checkout, or other repository subprojects are needed on the VPS.

Use a partial clone plus sparse checkout. The deployment guide can start from:

```bash
DEPLOY_BRANCH='rowan/evals-sharing-website'
EXPECTED_COMMIT='<reviewed-full-commit-sha>'
git clone --filter=blob:none --no-checkout --depth=1 \
  --branch "$DEPLOY_BRANCH" --single-branch --no-tags --no-recurse-submodules \
  <repo-url> results-site
cd results-site
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
```

Obtain the reviewed full commit SHA before cloning and replace the placeholder.
The direct cone configuration avoids Git 2.25.1's `clone --sparse` source-directory
bug and the clean-worktree requirement of `sparse-checkout init` after a
`--no-checkout` clone. Verify the commit before populating the sparse worktree.

These settings persist. Ordinary fast-forward pulls should preserve the sparse checkout and partial-fetch filter. Cone-mode sparse checkout also includes root-level files. Git still stores commit/tree metadata; explicit operations inspecting excluded file contents/history may fetch them. Do not describe this as a hard access boundary. Verify the actual remote supports partial clone and check that excluded large content is absent before calling deployment complete.

Keep runtime data/secrets outside the checkout so code updates cannot overwrite them. Run the app under an unprivileged account, bind Uvicorn to localhost behind Caddy, and use HTTPS. Provide reproducible installation, migration, restart, and rollback instructions under the website folder. Set worker count and resource limits from the actual VPS capacity, not an arbitrary high-concurrency template.

Use short database transactions and appropriate SQLite busy handling. If WAL is enabled, account for journal growth and use a supported SQLite version. Budget for temporary uploads, journals, logs, backups, and free-space reserve as well as live recordings. Rotate logs and clean expired sessions/temporary uploads.

Back up the database consistently using SQLite's backup facilities, with an off-VPS copy and a tested restore procedure. Avoid an unsafe standalone copy of a live database/WAL arrangement. SQLite may reuse deleted space without immediately shrinking the database file; storage reporting and maintenance must account for that. Do not add a trash/restore feature to application deletion.

Deployment inputs still to obtain: VPS OS, RAM, available disk, domain/subdomain and existing reverse-proxy setup, GitHub account IDs/OAuth registration details, and backup destination. These do not block local implementation.

## 10. Implementation sequence and verification

Work in reviewable increments, explain changes and verification to the user, and incorporate revisions. No research experiments or longer training runs are authorized by this plan.

1. **Format and shared evaluation core.** Define the outer JSON contract with a small real/synthetic example. Consolidate the two evaluation paths while preserving existing behavior. Add all-episode serialization, checkpoint/run provenance, and the additional JSON output. Keep local previews working.
2. **Backend foundation.** Create the website package, models/migrations, GitHub sessions/authorization, validation/import service, and episode storage. Keep the sparse-checkout dependency boundary intact.
3. **Main workflow.** Build the home tabs, upload/review/association flow, run/evaluation pages, inline annotation editing, and confirmed permanent deletion with reference protection.
4. **Replay integration.** Share the existing canvas/adaptation code, add episode selection and exact checkpoint labels, remove uploaded-frame requirements, and handle unsupported/missing recordings honestly.
5. **Compatibility and operations.** Add the local legacy conversion path, documentation, deployment configuration, size/quota behavior, and backup/restore procedure. Download and optional about-page polish can follow the core workflow.
6. **Review and deployment.** Review the working site locally, then deploy to the VPS when its configuration is available and deployment is authorized. Record the actual deployment and verification results rather than claiming them in advance.

Required verification should cover meaningful behavior rather than tests that merely mirror implementation:

- Existing seeded rollout behavior, action/state alignment, recurrent-memory resets, cross-policy assignment, and shared reward accounting remain unchanged after refactoring.
- Specialized sushi sampled/greedy and perturbation behavior remains intact. Existing zero-noise equivalence and proposed/executed-action tests are useful references.
- An export includes every episode's full recorded states and correct metadata/statistics, without PNGs or weights. Old local GIF/HTML/replay-restyling paths still work.
- A+A registers one checkpoint used twice; A+B registers/reuses two. Later checkpoints share the run, and mismatched hashes cannot be silently attached. Repeated/concurrent imports do not duplicate records or leave partial data.
- Variable-participant import/rendering works with synthetic fixtures even though current simulators commonly have two agents.
- Anonymous reads succeed and unauthorized mutations fail. GitHub allowlisting, OAuth callback validation, session expiry/logout, CSRF, and safe text rendering are checked.
- The exact 20 MiB file limit, malformed/oversized JSON, non-finite values, invalid references, and structural sanity limits fail clearly without uncontrolled memory/disk use.
- Evaluation deletion removes only its dependent rows. Referenced checkpoints and runs cannot be deleted. A stale edit does not silently overwrite another user's change.
- Episode compression round-trips correctly and only the selected recording is loaded. Legacy missing recordings are displayed accurately.
- Browser checks cover upload/review, filters/tabs, inline edits, confirmation dialogs, toasts, replay controls, and basic responsive/keyboard behavior.
- The server installs from the sparse website checkout without simulation dependencies. Backup restoration and migration/deployment procedures are exercised before calling hosting complete.

Exact module names, API DTOs, styling details, schema-version spelling, validation-library choice, and test tooling can be decided during implementation. Preserve the behavior and boundaries above rather than turning incidental examples into additional product requirements.

## References

Repository starting points:

- [Repository instructions](../../AGENTS.md)
- [Simulator setup and evaluation guide](../../overcooked-v2/README.md)
- [General evaluator](../../overcooked-v2/evaluate.py)
- [Sushi evaluator](../../overcooked-v2/evaluate_sushi_policy.py)
- [General replay serialization](../../overcooked-v2/preview.py)
- [Sushi replay serialization](../../overcooked-v2/scenarios/oc2_sushi_1_3/preview.py)
- [Scene adapter](../../overcooked-v2/replay_scene.py)
- [Replay packaging](../../overcooked-v2/replay.py)
- [Renderer notes](../../overcooked-v2/assets/replay/README.md)
- [Evaluation tests](../../overcooked-v2/tests/test_evaluate.py)
- [Sushi evaluation tests](../../overcooked-v2/tests/test_sushi_policy_evaluation.py)

Relevant upstream documentation:

- [FastAPI templates](https://fastapi.tiangolo.com/advanced/templates/)
- [SQLAlchemy](https://docs.sqlalchemy.org/)
- [Alembic](https://alembic.sqlalchemy.org/en/latest/)
- [Authlib web integrations](https://docs.authlib.org/en/latest/oauth2/client/web/index.html)
- [GitHub OAuth authorization](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps)
- [Git partial clone](https://git-scm.com/docs/partial-clone) and [sparse checkout](https://git-scm.com/docs/git-sparse-checkout)
- [Caddy HTTPS](https://caddyserver.com/docs/automatic-https)
- [SQLite backup](https://www.sqlite.org/backup.html)
