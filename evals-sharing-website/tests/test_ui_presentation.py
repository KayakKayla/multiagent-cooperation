from pathlib import Path
from types import SimpleNamespace
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.presentation import format_progress
from app.views import checkpoint_dict

ROOT = Path(__file__).resolve().parents[1]
ENV = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape())


def test_progress_labels_preserve_raw_checkpoint_identity():
    for value, unit, expected in [(9961472, "environment_steps", "9.96M steps"), (0, "optimizer_updates", "0 updates"), (None, None, "unknown progress"), (1500, None, "1.5K units"), (1000000000, "steps", "1B steps")]:
        policy = SimpleNamespace(id="private-id", source_checkpoint_id="source-id", progress=value, sha256="a" * 64)
        run = SimpleNamespace(id="run-id", name="My baseline", progress_unit=unit)
        result = checkpoint_dict(policy, run)
        assert result["label"] == "My baseline \u00b7 " + expected
        assert result["progress"] == value
        assert result["sha256"] == "a" * 64
        assert result["id"] == "private-id"


def test_annotation_icons_are_accessible_and_notes_are_optional():
    template = ENV.from_string('{% from "macros.html" import annotation,attribution with context %}{{ annotation(record,"training-runs",user) }}{{ attribution(record) }}')
    record = dict(id="r", name="<unsafe>", notes="  ", version=3, updated_at="SECRET-DATE", editor=dict(login="Editor"))
    html = template.render(record=record, user={"login":"me"}, root_path="/sims")
    assert 'data-record-notes hidden' in html
    assert 'aria-label="Edit annotations" title="Edit annotations"' in html
    assert 'aria-label="Delete record" title="Delete record"' in html
    assert html.count('<svg aria-hidden="true"') == 2
    assert 'data-delete-url="/sims/api/training-runs/r"' in html
    assert 'name="version" value="3"' in html
    assert 'SECRET-DATE' not in html
    assert '&lt;unsafe&gt;' in html
    assert '<unsafe>' not in html
    assert 'data-edit-toggle' not in template.render(record=record, user=None)


def test_datetime_macro_keeps_epoch_zero_and_unknown_values():
    template = ENV.from_string('{% from "macros.html" import local_datetime %}{{ local_datetime(value) }}')
    assert 'data-local-datetime="0"' in template.render(value=0)
    assert template.render(value=None) == "Unknown"
    assert template.render(value="") == "Unknown"


def test_training_checkpoint_identifiers_are_only_in_collapsed_evidence():
    run = dict(id="run-id",name="Baseline",notes="",version=1,algorithm="PPO",seed=0,progress_unit="environment_steps",config={},provenance={},device={})
    checkpoint = dict(id="private-id",sha256="a"*64,source_checkpoint_id="source-id",progress=9961472,label="Baseline \u00b7 9.96M steps")
    html = ENV.get_template("training_run.html").render(run=run,checkpoints=[checkpoint],evaluations=[],user=None)
    normal = html.split('<details class="card evidence">')[0]
    assert "Baseline \u00b7 9.96M steps" in normal
    assert "private-id" not in normal
    assert "a" * 64 not in normal
    assert "source-id" not in normal
    assert "environment_steps" not in normal
    assert "private-id" in html


def test_upload_action_is_home_collection_action_and_signed_in_only():
    common = dict(tab="evaluations",filters=dict(q="",layout="",algorithm=""),evaluations=[],training_runs=[],page=1,pages=1,root_path="/sims")
    html = ENV.get_template("home.html").render(**common,user=dict(login="me"))
    header = html.split('</header>')[0]
    assert '/sims/upload' not in header
    assert 'class="button primary" href="/sims/upload"' in html
    assert '/sims/upload' not in ENV.get_template("home.html").render(**common,user=None)


def test_readable_numbers_have_bounded_fallback_and_preserve_other_values():
    template = ENV.from_string('{% from "macros.html" import readable_number %}{{ readable_number(value) }}')
    assert '>1,234.57</span>' in template.render(value=1234.5678912345)
    assert 'title="1234.5678912345"' in template.render(value=1234.5678912345)
    assert '>0</span>' in template.render(value=0)
    assert template.render(value=True) == 'Yes'
    assert template.render(value=False) == 'No'
    assert template.render(value='1.23456789') == '1.23456789'
    assert template.render(value='<tag>') == '&lt;tag&gt;'
    assert template.render(value=None) == 'Unknown'


def test_delete_run_conflict_references_use_readable_checkpoint_labels(tmp_path):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app
    from app.models import Base, User, TrainingRun, Policy
    from app.auth import new_session, COOKIE
    app = create_app(Settings(data_dir=tmp_path, base_url="http://testserver", allowed_github_ids=frozenset({123}), reserve_bytes=0))
    Base.metadata.create_all(app.state.engine)
    with app.state.db() as db:
        user = User(github_id=123, login="me", display_name="Me")
        db.add(user); db.flush()
        token, session = new_session(db, app.state.settings, user.id)
        run = TrainingRun(source_run_id="source", name="Jeff", progress_unit="environment_steps", evidence={}, created_by=user.id, updated_by=user.id)
        db.add(run); db.flush()
        policy = Policy(run_id=run.id, source_checkpoint_id="cp-source", sha256="a"*64, progress=9961472)
        db.add(policy); db.commit()
        run_id, policy_id, csrf = run.id, policy.id, session.csrf_token
    try:
        with TestClient(app) as client:
            client.cookies.set(COOKIE, token)
            response = client.delete(f"/api/training-runs/{run_id}", headers={"x-csrf-token":csrf})
        assert response.status_code == 409
        reference = response.json()['detail']['references'][0]
        assert reference['id'] == policy_id
        assert reference['name'] == 'Jeff \u00b7 9.96M steps'
        assert 'a'*12 not in reference['name']
    finally:
        app.state.engine.dispose()


def test_static_asset_version_tracks_code_and_styles_only(tmp_path):
    from app.main import static_asset_version
    (tmp_path/'nested').mkdir()
    code = tmp_path/'nested'/'episode.js'
    code.write_text('first',encoding='utf-8')
    first = static_asset_version(tmp_path)
    assert len(first) == 16
    assert static_asset_version(tmp_path) == first
    (tmp_path/'art.png').write_bytes(b'image')
    assert static_asset_version(tmp_path) == first
    code.write_text('second',encoding='utf-8')
    second = static_asset_version(tmp_path)
    assert second != first
    (tmp_path/'site.css').write_text('body{}',encoding='utf-8')
    assert static_asset_version(tmp_path) != second


def test_rendered_assets_use_content_version_and_preserve_prefix(tmp_path):
    import re
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app, static_asset_version
    from app.models import Base
    for prefix in ('', '/sims'):
        app = create_app(Settings(data_dir=tmp_path/('prefix' if prefix else 'root'), base_url='http://testserver'+prefix, reserve_bytes=0))
        Base.metadata.create_all(app.state.engine)
        try:
            with TestClient(app) as client:
                response = client.get(prefix+'/')
                assert response.status_code == 200
                assets = re.findall(r'(?:src|href)="([^"]+/static/[^" ]+)"',response.text) if prefix else re.findall(r'(?:src|href)="(/static/[^" ]+)"',response.text)
                assert len(assets) == 2
                for url in assets:
                    assert url.startswith(prefix+'/static/')
                    assert url.endswith('?v='+static_asset_version(ROOT/'static'))
                    assert client.get(url).status_code == 200
        finally:
            app.state.engine.dispose()


def test_training_metadata_is_compact_inline_and_has_no_back_link():
    run = dict(id="run",name="Jeff",notes="",version=1,algorithm="PPO",seed=0,progress_unit="steps",config={},provenance={},device={},created_at="2026-11-03T12:00:00Z",uploader=dict(login="Uploader"),editor=dict(login="Editor"))
    html = ENV.get_template("training_run.html").render(run=run,checkpoints=[],evaluations=[],user=None)
    assert html.count('class="muted record-metadata"') == 1
    assert 'data-local-datetime="2026-11-03T12:00:00Z"' in html
    assert 'by Uploader. Last edited by Editor.' in html
    assert '?tab=training-runs' not in html
    fragment = ENV.from_string('{% from "macros.html" import attribution %}{{ attribution(record) }}').render(record=run)
    assert '<p' not in fragment
