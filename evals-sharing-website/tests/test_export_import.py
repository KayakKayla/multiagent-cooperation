"""Actual producer-to-API regression: later resumed checkpoint shares its run."""
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select, func

from app.auth import COOKIE, new_session
from app.config import Settings
from app.main import create_app
from app.models import Base, User, TrainingRun, Policy, Evaluation
from evals_shared.provenance import source_records


def test_actual_export_records_import_after_resume_without_changing_original_evidence(tmp_path):
    data=tmp_path/"site"
    app=create_app(Settings(data_dir=data,base_url="http://testserver",allowed_github_ids=frozenset({123}),reserve_bytes=0))
    Base.metadata.create_all(app.state.engine)
    with app.state.db() as db:
        user=User(github_id=123,login="teammate",display_name="Teammate")
        db.add(user);db.flush()
        token,session=new_session(db,app.state.settings,user.id)
    root=tmp_path/"training";root.mkdir()
    config={"ALGORITHM":"ippo_rnn","SEED":42}
    metadata={"source_run_id":"run:resume-integration", "device":{"backend":"cpu"},"resumes":[]}
    documents=[]
    with TestClient(app) as client:
        client.cookies.set(COOKIE,token)
        for index,weights in enumerate((b"first checkpoint bytes",b"later checkpoint bytes")):
            if index:
                metadata["resumes"].append({"at":"2026-11-03T00:00:00+00:00","from_update":3,"backend":"cpu"})
            (root/"metadata.json").write_text(json.dumps(metadata))
            (root/"checkpoint.json").write_text(json.dumps({"sha256":hashlib.sha256(weights).hexdigest(),"progress":48+16*index,"progress_unit":"environment_steps"}))
            run,checkpoint=source_records(root,config,weights)
            doc=json.loads((Path(__file__).parent/"fixtures/evaluation-v1.json").read_text(encoding="utf-8"))
            doc["source_evaluation_id"]="evaluation:resume-integration-"+str(index)
            doc["training_runs"]=[run];doc["checkpoints"]=[checkpoint]
            for participant in doc["participants"]:
                participant["source_checkpoint_id"]=checkpoint["source_checkpoint_id"]
            doc["provenance"]["training_checkpoint_provenance"]={checkpoint["source_checkpoint_id"]:checkpoint["provenance"]}
            documents.append(doc)
            preview=client.post("/api/uploads",content=json.dumps(doc),headers={"content-type":"application/json","x-csrf-token":session.csrf_token})
            assert preview.status_code==200,preview.text
            assert preview.json()["conflicts"]==[]
            imported=client.post("/api/imports",json={"preview_id":preview.json()["id"]},headers={"x-csrf-token":session.csrf_token})
            assert imported.status_code==200,imported.text
    assert documents[0]["training_runs"]==documents[1]["training_runs"]
    with app.state.db() as db:
        assert db.scalar(select(func.count()).select_from(TrainingRun))==1
        assert db.scalar(select(func.count()).select_from(Policy))==2
        assert db.scalar(select(func.count()).select_from(Evaluation))==2
        stored=db.scalar(select(TrainingRun))
        assert "resumes" not in stored.evidence["provenance"]
        evaluations=db.scalars(select(Evaluation)).all()
        provenance=[e.metadata_json["provenance"]["training_checkpoint_provenance"] for e in evaluations]
        assert sorted(len(next(iter(p.values()))["training_resumes"]) for p in provenance)==[0,1]
    app.state.engine.dispose()
