from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import JSON, Boolean, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def uid():
    return str(uuid4())


def now():
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    github_id: Mapped[int] = mapped_column(Integer, unique=True)
    login: Mapped[str] = mapped_column(String)
    display_name: Mapped[str] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[str] = mapped_column(String, default=now)
    updated_at: Mapped[str] = mapped_column(String, default=now)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    source_run_id: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str] = mapped_column(String)
    notes: Mapped[str] = mapped_column(Text, default="")
    algorithm: Mapped[str | None] = mapped_column(String, index=True)
    seed: Mapped[int | None] = mapped_column(Integer)
    progress_unit: Mapped[str | None] = mapped_column(String)
    evidence: Mapped[dict] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[str] = mapped_column(String, default=now)
    updated_at: Mapped[str] = mapped_column(String, default=now)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    updated_by: Mapped[str] = mapped_column(ForeignKey("users.id"))


class Policy(Base):
    __tablename__ = "policies"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    source_checkpoint_id: Mapped[str] = mapped_column(String, unique=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id", ondelete="RESTRICT"), index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    progress: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String, default=now)


class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    source_evaluation_id: Mapped[str] = mapped_column(String, unique=True)
    digest: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String)
    notes: Mapped[str] = mapped_column(Text, default="")
    layout: Mapped[str] = mapped_column(String, index=True)
    environment_name: Mapped[str] = mapped_column(String, index=True)
    episode_count: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[dict] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[str] = mapped_column(String, default=now)
    updated_at: Mapped[str] = mapped_column(String, default=now)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    updated_by: Mapped[str] = mapped_column(ForeignKey("users.id"))


class RunIdentity(Base):
    __tablename__ = "run_identities"
    source_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"), index=True)


class CheckpointIdentity(Base):
    __tablename__ = "checkpoint_identities"
    source_id: Mapped[str] = mapped_column(String, primary_key=True)
    policy_id: Mapped[str] = mapped_column(ForeignKey("policies.id", ondelete="CASCADE"), index=True)


class Participant(Base):
    __tablename__ = "evaluation_participants"
    __table_args__ = (UniqueConstraint("evaluation_id", "agent_id"), UniqueConstraint("evaluation_id", "display_order"))
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluations.id", ondelete="CASCADE"), index=True)
    policy_id: Mapped[str] = mapped_column(ForeignKey("policies.id", ondelete="RESTRICT"), index=True)
    agent_id: Mapped[str] = mapped_column(String)
    display_order: Mapped[int] = mapped_column(Integer)
    observation_settings: Mapped[dict] = mapped_column(JSON)
    action_settings: Mapped[dict] = mapped_column(JSON)


class Episode(Base):
    __tablename__ = "episodes"
    __table_args__ = (UniqueConstraint("evaluation_id", "index"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluations.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int | None] = mapped_column(Integer)
    steps: Mapped[int] = mapped_column(Integer)
    termination: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    recording_available: Mapped[bool] = mapped_column(Boolean)
    encoding: Mapped[str | None] = mapped_column(String)
    raw_size: Mapped[int] = mapped_column(Integer, default=0)
    compressed_size: Mapped[int] = mapped_column(Integer, default=0)
    digest: Mapped[str | None] = mapped_column(String(64))
    recording: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True, deferred_raiseload=True)


class LoginSession(Base):
    __tablename__ = "sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    csrf_token: Mapped[str] = mapped_column(String)
    expires_at: Mapped[int] = mapped_column(Integer, index=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)


class UploadPreview(Base):
    __tablename__ = "upload_previews"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[int] = mapped_column(Integer, index=True)
    size: Mapped[int] = mapped_column(Integer)
    document: Mapped[bytes] = mapped_column(LargeBinary, deferred=True, deferred_raiseload=True)
