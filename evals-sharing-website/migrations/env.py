from alembic import context
from app.config import Settings
from app.db import database
from app.models import Base

settings = context.config.attributes.get("settings") or Settings()
settings.prepare()
engine, _ = database(settings)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
