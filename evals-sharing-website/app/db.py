from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


def database(settings):
    engine = create_engine("sqlite:///" + settings.database_path.as_posix(), connect_args={"check_same_thread": False, "timeout": 15})

    @event.listens_for(engine, "connect")
    def pragmas(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        # DELETE journaling bounds persistent journal growth for this small archive.
        conn.execute("PRAGMA journal_mode=DELETE")

    return engine, sessionmaker(engine, expire_on_commit=False)
