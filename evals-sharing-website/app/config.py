from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("EVALS_DATA_DIR", ".data")).resolve())
    base_url: str = field(default_factory=lambda: os.getenv("EVALS_BASE_URL", "http://127.0.0.1:8000").rstrip("/"))
    github_client_id: str = field(default_factory=lambda: os.getenv("GITHUB_CLIENT_ID", ""))
    github_client_secret: str = field(default_factory=lambda: os.getenv("GITHUB_CLIENT_SECRET", ""))
    allowed_github_ids: frozenset[int] = field(default_factory=lambda: frozenset(int(x) for x in os.getenv("GITHUB_ALLOWED_IDS", "").split(",") if x.strip()))
    session_seconds: int = 12 * 60 * 60
    preview_seconds: int = 30 * 60
    max_storage_bytes: int = field(default_factory=lambda: int(os.getenv("EVALS_STORAGE_BYTES", str(2 * 1024**3))))
    reserve_bytes: int = field(default_factory=lambda: int(os.getenv("EVALS_RESERVE_BYTES", str(512 * 1024**2))))
    max_previews: int = 20

    @property
    def secure_cookie(self):
        return self.base_url.startswith("https://")

    @property
    def host(self):
        return urlparse(self.base_url).hostname

    @property
    def database_path(self):
        return self.data_dir / "evaluations.sqlite3"

    def prepare(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
            raise ValueError("EVALS_BASE_URL must be an absolute HTTP(S) origin")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            raise ValueError("HTTPS is required outside localhost")
        self.data_dir.mkdir(parents=True, exist_ok=True)
