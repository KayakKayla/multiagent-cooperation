from dataclasses import dataclass, field
import os
import re
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
    def root_path(self):
        return urlparse(self.base_url).path.rstrip("/")

    @property
    def origin(self):
        parsed = urlparse(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @property
    def cookie_path(self):
        return self.root_path + "/"

    def app_path(self, path="/"):
        """External site path; proxy strips root_path before ASGI routing."""
        return self.root_path + "/" + path.lstrip("/")

    @property
    def database_path(self):
        return self.data_dir / "evaluations.sqlite3"

    def prepare(self):
        self.base_url = self.base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*", parsed.path) is None):
            raise ValueError("EVALS_BASE_URL must be an absolute HTTP(S) site URL with an optional simple path prefix")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            raise ValueError("HTTPS is required outside localhost")
        self.data_dir.mkdir(parents=True, exist_ok=True)
