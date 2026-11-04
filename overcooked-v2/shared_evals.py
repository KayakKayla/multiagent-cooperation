"""Bootstrap the lightweight package for local simulator scripts."""
from pathlib import Path
import sys

SHARED = Path(__file__).resolve().parents[1] / "evals-sharing-website" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from evals_shared.provenance import device_info, recorded_source_identity
