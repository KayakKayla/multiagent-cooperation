"""Compatibility entry point for the shared replay implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals-sharing-website" / "shared"))
from evals_shared.replay_scene import *  # noqa: F401,F403
