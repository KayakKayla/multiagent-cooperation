"""Package recorded episode data and PNGs as a portable, offline HTML replay."""

import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image


def save_replay(data, frames, output: Path):
    """Keep the versioned payload independent of the browser's playback controls."""
    if len(frames) != len(data["steps"]):
        raise ValueError("Each replay state needs exactly one kitchen image.")
    encoded = []
    for frame in frames:
        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    payload = {**data, "frames": encoded}
    # Escaping '<' prevents metadata from closing the embedded JSON script tag.
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("replay_template.html").read_text(encoding="utf-8")
    (output / "replay.html").write_text(template.replace("__REPLAY_DATA__", serialized), encoding="utf-8")
