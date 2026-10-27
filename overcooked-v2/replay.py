"""Package recorded episode data and PNGs as a portable, offline HTML replay."""

import base64
from io import BytesIO
import json
from pathlib import Path
import re

from replay_scene import build_presentation


def write_replay(payload, path: Path):
    """Restyle saved state/images without replaying actions or loading a policy."""
    if (payload.get("schema_version") != 1 or not payload.get("steps")
            or len(payload["steps"]) != len(payload.get("frames", []))):
        raise ValueError("Unsupported or incomplete replay payload.")
    payload = {**payload, "presentation": build_presentation(payload)}
    # Escaping '<' prevents metadata from closing the embedded JSON script tag.
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    root = Path(__file__).parent
    template = (root / "replay_template.html").read_text(encoding="utf-8")
    for placeholder, name in (("__KITCHEN_ATLAS__", "kitchen-atlas.png"),
                              ("__SUSHI_ATLAS__", "sushi-atlas.png")):
        encoded = base64.b64encode((root / "assets/replay" / name).read_bytes()).decode("ascii")
        template = template.replace(placeholder, "data:image/png;base64," + encoded)
    path.write_text(template.replace("__REPLAY_DATA__", serialized), encoding="utf-8")


def read_replay(path: Path):
    """Read only the inert JSON block; never evaluate saved HTML or JavaScript."""
    match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',
                      path.read_text(encoding="utf-8"), re.S)
    if match is None:
        raise ValueError(f"No recorded replay payload in {path}")
    return json.loads(match[1])


def save_replay(data, frames, output: Path):
    """Keep the versioned payload independent of the browser's playback controls."""
    from PIL import Image

    if len(frames) != len(data["steps"]):
        raise ValueError("Each replay state needs exactly one kitchen image.")
    encoded = []
    for frame in frames:
        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    payload = {**data, "frames": encoded}
    write_replay(payload, output / "replay.html")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply the shared UI to saved replays; no simulation or training.")
    parser.add_argument("paths", nargs="+", type=Path, help="Replay HTML files or directories to search recursively")
    args = parser.parse_args()
    paths = sorted({file for path in args.paths for file in
                    (path.rglob("replay.html") if path.is_dir() else [path])})
    for path in paths:
        payload = read_replay(path)
        write_replay(payload, path)
        print(path)
