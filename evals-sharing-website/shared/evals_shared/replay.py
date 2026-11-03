"""One renderer source for the website and portable offline replays; no JAX."""
import base64
from io import BytesIO
import json
from pathlib import Path
import re

from .replay_scene import build_presentation

STATIC_ROOT = Path(__file__).resolve().parents[2] / "static" / "replay"


def build_web_payload(recording, participants=None):
    """Adapt one selected recording without requiring or sending PNG frames."""
    if recording is None:
        return {"recording": None}
    if recording.get("schema_version") != 1 or not recording.get("steps"):
        raise ValueError("Unsupported or incomplete replay payload.")
    data = {key: value for key, value in recording.items() if key not in ("frames", "presentation")}
    data["presentation"] = build_presentation(data, generic_agents=True)
    data["participants"] = participants or []
    return {"recording": data}


def write_replay(payload, path):
    """Restyle state/images without replaying actions or loading a policy."""
    if (payload.get("schema_version") != 1 or not payload.get("steps")
            or len(payload["steps"]) != len(payload.get("frames", []))):
        raise ValueError("Unsupported or incomplete replay payload.")
    payload = {**payload, "presentation": build_presentation(payload)}
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    viewer = (STATIC_ROOT / "viewer.html").read_text(encoding="utf-8")
    for name in ("kitchen-atlas.png", "sushi-atlas.png"):
        image = "data:image/png;base64," + base64.b64encode((STATIC_ROOT / name).read_bytes()).decode("ascii")
        viewer = viewer.replace('data-replay-asset="' + name + '"', 'src="' + image + '"')
    style = (STATIC_ROOT / "viewer.css").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "viewer.js").read_text(encoding="utf-8")
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Kitchen Lab · Episode viewer</title><style>' + style + '</style></head>'
        '<body><main class="replay-viewer" id="replay-root">' + viewer + '</main>'
        '<script id="replay-data" type="application/json">' + serialized + '</script>'
        '<script>' + script + '\nmountReplay(document.getElementById("replay-root"),'
        'JSON.parse(document.getElementById("replay-data").textContent));</script></body></html>')
    Path(path).write_text(document, encoding="utf-8")


def read_replay(path):
    """Extract only inert JSON; never execute saved HTML or JavaScript."""
    match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',
                      Path(path).read_text(encoding="utf-8"), re.S)
    if match is None:
        raise ValueError(f"No recorded replay payload in {path}")
    return json.loads(match[1])


def save_replay(data, frames, output):
    from PIL import Image
    if len(frames) != len(data["steps"]):
        raise ValueError("Each replay state needs exactly one kitchen image.")
    encoded = []
    for frame in frames:
        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    write_replay({**data, "frames": encoded}, Path(output) / "replay.html")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Restyle saved replays; no simulation or training.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    paths = sorted({file for path in args.paths for file in
                    (path.rglob("replay.html") if path.is_dir() else [path])})
    for path in paths:
        write_replay(read_replay(path), path)
        print(path)
