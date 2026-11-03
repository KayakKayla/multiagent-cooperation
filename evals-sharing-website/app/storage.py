import gzip
import hashlib
import json


def pack_recording(recording):
    if recording is None:
        return dict(recording_available=False, recording=None, encoding=None, raw_size=0, compressed_size=0, digest=None)
    raw = json.dumps(recording, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    packed = gzip.compress(raw, mtime=0)
    return dict(recording_available=True, recording=packed, encoding="gzip-json-v1", raw_size=len(raw), compressed_size=len(packed), digest=hashlib.sha256(raw).hexdigest())


def unpack_recording(episode):
    if not episode.recording_available:
        return None
    if episode.encoding != "gzip-json-v1":
        raise ValueError("Unsupported recording encoding")
    raw = gzip.decompress(episode.recording)
    if len(raw) != episode.raw_size or hashlib.sha256(raw).hexdigest() != episode.digest:
        raise ValueError("Recording integrity check failed")
    return json.loads(raw)
