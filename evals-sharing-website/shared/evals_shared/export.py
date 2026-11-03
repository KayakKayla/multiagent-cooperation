"""Atomic, episode-at-a-time JSON evidence export."""
import json
import math
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from .format import FORMAT_NAME, FORMAT_VERSION, MAX_UPLOAD_BYTES, MAX_EPISODES, ValidationError, canonical_json, validate_document


class EvaluationWriter:
    """Spool plain episode JSON, never retaining native simulator trajectories.

    Only finish publishes evaluation.json. A crash leaves an unmistakable hidden
    temporary file, never a complete-looking partial export.
    """
    def __init__(self, destination):
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(mode="wb", prefix=".evaluation-episodes-", suffix=".tmp", dir=self.destination.parent, delete=False)
        self._spool = handle
        self._spool_path = Path(handle.name)
        self.count = 0

    def add_episode(self, episode):
        self._spool.write(canonical_json(episode) + b"\n")
        self._spool.flush()
        self.count += 1

    def finish(self, header):
        """Publish the complete document atomically; no upload-size truncation."""
        self._spool.close()
        header = dict(header)
        if not 1 <= self.count <= MAX_EPISODES or "episodes" in header:
            raise ValidationError("Export requires completed episodes and a header without episodes")
        header.setdefault("format", FORMAT_NAME)
        header.setdefault("version", FORMAT_VERSION)
        header.setdefault("source_evaluation_id", "evaluation:" + str(uuid4()))
        if len(header.get("protocol", {}).get("episode_seeds", [])) != self.count:
            raise ValidationError("Export episode count differs from protocol seeds")
        # Validate each episode with one-episode summaries to bound peak memory.
        # Full cross-episode totals are checked by the producer and importer.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=".evaluation-", suffix=".tmp", dir=self.destination.parent, delete=False) as output:
                temporary = Path(output.name)
                prefix = canonical_json(header)
                output.write(prefix[:-1] + b',"episodes":[')
                totals = {"team_reward": 0.0, "correct_dishes": 0, "wrong_dishes": 0}
                with self._spool_path.open("rb") as episodes:
                    for index, line in enumerate(episodes):
                        episode = json.loads(line)
                        if episode.get("index") != index or episode.get("seed") != header["protocol"]["episode_seeds"][index]:
                            raise ValidationError("Export episode order/seed disagrees")
                        for key in totals:
                            value = episode.get("metrics", {}).get(key)
                            totals[key] = totals[key] + value if value is not None and totals[key] is not None else None
                        check = dict(header, episodes=[dict(episode, index=0)], summary={})
                        check["protocol"] = dict(header["protocol"], episode_seeds=[episode["seed"]])
                        if episode.get("recording") is not None:
                            check["episodes"][0] = dict(check["episodes"][0], recording=dict(episode["recording"], metadata=dict(episode["recording"]["metadata"], episode=0)))
                        validate_document(check)
                        if index:
                            output.write(b",")
                        output.write(line.rstrip(b"\n"))
                summary = header.get("summary", {})
                if "episodes" in summary and summary["episodes"] != self.count:
                    raise ValidationError("Export summary episode count disagrees")
                for key in ("correct_dishes", "wrong_dishes"):
                    if "total_" + key in summary and totals[key] is not None and summary["total_" + key] != totals[key]:
                        raise ValidationError("Export summary delivery total disagrees")
                for key in ("mean_team_reward", "mean_raw_team_reward"):
                    if key in summary and (not self.count or not math.isclose(summary[key], totals["team_reward"] / self.count, rel_tol=1e-7, abs_tol=1e-6)):
                        raise ValidationError("Export summary mean reward disagrees")
                output.write(b"]}\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.destination)
            size = self.destination.stat().st_size
            return {"path": self.destination, "bytes": size, "within_upload_limit": size <= MAX_UPLOAD_BYTES}
        finally:
            self._spool_path.unlink(missing_ok=True)
            if temporary:
                temporary.unlink(missing_ok=True)

    def abort(self):
        self._spool.close()
        self._spool_path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.abort()


def write_document(destination, document):
    validate_document(document)
    with EvaluationWriter(destination) as writer:
        for episode in document["episodes"]:
            writer.add_episode(episode)
        return writer.finish({key: value for key, value in document.items() if key != "episodes"})
