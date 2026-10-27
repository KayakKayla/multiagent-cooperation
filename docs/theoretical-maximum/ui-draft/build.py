"""Build the standalone UI draft from the verified replay, without running JAX."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "media/18-soups/replay.html"
OUTPUT = HERE.parent / "media/18-soups-ui/replay.html"
sys.path.insert(0, str(HERE.parents[2] / "overcooked-v2"))
from replay import read_replay, write_replay


def main():
    data = read_replay(SOURCE)
    if len(data["steps"]) != 401 or data["steps"][-1]["totals"] != {
            "team_reward": 360.0, "correct_dishes": 18, "wrong_dishes": 0}:
        raise ValueError("Expected the verified 400-step, 18-soup reference")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_replay(data, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
