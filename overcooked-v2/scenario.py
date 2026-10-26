"""Run the local level 1-3 scripted prototype without training or Steam."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import jax
from PIL import Image

from replay import save_replay
from scenarios import make_env
from scenarios.oc2_sushi_1_3.preview import render, replay_data
from scenarios.oc2_sushi_1_3.scripted import Script, compare_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["oc2_sushi_1_3"], default="oc2_sushi_1_3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    env = make_env(args.layout)
    output = args.output or Path(__file__).parent / "outputs/scenarios" / (
        args.layout + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=True)
    print("Running two-chef recipe and plate-reuse script (no training).", flush=True)
    script = Script(env).run()
    comparison = compare_reference(script)
    data = replay_data(script)
    frames = [render(env, state) for state in script.states]
    save_replay(data, frames, output)
    Image.fromarray(frames[0]).save(output / "initial.png")
    Image.fromarray(frames[-1]).save(output / "final.png")
    assembly_step = next(m["step"] for m in script.milestones if m["name"] == "Dish 1: assembled sushi")
    Image.fromarray(frames[assembly_step]).save(output / "assembled.png")
    trace = dict(metadata=data["metadata"], rules=asdict(env.rules), milestones=script.milestones,
                 actions=script.actions, events=script.events, steps=data["steps"])
    (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    summary = dict(scenario=env.name, schema_version=env.schema_version, backend=jax.default_backend(),
                   purpose="Scripted simulator prototype; not training or a research benchmark",
                   steps=int(script.state.time), deliveries=int(script.state.deliveries),
                   prototype_reward=float(script.state.score), plates=4, pots=3,
                   reused_plate_slot=3, invariants_passed_each_step=True,
                   food_spawn_counts=script.spawn_counts.tolist(),
                   rules=asdict(env.rules),
                   provisional=["walkable geometry and spawn rounding", "step duration", "chopping and washing durations",
                                "plate return delay", "one-slot return/sink/drying capacities", "fixed order and reward",
                                "no burning; cooking progress is capped and retained off heat"],
                   milestones=script.milestones)
    summary["real_game_reference_comparison"] = comparison
    sources = Path(__file__).parent / "scenarios/oc2_sushi_1_3"
    summary["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(sources.glob("*.py"))}
    summary["source_sha256"]["reference_workflow.json"] = hashlib.sha256(
        (sources / "reference_workflow.json").read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {summary['steps']} steps, {summary['deliveries']} deliveries; same plate washed and reused.")
    print("Replay:", (output / "replay.html").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
