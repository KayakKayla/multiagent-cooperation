"""Play a frozen policy and save scores plus one interactive replay (no learning)."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from preview import save_preview
from train import DEFAULT_CONFIG, SCRIPT_DIR, load_baseline
from scenarios import make_env


from evaluation_core import (initialize_policy, check_run_revision, check_partner_config,
                             load_checkpoint, make_episode as _make_episode,
                             iterate_episodes, EvaluationExport, policy_config_path)


def make_episode(module, config, env, network, record_action_probs=False):
    return _make_episode(module, config, env, network,
                         record_action_probs=record_action_probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--untrained", action="store_true", help="Use fresh baseline weights.")
    source.add_argument(
        "--run", type=Path,
        help="Training directory containing config.json and params.msgpack.",
    )
    parser.add_argument(
        "--partner-run", type=Path,
        help="Optional saved policy for chef 2 (blue); --run controls chef 1 (red).",
    )
    parser.add_argument("--seed", type=int, help="Policy initialization seed (untrained only).")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--eval-seed", type=int, default=1000,
        help="First test seed; independent of training.",
    )
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.eval_seed <= 2**32 - args.episodes:
        parser.error("Use a positive episode count and unsigned 32-bit test seeds.")
    if args.run and args.seed is not None:
        parser.error("A saved policy uses its saved training seed; omit --seed.")

    if args.partner_run and not args.run:
        parser.error("--partner-run requires --run; it cannot accompany --untrained.")
    run = args.run.resolve() if args.run else None
    partner_run = args.partner_run.resolve() if args.partner_run else None
    revision = (SCRIPT_DIR / "UPSTREAM_REVISION").read_text().strip()
    for source_run in (run, partner_run):
        if source_run:
            check_run_revision(source_run, revision)
    module, config = load_baseline(
        policy_config_path(run) if run else DEFAULT_CONFIG, args.seed
    )
    partner_config = config
    if partner_run:
        partner_config = json.loads(policy_config_path(partner_run).read_text())
        check_partner_config(config, partner_config)
    env = make_env(env_name=config["ENV_NAME"], **config["ENV_KWARGS"])
    if partner_run and env.num_agents != 2:
        raise ValueError("Cross-policy evaluation requires exactly two chefs.")
    network, params = initialize_policy(module, config, env)
    partner_params = load_checkpoint(partner_run, params) if partner_run else None
    if run:
        params = load_checkpoint(run, params)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
        raise ValueError("Policy contains non-finite weights.")
    checkpoint_bytes = (run / "params.msgpack").read_bytes() if run else serialization.to_bytes(jax.device_get(params))

    mode = "cross-policy" if partner_run else ("trained" if run else "untrained")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"{config['ENV_KWARGS']['layout']}-{mode}-{timestamp}"
    output = SCRIPT_DIR / "outputs/evaluate" / run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if not run:
        # Keep the actual before weights, not just an initialization seed.
        (output / "params.msgpack").write_bytes(checkpoint_bytes)
    if partner_run:
        (output / "partner-config.json").write_text(json.dumps(partner_config, indent=2) + "\n")
    policy_seeds = [config["SEED"], partner_config["SEED"]]
    partner_bytes = (
        (partner_run / "params.msgpack").read_bytes()
        if partner_run else checkpoint_bytes
    )
    # Record both checkpoint identities and their fixed assignment to chef colors.
    policies = [
        {"agent": agent, "chef": chef + 1, "color": ("red", "blue")[chef],
         "source_run": str(source_run) if source_run else None,
         "training_seed": policy_seeds[chef],
         "policy_sha256": hashlib.sha256(data).hexdigest()}
        for chef, (agent, source_run, data) in enumerate(zip(
            env.agents, (run, partner_run or run), (checkpoint_bytes, partner_bytes)
        ))
    ]
    seeds = list(range(args.eval_seed, args.eval_seed + args.episodes))
    frame_ms = round(env.rules.step_seconds * 1000) if env.name == "oc2_sushi_1_3" else 100
    gif_stride = 2 if env.name == "oc2_sushi_1_3" else 1
    protocol = {
        "policy": mode, "source_run": str(run) if run else None,
        "partner_run": str(partner_run) if partner_run else None, "policies": policies,
        "policy_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "upstream_revision": revision, "backend": jax.default_backend(),
        "episode_seeds": seeds, "action_selection": "sample", "recorded_episode": 0,
        "reward": "shared raw environment reward counted once; no shaping",
        "gif_stride": gif_stride, "gif_frame_ms": gif_stride * frame_ms,
        "replay_file": "replay.html", "replay_schema_version": 1,
        "replay_stride": 1, "replay_frame_ms": frame_ms,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        f"Evaluating {mode} policy, {len(seeds)} episodes; no training. Results: {output}",
        flush=True,
    )
    print(f"Chef 1 (red): seed {policy_seeds[0]}; chef 2 (blue): seed {policy_seeds[1]}", flush=True)
    play = make_episode(module, config, env, network, record_action_probs=True)
    export = EvaluationExport(output, env, config, mode, protocol, [
        {"run": run, "config": config, "weights": checkpoint_bytes, "initialization": not run},
        {"run": partner_run or run, "config": partner_config, "weights": partner_bytes, "initialization": not run},
    ])
    rows = []
    recorded = None
    for episode, seed, (initial, trace, action_probs) in iterate_episodes(play, params, seeds, partner_params):
        states, actions, rewards, correct, wrong, finite = trace
        np.savez_compressed(
            output / f"decisions-{episode:03d}.npz",
            action_probs=action_probs,
            actions=actions,
            agent_ids=np.asarray(env.agents),
            evaluation_seed=np.uint32(seed),
        )
        if not finite.all() or not np.isfinite(rewards).all() or not bool(states.done[-1]):
            raise ValueError("Invalid policy predictions, rewards, or unfinished episode.")
        row = {
            "episode": episode, "seed": seed, "steps": len(actions),
            "team_reward": float(rewards.sum()),
            "correct_dishes": int(correct.sum()), "wrong_dishes": int(wrong.sum()),
        }
        rows.append(row)
        export.add_episode(episode, seed, initial, trace, row)
        print(f"Episode {episode + 1}/{len(seeds)}: {row}", flush=True)
        if episode == 0:
            recorded = (initial, trace)
            recorded_probs = action_probs

    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # Always record the first episode; do not pick whichever looks most successful.
    print("Rendering the first episode with spectator counters.", flush=True)
    if env.name == "oc2_sushi_1_3":
        from scenarios.oc2_sushi_1_3.preview import save_policy_preview
        save_policy_preview(env, *recorded, output, mode, seeds[0])
    else:
        save_preview(*recorded, output, config, mode, seeds[0], policy_seeds=policy_seeds, action_probs=recorded_probs)
    summary = {
        "status": "complete", "policy": mode, "episodes": len(rows),
        "mean_team_reward": float(np.mean([row["team_reward"] for row in rows])),
        "mean_correct_dishes": float(np.mean([row["correct_dishes"] for row in rows])),
        "total_correct_dishes": sum(row["correct_dishes"] for row in rows),
        "total_wrong_dishes": sum(row["wrong_dishes"] for row in rows),
        "episodes_with_a_correct_delivery": sum(row["correct_dishes"] > 0 for row in rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export.finish(summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Replay (open in a browser): {output / 'replay.html'}", flush=True)
    print(f"GIF: {output / 'rollout.gif'}", flush=True)


if __name__ == "__main__":
    main()
