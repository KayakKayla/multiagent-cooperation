"""Evaluate a saved neural sushi policy on CPU, without an expert or planner.

Example (from overcooked-v2/):
  JAX_PLATFORMS=cpu python evaluate_sushi_policy.py --run outputs/train/RUN
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

# Evaluation intentionally avoids the experimental Metal scan/readback path.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

import checkpoint as ckpt
from evaluate import initialize_policy
from scenarios import make_env
from scenarios.oc2_sushi_1_3.environment import Action, Item, QUEUED, RECIPE
from scenarios.oc2_sushi_1_3.layout import Tile
from train import ALGORITHMS, SCRIPT_DIR, load_baseline
from upload_evaluation import upload_evaluation


from evaluation_core import (make_episode as _make_episode, validate_perturbations,
                             perturb_initial_state, iterate_episodes, EvaluationExport,
                             load_checkpoint, policy_config_path)


def make_episode(module, config, env, network, action_selection="sampled", **options):
    return _make_episode(module, config, env, network, action_selection=action_selection,
                         diagnostics=True, **options)


def summarize_episode(initial, trace, diagnostics, env, seed, episode):
    """Audit actual transitions rather than treating predictions as deliveries."""
    states, actions, rewards, deliveries, _, finite = trace
    delivered_slots, washed_slots, invalid, blocked = map(np.asarray, diagnostics[:4])
    actions, rewards, deliveries = map(np.asarray, (actions, rewards, deliveries))
    if len(diagnostics) > 4:
        proposed, drop_mask, random_mask = map(np.asarray, diagnostics[4:])
    else:  # Allow older recorded diagnostics to retain their no-noise meaning.
        proposed = actions
        drop_mask = random_mask = np.zeros_like(actions, dtype=bool)
    if proposed.shape != actions.shape or drop_mask.shape != actions.shape or random_mask.shape != actions.shape:
        raise ValueError("Proposed/executed action and perturbation masks have inconsistent shapes")
    if not np.asarray(finite).all() or not np.isfinite(rewards).all():
        raise ValueError("Non-finite or unnormalized policy predictions/rewards")
    if len(actions) != env.max_steps or not bool(np.asarray(states.done)[-1]):
        raise ValueError("Evaluation did not finish the requested episode")
    if np.any(actions < 0) or np.any(actions >= env.num_actions):
        raise ValueError("Policy emitted an invalid action code")
    if np.any(deliveries < 0) or not np.array_equal(deliveries, delivered_slots.sum(axis=1)):
        raise ValueError("Delivery counter disagrees with completed plate transitions")
    actual_delta = np.diff(np.r_[int(initial.deliveries), np.asarray(states.deliveries)])
    if not np.array_equal(deliveries, actual_delta):
        raise ValueError("Recorded delivery events disagree with actual state")
    if not np.allclose(rewards, deliveries * env.rules.delivery_reward, rtol=0, atol=1e-6):
        raise ValueError("Raw reward disagrees with the actual completed dishes")
    kind = np.asarray(states.kind)
    if not np.all(np.sum(kind == Item.POT, axis=1) == np.sum(np.asarray(initial.kind) == Item.POT)):
        raise ValueError("Pot conservation failed")
    plate_count = np.sum((kind == Item.PLATE) | (kind == Item.DIRTY_PLATE), axis=1)
    initial_plates = np.sum((np.asarray(initial.kind) == Item.PLATE) | (np.asarray(initial.kind) == Item.DIRTY_PLATE))
    if not np.all(plate_count == initial_plates):
        raise ValueError("Plate conservation failed")

    times = [step + 1 for step, count in enumerate(deliveries) for _ in range(int(count))]
    prior_deliveries = np.zeros(delivered_slots.shape[1], dtype=np.int32)
    washed_since_delivery = np.zeros_like(prior_deliveries, dtype=bool)
    reused, washed_reused = 0, 0
    for delivered, washed in zip(delivered_slots, washed_slots):
        reused += int(np.sum(delivered & (prior_deliveries > 0)))
        washed_reused += int(np.sum(delivered & (prior_deliveries > 0) & washed_since_delivery))
        prior_deliveries += delivered.astype(np.int32)
        washed_since_delivery[delivered] = False
        washed_since_delivery[washed & (prior_deliveries > 0)] = True
    count = int(deliveries.sum())
    return {
        "episode": int(episode), "seed": int(seed), "steps": len(actions),
        "raw_team_reward": float(rewards.sum()), "correct_dishes": count,
        "success_at_least_one": count >= 1, "success_at_least_two": count >= 2,
        "delivery_steps": times,
        "first_delivery_step": times[0] if times else None,
        "second_delivery_step": times[1] if len(times) > 1 else None,
        "first_delivery_seconds": times[0] * env.rules.step_seconds if times else None,
        "second_delivery_seconds": times[1] * env.rules.step_seconds if len(times) > 1 else None,
        "plate_washes": int(washed_slots.sum()), "reused_plate_deliveries": reused,
        "washed_then_reused_plate_deliveries": washed_reused,
        "invalid_interactions": int(invalid.sum()), "blocked_movements": int(blocked.sum()),
        "finite_predictions": True,
        "initial_positions": np.asarray(initial.positions).tolist(),
        "initial_facings": np.asarray(initial.directions).tolist(),
        "proposed_actions": proposed.tolist(), "executed_actions": actions.tolist(),
        "action_dropout_mask": drop_mask.astype(bool).tolist(),
        "random_action_mask": random_mask.astype(bool).tolist(),
        "dropout_selections": int(drop_mask.sum()),
        "random_action_selections": int(random_mask.sum()),
        "perturbed_actions": int(np.count_nonzero(proposed != actions)),
    }


def summarize(rows):
    if not rows:
        raise ValueError("At least one completed episode is required")
    first = [r["first_delivery_step"] for r in rows if r["first_delivery_step"] is not None]
    second = [r["second_delivery_step"] for r in rows if r["second_delivery_step"] is not None]
    return {
        "status": "complete", "episodes": len(rows),
        "mean_raw_team_reward": float(np.mean([r["raw_team_reward"] for r in rows])),
        "mean_correct_dishes": float(np.mean([r["correct_dishes"] for r in rows])),
        "total_correct_dishes": sum(r["correct_dishes"] for r in rows),
        "success_rate_at_least_one": float(np.mean([r["success_at_least_one"] for r in rows])),
        "success_rate_at_least_two": float(np.mean([r["success_at_least_two"] for r in rows])),
        "mean_first_delivery_step_among_successes": float(np.mean(first)) if first else None,
        "mean_second_delivery_step_among_successes": float(np.mean(second)) if second else None,
        "episodes_with_plate_washing": sum(r["plate_washes"] > 0 for r in rows),
        "episodes_with_washed_plate_reuse": sum(r["washed_then_reused_plate_deliveries"] > 0 for r in rows),
        "finite_predictions": all(r["finite_predictions"] for r in rows),
        "total_dropout_selections": sum(r.get("dropout_selections", 0) for r in rows),
        "total_random_action_selections": sum(r.get("random_action_selections", 0) for r in rows),
        "total_perturbed_actions": sum(r.get("perturbed_actions", 0) for r in rows),
    }


def evaluate_run(run, output=None, episodes=20, eval_seed=1000, action_selection="sampled",
                 steps=600, replay=True, *, action_dropout=0., random_action_prob=0.,
                 spawn_jitter=0, random_facing=False):
    if jax.default_backend().lower() != "cpu":
        raise ValueError("Use JAX_PLATFORMS=cpu for this evaluator")
    if episodes < 1 or not 0 <= eval_seed <= 2**32 - episodes or steps < 1:
        raise ValueError("Use positive episodes/steps and unsigned 32-bit seeds")
    if action_selection not in ("sampled", "greedy"):
        raise ValueError("action_selection must be sampled or greedy")
    validate_perturbations(action_dropout, random_action_prob, spawn_jitter)
    run = Path(run).resolve()
    module, training_config = load_baseline(policy_config_path(run), None)
    config = copy.deepcopy(training_config)
    if config["ENV_KWARGS"]["layout"] != "oc2_sushi_1_3":
        raise ValueError("This evaluator is only for oc2_sushi_1_3")
    config["ENV_KWARGS"].setdefault("rules", {})["max_steps"] = steps
    env = make_env(env_name=config["ENV_NAME"], **config["ENV_KWARGS"])
    network, template = initialize_policy(module, config, env)
    weights = (run / "params.msgpack").read_bytes()
    params = load_checkpoint(run, template)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
        raise ValueError("Saved model contains non-finite weights")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output).resolve() if output else SCRIPT_DIR / "outputs/evaluate-sushi" / f"{run.name}-{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    protocol = {
        "layout": env.name, "source_run": str(run), "backend": "cpu",
        "policy_sha256": hashlib.sha256(weights).hexdigest(),
        "evaluator_source_sha256": ckpt.file_hash(Path(__file__)),
        "algorithm_source_sha256": ckpt.file_hash(ALGORITHMS[config["ALGORITHM"]]),
        "environment_source_sha256": ckpt.file_hash(Path(__file__).parent / "scenarios/oc2_sushi_1_3/environment.py"),
        "upstream_revision": (SCRIPT_DIR / "UPSTREAM_REVISION").read_text().strip(),
        "action_selection": action_selection, "episode_seeds": list(range(eval_seed, eval_seed + episodes)),
        "episode_steps": steps, "training_episode_steps": training_config["ENV_KWARGS"].get("rules", {}).get("max_steps", 600),
        "observation_shape": list(env.observation_space().shape), "initial_state": "standard empty kitchen; fixed chef starts",
        "seed_scope": "Reset ignores seeds. Seeds vary policy action sampling only; no unseen layouts or randomized starts.",
        "greedy_scope": "Greedy repetitions use the identical deterministic initial situation and are not independent starting-state tests.",
        "reward": "raw shared environment reward counted once; no training shaping",
        "expert_or_planner_used": False, "learning_during_evaluation": False,
        "replay_episode": 0 if replay else None,
        "action_dropout": float(action_dropout), "random_action_prob": float(random_action_prob),
        "spawn_jitter": spawn_jitter, "spawn_jitter_distance": "Manhattan",
        "random_facing": bool(random_facing),
        "perturbation_scope": "Independent per-chef/per-step exogenous action overrides; initial jitter changes only valid chef poses on this fixed layout.",
        "perturbation_rng": "Independent folded streams from eval seed; original policy/reset key chain is unchanged.",
    }
    if action_dropout or random_action_prob or spawn_jitter or random_facing:
        protocol["seed_scope"] = ("Base reset ignores seeds. Eval seeds independently seed policy sampling, "
                                  "action perturbations and initial pose jitter. This is a fixed-layout robustness intervention, not an unseen-layout test.")
        protocol["greedy_scope"] = ("Greedy removes policy sampling; declared action/pose perturbations still vary "
                                    "with eval seed. These are robustness interventions on the same layout.")
    if spawn_jitter or random_facing:
        protocol["initial_state"] = "standard empty kitchen with declared chef pose perturbations"
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    play = make_episode(module, config, env, network, action_selection,
                        action_dropout=action_dropout, random_action_prob=random_action_prob,
                        spawn_jitter=spawn_jitter, random_facing=random_facing)
    export = EvaluationExport(output, env, config, f"trained-{action_selection}", protocol,
        [{"run": run, "config": training_config, "weights": weights}] * env.num_agents)
    rows, recorded = [], None
    for episode, seed, (initial, trace, diagnostics) in iterate_episodes(play, params, protocol["episode_seeds"]):
        row = summarize_episode(initial, trace, diagnostics, env, seed, episode)
        rows.append(row)
        export.add_episode(episode, seed, initial, trace, row, diagnostics)
        print(f"Episode {episode + 1}/{episodes}: {row['correct_dishes']} dishes; "
              f"first={row['first_delivery_step']}; second={row['second_delivery_step']}", flush=True)
        if recorded is None:
            recorded = (initial, trace)
    protocol["initial_conditions"] = [{"seed": r["seed"], "positions": r["initial_positions"],
                                       "facings": r["initial_facings"]} for r in rows]
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for name, value in row.items()})
    result = summarize(rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if replay:
        from scenarios.oc2_sushi_1_3.preview import save_policy_preview
        save_policy_preview(env, *recorded, output, f"trained-{action_selection}", eval_seed)
    export.finish(result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Results: {output}", flush=True)
    return output, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--action-selection", choices=("sampled", "greedy"), default="sampled")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--no-upload", action="store_true", help="Keep the evaluation local.")
    parser.add_argument("--action-dropout", type=float, default=0.,
                        help="Independent probability of replacing each chef action with STAY.")
    parser.add_argument("--random-action-prob", type=float, default=0.,
                        help="Separate sweep: replace each chef action uniformly over all action codes.")
    parser.add_argument("--spawn-jitter", type=int, default=0, choices=(0, 1, 2),
                        help="Initial valid FLOOR pose jitter within this Manhattan radius.")
    parser.add_argument("--random-facing", action="store_true",
                        help="Independently randomize initial facing directions.")
    args = parser.parse_args()
    output, _ = evaluate_run(
        args.run, args.output, args.episodes, args.eval_seed,
        args.action_selection, args.steps, not args.no_replay,
        action_dropout=args.action_dropout, random_action_prob=args.random_action_prob,
        spawn_jitter=args.spawn_jitter, random_facing=args.random_facing,
    )
    if not args.no_upload:
        upload_evaluation(output / "evaluation.json")


if __name__ == "__main__":
    main()
