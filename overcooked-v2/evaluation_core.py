"""Shared frozen-policy loading, execution, iteration and portable export adapters."""
import hashlib
import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jaxmarl.environments.overcooked_v2.common import DynamicObject, OvercookedActionsEnum, StaticObject

# Local tools need only the stdlib shared package, never the web dependencies.
SHARED = Path(__file__).resolve().parents[1] / "evals-sharing-website" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
from evals_shared.export import EvaluationWriter
from evals_shared.provenance import device_info, portable_metadata, source_records, policy_run_root

def initialize_policy(module, config, env):
    """Recreate the upstream trainer's exact seed splitting and initial network."""
    network = module.ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    key = jax.random.split(jax.random.PRNGKey(config["SEED"]), 1)[0]
    _, parameter_key = jax.random.split(key)
    # Use the trainer's dummy batch too, so the before policy matches its start.
    batch = config["NUM_ENVS"]
    hidden = module.ScannedRNN.initialize_carry(batch, config["GRU_HIDDEN_DIM"])
    inputs = (
        jnp.zeros((1, batch, *env.observation_space().shape)),
        jnp.zeros((1, batch)),
    )
    params = jax.jit(lambda key: network.init(key, hidden, inputs))(parameter_key)
    return network, params


def check_run_revision(run, revision):
    """Reject saved training results that record a different JaxMARL code version."""
    metadata = json.loads((policy_run_root(run) / "metadata.json").read_text())
    if metadata["upstream_revision"] != revision:
        raise ValueError(f"Saved policy and local upstream revisions differ: {run}")


def policy_config_path(run):
    return policy_run_root(run) / "config.json"


def check_partner_config(config, partner_config):
    """Check compatibility with the evaluator's shared network and observation setup.

    Refactor or remove these restrictions when each policy can use its own network
    design and observation settings. Training seeds and hyperparameters may differ.
    """
    fields = ("ENV_NAME", "ENV_KWARGS", "ACTIVATION", "FC_DIM_SIZE", "GRU_HIDDEN_DIM")
    differences = [field for field in fields if config[field] != partner_config[field]]
    if config.get("ALGORITHM", "ippo_rnn") != partner_config.get("ALGORITHM", "ippo_rnn"):
        differences.append("ALGORITHM")
    if differences:
        raise ValueError("Policies have incompatible settings: " + ", ".join(differences))


def load_checkpoint(run, template):
    params = serialization.from_bytes(template, (run / "params.msgpack").read_bytes())
    for value, expected in zip(jax.tree.leaves(params), jax.tree.leaves(template), strict=True):
        if np.shape(value) != np.shape(expected) or not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Checkpoint has incompatible shapes or non-finite weights: {run}")
    return params


def delivery_counts(before, after, actions):
    """Count actual dish removals at serving stations, including simultaneous ones."""
    forward = jax.vmap(lambda agent: agent.get_fwd_pos())(before.agents)
    at_goal = before.grid[forward.y, forward.x, 0] == StaticObject.GOAL
    inventory = before.agents.inventory
    delivered = (
        (actions == OvercookedActionsEnum.interact)
        & at_goal
        & ((inventory & DynamicObject.COOKED) != 0)
        & (after.agents.inventory == DynamicObject.EMPTY)
    )
    correct_recipe = inventory == (
        before.recipe | DynamicObject.PLATE | DynamicObject.COOKED
    )
    return jnp.sum(delivered & correct_recipe), jnp.sum(delivered & ~correct_recipe)


def validate_perturbations(action_dropout, random_action_prob, spawn_jitter):
    for name, probability in (("action_dropout", action_dropout),
                              ("random_action_prob", random_action_prob)):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(name + " must be a finite probability between zero and one")
    if action_dropout and random_action_prob:
        raise ValueError("Use separate dropout and random-action sweeps")
    if not isinstance(spawn_jitter, int) or not 0 <= spawn_jitter <= 2:
        raise ValueError("spawn_jitter must be an integer radius from zero through two")


def perturb_initial_state(env, initial, seed, spawn_jitter=0, random_facing=False):
    """Jitter only chef poses, preserving the empty kitchen and its resources.

    Positions are sampled jointly and uniformly among valid distinct FLOOR pairs
    within the declared Manhattan radius. Facing uses a separate folded stream.
    """
    from scenarios.oc2_sushi_1_3.environment import Action
    from scenarios.oc2_sushi_1_3.layout import Tile
    pose_key = jax.random.fold_in(jax.random.PRNGKey(seed), 0x53555350)
    if spawn_jitter:
        offsets = jnp.array([(x, y) for y in range(-spawn_jitter, spawn_jitter + 1)
                             for x in range(-spawn_jitter, spawn_jitter + 1)
                             if abs(x) + abs(y) <= spawn_jitter], dtype=jnp.int32)
        candidates = initial.positions[:, None] + offsets[None]
        inside = ((candidates >= 0) & (candidates < jnp.array([env.width, env.height]))).all(axis=-1)
        cells = jnp.clip(candidates[:, :, 1] * env.width + candidates[:, :, 0], 0, env.width * env.height - 1)
        floor = inside & (env.flat_tiles[cells] == Tile.FLOOR)
        distinct = (candidates[0, :, None] != candidates[1, None, :]).any(axis=-1)
        valid = floor[0, :, None] & floor[1, None, :] & distinct
        # The unperturbed valid spawn pair is always among the candidates.
        logits = jnp.where(valid.reshape(-1), 0., -jnp.inf)
        choice = jax.random.categorical(jax.random.fold_in(pose_key, 1), logits)
        count = len(offsets)
        positions = jnp.stack([candidates[0, choice // count], candidates[1, choice % count]])
        initial = initial._replace(positions=positions)
    if random_facing:
        directions = jax.random.randint(jax.random.fold_in(pose_key, 2), (env.num_agents,), 0, 4)
        initial = initial._replace(directions=directions)
    return initial


def make_episode(module, config, env, network, action_selection="sampled", *,
                 action_dropout=0., random_action_prob=0., spawn_jitter=0,
                 random_facing=False, diagnostics=False, record_action_probs=False):
    """Return initial state and trace, then optional diagnostics and probabilities.

    Recording pre-action probabilities does not consume random keys or alter play.
    Default key splitting matches both original evaluation tools.
    """
    if action_selection not in ("sampled", "greedy"):
        raise ValueError("action_selection must be sampled or greedy")
    validate_perturbations(action_dropout, random_action_prob, spawn_jitter)
    sushi = env.name == "oc2_sushi_1_3"
    if (diagnostics or action_dropout or random_action_prob or spawn_jitter or random_facing) and not sushi:
        raise ValueError("Detailed diagnostics and perturbations currently require sushi")
    if sushi:
        from scenarios.oc2_sushi_1_3.environment import Action, Item, QUEUED, RECIPE

    @jax.jit
    def play(params, seed, partner_params=None):
        key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
        observations, initial = env.reset(reset_key)
        if spawn_jitter or random_facing:
            initial = perturb_initial_state(env, initial, seed, spawn_jitter, random_facing)
            observations = env.get_obs(initial)
        perturbation_key = jax.random.fold_in(jax.random.PRNGKey(seed), 0x53555341)
        hidden = module.ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])

        def step(carry, _):
            observations, state, hidden, key = carry
            key, action_key, step_key = jax.random.split(key, 3)
            batch = jnp.stack([observations[agent] for agent in env.agents])
            dones = jnp.full((1, env.num_agents), state.done)
            if partner_params is None:
                hidden, policy, values = network.apply(params, hidden, (batch[None], dones))
            else:
                memories, logits = [], []
                for chef, weights in enumerate((params, partner_params)):
                    inputs = (batch[None, chef:chef+1], dones[:, chef:chef+1])
                    memory, chef_policy, _ = network.apply(weights, hidden[chef:chef+1], inputs)
                    memories.append(memory)
                    logits.append(chef_policy.logits)
                hidden = jnp.concatenate(memories, axis=0)
                policy = module.distrax.Categorical(logits=jnp.concatenate(logits, axis=1))
                values = None
            proposed = policy.sample(seed=action_key)[0] if action_selection == "sampled" else jnp.argmax(policy.probs, axis=-1)[0]
            actions = proposed
            drop_mask = random_mask = jnp.zeros((env.num_agents,), dtype=jnp.bool_)
            if action_dropout or random_action_prob:
                noise_key = jax.random.fold_in(perturbation_key, state.time)
                if action_dropout:
                    drop_mask = jax.random.bernoulli(jax.random.fold_in(noise_key, 1), action_dropout, (env.num_agents,))
                    actions = jnp.where(drop_mask, int(Action.STAY), actions)
                if random_action_prob:
                    random_mask = jax.random.bernoulli(jax.random.fold_in(noise_key, 2), random_action_prob, (env.num_agents,))
                    random_actions = jax.random.randint(jax.random.fold_in(noise_key, 3), (env.num_agents,), 0, env.num_actions)
                    actions = jnp.where(random_mask, random_actions, actions)
            observations, after, rewards, _, info = env.step_env(step_key, state, dict(zip(env.agents, actions)))
            correct, wrong = (after.deliveries-state.deliveries, jnp.int32(0)) if sushi else delivery_counts(state, after, actions)
            reward = rewards[env.agents[0]]
            finite = jnp.all(jnp.isfinite(policy.probs))
            trace = (after, actions, reward, correct, wrong, finite)
            if diagnostics:
                finite = finite & jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(hidden))
                normalized = jnp.all(jnp.abs(policy.probs.sum(axis=-1)-1) < 1e-5)
                delivered = ((state.kind == Item.PLATE) & (state.contents == RECIPE)
                             & (after.kind == Item.DIRTY_PLATE) & (after.location == QUEUED))
                washed = (state.kind == Item.DIRTY_PLATE) & (after.kind == Item.PLATE)
                trace = (after, actions, reward, correct, wrong, finite & normalized)
                result = (trace, (delivered, washed, info["invalid_action"], info["blocked"], proposed, drop_mask, random_mask))
            else:
                result = trace
            if record_action_probs:
                result = (*result, policy.probs[0]) if diagnostics else (result, policy.probs[0])
            return (observations, after, hidden, key), result

        _, result = jax.lax.scan(step, (observations, initial, hidden, key), None, env.max_steps)
        return (initial, *result) if diagnostics or record_action_probs else (initial, result)
    return play


def iterate_episodes(play, params, seeds, partner_params=None):
    """Transfer/release one native trajectory at a time; memory resets in play."""
    for episode, seed in enumerate(seeds):
        result = play(params, np.uint32(seed)) if partner_params is None else play(params, np.uint32(seed), partner_params)
        yield episode, seed, jax.device_get(result)


class EvaluationExport:
    """Shared per-episode serialization and protocol/provenance composition."""
    def __init__(self, output, env, config, mode, protocol, policy_sources):
        self.writer = EvaluationWriter(Path(output) / "evaluation.json")
        self.env, self.config, self.mode, self.protocol = env, config, mode, protocol
        self.runs, self.checkpoints, self.participants = {}, {}, []
        for index, (agent, source) in enumerate(zip(env.agents, policy_sources, strict=True)):
            if source.get("initialization"):
                source = dict(source, identity_directory=source.get("identity_directory", output))
            run, checkpoint = source_records(**source)
            previous = self.runs.get(run["source_run_id"])
            if previous and {key: value for key, value in previous.items() if key not in ("name", "notes")} != {key: value for key, value in run.items() if key not in ("name", "notes")}:
                raise ValueError("Conflicting training evidence for one recorded source-run identity")
            previous_checkpoint = self.checkpoints.get(checkpoint["source_checkpoint_id"])
            if previous_checkpoint and previous_checkpoint != checkpoint:
                raise ValueError("Conflicting metadata for the same exact source checkpoint")
            self.runs[run["source_run_id"]] = run
            self.checkpoints[checkpoint["source_checkpoint_id"]] = checkpoint
            self.participants.append({"agent_id": agent, "display_order": index,
                "source_checkpoint_id": checkpoint["source_checkpoint_id"],
                "observation_settings": {"agent_view_size": config["ENV_KWARGS"].get("agent_view_size"),
                                         "shape": list(env.observation_space().shape)},
                "action_settings": {key: protocol[key] for key in ("action_selection", "action_dropout", "random_action_prob") if key in protocol}})
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.identity = "evaluation:" + str(uuid4())
        self.renderer = {}

    def add_episode(self, episode, seed, initial, trace, metrics, diagnostics=None):
        if self.env.name == "oc2_sushi_1_3":
            from scenarios.oc2_sushi_1_3.preview import build_policy_replay_data
            recording = build_policy_replay_data(self.env, initial, trace, self.mode, seed, episode)
        else:
            from preview import build_replay_data
            states, actions, rewards, correct, wrong, _ = trace
            states = jax.tree.map(lambda first, rest: np.concatenate(([first], rest)), initial, states)
            recording = build_replay_data(states, actions, rewards, correct, wrong, self.config, self.mode, seed, episode=episode)
        if diagnostics is not None and len(diagnostics) > 4:
            proposed, dropped, randomized = diagnostics[4:]
            for number, step in enumerate(recording["steps"][1:]):
                for index, agent in enumerate(step["agents"]):
                    agent.update(proposed_action_code=int(proposed[number, index]),
                                 executed_action_code=int(trace[1][number, index]),
                                 action_dropout=bool(dropped[number, index]), random_action=bool(randomized[number, index]))
        if "initial_state" in self.protocol:
            recording["metadata"]["starts"] = self.protocol["initial_state"]
        self.renderer = recording["metadata"].get("renderer", {})
        metrics = dict(metrics)
        metrics["team_reward"] = metrics.get("team_reward", metrics.get("raw_team_reward"))
        metrics.setdefault("wrong_dishes", 0)
        self.writer.add_episode({"index": episode, "seed": int(seed), "steps": len(trace[1]),
            "termination": {"done": bool(trace[0].done[-1]), "reason": "configured_horizon"},
            "metrics": {key: value for key, value in metrics.items() if key not in ("episode", "seed", "steps")}, "recording": recording})

    def finish(self, summary):
        import checkpoint as ckpt
        from train import ALGORITHMS, git_state
        protocol = portable_metadata(self.protocol)
        protocol.update(recording_coverage="all_episodes_full_resolution", state_action_convention="steps[0] initial; steps[t] after attached action t")
        if self.env.name == "oc2_sushi_1_3":
            from scenarios.oc2_sushi_1_3.environment import Action
            action_enum = Action
        else:
            action_enum = OvercookedActionsEnum
        sources = {"evaluation_core_sha256": ckpt.file_hash(Path(__file__)),
                   "environment_sha256": ckpt.file_hash(Path(inspect.getfile(type(self.env)))),
                   "algorithm_sha256": ckpt.file_hash(ALGORITHMS[self.config["ALGORITHM"]])}
        result = self.writer.finish({"source_evaluation_id": self.identity, "created_at": self.created_at,
            "title": f"{self.config['ENV_KWARGS']['layout']} · {self.mode}", "notes": "",
            "training_runs": list(self.runs.values()), "checkpoints": list(self.checkpoints.values()),
            "participants": self.participants,
            "environment": {"name": self.config["ENV_NAME"], "layout": self.config["ENV_KWARGS"]["layout"],
                            "config": portable_metadata(self.config["ENV_KWARGS"]), "renderer": self.renderer,
                            "action_meanings": {str(int(action)): action.name for action in action_enum}},
            "protocol": protocol, "summary": summary,
            "provenance": {"upstream_revision": self.protocol.get("upstream_revision"),
                           "training_checkpoint_provenance": {key: value.get("provenance", {}) for key, value in self.checkpoints.items()},
                           **git_state(), "sources": {**sources, **{key: value for key, value in protocol.items() if "sha256" in key and "policy" not in key}}},
            "device": device_info(backend=jax.default_backend(), devices=jax.devices())})
        print(f"Upload JSON: {result['path']} ({result['bytes']:,} bytes; " + ("within 20 MiB limit)" if result['within_upload_limit'] else "EXCEEDS 20 MiB upload limit; no episodes discarded)"), flush=True)
        return result
