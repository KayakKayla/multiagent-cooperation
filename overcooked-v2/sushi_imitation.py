"""Train the existing sushi CNN+GRU from audited action demonstrations.

This is behavior cloning, not a PPO result. Evaluation is always a separate,
closed-loop environment run: no teacher action or phase is supplied to the model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

import checkpoint as ckpt
from scenarios import make_env
from train import load_algorithm, load_config, git_state

ROOT = Path(__file__).resolve().parent


def load_demonstrations(path, padded_steps=512):
    """Pack whole trajectories; padding and injected waits receive no loss."""
    files = sorted(Path(path).glob('*.npz')) if Path(path).is_dir() else [Path(path)]
    if not files:
        raise ValueError('No demonstration NPZ files')
    episodes = []
    hashes = {}
    for file in files:
        with np.load(file, allow_pickle=False) as saved:
            obs = saved['observations']
            actions = saved['actions']
            mask = saved['supervision_mask'] if 'supervision_mask' in saved.files else np.ones_like(actions, dtype=np.float32)
            if obs.ndim != 5 or actions.shape != obs.shape[:2] or obs.shape[1] != 2:
                raise ValueError(f'{file}: expected [time,2,height,width,48] observations')
            if mask.shape != actions.shape or len(obs) > padded_steps:
                raise ValueError(f'{file}: invalid mask or trajectory longer than {padded_steps}')
            if (not np.isfinite(obs).all() or not np.isfinite(mask).all()
                    or not np.issubdtype(actions.dtype, np.integer)
                    or not np.isfinite(actions).all() or np.any((actions < 0) | (actions >= 7))):
                raise ValueError(f'{file}: non-finite or invalid actions')
            if np.any((mask < 0) | (mask > 1)) or mask.sum() == 0:
                raise ValueError(f'{file}: invalid supervision weights')
            n = len(obs)
            pad_obs = np.zeros((padded_steps, *obs.shape[1:]), np.float32)
            pad_actions = np.full((padded_steps, 2), 4, np.int32)
            pad_mask = np.zeros((padded_steps, 2), np.float32)
            pad_obs[:n], pad_actions[:n], pad_mask[:n] = obs, actions, mask
            episodes.append((pad_obs, pad_actions, pad_mask, n))
            hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    return {
        'observations': np.stack([e[0] for e in episodes]),
        'actions': np.stack([e[1] for e in episodes]),
        'mask': np.stack([e[2] for e in episodes]),
        'lengths': [e[3] for e in episodes], 'hashes': hashes,
    }


def make_updates(network, config, action_weights):
    """Full-trajectory recurrent cross entropy, masking padding/injected waits."""
    width = config['GRU_HIDDEN_DIM']

    def loss_fn(params, obs, labels, mask):
        # Incoming arrays are [episodes,time,chefs,...]. Preserve chef histories.
        obs = jnp.swapaxes(obs, 0, 1)
        labels, mask = jnp.swapaxes(labels, 0, 1), jnp.swapaxes(mask, 0, 1)
        steps, episodes, chefs = labels.shape
        obs = obs.reshape(steps, episodes * chefs, *obs.shape[3:]).astype(jnp.float32)
        labels, mask = labels.reshape(steps, -1), mask.reshape(steps, -1)
        hidden = jnp.zeros((episodes * chefs, width), jnp.float32)
        resets = jnp.zeros_like(labels).at[0].set(1)
        _, policy, _ = network.apply(params, hidden, (obs, resets))
        logits = policy.logits_parameter()
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        weights = mask * action_weights[labels]
        loss = (nll * weights).sum() / jnp.maximum(weights.sum(), 1)
        correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
        metrics = {'loss': loss, 'accuracy': (correct * mask).sum() / jnp.maximum(mask.sum(), 1),
                   'unweighted_nll': (nll * mask).sum() / jnp.maximum(mask.sum(), 1)}
        # Every supervised action should be learned, including waiting for a partner.
        for action in range(7):
            selected = mask * (labels == action).astype(jnp.float32)
            metrics[f'accuracy_action_{action}'] = (correct * selected).sum() / jnp.maximum(selected.sum(), 1)
        return loss, metrics

    @jax.jit
    def update(state, obs, labels, mask):
        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params, obs, labels, mask)
        return state.apply_gradients(grads=gradients), metrics

    return update, jax.jit(loss_fn)


def save_artifacts(output, state, key, history, config):
    """Reuse atomic full-state checkpoints; keep standard weights for evaluation."""
    update = int(state.step)
    params = jax.device_get(state.params)
    if not all(np.isfinite(x).all() for x in jax.tree.leaves(params)):
        raise ValueError('Non-finite model weights')
    parts = {name: np.asarray([row[name] for row in history], dtype=np.float32) for name in history[0]}
    path = ckpt.save_checkpoint(output, (state, key), [parts], update, config, Path(__file__), keep_last=3)
    temporary = output / 'params.msgpack.tmp'
    temporary.write_bytes(serialization.to_bytes(params))
    os.replace(temporary, output / 'params.msgpack')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/oc2_sushi_check.yaml')
    parser.add_argument('--updates', type=int, default=10000, help='Declared total update budget')
    parser.add_argument('--stop-after', type=int, help='Stop at this absolute update without changing the resume config')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--batch-episodes', type=int, default=1)
    parser.add_argument('--padded-steps', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init-weights', type=Path, help='Warm start a new optimizer from a previous policy; not a resume')
    parser.add_argument('--max-seconds', type=float, default=3600)
    args = parser.parse_args()
    if min(args.updates, args.save_every, args.batch_episodes, args.padded_steps, args.width) < 1 or args.lr <= 0:
        parser.error('All budget/shape settings and learning rate must be positive')
    if args.resume and args.init_weights:
        parser.error('Choose resume or warm-start weights, not both')
    initial_weights = None
    if args.init_weights:
        weight_file = args.init_weights / 'params.msgpack' if args.init_weights.is_dir() else args.init_weights
        initial_weights = weight_file.read_bytes()
    started = time.perf_counter()
    data = load_demonstrations(args.dataset, args.padded_steps)
    config = load_config(args.config, args.seed)
    config.update(TRAINING_METHOD='behavior_cloning', GRU_HIDDEN_DIM=args.width, FC_DIM_SIZE=args.width)
    config['ENV_KWARGS'] = {'layout': 'oc2_sushi_1_3', 'agent_view_size': None, 'rules': {'max_steps': 600}}
    config['RUNTIME'] = {'METAL_COMPATIBILITY': True}
    config['IMITATION'] = {'updates': args.updates, 'lr': args.lr, 'batch_episodes': args.batch_episodes,
                           'padded_steps': args.padded_steps, 'dataset_sha256': data['hashes'],
                           'supervision': 'expert real-action traces; injected waits and padding masked',
                           'observations': 'full_map', 'action_weighting': 'inverse_square_root_frequency',
                           'model_source_sha256': ckpt.file_hash(ROOT / 'algorithms/ippo_rnn.py'),
                           'environment_source_sha256': ckpt.file_hash(ROOT / 'scenarios/oc2_sushi_1_3/environment.py'),
                           'initial_weights_sha256': hashlib.sha256(initial_weights).hexdigest() if initial_weights else None}
    env = make_env(**config['ENV_KWARGS'])
    if tuple(data['observations'].shape[3:]) != env.observation_space().shape:
        raise ValueError('Dataset observation schema does not match the policy')
    module = load_algorithm(config['ALGORITHM'])
    network = module.ActorCriticRNN(env.num_actions, config=config)
    cpu = jax.devices('cpu')[0]
    device = jax.devices()[0]
    key = jax.device_put(jax.random.PRNGKey(args.seed), cpu)
    batch = args.batch_episodes * env.num_agents
    dummy = (jnp.zeros((1, batch, *env.obs_shape), jnp.float32), jnp.ones((1, batch), jnp.int32))
    hidden = jnp.zeros((batch, args.width), jnp.float32)
    # Keep unsupported QR initialization off Metal.
    params = jax.jit(lambda k: network.init(k, hidden, dummy), backend='cpu')(key)
    if initial_weights is not None:
        restored = serialization.from_bytes(params, initial_weights)
        for ref, value in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
            if ref.shape != value.shape or ref.dtype != value.dtype or not np.isfinite(np.asarray(value)).all():
                raise ValueError('Warm-start weights differ in schema or contain non-finite values')
        params = restored
    state = TrainState.create(apply_fn=network.apply, params=params,
                              tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr)))
    state, key = jax.device_put((state, key), device)
    counts = np.bincount(data['actions'][data['mask'] > 0].reshape(-1), minlength=7)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights /= weights.mean()
    update, calculate_loss = make_updates(network, config, jax.device_put(weights.astype(np.float32), device))
    output = args.output.resolve()
    history = []
    if args.resume:
        previous = json.loads((output / 'config.json').read_text())
        # Checkpoints restore the optimizer and policy. Keep the original warm
        # start's provenance without reading or applying its weights again.
        config['IMITATION']['initial_weights_sha256'] = previous['IMITATION'].get('initial_weights_sha256')
        if ckpt.config_hash(previous) != ckpt.config_hash(config):
            raise ValueError('Resume settings/dataset differ from saved run')
        path = ckpt.latest_checkpoint(output)
        if path is None:
            raise FileNotFoundError('No complete checkpoint to resume')
        (state, key), metrics, done = ckpt.load_checkpoint(path, (state, key), config, Path(__file__))
        history = [{name: float(value[i]) for name, value in metrics.items()} for i in range(done)]
        print('Restored update', done, flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (output / 'metadata.json').write_text(json.dumps({
            'upstream_revision': (ROOT / 'UPSTREAM_REVISION').read_text().strip(),
            'backend': jax.default_backend(), 'jax': jax.__version__, **git_state(),
            'parameter_count': sum(x.size for x in jax.tree.leaves(params)),
            'demonstration_lengths': data['lengths'], 'action_counts': counts.tolist(),
            'method': 'behavior cloning; no reward optimization or expert at evaluation',
        }, indent=2) + '\n')
    # Sampling key lives in checkpoint; NumPy only decodes its randomly selected indices.
    stop = min(args.updates, args.stop_after or args.updates)
    print('Backend', device, 'parameters', sum(x.size for x in jax.tree.leaves(params)),
          'demonstrations', len(data['lengths']), 'shape', data['observations'].shape, flush=True)
    while int(state.step) < stop:
        key, sample_key = jax.random.split(key)
        indices = np.asarray(jax.random.randint(sample_key, (args.batch_episodes,), 0, len(data['lengths'])))
        obs = jax.device_put(data['observations'][indices], device)
        labels = jax.device_put(data['actions'][indices], device)
        mask = jax.device_put(data['mask'][indices], device)
        state, metrics = update(state, obs, labels, mask)
        row = {name: float(value) for name, value in jax.device_get(metrics).items()}
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError('Non-finite training metrics')
        history.append(row)
        step = int(state.step)
        if step == 1 or step % 25 == 0:
            print(json.dumps({'update': step, 'seconds': time.perf_counter() - started, **row}), flush=True)
        if step % args.save_every == 0 or step == stop or time.perf_counter() - started >= args.max_seconds:
            path = save_artifacts(output, state, key, history, config)
            print('Saved', path, flush=True)
        if time.perf_counter() - started >= args.max_seconds:
            print('Stopped at requested wall-time bound', flush=True)
            break
    (output / 'summary.json').write_text(json.dumps({
        'status': 'stopped_at_budget', 'updates': int(state.step), 'seconds': time.perf_counter() - started,
        'last_training_metrics': history[-1], 'learning_success': 'not measured here; use evaluate_sushi_policy.py',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
