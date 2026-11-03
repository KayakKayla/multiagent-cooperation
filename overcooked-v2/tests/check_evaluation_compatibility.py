"""Short frozen-policy equivalence check against the pre-refactor source.

Run from overcooked-v2 with PYTHONPATH=. and JAX_PLATFORMS=cpu.
Uses tiny horizons and fresh weights; performs no optimizer update/training.
The reference Git revision must be present locally.
"""
import copy, types, subprocess, sys
from pathlib import Path
import jax
import numpy as np
from train import load_baseline
from scenarios import make_env
import evaluate
import evaluate_sushi_policy as sushi
from evaluation_core import EvaluationExport
from tempfile import TemporaryDirectory

def old(name):
    source=subprocess.check_output(["git","show",'4cd193f1b039bac943226cc0f4f9f51a83d8d858:overcooked-v2/'+name+".py"],text=True)
    module=types.ModuleType("old_"+name); module.__file__=str(Path(name+".py").resolve())
    exec(compile(source,module.__file__,"exec"),module.__dict__)
    return module

def equal(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))

module,config=load_baseline(Path("configs/oc2_sushi_check.yaml"),None)
env=make_env("oc2_sushi_1_3",rules={"max_steps":4},agent_view_size=2)
network,params=evaluate.initialize_policy(module,config,env)
original=old("evaluate"); old_sushi=old("evaluate_sushi_policy")
for seed in (1000,1001):
    equal(original.make_episode(module,config,env,network)(params,seed),evaluate.make_episode(module,config,env,network)(params,seed))
for options in ({},{"action_dropout":.25},{"random_action_prob":.4},{"spawn_jitter":1,"random_facing":True}):
    for mode in ("sampled","greedy"):
        equal(old_sushi.make_episode(module,config,env,network,mode,**options)(params,1001),sushi.make_episode(module,config,env,network,mode,**options)(params,1001))
config=copy.deepcopy(config);config["ENV_NAME"]="overcooked_v2";config["ENV_KWARGS"]={"layout":"coord_ring","max_steps":3,"agent_view_size":2,"random_agent_positions":True}
env=make_env(env_name=config["ENV_NAME"],**config["ENV_KWARGS"])
network,params=evaluate.initialize_policy(module,config,env)
for partner in (None,params):
    equal(original.make_episode(module,config,env,network)(params,1000,partner),evaluate.make_episode(module,config,env,network)(params,1000,partner))
from flax import serialization
with TemporaryDirectory() as tmp:
    protocol={"episode_seeds":[1000,1001],"action_selection":"sample","upstream_revision":Path("UPSTREAM_REVISION").read_text().strip()}
    export=EvaluationExport(tmp,env,config,"untrained",protocol,[{"run":None,"config":config,"weights":serialization.to_bytes(params),"initialization":True}]*2)
    for index,seed in enumerate(protocol["episode_seeds"]):
        initial,trace=jax.device_get(evaluate.make_episode(module,config,env,network)(params,seed))
        row={"team_reward":float(trace[2].sum()),"correct_dishes":int(trace[3].sum()),"wrong_dishes":int(trace[4].sum())}
        export.add_episode(index,seed,initial,trace,row)
    export.finish({"episodes":2,"mean_team_reward":0,"total_correct_dishes":0,"total_wrong_dishes":0})
print("PASS: 12 original-versus-refactored rollout comparisons + real general all-episode export")
