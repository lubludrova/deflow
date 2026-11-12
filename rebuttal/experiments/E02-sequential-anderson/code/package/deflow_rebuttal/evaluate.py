"""Evaluate a new-run checkpoint using its recorded solver, stochastic coverage."""
import argparse
from pathlib import Path
import json
import hashlib
import sys
import torch
from .run_docking import Recorder, make_actor, write_json
from .sequential_newton import NewtonConfig
from .precision import apply_policy, assert_policy, constructor_digest, source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--eval-seed',type=int,default=1)
    p.add_argument('--device',default='cuda')
    c=p.parse_args()
    sys.path.insert(0,str(c.repo.resolve()/'src/docking_full'))
    import evaluate_multigoal_docking_full as ev
    launch=ev.launcher; engine=launch.engine
    manifest=json.loads((c.run/'manifest.json').read_text())
    policy=manifest['precision_policy']
    apply_policy(policy)
    if manifest['source_sha256']!=launch.source_hashes(): raise ValueError('Training source mismatch')
    payload=torch.load(c.checkpoint,map_location='cpu',weights_only=False)
    if payload['args'].get('rebuttal_solver')!=manifest['solver']: raise ValueError('Checkpoint solver mismatch')
    if payload['args'].get('seed')!=manifest['configuration']['seed']: raise ValueError('Seed mismatch')
    if payload['args'].get('rebuttal_newton')!=manifest['newton']: raise ValueError('Newton settings mismatch')
    for key,value in manifest['configuration'].items():
        if payload['args'].get(key)!=value: raise ValueError(f'Checkpoint configuration mismatch: {key}')
    for name in ('sequential_newton.py','sequential_anderson.py','run_docking.py','precision.py','evaluate.py'):
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=manifest['package_sha256'][name]:
            raise ValueError(f'Solver package changed after training: {name}')
    anderson=manifest.get('anderson',dict(tol=1e-5,cap=25))
    if payload['args'].get('rebuttal_anderson',dict(tol=1e-5,cap=25))!=anderson:
        raise ValueError('Anderson settings mismatch')
    constructor=json.loads((c.run/'actor_kwargs.json').read_text())
    if constructor_digest(constructor)!=payload['args'].get('rebuttal_actor_kwargs_sha256'):
        raise ValueError('Effective constructor metadata mismatch')
    if payload['args'].get('precision_policy')!=policy:
        raise ValueError('Checkpoint precision policy mismatch')
    c.output.mkdir(parents=True,exist_ok=False)
    recorder=Recorder(c.output,1000,policy)
    recorder.engine_hashes=launch.source_hashes()
    engine._rebuttal_original_actor=launch.DEFlowActor
    actor_type=make_actor(launch.GatedDockingActor,engine,manifest['solver'],NewtonConfig(**manifest['newton']),recorder,anderson['tol'],anderson['cap'])
    actor=actor_type(**constructor).to(c.device).eval()
    actor.load_state_dict(payload['actor'])
    torch.manual_seed(c.eval_seed)
    try:
        episodes=ev.rollout(actor,'deflow',manifest['configuration']['env_id'],torch.device(c.device),c.episodes,c.eval_seed)
        write_json(c.output/'evaluation.json',dict(precision_policy=policy,runtime_precision=assert_policy(policy),
            package_sha256=source_hashes(),source_sha256=launch.source_hashes(),actor_kwargs_sha256=constructor_digest(constructor),solver=manifest['solver'],checkpoint_sha256=hashlib.sha256(c.checkpoint.read_bytes()).hexdigest(),
            checkpoint=str(c.checkpoint),episodes=episodes,summary=ev.summarize(episodes),eval_seed=c.eval_seed))
    finally: recorder.close()

if __name__=='__main__': main()
