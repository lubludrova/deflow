"""Generate a labeled reconstructed observation/noise bank from a historical policy.

Not historical replay or an original training minibatch. Episodes include failures.
Uses the reviewed original evaluator's policy reconstruction and fixed origins.
"""
import argparse
import inspect
from pathlib import Path
import sys
import hashlib
import torch
from .precision import apply_policy, assert_policy, assert_float32, constructor_digest, source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision-policy',choices=['strict_fp32','legacy_tf32'],required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--seed',type=int,default=26152)
    p.add_argument('--device',default='cpu')
    c=p.parse_args()
    apply_policy(c.precision_policy)
    if c.output.exists(): raise FileExistsError(c.output)
    sys.path.insert(0,str(c.repo.resolve()/'src/docking_full'))
    import evaluate_multigoal_docking_full as ev
    actor,method,sigma,seed,env,step,saved=ev.load_actor(c.checkpoint,torch.device(c.device))
    if method!='deflow': raise ValueError('Input preparation expects a historical DEFlow checkpoint')
    assert_float32(c.precision_policy,*actor.parameters())
    original=actor.act; observed=[]
    def record(obs,deterministic=False):
        assert_float32(c.precision_policy,obs)
        observed.append(obs.detach().cpu().clone())
        return original(obs,deterministic=deterministic)
    actor.act=record
    torch.manual_seed(c.seed)
    episodes=ev.rollout(actor,method,env,torch.device(c.device),c.episodes,c.seed)
    pool=torch.cat(observed)
    generator=torch.Generator().manual_seed(c.seed+1)
    indexes=torch.randperm(len(pool),generator=generator)[:c.batch_size]
    if len(indexes)!=c.batch_size: raise ValueError('Too few visited states. Increase episodes, do not silently duplicate.')
    kwargs={k:v.default for k,v in inspect.signature(ev.launcher.DEFlowActor.__init__).parameters.items()
            if k!='self' and v.default is not inspect.Parameter.empty}
    for k in list(kwargs):
        if hasattr(actor,k): kwargs[k]=getattr(actor,k)
    kwargs.update(obs_dim=pool.shape[1],act_dim=actor.act_dim,act_low=(actor.action_bias-actor.action_scale).cpu(),
                  act_high=(actor.action_bias+actor.action_scale).cpu(),hidden_dim=256,denoising_steps=actor.T,
                  gate_bias=saved['gate_bias'],u_scale=saved['u_scale'])
    z=torch.randn(c.batch_size,actor.act_dim,generator=generator)*actor.z_std
    c.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(obs=pool[indexes],z=z,actor_kwargs=kwargs,metadata=dict(kind='reconstructed_full_batch_not_historical_replay',
        precision_policy=c.precision_policy,runtime_precision=assert_policy(c.precision_policy),package_sha256=source_hashes(),
        checkpoint_sha256=hashlib.sha256(c.checkpoint.read_bytes()).hexdigest(),seed=c.seed,pool_size=len(pool),
        source_sha256=ev.launcher.source_hashes(),episodes=episodes,selected_pool_indexes=indexes.tolist())),c.output)
    print(c.output)

if __name__=='__main__': main()
