"""Validate a recovered reference locally using fresh parameter-perturbed solves."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import manipulation_numerical_verification as audit
from audit_manipulation_solver_reeval import gradient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--key',required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.set_defaults(seed=1)
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=args.root
    out=root/'numerical'/args.key
    data=json.loads((out/'result.json').read_text())
    followup=json.loads((out/'reference_followup.json').read_text())
    inputs=root/'bundle/inputs'/args.key
    engine=audit.load_module(inputs/'engine.py','branch_validation_engine')
    payload=torch.load(inputs/'checkpoint.pt',map_location='cpu',weights_only=False)
    w=payload['actor'];dim=w['action_scale'].numel();odim=w['vf.pre.weight'].shape[1]-dim-32
    actors=[]
    for dtype in [torch.float32,torch.float64]:
        actor=engine.deq_multistep_flow_actor(odim,dim,w['action_bias']-w['action_scale'],
             w['action_bias']+w['action_scale'],**data['job']['original_constructor'])
        actor.load_state_dict(w);actors.append(actor.to(dtype=dtype).eval())
    actor,actor64=actors
    bank=np.load(out/'inputs.npz')
    obs,noise=torch.from_numpy(bank['obs']),torch.from_numpy(bank['noise'])
    reference=torch.from_numpy(bank['reference']).clone()
    recovered=[]
    for r in followup['followups']:
        if r['starts']['tight100']['passed']:
            i=r['sample'];reference[i]=torch.tensor(r['starts']['tight100']['sequence'])
            recovered.append(i)
    checks=[];gradients=[];rows=[]
    theta=torch.nn.utils.parameters_to_vector(actor64.parameters()).detach().clone()
    rng=torch.Generator().manual_seed(args.seed)
    q1,q2=audit.critics(payload,torch.float64)
    def objective(o,z,seq):
        parts=audit.density(actor64,o,z,seq)
        lp=parts['guarded'];cap=payload['args'].get('logp_clip',0.)
        if cap>0: lp=lp.clamp(-cap,cap)
        action=seq[:,-1].tanh()*actor64.action_scale+actor64.action_bias
        inputs=torch.cat([o,action],dim=-1)
        value=(payload['log_alpha'].exp().double()*lp-torch.minimum(q1(inputs),q2(inputs)).squeeze(-1)).mean()
        norms=torch.linalg.matrix_norm(actor64.dt*parts['j'],ord=2)
        return float(value+payload['args']['lam_jac']*torch.relu(norms-actor64.jac_sigma_target).square().sum(-1).mean())
    for start in sorted({(i//8)*8 for i in recovered}):
        o,z=obs[start:start+8].double(),noise[start:start+8].double()
        seq=reference[start:start+8]
        grad=gradient(actor64,payload,o,z,seq)
        for direction_id in range(2):
            direction=torch.randn(theta.shape,generator=rng,dtype=torch.float64)
            direction/=direction.norm()
            expected=float(grad@direction)
            for epsilon in [1e-4,1e-5,1e-6]:
                losses=[];residuals=[]
                for sign in [-1,1]:
                    torch.nn.utils.vector_to_parameters(theta+sign*epsilon*direction,actor64.parameters())
                    perturbed=[]
                    for i in range(len(o)):
                        solution,residual=audit.reference_root(actor64,o[i],z[i],seq[i])
                        perturbed.append(solution);residuals.extend(residual)
                    losses.append(objective(o,z,torch.stack(perturbed)))
                observed=(losses[1]-losses[0])/(2*epsilon)
                error=abs(observed-expected)/max(abs(expected),abs(observed),1e-12)
                checks.append(dict(start=start,direction=direction_id,epsilon=epsilon,
                    analytic=expected,finite_difference=observed,relative_error=error,
                    max_root_residual=max(residuals),passed=error<1e-3 and max(residuals)<1e-8))
                torch.nn.utils.vector_to_parameters(theta.clone(),actor64.parameters())
        for label,cap,tol in [('original',25,1e-5),('tight100',100,1e-6)]:
            actor.max_iter_fwd,actor.tol_fwd=cap,tol
            value=gradient(actor,payload,o.float(),z.float())
            gradients.append(dict(start=start,arm=label,status='ok',reference_norm=float(grad.norm()),
                relative_error=float((value-grad).norm()/grad.norm()),
                cosine=float(torch.nn.functional.cosine_similarity(value,grad,dim=0))))
    for i in recovered:
        o,z=obs[i:i+1],noise[i:i+1]
        lp_ref=audit.density(actor64,o.double(),z.double(),reference[i:i+1])['guarded'][0]
        action_ref=reference[i,-1].tanh()*actor64.action_scale+actor64.action_bias
        for arm in ['original','tight100']:
            seq=torch.from_numpy(bank[arm][i:i+1])
            lp=audit.density(actor,o,z,seq)['guarded'][0]
            action=seq[0,-1].tanh()*actor.action_scale+actor.action_bias
            rows.append(dict(sample=i,arm=arm,reference_ok=True,

                action_linf=float((action.double()-action_ref).abs().max()),
                logp_abs_nats=float((lp.double()-lp_ref).abs())))
    result=dict(recovered_samples=recovered,checks=checks,rows=rows,gradients=gradients,
        accepted=bool(checks) and all(c['passed'] for c in checks),
        reference_followup_sha256=audit.digest(out/'reference_followup.json'),script_sha256=audit.digest(__file__),
        limitation='Local differentiable-branch validation only, not global uniqueness. Original-initialization failure and failed two-start-agreement gate retained; recovered root initialized from tighter Anderson and independently solved inFP64.')
    with (out/'local_branch_validation.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'key':args.key,'accepted':result['accepted'], 'max_gradient_fd_relative_error':max((c['relative_error'] for c in checks), default=None)}))


if __name__=='__main__':
    main()
