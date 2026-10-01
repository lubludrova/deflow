"""Preserve failed references; cross-check their roots from two other starts."""

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
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=args.root
    out=root/'numerical'/args.key
    data=json.loads((out/'result.json').read_text())
    inputs=root/'bundle/inputs'/args.key
    engine=audit.load_module(inputs/'engine.py','reference_followup_engine')
    payload=torch.load(inputs/'checkpoint.pt',map_location='cpu',weights_only=False)
    weights=payload['actor'];dim=weights['action_scale'].numel()
    odim=weights['vf.pre.weight'].shape[1]-dim-32
    actors=[]
    for dtype in [torch.float32,torch.float64]:
        a=engine.deq_multistep_flow_actor(odim,dim,weights['action_bias']-weights['action_scale'],
            weights['action_bias']+weights['action_scale'],**data['job']['original_constructor'])
        a.load_state_dict(weights);actors.append(a.to(dtype=dtype).eval())
    actor,actor64=actors
    bank=np.load(out/'inputs.npz')
    obs,noise=torch.from_numpy(bank['obs']),torch.from_numpy(bank['noise'])
    reference=torch.from_numpy(bank['reference']).clone()
    failed=[i for i,r in enumerate(data['root_residuals']) if max(r)>=1e-8]
    followups=[];rows=[]
    for i in failed:
        answers={}
        for label,initial in [('tight100',bank['tight100'][i]),('base',np.repeat(bank['noise'][i,None],actor.T,axis=0))]:
            seq,residuals=audit.reference_root(actor64,obs[i].double(),noise[i].double(),torch.from_numpy(initial).double())
            answers[label]={'sequence':seq.tolist(),'residuals':residuals,'passed':max(residuals)<1e-8}
        delta=float(np.max(np.abs(np.array(answers['tight100']['sequence'])-np.array(answers['base']['sequence']))))
        accepted=all(a['passed'] for a in answers.values()) and delta<1e-7
        followups.append(dict(sample=i,starts=answers,sequence_linf_between_starts=delta,accepted=accepted))
        if not accepted:
            continue
        reference[i]=torch.tensor(answers['tight100']['sequence'])
        ref_lp=audit.density(actor64,obs[i:i+1].double(),noise[i:i+1].double(),reference[i:i+1])['guarded'][0]
        ref_action=reference[i,-1].tanh()*actor64.action_scale+actor64.action_bias
        for arm in ['original','tight100']:
            seq=torch.from_numpy(bank[arm][i:i+1])
            lp=audit.density(actor,obs[i:i+1],noise[i:i+1],seq)['guarded'][0]
            action=seq[0,-1].tanh()*actor.action_scale+actor.action_bias
            rows.append(dict(sample=i,arm=arm,reference_ok=True,

                action_linf=float((action.double()-ref_action).abs().max()),
                logp_abs_nats=float((lp.double()-ref_lp).abs())))
    gradients=[]
    for start in sorted({(i//8)*8 for i in failed}):
        affected=[r for r in followups if start<=r['sample']<start+8]
        if not all(r['accepted'] for r in affected):
            continue
        o,z=obs[start:start+8],noise[start:start+8]
        ref=gradient(actor64,payload,o.double(),z.double(),reference[start:start+8])
        for arm,cap,tol in [('original',25,1e-5),('tight100',100,1e-6)]:
            actor.max_iter_fwd,actor.tol_fwd=cap,tol
            candidate=gradient(actor,payload,o,z)
            gradients.append(dict(start=start,arm=arm,status='ok',reference_norm=float(ref.norm()),
                relative_error=float((candidate-ref).norm()/ref.norm()),
                cosine=float(torch.nn.functional.cosine_similarity(candidate,ref,dim=0))))
    result=dict(original_audit_sha256=audit.digest(out/'result.json'),script_sha256=audit.digest(__file__),
        followups=followups,rows=rows,gradients=gradients,
        scope='Posthoc reference-solver diagnosis only; original failed audit retained. Accept only two converged independent starts agreeing1e-7. No sampler rerun or outcome-dependent model selection; not global uniqueness proof.')
    with (out/'reference_followup.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'key':args.key,'failed_initial':len(failed),'recovered':sum(r['accepted'] for r in followups)}))


if __name__=='__main__':
    main()
