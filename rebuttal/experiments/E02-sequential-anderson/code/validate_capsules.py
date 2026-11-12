"""Exclusive-GPU fixed-input diagnostics, without selecting successful rows."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'package'))
sys.path.insert(0, str(HERE/'DEFlow/src/docking_full'))
from deflow_rebuttal import audit, precision
from deflow_rebuttal.sequential_anderson import solve_sequential_anderson
from deflow_rebuttal.sequential_newton import NewtonConfig, solve_sequential
import run_multigoal_docking_full as launch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    precision.apply_policy('strict_fp32')
    torch.set_num_threads(1)
    engine = launch.engine
    report = dict(runtime=precision.snapshot(), device=torch.cuda.get_device_name(),
                  source_sha256=launch.source_hashes(), package_sha256=precision.source_hashes(),
                  timing='exclusive GPU, one warmup and three synchronized forward samples, no telemetry inside timer',
                  reference='FP64 sequential Newton cap100, independent previous-root starts, residual and singular-value masks',
                  limitations='Reference is diagnostic, not branch uniqueness. No finite-program derivative claim.',
                  capsules=[])
    for path in sorted(args.inputs.glob('*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['precision_policy'] == 'strict_fp32'
        assert payload['source_sha256'] == launch.source_hashes()
        assert payload['metadata']['role'] == 'training/actor_grad'
        assert len(payload['obs']) == 1024
        actor = launch.DEFlowActor(**payload['actor_kwargs']).cuda().eval()
        actor.load_state_dict(payload['actor'])
        obs, z = payload['obs'].cuda(), payload['z'].cuda()
        critics = []
        import sac_flow_parent
        for name in ('q1','q2'):
            critic = sac_flow_parent.parent_critic(obs.shape[1], z.shape[1]).cuda().eval()
            critic.load_state_dict(payload[name])
            critics.append(lambda inputs, net=critic: net(inputs[:,:obs.shape[1]], inputs[:,obs.shape[1]:]))
        reference_actor = copy.deepcopy(actor).double()
        qref, rstats = solve_sequential(reference_actor.vf, obs.double(), z.double(), steps=actor.T,
                                        dt=actor.dt, config=NewtonConfig(rtol=1e-10,atol=1e-12,max_iter=100))
        rabs, _ = audit.local_residual(reference_actor, obs.double(), z.double(), qref)
        jac = audit.jacobians(reference_actor, obs.double(), qref)
        singular = torch.linalg.svdvals(torch.eye(z.shape[1],device=z.device,dtype=torch.float64)-actor.dt*jac)[...,-1]
        valid = rstats['all_converged'] & (rabs < 1e-8).all(1) & (singular > 1e-8).all(1)
        refout = audit.readout(actor,obs,z,qref.float(),critics,payload['loss_config'])
        entry = dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     metadata=payload['metadata'],reference_valid=int(valid.sum()),samples=len(z),methods={})
        raw = dict(reference=qref.cpu(),reference_valid=valid.cpu(),reference_residual=rabs.cpu(),
                   reference_sigma_min=singular.detach().cpu(),methods={})
        for method in ('anderson','newton','sequential_anderson'):
            def solve():
                if method == 'sequential_anderson':
                    q, native, work = solve_sequential_anderson(actor.vf,obs,z,steps=actor.T,dt=actor.dt,
                        anderson=engine.anderson,m=actor.m_fwd,lam=actor.lam_fwd,
                        max_iter=25,tol=1e-5,beta=actor.beta_fwd)
                    return q,dict(native=native,work=work)
                return audit.solve(engine,actor,obs,z,method,NewtonConfig())
            solve()
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                q, native = solve()
                torch.cuda.synchronize()
                times.append(time.perf_counter()-started)
            absolute, _ = audit.local_residual(actor,obs,z,q)
            prev = torch.cat((z[:,None],q[:,:-1]),1)
            failed = ~torch.isfinite(absolute) | (absolute > 1e-7+1e-5*torch.maximum(q.norm(dim=-1),prev.norm(dim=-1)))
            out = audit.readout(actor,obs,z,q,critics,payload['loss_config'])
            finite = all(bool(torch.isfinite(x).all()) for x in [q,*out.values()])
            entry['methods'][method] = dict(finite=finite,seconds=times,
                failed_samples=int(failed.any(1).sum()),failed_local_steps=int(failed.sum()),
                local_residual=audit.stats(absolute),trajectory_error_valid=audit.stats((q.double()-qref).norm(dim=-1)[valid]),
                action_error_valid=audit.stats((out['action']-refout['action']).norm(dim=-1)[valid]),
                logp_error_valid=audit.stats((out['logp']-refout['logp']).abs()[valid]),
                full_batch_sac_loss=float(out['loss']),
                full_batch_sac_gradient_error_vs_reference=float((out['gradient']-refout['gradient']).norm()/refout['gradient'].norm().clamp_min(1e-20)),
                gradient_scope='full batch including reference-invalid rows, diagnostic only')
            raw['methods'][method] = dict(trajectory=q.cpu(),native=audit.clean(native),
                failed=failed.cpu(),absolute=absolute.cpu(),readout={k:v.cpu() for k,v in out.items()})
        torch.save(raw,args.output/(path.stem+'_raw.pt'))
        report['capsules'].append(entry)
        (args.output/'report.json').write_text(json.dumps(audit.clean(report),indent=2,allow_nan=False)+'\n')
        print(json.dumps({'capsule':path.name,'reference_valid':int(valid.sum()),'methods':entry['methods']}),flush=True)
    report['implementation_gate_pass'] = len(report['capsules']) == 2 and all(
        m['finite'] for c in report['capsules'] for m in c['methods'].values())
    (args.output/'report.json').write_text(json.dumps(audit.clean(report),indent=2,allow_nan=False)+'\n')
    assert report['implementation_gate_pass'], 'Nonfinite solver/readout/gradient'


if __name__ == '__main__':
    main()
