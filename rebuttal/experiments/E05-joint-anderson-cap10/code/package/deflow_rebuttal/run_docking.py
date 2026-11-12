"""Fresh wide-tight docking training with joint/sequential Anderson or sequential Newton.

Run --help for commands. Default is a plan, --execute explicitly starts work.
The original engine is imported from --repo and is never edited.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
import torch
import numpy as np
from .sequential_newton import NewtonConfig, solve_sequential
from .sequential_anderson import solve_sequential_anderson
from .precision import assert_policy, assert_float32, constructor_digest, source_hashes


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not __import__('math').isfinite(value):
        return None
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False)+'\n')


def quantiles(x):
    x = x.detach().flatten().double()
    finite = x[torch.isfinite(x)]
    return dict(nonfinite=int((~torch.isfinite(x)).sum()), count=x.numel(),
                p50=float(finite.median()) if len(finite) else None,
                p95=float(torch.quantile(finite,.95)) if len(finite) else None,
                max=float(finite.max()) if len(finite) else None)


def anderson_status(native, tolerance, cap):
    """Native stopping uses batch mean, not every sample's residual."""
    residual = native.get('res_last')
    converged = residual is not None and __import__('math').isfinite(residual) and residual < tolerance
    limit_reached = native['n_iter'] >= cap
    return dict(tolerance=tolerance, iteration_limit=cap,
                mean_criterion_met=converged, iteration_limit_reached=limit_reached,
                budget_exhausted_without_convergence=limit_reached and not converged,
                stop_reason='mean_tolerance' if converged else 'iteration_limit',
                convergence_scope='native batch mean, not all samples')


class Recorder:
    def __init__(self, output, every, policy="strict_fp32", args=None):
        self.output, self.every = output, every
        self.policy, self.args = policy, args
        self.runtime = dict(policy=policy, events={})
        self.stream = (output/'solver_calls.jsonl').open('x')
        self.count = 0
        self.roles = {}
        self.tail_saved = 0

    def observe(self, event, *tensors):
        assert_float32(self.policy, *tensors)
        if event not in self.runtime['events']:
            self.runtime['events'][event] = dict(actual=assert_policy(self.policy),
                dtypes=sorted({str(t.dtype) for t in tensors}),
                devices=sorted({str(t.device) for t in tensors}))
            write_json(self.output/'runtime_precision.json', self.runtime)

    def record(self, actor, obs, z, q, native, raw, seconds, role):
        self.observe('solver_result', obs, z, q)
        self.count += 1
        frame = inspect.currentframe()
        step = None
        context = {}
        evaluation = False
        try:
            while frame:
                evaluation = evaluation or frame.f_code.co_name in ('eval_policy','rollout')
                if frame.f_code.co_name == 'main' and 'global_step' in frame.f_locals:
                    step = int(frame.f_locals['global_step'])
                    context = frame.f_locals.copy()
                    break
                frame = frame.f_back
        finally:
            del frame
        role = ('evaluation/' if evaluation else 'training/') + role
        with torch.no_grad():
            B,T,d = q.shape
            previous = torch.cat((z[:,None],q[:,:-1]),1)
            times = torch.arange(1,T+1,device=q.device,dtype=q.dtype)[None,:,None].expand(B,T,1)*actor.dt
            values = actor.vf(q.reshape(B*T,d),obs[:,None].expand(B,T,obs.shape[1]).reshape(B*T,-1),times.reshape(B*T,1)).reshape(B,T,d)
            residual = q-previous-actor.dt*values
            absolute = residual.norm(dim=-1)
            scale = torch.maximum(q.norm(dim=-1),previous.norm(dim=-1))
            failed = ~torch.isfinite(absolute) | (absolute > actor.audit_atol+actor.audit_rtol*scale)
            samples_failed = failed.any(1)
        totals = self.roles.setdefault(role,dict(calls=0,samples=0,local_steps=0,failed_samples=0,failed_local_steps=0,cap_calls=0,mean_converged_calls=0,unconverged_budget_calls=0,newton_status_counts={},newton_converged_samples=0,newton_budget_exhausted_calls=0))
        totals['calls'] += 1
        totals['samples'] += B
        totals['local_steps'] += B*T
        totals['failed_samples'] += int(samples_failed.sum())
        totals['failed_local_steps'] += int(failed.sum())
        cap = bool(native.get('cap_hit',native.get('n_iter',0)>=actor.max_iter_fwd))
        totals['cap_calls'] += int(cap)
        totals['mean_converged_calls'] += int(native.get('mean_criterion_met',False))
        if actor.solver_name in ('anderson', 'sequential_anderson'):
            totals['unconverged_budget_calls'] += int(native.get('budget_exhausted_without_convergence',False))
        else:
            for status, count in native['status_counts'].items():
                totals['newton_status_counts'][status] = totals['newton_status_counts'].get(status, 0) + count
            totals['newton_converged_samples'] += B - native['failed_samples']
            totals['newton_budget_exhausted_calls'] += int(native['status_counts'].get('1', 0) > 0)
        entry = dict(call=self.count,global_step=step,role=role,batch=B,steps=T,
                     solver=actor.solver_name,solver_seconds=seconds,cap_hit=cap,
                     common_abs_residual=quantiles(absolute),failed_samples=int(samples_failed.sum()),
                     failed_local_steps=int(failed.sum()),native=native,
                     work=({k:v for k,v in raw.items() if k.endswith('_total')} if raw else {}),
                     telemetry_extra_field_positions=B*T,totals=totals.copy())
        self.stream.write(json.dumps(clean(entry),allow_nan=False)+'\n')
        if self.count % 100 == 0: self.stream.flush()
        periodic = self.count == 1 or self.count % self.every == 0
        tail = bool(samples_failed.any()) and self.tail_saved < 12
        if periodic or tail:
            if tail: self.tail_saved += 1
            payload = dict(obs=obs.detach().cpu(),z=z.detach().cpu(),trajectory=q.detach().cpu(),
                           local_residual=residual.cpu(),actor_kwargs=actor.audit_actor_kwargs,
                           actor= {k:v.detach().cpu() for k,v in actor.state_dict().items()},
                           precision_policy=self.policy, runtime_precision=self.runtime,
                           source_sha256=getattr(actor,'audit_source_sha256',{}), package_sha256=source_hashes(),
                           actor_kwargs_sha256=constructor_digest(actor.audit_actor_kwargs),
                           metadata=entry,per_step=clean(raw.get('per_step',{})) if raw else {},
                           sampling='periodic' if periodic else 'first_12_failing_calls')
            for name in ('q1','q2','q1_targ','q2_targ'):
                if name in context:
                    payload[name] = {k:v.detach().cpu() for k,v in context[name].state_dict().items()}
            for name in ('b_obs','b_act','b_next_obs','b_rew','b_done','log_alpha'):
                if name in context: payload[name] = context[name].detach().cpu()
            if 'args' in context:
                args = context['args']
                payload['args'] = vars(args).copy()
                payload['loss_config'] = dict(alpha=float(context['log_alpha'].detach().exp()),
                    logp_lower=context.get('actor_rail_lower',-float('inf')),
                    logp_upper=context.get('actor_rail_upper',float('inf')),
                    effective_lam_jac=args.lam_jac*min(1.,step/max(1,args.jac_warmup)))
            torch.save(payload,self.output/f'capsule_{self.count:08d}.pt')
        write_json(self.output/'telemetry_summary.json',dict(calls=self.count,roles=self.roles)) if periodic else None
        if not torch.isfinite(q).all():
            self.stream.flush()
            raise FloatingPointError('Nonfinite solver trajectory, capsule preserved')

    def close(self):
        write_json(self.output/'telemetry_summary.json',dict(calls=self.count,roles=self.roles))
        self.stream.close()


def make_actor(base, engine, solver, config, recorder, anderson_tol=1e-5, anderson_cap=25):
    class InstrumentedActor(base):
        def __init__(self,*args,**kwargs):
            recorder.observe("actor_initialization")
            super().__init__(*args,**kwargs)
            self.tol_fwd, self.max_iter_fwd = anderson_tol, anderson_cap
            self.solver_name = solver
            self.audit_rtol,self.audit_atol=config.rtol,config.atol
            bound = inspect.signature(engine._rebuttal_original_actor.__init__).bind(self,*args,**{**kwargs,'hidden_dim':256,'max_iter_fwd':anderson_cap,'tol_fwd':anderson_tol,'exact_backward':True})
            bound.apply_defaults()
            self.audit_actor_kwargs = clean({k:v for k,v in bound.arguments.items() if k!='self'})
            write_json(recorder.output/'actor_kwargs.json',self.audit_actor_kwargs)
            if recorder.args is not None:
                recorder.args.rebuttal_actor_kwargs_sha256 = constructor_digest(self.audit_actor_kwargs)
            recorder.runtime['actor_kwargs_sha256'] = constructor_digest(self.audit_actor_kwargs)
            write_json(recorder.output/'runtime_precision.json', recorder.runtime)
            self.audit_source_sha256 = getattr(recorder, 'engine_hashes', {})
            self._role = 'unknown'

        def _forward_core(self,obs,need_logprob,need_jacobian_penalty=False):
            self._role = 'actor_grad' if torch.is_grad_enabled() else ('target_or_eval_logp' if need_logprob else 'rollout_or_eval_action')
            recorder.observe('actor_forward', obs, *self.parameters())
            if need_logprob or need_jacobian_penalty:
                recorder.observe('density_entry', obs)
            result = super()._forward_core(obs,need_logprob,need_jacobian_penalty)
            if need_logprob or need_jacobian_penalty:
                outputs = result if isinstance(result, tuple) else (result,)
                recorder.observe('density_result', *(value for value in outputs if value is not None))
            recorder.stream.write(json.dumps(clean(dict(event='readout',call=recorder.count,
                attachment=self.last_fwd,density_evaluated=bool(need_logprob or need_jacobian_penalty),
                density=self.last_diag if need_logprob or need_jacobian_penalty else None)),allow_nan=False)+'\n')
            return result

        def act(self,obs,deterministic=False):
            if not deterministic: return super().act(obs,deterministic=False)
            with torch.no_grad():
                self._role='deterministic_action'
                z=torch.zeros((len(obs),self.act_dim),device=obs.device,dtype=obs.dtype)
                q=self._solve_zstar_with_implicit_grad(obs,self._make_a0_from_z(z))
                return torch.tanh(q.reshape(-1,self.T,self.act_dim)[:,-1])*self.action_scale+self.action_bias

        def _solve_zstar_with_implicit_grad(self,obs,a0):
            recorder.observe("solver_entry", obs, a0, *self.parameters())
            original = engine.anderson
            if not self.exact_backward: raise ValueError('Shared native dense backward is required')
            def forward_only(f,x0,**kwargs):
                if x0.device.type=='cuda': torch.cuda.synchronize(x0.device)
                started=time.perf_counter()
                raw=None
                if solver=='newton':
                    trajectory,raw=solve_sequential(self.vf,obs,a0[:,:self.act_dim],steps=self.T,dt=self.dt,config=config)
                    q=trajectory.flatten(1)
                    ps=raw['per_step']
                    native=dict(n_iter=int(ps['iterations'].max()),cap_hit=bool(ps['cap_hit'].any()),
                                failed_samples=int((~raw['all_converged']).sum()),
                                status_counts={str(k):int((ps['status']==k).sum()) for k in raw['status_names']},
                                backtracks_total=int(ps['backtracks'].sum()))
                elif solver=='sequential_anderson':
                    trajectory,native,raw=solve_sequential_anderson(
                        self.vf,obs,a0[:,:self.act_dim],steps=self.T,dt=self.dt,
                        anderson=original,**{k:v for k,v in kwargs.items() if k!='return_stats'})
                    q=trajectory.flatten(1)
                else:
                    evaluations=0
                    def counted(x):
                        nonlocal evaluations
                        evaluations += x.shape[0]*self.T
                        return f(x)
                    q,native=original(counted,x0,**kwargs)
                    native.update(anderson_status(native,self.tol_fwd,self.max_iter_fwd))
                    raw={'nfe_total':evaluations,'jacobian_evals_total':0,'linear_solves_total':0}
                    native['anderson_history_lstsq_count']=max(0,native.get('n_iter',2)-2)
                    trajectory=q.reshape(-1,self.T,self.act_dim)
                if x0.device.type=='cuda': torch.cuda.synchronize(x0.device)
                elapsed=time.perf_counter()-started
                recorder.record(self,obs,a0[:,:self.act_dim],trajectory,native,raw,elapsed,self._role)
                # Full structured diagnostics are already preserved by Recorder.
                # The native engine sends last_fwd entries to TensorBoard add_scalar.
                return q,{k:v for k,v in native.items() if isinstance(v,(int,float)) or k=='res_curve'}
            engine.anderson=forward_only
            try:
                result = super()._solve_zstar_with_implicit_grad(obs,a0)
                if result.requires_grad:
                    def record_backward(grad):
                        recorder.observe('backward', grad)
                        return grad
                    result.register_hook(record_backward)
                return result
            finally:
                engine.anderson=original
    return InstrumentedActor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--solver',choices=['newton','anderson','sequential_anderson'],default='newton')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--precision-policy',choices=['strict_fp32'],required=True)
    p.add_argument('--steps',type=int,default=200000)
    p.add_argument('--rtol',type=float,default=1e-5)
    p.add_argument('--atol',type=float,default=1e-7)
    p.add_argument('--max-iter',type=int,default=25)
    p.add_argument('--anderson-tol',type=float,default=1e-5)
    p.add_argument('--anderson-cap',type=int,default=25)
    p.add_argument('--capsule-every',type=int,default=1000)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--execute',action='store_true')
    c=p.parse_args()
    if not __import__('math').isfinite(c.anderson_tol) or c.anderson_tol<=0 or c.anderson_cap<3:
        p.error('Anderson tolerance must be finite/positive and cap at least3')
    if c.capsule_every<1: p.error('capsule-every must be positive')
    if c.plan and c.execute: p.error('Choose plan or execute')
    source=c.repo.resolve()/'src/docking_full'
    sys.path.insert(0,str(source))
    import run_multigoal_docking_full as launch
    engine=launch.engine
    engine._rebuttal_original_actor=launch.DEFlowActor
    output=c.output.resolve()
    arm=c.solver
    if c.solver=='anderson' and (c.anderson_tol!=1e-5 or c.anderson_cap!=25):
        arm=f'anderson_tol{c.anderson_tol:g}_cap{c.anderson_cap}'
    run_id=f'{arm}_wide_tight_s{c.seed}'
    args=launch.configure('deflow','0.03',c.seed,256 if c.smoke else c.steps,c.smoke,run_id)
    args.precision_policy=c.precision_policy
    args.density_diag_every=100
    args.density_diag_batch=32
    args.rebuttal_solver=c.solver
    args.rebuttal_anderson=dict(tol=c.anderson_tol,cap=c.anderson_cap)
    args.rebuttal_newton=asdict(NewtonConfig(rtol=c.rtol,atol=c.atol,max_iter=c.max_iter))
    if c.smoke:
        args.learning_starts=64
        args.batch_size=8
        args.eval_interval=0
        args.buffer_size=512
    cfg=NewtonConfig(rtol=c.rtol,atol=c.atol,max_iter=c.max_iter)
    expected=json.loads((Path(__file__).parent.parent/'engine_sha256.json').read_text())
    if launch.source_hashes()!=expected: raise RuntimeError('Engine source differs from reviewed snapshot. Review/revalidate before changing hash manifest.')
    manifest=dict(precision_policy=c.precision_policy,run_id=run_id,solver=c.solver,anderson=args.rebuttal_anderson,newton=asdict(cfg),configuration=asdict(args),
                  source_sha256=launch.source_hashes(),package_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')},
                  torch=torch.__version__,device='selected by original engine, CUDA required outside smoke',
                  scope='fresh-only, native mixed-point shifted dense backward, no solver differentiation',
                  telemetry='All forward calls, grouped by role; synchronous diagnostic overhead included in training elapsed. Not clean timing.',
                  resume='Unsupported by docking engine. Preserve failure, no automatic retry.',output=str(output))
    if not c.execute:
        print(json.dumps(manifest,indent=2)); return
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.json',manifest)
    recorder=Recorder(output,c.capsule_every,c.precision_policy,args)
    recorder.engine_hashes=launch.source_hashes()
    engine.deq_multistep_flow_actor=make_actor(launch.GatedDockingActor,engine,c.solver,cfg,recorder,c.anderson_tol,c.anderson_cap)
    import os
    old=Path.cwd()
    os.chdir(output)
    started=time.time()
    try:
        engine.main(args)
        checkpoint=output/'runs'/run_id/f'ckpt_{args.total_timesteps}.pt'
        # The engine names checkpoints with zero padding. Verify actual contents.
        found=[]
        for f in (output/'runs'/run_id).glob('*.pt'):
            state=torch.load(f,map_location='cpu',weights_only=False)
            actual=state.get('global_step',-1)
            if args.total_timesteps<=actual<args.total_timesteps+args.num_envs: found.append(dict(path=str(f),global_step=actual))
        if not found: raise RuntimeError('Engine returned without expected-horizon checkpoint')
        write_json(output/'DONE.json',dict(exit_code=0,elapsed_seconds=time.time()-started,checkpoints=found))
    except BaseException:
        write_json(output/'FAILED.json',dict(elapsed_seconds=time.time()-started,traceback=traceback.format_exc()))
        raise
    finally:
        recorder.close()
        os.chdir(old)

if __name__=='__main__': main()
