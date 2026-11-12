"""Numerical policy and actual instrumented native actor integration regressions."""
import ast
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import jacrev, vmap
from .precision import apply_policy, assert_policy, assert_float32, constructor_digest
from .run_docking import Recorder, make_actor
from .sequential_newton import NewtonConfig


class PrecisionTests(unittest.TestCase):
    def tearDown(self):
        apply_policy('strict_fp32')

    def test_policy_restores_contaminated_backend(self):
        apply_policy('legacy_tf32')
        with self.assertRaises(RuntimeError):
            assert_policy('strict_fp32')
        state=apply_policy('strict_fp32')
        self.assertEqual(state['matmul_precision'],'highest')
        self.assertFalse(state['matmul_allow_tf32'])
        self.assertFalse(state['cudnn_allow_tf32'])

    def test_overrides_autocast_and_non_fp32_rejected(self):
        for name in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'):
            with patch.dict(os.environ,{name:'0'}):
                with self.assertRaises(RuntimeError):
                    apply_policy('strict_fp32')
        apply_policy('strict_fp32')
        with torch.autocast('cpu'):
            with self.assertRaises(RuntimeError):
                assert_policy('strict_fp32')
        with self.assertRaises(RuntimeError):
            assert_float32('strict_fp32',torch.ones(1,dtype=torch.float64))

    def test_engine_override_point_honors_args(self):
        path=os.environ.get('DEFLOW_ENGINE')
        if not path:
            self.skipTest('DEFLOW_ENGINE required')
        tree=ast.parse(Path(path).read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.If) and 'args.precision_policy' in ast.unparse(n.test))
        end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='env' for t in n.targets))
        fragment=ast.Module(body=main.body[start:end],type_ignores=[])
        for policy in ('strict_fp32','legacy_tf32'):
            apply_policy('legacy_tf32' if policy=='strict_fp32' else 'strict_fp32')
            exec(compile(fragment,path,'exec'),dict(torch=torch,os=os,args=types.SimpleNamespace(precision_policy=policy)))
            assert_policy(policy)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        path=os.environ.get('DEFLOW_ENGINE')
        if not path:
            self.skipTest('DEFLOW_ENGINE required')
        apply_policy('strict_fp32')
        wanted={'sinusoidal_pos_emb','anderson','actor_vector_field_torch','deq_multistep_flow_actor'}
        tree=ast.parse(Path(path).read_text())
        tree.body=[n for n in tree.body if getattr(n,'name','') in wanted]
        self.engine=types.ModuleType('native_test_engine')
        self.engine.__dict__.update(torch=torch,nn=nn,F=F,math=math,jacrev=jacrev,vmap=vmap)
        exec(compile(tree,path,'exec'),self.engine.__dict__)
        self.engine._rebuttal_original_actor=self.engine.deq_multistep_flow_actor
        launch=ast.parse((Path(path).parent/'run_multigoal_docking_full.py').read_text())
        launch.body=[n for n in launch.body if getattr(n,'name','')=='GatedDockingActor']
        ns={'DEFlowActor':self.engine.deq_multistep_flow_actor}
        exec(compile(launch,path,'exec'),ns)
        self.base=ns['GatedDockingActor']
        self.temp=tempfile.TemporaryDirectory()
        self.output=Path(self.temp.name)
        self.recorder=Recorder(self.output,100000,'strict_fp32',types.SimpleNamespace())

    def tearDown(self):
        if hasattr(self,'recorder'):
            self.recorder.close()
            self.temp.cleanup()

    def actor(self,solver='newton',tol=1e-5,cap=25):
        cls=make_actor(self.base,self.engine,solver,NewtonConfig(),self.recorder,tol,cap)
        return cls(3,2,[-1.,-1.],[1.,1.],denoising_steps=2,time_emb_dim=4)

    def test_four_arm_effective_constructor_roundtrip(self):
        for solver,tol,cap in [('newton',1e-5,25),('anderson',1e-5,25),('anderson',1e-5,500),('anderson',1e-6,100)]:
            a=self.actor(solver,tol,cap)
            kwargs=json.loads((self.output/'actor_kwargs.json').read_text())
            self.assertEqual(kwargs['hidden_dim'],256)
            self.assertEqual((a.max_iter_fwd,a.tol_fwd,a.exact_backward),(cap,tol,True))
            self.assertEqual((kwargs['max_iter_fwd'],kwargs['tol_fwd'],kwargs['exact_backward']),(cap,tol,True))
            self.assertEqual(self.recorder.args.rebuttal_actor_kwargs_sha256,constructor_digest(kwargs))
            b=type(a)(**kwargs)
            b.load_state_dict(a.state_dict())
            obs=torch.zeros(2,3)
            z=torch.zeros(2,2)
            a._sample_base_z=b._sample_base_z=lambda *args:z
            first=a.forward_with_logprob(obs)
            second=b.forward_with_logprob(obs)
            for x,y in zip(first,second):
                torch.testing.assert_close(x,y)
            ga=torch.autograd.grad(sum(x.sum() for x in first),list(a.parameters()),allow_unused=True)
            gb=torch.autograd.grad(sum(x.sum() for x in second),list(b.parameters()),allow_unused=True)
            for x,y in zip(ga,gb):
                if x is not None: torch.testing.assert_close(x,y)

    def test_newton_dispatch_native_backward_and_exception_restoration(self):
        a=self.actor()
        original=self.engine.anderson
        def forbidden(*args,**kwargs):
            raise AssertionError('Native forward Anderson dispatched for Newton')
        self.engine.anderson=forbidden
        obs=torch.zeros(2,3)
        a.act(obs,deterministic=True)
        a.act(obs,deterministic=False)
        penalty_action=a.forward_with_jacobian_penalty(obs)
        self.assertTrue(torch.isfinite(penalty_action).all())
        penalty_action.square().sum().backward()
        a.zero_grad(set_to_none=True)
        action,logp=a.forward_with_logprob(obs)
        (action.square().sum()+.1*logp.sum()).backward()
        self.assertEqual(a.last_bwd['n_iter'],0)
        self.assertIs(self.engine.anderson,forbidden)
        self.assertIn('backward',self.recorder.runtime['events'])
        self.assertIn('density_result',self.recorder.runtime['events'])
        with patch('deflow_rebuttal.run_docking.solve_sequential',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                a.act(obs,deterministic=True)
        self.assertIs(self.engine.anderson,forbidden)
        self.engine.anderson=original

    def test_newton_budget_accounting_separates_cap_from_failure(self):
        a=self.actor()
        obs=torch.zeros(2,3)
        z=torch.zeros(2,2)
        q=torch.zeros(2,2,2)
        for status in ('0','1','4'):
            native=dict(cap_hit=True,failed_samples=0 if status=='0' else 1,status_counts={status:4})
            self.recorder.record(a,obs,z,q,native,{},0.,'fixture')
        total=self.recorder.roles['training/fixture']
        self.assertEqual(total['cap_calls'],3)
        self.assertEqual(total['newton_budget_exhausted_calls'],1)
        self.assertEqual(total['newton_status_counts'],{'0':4,'1':4,'4':4})
        self.assertEqual(total['mean_converged_calls'],0)


if __name__=='__main__':
    unittest.main()
