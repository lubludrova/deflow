"""Optional native parity test without simulator dependencies.

Set DEFLOW_ENGINE to the engine used for runs. This extracts its actual actor
AST unchanged, avoiding unrelated gym/tensorboard imports in this unit test.
"""
import ast
import json
import numpy as np
import math
import os
import types
import unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import jacrev, vmap
from .audit import readout, solve, reference, local_residual, NewtonConfig, clean


class ConvergenceStatusTests(unittest.TestCase):
    def test_last_iteration_can_converge(self):
        from .run_docking import anderson_status
        status = anderson_status({'res_last': 1e-6, 'n_iter': 25}, 1e-5, 25)
        self.assertTrue(status['mean_criterion_met'])
        self.assertTrue(status['iteration_limit_reached'])
        self.assertFalse(status['budget_exhausted_without_convergence'])

    def test_failed_and_nonfinite_are_not_convergence(self):
        from .run_docking import anderson_status
        for residual in (1e-3, float('nan'), float('inf'), None):
            status = anderson_status({'res_last': residual, 'n_iter': 500}, 1e-5, 500)
            self.assertFalse(status['mean_criterion_met'])
            self.assertTrue(status['budget_exhausted_without_convergence'])


class SerializationTests(unittest.TestCase):
    def test_historical_constructor_metadata_strict_json(self):
        data = {'act_low': torch.tensor([-1., -1.]),
                'act_high': np.array([1., 1.], dtype=np.float32),
                'counter': np.int64(8), 'values': (torch.tensor(float('nan')), np.float64(float('inf'))),
                'path': Path('/tmp/fixture')}
        recovered = json.loads(json.dumps(clean(data), allow_nan=False))
        self.assertEqual(recovered, {'act_low': [-1., -1.], 'act_high': [1., 1.],
                                   'counter': 8, 'values': [None, None], 'path': '/tmp/fixture'})


class AuditTests(unittest.TestCase):
    def setUp(self):
        path = os.environ.get('DEFLOW_ENGINE')
        if not path:
            self.skipTest('Set DEFLOW_ENGINE for native-source parity tests')
        wanted = {'sinusoidal_pos_emb', 'anderson', 'actor_vector_field_torch', 'deq_multistep_flow_actor'}
        source = ast.parse(Path(path).read_text())
        source.body = [node for node in source.body if getattr(node, 'name', '') in wanted]
        namespace = dict(torch=torch, nn=nn, F=F, math=math, jacrev=jacrev, vmap=vmap)
        exec(compile(source, path, 'exec'), namespace)
        self.engine = types.SimpleNamespace(**namespace)
        torch.manual_seed(13)
        self.actor = self.engine.deq_multistep_flow_actor(3, 2, torch.full((2,),-1.),torch.ones(2),
            hidden_dim=8, time_emb_dim=4, denoising_steps=2, exact_backward=True,
            tol_fwd=1e-6, max_iter_fwd=40).double()
        self.obs, self.z = torch.randn(3,3,dtype=torch.float64), torch.randn(3,2,dtype=torch.float64)

    def test_common_readout_matches_native_action_density_gradient(self):
        a, obs, z = self.actor,self.obs,self.z
        a._sample_base_z = lambda *args:z
        action, logp = a.forward_with_logprob(obs)
        loss = (action.square().sum(-1)+.1*logp).mean()
        gs = torch.autograd.grad(loss,list(a.parameters()),allow_unused=True)
        gradient = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                              for p,g in zip(a.parameters(),gs)])
        seq,_ = solve(self.engine,a,obs,z,'anderson',NewtonConfig())
        audited = readout(a,obs,z,seq)
        torch.testing.assert_close(audited['action'], action)
        torch.testing.assert_close(audited['logp'], logp)
        torch.testing.assert_close(audited['gradient'], gradient)

    def test_native_high_cap_early_stop_and_unconverged_limit(self):
        from .run_docking import anderson_status
        initial = torch.zeros(3, 2, dtype=torch.float64)
        for cap in (25, 100, 500):
            _, native = self.engine.anderson(lambda x: x+1, initial,
                max_iter=cap, tol=1e-12, return_stats=True)
            status = anderson_status(native, 1e-12, cap)
            self.assertEqual(native['n_iter'], cap)
            self.assertTrue(status['budget_exhausted_without_convergence'])
        _, native = self.engine.anderson(lambda x: torch.ones_like(x), initial,
            max_iter=500, tol=1e-5, return_stats=True)
        status = anderson_status(native, 1e-5, 500)
        self.assertTrue(status['mean_criterion_met'])
        self.assertLess(native['n_iter'], 500)

    def test_reference_is_validated_by_equations(self):
        seq, info = reference(self.actor,self.obs,self.z)
        self.assertTrue(bool(info['valid'].all()))
        absolute,_ = local_residual(self.actor,self.obs,self.z,seq)
        self.assertLess(float(absolute.max()), 1e-9)
        self.assertTrue(bool((info['sigma_min']>0).all()))

    def test_candidate_newton_refinement(self):
        seq,_=solve(self.engine,self.actor,self.obs,self.z,'newton',NewtonConfig(rtol=1e-10,atol=1e-12,max_iter=30))
        absolute,_=local_residual(self.actor,self.obs,self.z,seq)
        self.assertLess(float(absolute.max()),1e-8)


if __name__=='__main__':
    unittest.main()
