"""Bounded CPU regression tests for preflight classification and parity."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import jacrev, vmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'package'))
spec = importlib.util.spec_from_file_location('numerical_preflight', HERE / 'numerical_preflight.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
from deflow_rebuttal import audit, precision


class BoundaryTests(unittest.TestCase):
    def case(self, fresh, reference):
        q = torch.ones(1, 1, 2)
        z = torch.ones(1, 2)
        tau = preflight.tolerance(q, z).item()
        residual = torch.tensor([[[fresh * tau, 0.]]])
        residual64 = torch.tensor([[[reference * tau, 0.]]], dtype=torch.float64)
        return preflight.classify_boundary(torch.tensor([[.999 * tau]]), torch.zeros(1, 1, dtype=torch.long),
                                            {'fresh': residual}, residual64, q, z)

    def test_exact_masks_retained_for_explained_boundary(self):
        result = self.case(1.001, 1.0005)
        self.assertTrue(result['explained_boundary'].item())
        self.assertFalse(result['exact_pass_masks']['fresh'].item())
        self.assertTrue(result['exact_pass_masks']['stored_native'].item())
        self.assertFalse(result['unexplained_native_success'].item())

    def test_large_precision_discrepancy_blocks(self):
        result = self.case(2., 1.5)
        self.assertTrue(result['unexplained_native_success'].item())
        self.assertFalse(result['explained_boundary'].item())

    def test_agreement_needs_no_boundary_waiver(self):
        result = self.case(.5, .5)
        self.assertFalse(result['disagreement'].item())

    def test_hard_failure_is_not_reclassified_as_rounding(self):
        q, z = torch.ones(1, 1, 2), torch.ones(1, 2)
        result = preflight.classify_boundary(torch.ones(1, 1), torch.full((1, 1), 4),
                                             {'fresh': torch.ones(1, 1, 2)}, torch.ones(1, 1, 2).double(), q, z)
        self.assertFalse(result['explained_boundary'].item())


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        precision.apply_policy('strict_fp32')
        source_path = HERE.parent / 'DEFlow/src/docking_full/sac_anderson_flow.py'
        source = ast.parse(source_path.read_text())
        wanted = {'sinusoidal_pos_emb', 'anderson', 'actor_vector_field_torch', 'deq_multistep_flow_actor'}
        source.body = [node for node in source.body if getattr(node, 'name', '') in wanted]
        namespace = dict(torch=torch, nn=nn, F=F, math=math, jacrev=jacrev, vmap=vmap)
        exec(compile(source, str(source_path), 'exec'), namespace)
        cls.engine = types.SimpleNamespace(**namespace)
        # The compiled native method resolves anderson in this dictionary.
        cls.namespace = namespace

    def actor(self):
        torch.manual_seed(18)
        return self.engine.deq_multistep_flow_actor(3, 2, -torch.ones(2), torch.ones(2),
            hidden_dim=8, time_emb_dim=4, denoising_steps=4, exact_backward=True)

    def test_all_layouts_and_full_jacobian_fd(self):
        actor = self.actor().double()
        obs, z = torch.randn(8, 3).double(), torch.randn(8, 2).double()
        q, _ = audit.solve(self.engine, actor, obs, z, 'newton', audit.NewtonConfig())
        residuals = preflight.residual_layouts(actor, obs, z, q)
        for value in residuals.values():
            torch.testing.assert_close(value, residuals['flattened'], atol=1e-12, rtol=1e-9)
        result = preflight.finite_difference_gate(actor, obs, q, audit)
        self.assertTrue(result['pass_all'])
        self.assertEqual(result['absolute_error'].shape, (3, 8, 4))

    def test_full1024_native_and_audit_sac_backward(self):
        actor = self.actor()
        obs, z = torch.randn(1024, 3), torch.randn(1024, 2)
        q, _ = audit.solve(self.engine, actor, obs, z, 'newton', audit.NewtonConfig())
        critics = [nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 1)) for _ in range(2)]
        payload = {name: {f'net.{k}': v for k, v in net.state_dict().items()}
                   for name, net in zip(('q1', 'q2'), critics)}
        payload['loss_config'] = dict(alpha=.2, logp_lower=-100., logp_upper=100., effective_lam_jac=0.)
        # Forward global lookup must see the native_readout fixed-q hook.
        class EngineProxy:
            def __getattr__(self, key):
                return NativeTests.namespace[key]
            def __setattr__(self, key, value):
                NativeTests.namespace[key] = value
        result = preflight.parity_gate(EngineProxy(), actor, obs, z, q, payload, audit)
        self.assertTrue(result['pass_all'], result['gradient_relative'])
        self.assertEqual(result['native']['action'].shape, (1024, 2))

    def test_independent_trace_reports_real_line_search_failure(self):
        class Field:
            dt = 1.
            def vf(self, x, obs, times):
                return x - x ** 3 + 1 - obs
        previous = torch.tensor([.1], dtype=torch.float64)
        trace = preflight.trace_local(Field(), previous, previous, 0, audit.NewtonConfig(max_backtracks=0))
        self.assertEqual(trace['status'], 'line_search_failed')
        torch.testing.assert_close(trace['final'], previous)
        self.assertFalse(trace['trace'][0]['trials'][0]['accepted'])


if __name__ == '__main__':
    unittest.main()
