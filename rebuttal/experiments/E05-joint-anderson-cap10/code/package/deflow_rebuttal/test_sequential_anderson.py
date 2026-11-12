"""Numerical and integration checks against the actual unchanged E01 engine."""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from .precision import apply_policy
from .run_docking import Recorder, make_actor
from .sequential_anderson import solve_sequential_anderson
from .sequential_newton import NewtonConfig


class SequentialAndersonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(os.environ['DEFLOW_ENGINE']).parent))
        cls.launch = importlib.import_module('run_multigoal_docking_full')
        cls.engine = cls.launch.engine
        cls.engine._rebuttal_original_actor = cls.launch.DEFlowActor

    def setUp(self):
        apply_policy('strict_fp32')
        torch.manual_seed(17)

    def solve(self, field, obs, z, **kwargs):
        return solve_sequential_anderson(field, obs, z, anderson=self.engine.anderson, **kwargs)

    def test_affine_closed_form_and_work(self):
        z = torch.tensor([[.5, -.3], [1.2, 2.]], dtype=torch.float64)
        obs = torch.tensor([[.3], [.7]], dtype=z.dtype)
        matrix = torch.tensor([[-.8, .3], [.1, -.4]], dtype=z.dtype)
        q, native, raw = self.solve(lambda x,o,t: x @ matrix.T + o + t,
                                    obs, z, steps=4, tol=1e-10, max_iter=100)
        previous = z
        for k in range(4):
            previous = torch.linalg.solve(torch.eye(2, dtype=z.dtype)-.25*matrix,
                                           (previous+.25*(obs+(k+1)*.25)).T).T
            torch.testing.assert_close(q[:,k], previous, atol=1e-8, rtol=1e-8)
        self.assertTrue(native['mean_criterion_met'])
        self.assertEqual(raw['nfe_total'], len(z)*native['iterations_sum'])
        self.assertEqual(raw['jacobian_evals_total'], 0)
        self.assertFalse(q.requires_grad)

    def test_one_step_is_native_kernel(self):
        z = torch.randn(7,2)
        field = lambda x,o,t: -.2*x + o
        kwargs = dict(m=4, lam=.02, max_iter=25, tol=1e-5, beta=.6)
        q, native, raw = self.solve(field, z, z, steps=1, **kwargs)
        expected, stats = self.engine.anderson(lambda x: z+field(x,z,None), z,
                                               return_stats=True, **kwargs)
        # CPU lstsq may round differently on repeated identical calls.
        torch.testing.assert_close(q[:,0], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(native['per_step'][0]['n_iter'], stats['n_iter'])
        self.assertEqual(raw['nfe_total'], len(z)*stats['n_iter'])

    def test_failure_at_early_step_remains_visible(self):
        z = torch.ones(2,1,dtype=torch.float64)
        q, native, raw = self.solve(lambda x,o,t: torch.where(t < .75, 2*x+2, torch.zeros_like(x)),
                                    z, z, steps=2, max_iter=3, tol=1e-10)
        self.assertFalse(native['per_step'][0]['mean_criterion_met'])
        self.assertTrue(native['per_step'][1]['mean_criterion_met'])
        self.assertFalse(native['mean_criterion_met'])
        self.assertTrue(native['cap_hit'])
        self.assertTrue(native['budget_exhausted_without_convergence'])
        self.assertEqual(raw['nfe_total'], 12)

    def test_init_is_previous_root_and_history_is_reset(self):
        initial = torch.ones(2,2)
        starts = []
        def kernel(f, x, **kwargs):
            starts.append(x.clone())
            return f(x), dict(n_iter=3, res_last=0.)
        q, _, _ = solve_sequential_anderson(lambda x,o,t: torch.ones_like(x), initial, initial,
                                            steps=4, anderson=kernel)
        for k, start in enumerate(starts):
            torch.testing.assert_close(start, initial if k==0 else q[:,k-1])

    def test_native_backward_and_density_at_identical_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(Path(tmp), 1000)
            actor_type = make_actor(self.launch.GatedDockingActor, self.engine,
                                    'sequential_anderson', NewtonConfig(), recorder)
            actor = actor_type(3,2,torch.full((2,),-1.),torch.ones(2), denoising_steps=4)
            obs, z = torch.randn(4,3), torch.randn(4,2)
            actor._sample_base_z = lambda *args: z
            original = self.engine.anderson
            q, _, _ = self.solve(actor.vf,obs,z,steps=actor.T,dt=actor.dt,
                                  m=actor.m_fwd,lam=actor.lam_fwd,beta=actor.beta_fwd)
            action, logp = actor.forward_with_logprob(obs)
            loss = (action.square().sum(-1) + .1*logp).mean()
            gradient = torch.autograd.grad(loss, tuple(actor.parameters()))
            self.assertIs(self.engine.anderson, original)
            plain = self.launch.GatedDockingActor(**actor.audit_actor_kwargs)
            plain.load_state_dict(actor.state_dict())
            plain._sample_base_z = lambda *args: z
            try:
                self.engine.anderson = lambda *args, **kwargs: (q.flatten(1), {'n_iter':0,'res_last':0.})
                action2, logp2 = plain.forward_with_logprob(obs)
                grad2 = torch.autograd.grad((action2.square().sum(-1)+.1*logp2).mean(), tuple(plain.parameters()))
            finally:
                self.engine.anderson = original
                recorder.close()
            torch.testing.assert_close(action,action2)
            torch.testing.assert_close(logp,logp2)
            for first, second in zip(gradient,grad2):
                torch.testing.assert_close(first,second)
                self.assertTrue(torch.isfinite(first).all())


if __name__ == '__main__':
    unittest.main()
