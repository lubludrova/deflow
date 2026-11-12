"""Run with python -m unittest deflow_rebuttal.test_solver -v."""
import unittest
import torch
from .sequential_newton import NewtonConfig, solve_sequential


class SequentialNewtonTests(unittest.TestCase):
    def test_linear_affine_matches_closed_form_and_exact_budget(self):
        dtype = torch.float64
        matrix = torch.tensor([[-2., .3], [.1, -1.]], dtype=dtype)
        z = torch.tensor([[.5, -.3], [1.2, 2.]], dtype=dtype)
        obs = torch.tensor([[.3], [.7]], dtype=dtype)
        def field(x, o, t):
            return x @ matrix.T + o + t
        trajectory, stats = solve_sequential(field, obs, z, steps=4,
            config=NewtonConfig(atol=1e-12, rtol=1e-12, max_iter=1))
        previous = z
        for k in range(4):
            previous = torch.linalg.solve(torch.eye(2, dtype=dtype)-.25*matrix,
                (previous+.25*(obs+(k+1)*.25)).T).T
            torch.testing.assert_close(trajectory[:,k], previous, atol=1e-12, rtol=1e-12)
        self.assertTrue(stats['all_converged'].all())
        self.assertTrue(stats['per_step']['cap_hit'].all())
        self.assertEqual(stats['nfe_total'], 2*4*3)
        self.assertEqual(stats['jacobian_evals_total'], 8)

    def test_armijo_damps_cubic_and_reports_exhaustion(self):
        z = torch.tensor([[.1]], dtype=torch.float64)
        obs = z.clone()
        def field(x, o, t):
            return x - x**3 + 1 - o
        cfg = NewtonConfig(atol=1e-12, rtol=1e-12, max_iter=30)
        trajectory, stats = solve_sequential(field, obs, z, steps=1, config=cfg)
        torch.testing.assert_close(trajectory[:,0], torch.ones_like(z), atol=1e-11, rtol=1e-11)
        self.assertTrue(stats['all_converged'].all())
        self.assertLess(float(stats['per_step']['accepted_damping_min'][0,0]), 1.)
        self.assertGreater(int(stats['per_step']['backtracks'][0,0]), 0)
        failed, failure = solve_sequential(field, obs, z, steps=1,
            config=NewtonConfig(max_backtracks=0))
        torch.testing.assert_close(failed[:,0], z)
        self.assertEqual(int(failure['per_step']['status'][0,0]), 4)
        self.assertFalse(bool(failure['all_converged'][0]))

    def test_singular_nonfinite_and_good_rows_do_not_contaminate(self):
        z = torch.tensor([[1.], [1.], [1.]], dtype=torch.float64)
        obs = torch.tensor([[0.], [1.], [2.]], dtype=torch.float64)
        # Avoid undefined masked Jacobians: define smooth per-row coefficients.
        def field(x, o, t):
            coefficient = torch.where(o == 0, -torch.ones_like(o), torch.ones_like(o))
            bias = torch.where(o == 1, torch.ones_like(o), torch.zeros_like(o))
            bias = torch.where(o == 2, torch.full_like(o, float('nan')), bias)
            return coefficient*x+bias
        trajectory, stats = solve_sequential(field, obs, z, steps=1)
        self.assertEqual(stats['per_step']['status'][:,0].tolist(), [0,3,2])
        self.assertAlmostEqual(float(trajectory[0,0,0]), .5)
        self.assertTrue(torch.isfinite(trajectory).all())
        self.assertEqual(stats['per_step']['jacobian_evals'][:,0].tolist(), [1,1,0])

    def test_zero_field_initial_convergence_and_detached_output(self):
        z = torch.ones((3,2), dtype=torch.float64, requires_grad=True)
        obs = torch.zeros((3,1), dtype=torch.float64)
        parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        trajectory, stats = solve_sequential(lambda x,o,t: parameter*x, obs, z, steps=4)
        torch.testing.assert_close(trajectory, z[:,None].expand(-1,4,-1))
        self.assertTrue(stats['all_converged'].all())
        self.assertEqual(stats['nfe_total'], 12)
        self.assertEqual(stats['jacobian_evals_total'], 0)
        self.assertFalse(trajectory.requires_grad)
        self.assertIsNone(parameter.grad)
        self.assertIsNone(z.grad)

    def test_iteration_exhaustion_has_finite_last_accepted_iterate(self):
        z = torch.tensor([[.1]], dtype=torch.float64)
        trajectory, stats = solve_sequential(lambda x,o,t: x-x**3+1-o, z, z, steps=1,
            config=NewtonConfig(max_iter=1, rtol=1e-12, atol=1e-12))
        self.assertEqual(int(stats['per_step']['status'][0,0]), 1)
        self.assertTrue(stats['per_step']['cap_hit'].all())
        self.assertGreater(float(trajectory[0,0,0]), .1)
        self.assertTrue(torch.isfinite(trajectory).all())

    def test_batch_permutation_and_single_sample_equivalence(self):
        z = torch.tensor([[.1], [2.], [1.]], dtype=torch.float64)
        field = lambda x,o,t: x-x**3+1-o
        cfg = NewtonConfig(rtol=1e-11, atol=1e-12)
        batched, stats = solve_sequential(field, z, z, steps=1, config=cfg)
        for i in range(3):
            one, one_stats = solve_sequential(field, z[i:i+1], z[i:i+1], steps=1, config=cfg)
            torch.testing.assert_close(batched[i:i+1], one)
            for key in stats['per_step']:
                torch.testing.assert_close(stats['per_step'][key][i:i+1], one_stats['per_step'][key])
        permutation = torch.tensor([2,0,1])
        permuted, perm_stats = solve_sequential(field, z[permutation], z[permutation], steps=1, config=cfg)
        torch.testing.assert_close(permuted, batched[permutation])
        torch.testing.assert_close(perm_stats['all_converged'], stats['all_converged'][permutation])

    def test_failed_early_step_is_not_erased_by_later_convergence(self):
        z = torch.ones((1,1), dtype=torch.float64)
        def field(x, o, t):
            return torch.where(t < .75, 2*x+1, -x)
        _, stats = solve_sequential(field, z, z, steps=2)
        self.assertEqual(stats['per_step']['status'][0].tolist(), [3,0])
        self.assertFalse(bool(stats['all_converged'][0]))

    def test_finite_value_nonfinite_jacobian_is_reported(self):
        z = torch.zeros((1,1), dtype=torch.float64)
        _, stats = solve_sequential(lambda x,o,t: torch.sqrt(x.square())+1, z, z, steps=1)
        self.assertEqual(int(stats['per_step']['status'][0,0]), 2)
        self.assertEqual(stats['linear_solves_total'], 0)
        self.assertFalse(bool(stats['all_converged'][0]))

    def test_fp32_supported(self):
        z = torch.tensor([[.1, 3.]], dtype=torch.float32)
        result, stats = solve_sequential(lambda x,o,t: -2*x, z, z, steps=4)
        torch.testing.assert_close(result[:,-1], z/(1+.5)**4, atol=1e-6, rtol=1e-5)
        self.assertTrue(stats['all_converged'].all())


if __name__ == '__main__':
    unittest.main()
