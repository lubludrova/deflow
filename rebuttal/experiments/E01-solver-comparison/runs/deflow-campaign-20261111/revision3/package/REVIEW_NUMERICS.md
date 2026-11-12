# Numerical implementation review

Reviewer: solver agent, independent review of the integration and audit code after implementing the forward solver. Date: 2026-11-10. No training campaign launched.

## Checked and fixed

- Sequential implicit-Euler residual is `x - previous - dt * v(x,s,t)`. The Jacobian is `I - dt * dv/dx`, and the Newton direction solves `J p = -residual`.
- Per-row Armijo tests the squared-residual merit using its actual directional derivative. Failed line searches preserve the last accepted iterate and report failure. There is no undisclosed clipping, diagonal shift or alternate solver in the forward Newton solve.
- Initial and final-iteration convergence are tested independently of iteration-cap events. Failure of an earlier implicit step remains visible in the trajectory-level status even if later local steps converge.
- Native `g(q)` attachment, shifted dense adjoint and density safeguards are shared by both candidate forward solvers. Their finite-root limitations are explicitly retained rather than accidentally changed in a solver ablation.
- The original deterministic `act` bypassed the patched solver. The new override routes zero-noise evaluation through the recorded solver and instrumentation.
- Training may overshoot the requested transition horizon by fewer than `num_envs` transitions under NEXT_STEP autoreset. Completion now accepts and records this legitimate overshoot.
- Input preparation uses all visited states, including unsuccessful episodes, selects without replacement and labels the bank as reconstructed rather than historical replay.
- The separate evaluator reconstructs the selected solver and reuses the original stochastic coverage protocol. It does not use the legacy default-Anderson reconstruction for new Newton runs.
- Evaluator now verifies checkpoint Newton settings, every manifest configuration value and the recorded `sequential_newton.py` / `run_docking.py` hashes before loading the policy.
- Action-only calls now record `density_evaluated=false` and `density=null`. Density diagnostics cannot silently reuse the preceding call. Cached native matrix probes retain their probe tick.
- Source hashes are checked by the training entrypoint. Calls carry local residual failure denominators, solver status and operation counts. Capsules now also capture available critic, target-critic, temperature and minibatch context.

## Verification

Nine solver CPU tests pass using PyTorch 2.13.0 at `/Users/ruslanageev/dev/signal-flows/.venv/bin/python`. Tests cover affine closed-form solutions, nonlinear damping, line-search exhaustion, singular/nonfinite row isolation, nonfinite Jacobians, exact-cap convergence, failure propagation, sample/batch/permutation consistency, detached outputs and FP32. GPU behavior/performance is not certified by these tests. The root agent separately reports successful bounded training/evaluation and historical-input-preparation smoke checks.

## Remaining interpretation boundaries

1. A reconstructed input bank records the current engine hash. This does not by itself establish the identity of the historical training engine. Interpret results as a controlled frozen-policy comparison under the declared runtime unless historical source identity is separately verified.
2. Converged FP64 reference readouts retain the common density safeguards and shifted adjoint. They measure agreement with that converged safeguarded computation, not a certificate of exact unshifted gradients or finite-sampler likelihood.
3. Full training nonconvergence frequencies come from new all-call telemetry. Historical checkpoints/reconstructed observation banks cannot recover the original training-call denominator or prove harmlessness of numerical errors throughout learning.

The concrete integration issues raised during review are fixed in the reviewed code. No immediate numerical implementation blocker remains for a bounded server preflight. The three interpretation boundaries above must remain in scientific reports. GPU preflight and measured timing remain necessary before the training campaign.
