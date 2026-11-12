# Final numerical code review, revision3

Review status: conditionally acceptable for the prescribed CUDA preflight, not authorization to skip those gates. No forward-solver formula change is warranted from the available evidence. The small penalty-only observer regression found during review has been fixed and independently retested. Full-batch production CUDA boundary attribution remains unperformed by this reviewer.

## Independently checked

Compared revision3 runner, evaluator, audit, input-preparation and precision helper against revision2, and engine against the original local DEFlow engine. Newton source is byte-identical, SHA256 e7c24ebef16db7bd2e16101c08625294ba4902449e774281b9f5f0fca0a3a6d8. Engine changes add the explicit precision argument and replace the legacy override in its actual execution location. Backward attachment, dense shift, density safeguards, reward/loss configuration and Newton stopping equations are unchanged.

I independently ran the three scientific test files with DEFLOW_ENGINE pointing at revision3 under local Torch2.5.1 CPU: 22 passed, zero skipped. Also ran seven standalone numerical-preflight regression tests and a complete all1024 small-width synthetic CPU case through independent SciPy validation, all-row Jacobian FD and native/audit SAC backward parity. The synthetic case passed and all1024 independent references validated. These are real code tests, not evidence that production GPU capsules satisfy the gates.

## Precision and dispatch findings

The strict policy is applied at the engine's former TF32 override, before model creation. Highest matmul precision and both TF32 flags false are explicit. Autocast and backend override environment variables are rejected. The shared helper also rejects a non-FP32 default dtype and checks actual actor/input/output dtypes at runtime. These checks cover actual solver and density boundaries, and the new backward hook runs after the native hook registration, observing the native transformed gradient without changing it. The helper policy is process-wide, so the unchanged FP32 critic construction and ordinary critic operations receive the same precision policy. No mixed-precision cast or autocast context was found in the relevant native paths.

Newton dispatch still replaces only engine.anderson inside a try/finally, then executes the native attachment/backward. The tests cover deterministic action, stochastic action, logp/gradient path, exception restoration and all four effective constructor variants. Runtime observer bookkeeping does not mutate numerical outputs. The effective constructor is hashed into checkpoints and matched on evaluator reconstruction. The evaluator applies recorded precision before building the actor, checks source/package/checkpoint/configuration and constructor consistency, and records final effective policy.

The audit's campaign mode requires identical input/checkpoint bytes and recorded SAC critics/loss, rejects constructor/loss overrides, and checks policy plus source/implementation hashes. Historical cross-checkpoint operation remains explicit and labeled. FP64 reference readout is explicitly marked so strict policy enforcement checks the backend while allowing its reference tensors to remain FP64. Audit actor hooks now record solver, density and adjoint boundaries. Reference invalidity still disables the full-batch reference-gradient comparison.

## Telemetry and scope

New Newton-specific status totals and budget-exhausted counters correctly distinguish status1 from a converged last iteration or line-search failure. Common residual failure and native status remain separate. This is the right minimal change: no residual-refresh intervention has been inserted into the Newton stopping program. The frozen bank must decide whether the one pilot discrepancy is genuinely a narrow FP32 threshold boundary.

This instrumentation still reports historical shifted mixed-point implicit gradients and guarded density. It does not certify finite-program gradients, exact finite-sampler likelihood, global root uniqueness, all-row convergence or a clean timing benchmark. Policy-change retraining is a new numerical intervention across all arms and cannot be pooled with old TF32-trained results.

## Review finding resolved before freeze

P1 found and resolved: density_result previously forwarded the entire return tuple to the dtype observer. The supported forward_with_jacobian_penalty path returns (action,None), which made assert_float32 call None.is_floating_point(). The owner now filters None from observation only, preserving the native forward/penalty calculation. The dispatch test invokes the penalty-only path and backward. I independently reran all22 scientific tests after this change and they passed with zero skips. No outstanding code-review blocker remains at the hashes below. Production CUDA preflight remains a separate mandatory release gate.

## Standalone numerical gate limitations

The standalone preflight preserves all three historical full batches and all rows. Fresh bank requires latest full actor-gradient capsule for each of four arms and verifies provenance. It keeps exact scientific masks, uses measured fixed-q layout/FP64 differences for a separately labeled boundary classification, and blocks any unreviewed hard failure. Only the exact known A2 capsule SHA, row473, localstep2 is preregistered as an accepted implementation failure fixture, requiring both status4 and invalid independent reference. It does not become converged in reports.

Full-batch FP64 reference gradients are not computed by this bounded preflight. It records whether every reference row validates, and the later audit performs the corresponding gradient comparison. Independent hard-row traces reconstruct the scalar FP64 Newton rules and are labeled as such, not presented as internal traces captured from the production vectorized solver. CPU synthetic success does not remove the need for CUDA policy assertions, actual full-batch residual checks, fresh four-arm pilots, checkpoint evaluation and measured throughput.

## Reviewed hashes

```json
{
  "package/deflow_rebuttal/precision.py": "538963ed417e294db4c4bf78605671463e6f9263222073b5e0e2404e3a3dbe92",
  "package/deflow_rebuttal/run_docking.py": "e6f32b01bbbc00bffaa1852c757d8d940a4fe9e3ab792b90e063013cf7ce0cf2",
  "package/deflow_rebuttal/evaluate.py": "b5e50a8fc7131fd765c4207e837d310f3168b5e897f004696d16746e9984b8f1",
  "package/deflow_rebuttal/audit.py": "6dcff6cb6b5c2e3f58b4c98b22c99551598c8512f1c08ebaf9c6e0ecbcd7d48d",
  "package/deflow_rebuttal/prepare_inputs.py": "13c9ffbac77395c8d2b9cd153408214d3c5d212a341250b042637509aa58030b",
  "package/deflow_rebuttal/sequential_newton.py": "e7c24ebef16db7bd2e16101c08625294ba4902449e774281b9f5f0fca0a3a6d8",
  "package/deflow_rebuttal/test_precision.py": "4aa05f012ee04290f4b3e7d5230adf805c9b4bcba5dfe8d93ff1383a22968da5",
  "DEFlow/src/docking_full/sac_anderson_flow.py": "ba94c8e1f66a41bc6b828ec78d52a1d0ca53573428a771c52c73ea7196205bb5"
}
```
