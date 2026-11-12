# Independent implementation and runtime review

Reviewed by audit subagent on 2026-11-10. Scope: sequential solver, original-actor integration, saved diagnostics, frozen audit, and local CPU execution. This is an independent task review within the same agent team, not cross-model review.

## Findings and dispositions

- Native engine real-transition counts can overshoot the requested horizon. Completion now accepts requested <= actual < requested + num_envs and records actual steps.
- The original actor uses an extra g(q) attachment and a shifted dense adjoint evaluated at g(q). Both candidate solvers preserve these semantics. Native action, log-density, and full parameter gradient parity were tested.
- Capsules preserve the actor at the time of the input call. Use a capsule as both --checkpoint and --inputs for exact snapshot matching. Nearby training checkpoints are not interchangeable.
- Capsules now retain critics and effective loss configuration. Audit automatically uses embedded loss settings, otherwise explicitly reports a probe objective. A target/rollout capsule is a frozen-bank SAC-objective diagnostic, not a replay of an actor optimizer update.
- Forward field-work accounting and recorder overhead are separate. Audit supports warmup and repeated synchronized solver-only timings. It does not establish clean whole-training throughput.
- Root failures retain the last accepted iterate, explicit status and counters. No silent rescue or seed replacement was introduced.
- Source files are checked against engine_sha256.json before execution.
- Audit strict-JSON serialization originally failed on historical constructor tensor bounds. Fixed for tensors, NumPy arrays/scalars, tuples, paths and nonfinite numbers, with a regression test. Raw tensors preserve nonfinite values independently of JSON nulls.

## Executed validation

Interpreter: /Users/ruslanageev/dev/signal-flows/.venv/bin/python. The original simulator dependency was made available through /private/tmp/deflow-runtime-deps. No persistent environment modifications were made by this reviewer.

1. Four audit tests passed: native action/density/gradient parity, independent FP64 reference validation, Newton refinement, and strict metadata serialization.
2. A 256-transition CPU Newton smoke completed, including 225 solver calls across training roles. Exact outputs: /private/tmp/deflow-review-smoke-v2/. DONE.json and checkpoints exist.
3. Frozen audit of that smoke's first capsule passed with embedded configured SAC loss. All eight reference rows validated, both solver gradients completed.
4. A historical wide-tight seed2 checkpoint at step50010 passed the complete audit CLI using /private/tmp/deflow-historical-bank.pt. Outputs: /private/tmp/deflow-historical-audit-v2/. All eight reference rows validated. This was a small B8 probe-loss compatibility check, not a native B1024 training comparison and not scientific solver-performance evidence.
5. Tiny fixture audit with optional finite-program directional finite differences completed.

All processes started by this reviewer exited. No scientific 200k run or GPU benchmark was launched.

## Remaining validation limits

CUDA runtime/performance must pass the runner's preflight. Historical checkpoint compatibility does not independently prove historical source identity. Three-sample fixtures and B8 smoke banks cannot establish numerical reliability throughout training. Reference comparison concerns a safeguarded equilibrium readout and common shifted adjoint, not exact finite-program likelihood. The finite-difference option is one configurable FP32 direction/step and explicitly not gradient certification. The solver-aware final evaluator and input-bank builder were prepared separately by the primary agent and are outside this review's executed unit tests.
