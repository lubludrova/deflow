# Independent final review: reviewer coverage and runnable protocol

Reviewer: evidence/coverage agent. Reviewed `run_docking.py`, `audit.py`, the current original engine, Mike's report, and dkt6/VrBX comments on 10 November 2026. This is a separate read-only review of implementation files, not a third independent reimplementation. Files may be updated after these findings. No RL training was launched by this reviewer.

## Scope that this package can support

| Reviewer request | Delivered mechanism | Limit that must remain explicit |
|---|---|---|
| dkt6 W1: sequential implicit alternative | Batched sequential Newton forward, shared native backward, same docking recipe | Same equations does not imply same selected root or finite accuracy. Shared backward isolates forward, not total best-implementation advantage |
| dkt6 W1: matched compute | Synchronized forward timing and work counters, original checkpoints available | Single timings are not repeated performance statistics or a controlled equal-time learning result |
| dkt6 Q: nonconvergence frequency | Per-call cumulative local residual/cap counters in new runs | Historical rate unavailable from exported checkpoint/evaluation files. Call phase must separate training and evaluation |
| dkt6 Q: objective effects | Audit supports configured frozen SAC actor loss with critics/alpha | Default audit loss is a labeled probe, not SAC. Original replay/target critics are absent in old checkpoints |
| dkt6 Q: tol/cap/regularization learning sensitivity | Tolerance/cap settings and offline comparison | No causal full-training sensitivity experiment is completed. Jacobian penalty interventions change the objective |
| VrBX Q1: finite-solver derivatives | Optional actual-program directional finite difference | One FP32 direction/step cannot certify full gradients, branch differentiability, or likelihood |
| VrBX W1: finite sampler density | Guarded implicit readout compared with validated reference | This is not global finite-program likelihood certification |
| dkt6 W2 / VrBX Q2: cost-benefit | Timing/provenance fields and a bounded plan | Instrumented Newton versus historical joint is neither clean same-hardware timing nor equal-time SAC/DIME comparison |
| VrBX W2: seed uncertainty | Fixed1/2/3 with failure retention and per-seed reporting | n3 cannot establish equivalence or robust superiority. Do not choose surviving-only seeds |

## Blocking issues reported to the implementation owner

1. Original `act(deterministic=True)` calls Anderson directly and bypasses the instrumented `_solve_zstar_with_implicit_grad`. Without an override, online deterministic evaluation during Newton training silently evaluates joint Anderson and bypasses telemetry. The parent is implementing the solver-aware path and final evaluator.
2. Initial role names combined target/evaluation and rollout/evaluation. Therefore the original manifest's claim of all-forward-call training denominators required explicit phase classification. The parent is updating phase handling.
3. The frozen audit requires a trusted obs/noise bank, actor constructor configuration, and optional explicit loss configuration. A documented bank-generation path and concrete audit invocation are needed before the historical-checkpoint audit is turnkey. A small-batch smoke capsule is not a historical batch1024 audit. The parent is addressing input construction/capture.
4. The default audit uses a labeled probe loss. Claims about SAC actor gradients must use critics and explicit alpha/effective Jacobian penalty/clip configuration. Stored reference readout retains implementation safeguards, so it is not an unshifted exact-equilibrium gradient benchmark. Keep this distinction in README and result interpretation.
5. Conditioning fields depend on enabling numerical diagnostic probes. Native `density_diag_every=0` must not silently yield a claimed comprehensive conditioning audit.
6. The audit's single-call timings must not be presented as a repeated matched-precision/time benchmark. For performance claims, add warmup/repeats and report timing distribution separately from heavy telemetry.

These findings are not evidence that fixes were validated. The owner should append final dispositions and corresponding test evidence before marking the package ready.

## Execution check performed

Tried the documented read-only `python -m deflow_rebuttal.run_docking ... --plan` with `/Users/ruslanageev/dev/signal-flows/.venv/bin/python` (PyTorch2.13.0). Parsing reached import of the original launcher, then failed because this environment lacks `gymnasium`. No output experiment directory was created. This demonstrates a dependency gap in that local runtime, not a solver result. Full integration smoke needs the complete docking runtime. Dependency installation and further smoke validation belong to the parent run record.

## Existing evidence verified independently

- Current docking export:647 checkpoint files,592 JSON,zero event files.
- Wide-tight DEFlow has27 checkpoint files across original seeds1/2/3. Earlier/middle/final offline probes are possible without retraining.
- Correct environment is wide navigation with sigma.03 and success distance.06.
- Final-cohort DEFlow U4 .588860±.128453, DIME .501726±.010568, explicit .412145±.084034 with only seeds3/4 completed. Partial explicit1/2 retained.
- Historical checkpoint payload lacks replay, target critics, optimizers and RNG. No faithful resume or original-batch reconstruction claim is supported.

## Final launch gates

All three pre-fixed seeds may run only after the selected runtime passes tests, both solver paths pass integration smoke, the solver-aware evaluator is exercised, and the produced telemetry/checkpoints can be read by the audit tool. The final report must distinguish training completion from final benchmark-evaluation completion. Server/resource ownership and exact command approval are supplied separately by the user to the runner.

## Follow-up inspection after owner fixes

The coverage reviewer inspected updated source and successfully executed `--help` for `prepare_inputs`, `evaluate`, and `audit`. The owner separately reports CPU execution of an8-sample reconstructed bank and4-episode evaluation. Those integration executions were not independently repeated by this reviewer.

- **Deterministic path fixed in source:** `InstrumentedActor.act` routes zero-noise deterministic actions through the selected instrumented solver. Final `evaluate.py` reads manifest solver identity and checks checkpoint/source/seed consistency.
- **Phase classification added:** recorder labels evaluation when the stack contains `eval_policy` or the evaluator's `rollout`. Training/evaluation denominators are separated. This is an implementation-specific stack convention, which must be rechecked if the engine is refactored.
- **Inputs now runnable:** `prepare_inputs.py` builds and hashes a full reconstructed state/noise bank. This addresses offline-input availability, not recovery of historical replay. Exact CLI templates are now in RUNNER_PROMPT.md.
- **SAC capture added:** periodic/tail capsules collect online/target critics, available replay minibatch tensors, alpha, args and effective actor-loss settings. `audit.py` automatically uses capsule loss metadata unless `--probe-loss` is requested. Historical generated banks still default to the labeled probe unless a verified explicit loss config is supplied. Capture of target networks does not itself mean the audit evaluates all Bellman-target/temperature error paths.
- **Conditioning enabled:** new training uses numerical diagnostic cadence100 and audit sets cadence1. Reports must state the subset/denominator of matrix probes rather than claim every matrix is checked.
- **Repeated timings added:** audit has warmup/repeats, synchronized solver timing distributions, and explicit exclusion of readout/telemetry. It remains solver-only timing, not full actor-update timing or equal-time training evidence.

The initially reported code blockers have corresponding implementation changes. Remaining scope limits are scientifically material: original-batch training frequency cannot be recovered from reconstructed banks, finite-sampler likelihood is not certified, single-direction FD is limited, and the three Newton runs alone do not establish controlled superiority over historical joint runs. Comprehensive test/integration pass status must come from the owner's final validation record.
