# Final scientific and operational integration review

Read-only review on 2026-11-11. This review is tied to the file hashes below. Other agents were actively updating these files, so later hashes require a focused recheck. No server or GPU operations performed.

## Verdict

No remaining source-level release blockers were identified in the final focused recheck. Numerical and operational integration are approved for frozen deployment and the prescribed pinned CUDA preflight. This is conditional approval for validation, not a claim that the CUDA gates or scientific campaign have passed. The scientific optional-None regression and operational failure-finalization race are resolved. Successful CPU tests do not substitute for the pinned CUDA release gate.

## Findings

**Resolved P1: failure finalization and synchronization.** Early worker failure emits PREFLIGHT_FAILED as an admission latch only. The top-level preflight failure handler emits PREFLIGHT_TERMINAL_FAILED after ThreadPoolExecutor contexts have joined sibling workers, ACTIVE is empty, and no CLEANUP_FAILED marker exists. Every normally completed owned child is checked for non-zombie process-group descendants before its execution receipt succeeds. Lingering owned group members are checked against boot/session/start identity, terminated within bounded grace periods, and unresolved cleanup vetoes failure finalization. The local mirror now requires the verified terminal preflight marker and, when the coordinator has started, verified COORDINATOR_FAILED before stopping on preflight failure. An initial failure latch alone no longer stops sync. CLEANUP_FAILED vetoes all sync termination. Successful completion still requires verified FINAL_SYNC_ACK and exact agreement with its covered terminal/report hashes.

**Resolved P1: density observer mishandled supported None output.** The numerical reviewer found `forward_with_jacobian_penalty` returns `(action, None)` while new density-result instrumentation assumes every tuple entry is a tensor. The implementation owner filtered optional None when observing output dtypes and extended the dispatch regression to execute this path and its backward. Current scientific campaign uses the logprob path, so this is not evidence that existing smoke/evaluation results are wrong. The corrected package hashes are recorded in the regenerated scientific source map.

**Resolved during review: unreviewed shared hard failures were accepted.** Earlier numerical code accepted arbitrary status1/4 failures whenever FP64 also failed, under the label intrinsic. Current reviewed script blocks any shared failure except the exact preregistered historical A2 capsule hash, row473, localstep1. That exception requires status4 in both precisions and an invalid independent reference. FP32-only and FP64-only hard failures also block. This is an appropriate bounded exception, not a universal convergence claim. The unused adjoint threshold was removed.

**Resolved during review: server validation coverage and exact CUDA pin.** Current preflight runs original16 tests, precision tests, numerical gate tests, and operational lifecycle tests. It requires Torch2.5.1+cu124 and CUDA12.4. `--phase fixed` precedes smoke/pilots and `--phase fresh --fresh-bank ...` follows all four full-batch pilots. The runtime event names and expected dtypes match the scientific recorder.

## Numerical acceptance assessment

Fixed-q residual checks retain the entire batch and all local equations. They evaluate flattened, permuted, chunk1/32/1024, and per-step layouts without resolving favorable subsets. The FP64 fixed-q recomputation is distinct from the independent FP64 re-solve and SciPy reference. Exact pass masks remain in raw outputs. No solver tolerance changes or row exclusions were found.

The explained-boundary category is a preregistered measured discrepancy allowance capped at10% of the equation tolerance. It does not prove that failing rows actually satisfy tolerance. Its uncertainty includes the observed discrepancy to FP64 and stored native residual, so being close enough under this rule is evidence of a small numerical disagreement, not an independent causal proof of rounding. Keep exact failure masks and the explicit cap in reports. The existing synthetic full1024 fixture demonstrates this honestly: all native statuses succeed, one chunked/FP64 fixed-q residual fails, and the release category labels one boundary discrepancy while preserving the failure. This is not a silent relaxation of the solver stopping rule.

Native-versus-audit parity uses identical frozen q and z, actual native attachment/backward via temporary forward-solver interception, the same SAC critics and loss metadata, and parameter gradients for the full batch. The interception and parameter hooks are restored in finally. Both action/logp/loss and relative gradient errors are tested, with finite values required. It tests preservation of the implemented estimator, not exact finite-program differentiation or exact sampler density.

FP64 finite differences cover all B*T local Jacobians with three prespecified step sizes. All estimates/errors are retained. Independent SciPy validation covers all1024 rows and retains invalid references. FP64 reference gradients are explicitly not computed in preflight. Therefore this gate does not certify candidate-gradient agreement with the independent equilibrium reference. That evidence belongs in the later frozen audit and must be unavailable rather than fabricated when reference rows are invalid. A reference invalidity by itself is reported, not automatically a release blocker, which is acceptable only with this limited implementation/policy verdict.

Fresh capsules are bound to current package/source hashes and constructor digest, and must be full1024 training/actor_grad with matching actual steps. All four pilot arms are required, and omission of later eligible capsules is rejected. Historical banks are bound to three preregistered hashes. Native FP64 models are explicitly labeled as references rather than mistakenly checked as FP32 scientific tensors.

## Operational integration assessment

`train_argv` appends the explicit strict policy without altering the original scientific arm argv. Evaluator restores the recorded policy and the ops receipt validator requires strict backend flags and CUDA float32 execution at solver boundaries. Training additionally requires density and backward receipts. Horizon, checkpoint batch, finite model/critic state, actor updates, and evaluation protocol are checked. Audit commands pass the required policy and same-capsule enforcement.

Frozen source contract checks, externally supplied contract hash, per-slot locking, new output roots, and no automatic retries are present. Paired audits preserve actual capsule steps and source-run state. Compact mirror deliberately excludes most raw capsules and raw.pt, so final local reports must not claim every raw scientific artifact was mirrored. The original server copies are preserved and selected audit capsules are included.

Final process cleanup checks include group members after leader exit, ownership matching, bounded termination, and a persistent cleanup-failure latch. The14 local lifecycle tests pass. They use mocks and do not constitute a live Linux process-tree or remote shutdown test. The actual campaign engine uses synchronous environments and no explicit persistent worker pool. Processes that deliberately detach into a different session are outside the current owned-group model, which matches these entrypoints.

## Bounded local validation

Executed numerical gate unit tests:7 passed. Executed final operational lifecycle unit tests:14 passed after the failure-finalization and owned-descendant changes. Earlier10-test result is superseded. Interpreter /opt/anaconda3/bin/python, Python3.11.8, Torch2.5.1 CPU. Inspected the existing full1024 synthetic pipeline artifact, which passes FD/native-SAC parity and preserves one explained boundary failure. No scientific capsule was modified.

Final focused delta review: the pilot-derived disk estimate sums four pilot artifact footprints, scales by200000/8000 and three seeds, records the estimate and current free space, and requires an additional30GiB reserve. Its receipt is included in preflight evidence hashes. The formula correctly represents12 runs under a linear pilot extrapolation. It explicitly excludes uncertain evaluation/audit growth and does not delete artifacts. This is an operational admission estimate, not a storage guarantee. No new blocker identified.

## Reviewed SHA256 values

- `validation/numerical_preflight.py`: `91e70af30c5ba6e271248581891e7e6583d2fcff13c2678a590216674825186f`
- `validation/test_numerical_preflight.py`: `4b22aed2345a23a0bbd5bd86a5d7cb105648fa05932d8b8794ed38a0199e8be3`
- `ops/common.py`: `d948480047b9b84b5b439a0ab53ec22890e26aa4cfe1527a178f93976c082bda`
- `ops/preflight.py`: `14477e49c38e70c7c13d091603e98441c908c522bcfb110462a50f2c5716221f`
- `ops/campaign.py`: `8e6b9eac80476dd2d7435d320e88e31bc9c2180d1c7bf4636d5dc43f1bbdf1bd`
- `ops/sync_index.py`: `689d577e17a4d86d6848ca41d04dfe97ab4aaeb658ebba58eb3367240f2a3f47`
- `ops/sync_local.py`: `e6106fe590c1a65ad87f32e4c01faa7a5fd99aac5b5ecef8179c94180727dec7`
- `ops/test_lifecycle.py`: `48b82da1486aa4d119e2e0a868b38f4cf9bde7e4f8e5fe5620857e8eb91b1333`
