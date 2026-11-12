# DEFlow sequential solver and numerical audit

This package adds one production candidate: **batched exact sequential Newton with per-sample Armijo backtracking**, compared with the existing native joint Anderson solver. SciPy FP64 `hybr` is an independent offline reference, not a speed baseline. Scientific training has not been launched.

The code does not edit the DEFlow repository. Training loads the pinned `src/docking_full` snapshot, uses its recipe and native mixed-point `g(q)` attachment, shifted dense adjoint and guarded density for both forward solvers. It intentionally does not fix historical gradient semantics while replacing the solver. Each process runs one solver. The scoped forward dispatch is not thread-safe, so do not call actors concurrently in one process.

## Environment and preflight

Use the original repository's `requirements.txt` in an isolated server environment. Local checks used Torch 2.13.0 rather than the repository's pinned 2.5.1. CUDA and pinned-version parity must pass on the provided server before the campaign. No source edits or dependency upgrades in a running environment.

From this directory:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DEFLOW_ENGINE=/path/to/DEFlow/src/docking_full/sac_anderson_flow.py
python -m pytest -q deflow_rebuttal/test_solver.py deflow_rebuttal/test_audit.py
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --output /new/path/preflight --solver newton --seed 0 --smoke --execute
```

`--smoke` is a 256-transition CPU-capable integration check with small batches, not scientific evidence. Normal runs require CUDA via the original device selector. `--plan` or omission of `--execute` prints the exact recipe without training. Source mismatch fails closed against `engine_sha256.json`.

## Proposed first campaign

Wide-tight = `MultiGoalDockTight-v0`, sigma .03, actual success tolerance .06. Three preselected seeds, 200k real transitions each:

```bash
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --output /new/path/newton_s1 --solver newton --seed 1 --steps 200000 --execute
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --output /new/path/newton_s2 --solver newton --seed 2 --steps 200000 --execute
python -m deflow_rebuttal.run_docking --repo /path/to/DEFlow --output /new/path/newton_s3 --solver newton --seed 3 --steps 200000 --execute
```

Run after server-specific authorization/preflight. No automatic retries or resume: original docking engine is fresh-only. Sequential commands are safe defaults until GPU/RAM concurrency is measured. Retain all failures and partial runs, do not substitute seeds.

Outputs include manifest/source+package hashes, native events and checkpoints, all-call `solver_calls.jsonl`, per-role cumulative denominators, readout/attachment diagnostics, periodic and early-failure full-batch capsules, and `DONE.json` or `FAILED.json`. Training timing includes synchronous telemetry and is not a clean equal-time benchmark. Newton tol1e-5/cap25 is an initial recipe, not a claim of equal precision/cost to Anderson tol1e-5/cap25.

Capsules preserve exact actor weights, inputs/noise, raw trajectories/residuals, constructor, online/target critics when in the training loop, sampled transitions, alpha and effective actor-loss settings. Roles distinguish training/evaluation and action/logp/gradient calls. No claim that periodic capsules represent every training batch. Failure tails are first12 failing calls, not unbiased tail sampling. Native density conditioning is probed every100 density calls; cached matrix values retain the native tick. Full causal sensitivity still needs interventions.

## Frozen stability audit

Historical input construction and audit:

```bash
python -m deflow_rebuttal.prepare_inputs --repo /path/to/DEFlow --checkpoint /path/to/historical.pt --output /new/path/bank.pt --episodes 100 --batch-size 1024 --device cuda
python -m deflow_rebuttal.audit --engine "$DEFLOW_ENGINE" --checkpoint /path/to/historical.pt --inputs /new/path/bank.pt --output /new/path/audit --device cuda --probe-loss --repeats 10 --warmup 2
```

This bank samples visited states from complete rollout episodes including failures. It is reconstructed, not historical replay. All batch rows are preserved in comparison. Historical engine identity must still be checked, strict state loading alone does not prove provenance.

For new-run captured actor calls use the SAME capsule as `--checkpoint` and `--inputs`. Embedded critics/loss settings enable configured SAC loss automatically. Select metadata role ending `actor_grad` to describe actual actor-call context. A target-call bank supports a hypothetical actor-loss diagnostic, not actual actor-update replay. Capsule state hashes must not be silently replaced by a nearby saved checkpoint.

Audit reports raw/attached residuals, action/logp/gradient differences, residual/conditioning validation of independent FP64 roots, reference exclusions, shifted/unshifted adjoint residuals, native guards and repeated synchronized solver-only timing. Use `--fd-step 1e-4` for a limited centered finite-program directional check. The FP64 reference readout retains safeguards and adjoint shift: it is a converged safeguarded reference, not certification of exact finite-sampler likelihood. Different converged branches are possible. Only fully valid reference batches receive comparable full-batch gradient statistics.

## Solver-aware evaluation

Do not use the original evaluator to reconstruct Newton: it defaults to Anderson. Use:

```bash
python -m deflow_rebuttal.evaluate --repo /path/to/DEFlow --run /new/path/newton_s1 --checkpoint /actual/checkpoint/from/DONE.json --output /new/path/newton_s1_eval --episodes 1000 --device cuda
```

Read the exact completed checkpoint path from `DONE.json`: real-transition accounting can finish up to7 steps beyond200000. The evaluator checks source/solver code/configuration identity, restores the recorded solver and uses the original stochastic coverage protocol. It preserves per-episode data and U4/success/return.

## Evidence and review

- `EVIDENCE_AND_PLAN.md`: actual artifact inventory, variant selection, experiment order and conditional cost estimates.
- `RUNNER_PROMPT.md`: server-agent handoff.
- `REVIEW_NUMERICS.md`, `REVIEW_RUNTIME.md`, `REVIEW_COVERAGE.md`: independent review and fix dispositions.
- `rebuttal/REVIEWS_ALL_2026-11-10.png`: user-provided full review screenshot.

Existing intermediate docking checkpoints permit frozen audits without retraining. Current local exports do not support historical all-call training nonconvergence frequencies. New Newton3 alone versus old joint results is not a controlled wall-clock learning comparison. The current night queue includes matched instrumented joint3, subject to preflight.

Primary cross-check sources: [DEFlow](https://arxiv.org/html/2610.16597v2), [DEQ reference solver](https://github.com/locuslab/deq/blob/master/lib/solvers.py), [SciPy Armijo implementation v1.14.1](https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_nonlin.py), [Kelley chapter8](https://epubs.siam.org/doi/10.1137/1.9781611970944.ch8). Our Newton is independently implemented with the actual squared-residual merit derivative, exact local Jacobians, no silent full-step fallback, and explicit row-wise failures.

Tolerance/cap sensitivity can be run offline with `audit --anderson-tol ... --anderson-cap ... --newton-tol ... --newton-cap ...`, each in a new output directory. Change one factor at a time for attribution. Solver sensitivity on frozen policies does not establish training sensitivity to Jacobian regularization.

Local installation: `python install_into_deflow.py --repo /Users/ruslanageev/dev/DEFlow`. This creates `rebuttal/solver_tools` and copies the screenshot to `rebuttal/REVIEWS_ALL_2026-11-10.png`, refusing overwrite. Add `--move-original-screenshot /path/to/original.png` only to complete the user's requested move after hash verification. The current Codex sandbox cannot write to that repository, so the prepared package remains in the current workspace.

## Current night queue (11 November)

Current convergence extension: `night_campaign.json` now contains twelve prepared jobs, with the three same-tolerance high-cap Anderson jobs ahead of the three tighter jobs. `RUNNER_PROMPT.md` is the short handoff. Training accepts `--anderson-tol` and `--anderson-cap`, persists them and restores them for evaluation. Frozen `audit --anderson-caps 25 100 500 --anderson-tol 1e-5` compares identical inputs with one FP64 reference. High cap means a safety limit with early stopping by native batch mean, not guaranteed full convergence. See `RUNNER_COMMANDS.md` for interpretation and commands. The current suite has16 tests.

Latest validation: `VALIDATION_CONVERGENCE.json` records the current16-test CPU suite, cap500 training/evaluation smoke, paired SAC cap sweep and current code hashes. `VALIDATION.json` and the independent review notes are historical snapshots of the initial package. CUDA validation remains pending.
