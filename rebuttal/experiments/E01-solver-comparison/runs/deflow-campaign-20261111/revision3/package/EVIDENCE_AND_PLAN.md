# Docking evidence and the next experiment

Prepared 10 November 2026 from the current local DEFlow artifacts and Mike's Chinese report of 9 November. This document does not authorize a remote launch. The user will supply the server. The first proposed training scope is three sequential-Newton runs, not Mike's entire 45-run campaign.

## Recommended environment

Use **wide navigation, tight docking**: `src/docking_full/multigoal_docking_env.py`, Gym ID `MultiGoalDockTight-v0`, launcher sigma **0.03**, actual success distance **0.06** because the environment tests `distance <= 2 * docking_sigma`. Do not confuse the launcher sigma with the success tolerance or substitute a narrow-navigation environment.

The table below was recomputed from each seed's last evaluation JSON only when `checkpoint_global_step >= 200000`. Values are mean U4, with sample SD. Partial failed runs are excluded from these published-style means but explicitly retained in the inventory.

| Variant | DEFlow | DIME | Matched explicit | DEFlow minus DIME |
|---|---:|---:|---:|---:|
| Wide loose | .671009 ± .002029 | .633984 ± .021405 | .506802 ± .030448, n=3 | +.037024 |
| Narrow loose | .661455 ± .020139 | .632414 ± .011858 | .498697 ± .099606, n=3 | +.029041 |
| **Wide tight** | **.588860 ± .128453** | **.501726 ± .010568** | **.412145 ± .084034, n=2** | **+.087134** |
| Narrow tight | .552462 ± .112824 | .490807 ± .008606 | .417206 ± .152976, n=3 | +.061655 |

DIME is the strongest completed-mean alternative in each variant. Wide-tight has the largest observed gap to DIME and to matched explicit among these four completed-cohort comparisons. This is a deliberately selected stress case, not an independent confirmation of general superiority. Large DEFlow SD means a new three-seed experiment may be inconclusive.

Wide-tight DEFlow U4 by original seeds 1/2/3: `.663564 / .440537 / .662480`. Matched explicit completed seeds are 3/4. Seeds 1/2 have last saved evaluations at 175025/150016 transitions and must not be paired as completed runs. Wide-loose explicit has completed seeds1/2/4 and partial seed3 at175017. Failure causes require the original logs and must not be inferred solely from partial files.

## Current local evidence: what exists and what does not

The inventory used `pathlib.rglob`, not default `rg --files`, because checkpoint files are ignored by repository search rules.

- `results/docking/` contains **647 `.pt` files and 592 `.json` files**, with **zero TensorBoard event files**. The JSONs contain episode-level evaluations, not solver-call telemetry.
- Each wide-tight DEFlow seed1/2/3 has **nine checkpoint files**, comprising eight approximately25k-spaced checkpoints through200k and one named final serialization, approximately62.5 MB per seed. Thus early/mid/final frozen audits do not require new training.
- A directly inspected 25k wide-tight checkpoint contains `actor`, `q1`, `q2`, `log_alpha`, `global_step`, and `args`. It does **not** contain replay, target critics, optimizer/RNG state, original training batches, or per-call solver histories. Stored flags include `save_full_state=False`, `density_diag_every=0`, and `root_probe_every=0`.
- `docs/run_contracts.json` and `docs/checkpoint_seed_changes_2026-09-29.json` must accompany checkpoints. Display seed IDs can differ from original seeds elsewhere. The inventory maps wide-tight DEFlow1/2/3 to original1/2/3.
- `docs/sacflow_artifact_inventory_2026-09-29.md` records that original source runs previously had events/logs in `/Users/ruslanageev/dev/sacflow`. That checkout is absent now. The historical inventory is a retrieval lead, not proof that those logs are present in this package.
- `results/numerical/integrator_depth/` is a one-pair frozen depth illustration. `results/numerical/solver_accuracy/` is manipulation evidence, not a docking training-convergence audit.
- The present engine records `last_fwd` and `last_diag` snapshots at TensorBoard intervals. Even if equivalent old event files are recovered, last-call snapshots do not give the denominator for every training actor/critic-target/rollout call. Current logging code also does not prove the historical engine emitted every current metric.

### Do old experiments need rerunning?

**No** for comparing solvers on frozen early/mid/final policies, estimating action/logp/gradient discrepancies on newly generated evaluation inputs, and reproducing stored episode aggregates. Retrieve and verify checkpoint/source identity, then use the existing weights.

**Yes, or recover richer original artifacts**, for actual historical training call failure frequency, original replay-batch distributions, exact target-critic effects, full optimizer-step replay, and causal effects of tighter solving throughout learning. A new audit bank is not the original replay distribution. Do not label reconstructed losses as exact historical SAC targets.

The proposed three new Newton runs characterize Newton learning. Comparing them to old joint runs is a historical comparison, not a controlled equal-time or same-instrumentation comparison. If that stronger claim is needed, the next smallest addition is three fresh joint-Anderson runs with the same new source, telemetry, recipe and hardware allocation. Do not force weights-only resume to masquerade as continuity.

## Fixed scientific recipe for phase 1

Reuse `run_multigoal_docking_full.configure('deflow', '0.03', seed, 200000)` and change only the solver plus explicitly documented instrumentation/output handling. The historical launcher CLI lacks a `--seed` option even though `configure` accepts it. Use the new entrypoint's explicit seed argument.

- Seeds: 1,2,3, fixed before outcomes. No replacement of numerical failures with favorable seeds.
- Environment: wide-tight, sigma.03, success tolerance.06,200000 actual replay transitions, eight environments.
- Actor: hidden256, T4, u_scale4, gate_bias1, implicit integration, exact-density readout, common implicit backward implementation.
- Replay500000, batch1024, learning_starts4000, UTD.5, policy_frequency1.
- Actor/critic/alpha LR3e-4, gamma.8, tau.01, alpha_init.2, target_entropy_scale0.
- grad_clip1, lam_jac.01, jac_sigma_target.9, jac_warmup100000.
- Historical Anderson center: tol1e-5,cap25. Newton's stopping norm and cap must come from the tested new solver defaults and be written in the run contract. Equal nominal tolerance/cap does not establish equal work or accuracy.
- Save checkpoints around25k intervals, cumulative telemetry throughout training, final original-protocol stochastic evaluation of1000 episodes per seed, per-goal success and U4. A model checkpoint or process exit alone does not establish benchmark completion.
- Keep source hashes, actual effective actor attributes, device/runtime versions, failed attempts, and timings. This package uses fresh runs and ordinary weights-only checkpoints. Resume is not supported by the campaign entrypoint. A future full-state implementation must separately verify custom-actor restoration, replay, optimizer, RNG and environment continuity.

## Sequence and decision points

1. **Preflight:** verify source/weights and imports, run tests, execute the new entrypoint's bounded256-transition nonbenchmark smoke (learning_starts64, batch8, CPU allowed) that includes actor updates, preserve outputs. This is required before any200k campaign.
2. **Frozen pilot:** same states/noise and actual batch composition on early/mid/final wide-tight checkpoints, joint versus sequential Newton. Verify reference residuals and selected roots. Measure sampling and complete actor-update costs independently. Report failures instead of discarding them. Use this pilot to estimate compute hours, not to select favorable learning seeds or tune by return.
3. **First campaign:** three Newton seeds1/2/3 to200k. Evaluate U4, per-goal probabilities, successful coverage, return, convergence, failures and time. Compare to historical joint results only with their provenance limitations.
4. **Next, conditional on the remaining question:** three contemporaneous joint-Anderson seeds if a controlled training/numerical-frequency comparison is needed. Center-versus-tighter training is a different intervention and should be a subsequent focused arm only if offline discrepancies warrant it.
5. **Scope extension:** PushWall frozen numerical audit addresses the stress case explicitly named by reviewers. Broader environments and equal-time SAC-Flow/DIME comparisons answer practical-benefit questions, not the isolated solver question, and require a separate budget decision.

## Numerical reporting requirements

Report sample/step residuals and the native batch stopping rule separately, call counts by rollout/target/actor/evaluation path, cap events versus residual failures, Newton linear-solve and line-search failures, finite values, actual iteration/evaluation work, and tails. Preserve inputs/checkpoint identity for audit samples and failures. Explain whether density/gradient comparisons concern equilibrium reference, actual finite-program derivatives, or merely two implicit readouts. They are not interchangeable.

The finite-program/global-likelihood issue cannot be closed by a converged-root reference alone. Likewise, frozen actor-gradient agreement does not establish harmless accumulated training error. Every report must retain those limits.

## Time budget

No server/device has been assigned and no Newton GPU pilot has run. Mike's historical MultiGoal T4 estimate is2–4.5 trainer-hours per200k run on packed4090 workloads. It is **not** a measured Newton estimate and is not physical GPU-hours.

For planning only, if Newton's end-to-end cost matches that historical center, three runs would total6–13.5 trainer-hours. If it costs twice as much,12–27 trainer-hours. Add measured pilot/final evaluation/artifact transfer and20% contingency separately. GPU occupancy hours are device-time interval unions, not trainer-hour sums under packing. Obtain ETA from the new engine's measured transitions/time after warmup and again after representative actor updates. Do not promise a calendar duration from these historical bounds.

## Relationship to Mike's plan

This deliberately narrows E2 to the user's requested first three-seed sequential campaign plus reusable offline diagnostics. It preserves Mike's same-root qualification, common local residual, full-cost accounting, source/recipe checks, failure retention and limits of finite-sampler validation. It does not claim completion of E1 equal-time learning, E3 seed expansion, all E5 sensitivity arms, or the45-run package.

## Ready command interfaces and evidence labels

See RUNNER_COMMANDS.md for the `--help` interfaces and command templates for `run_docking`, `prepare_inputs`, `audit`, and solver-aware `evaluate`.

Historical checkpoints now have an explicit reconstructed-bank workflow. Its observations come from fresh historical-policy rollouts, not recovered historical replay. Historical gradient audits default to a clearly labeled action/logp probe. Newly recorded actor-update capsules can provide matched actor/critics/alpha/loss settings and automatically activate the configured SAC actor-loss audit. Neither path automatically certifies finite-sampler density or causal training harmlessness. `DONE.json` marks training/checkpoint completion, with final1000-episode benchmark evaluation separately required.

## Updated convergence-budget experiment (11 November)

The current queue supersedes the earlier three-Newton-only sequence above: prepare twelve jobs, prioritize the six Newton/default Anderson controls, then three high-cap Anderson jobs, then the three tighter jobs if time permits. High-cap uses tol=1e-5 and cap=500, changing only the forward iteration budget relative to cap=25. It retains the original batch-mean stopping rule, so it is not an all-sample convergence guarantee. Tighter tol=1e-6/cap=100 remains a separate combined intervention.

The paired frozen audit sweeps caps25/100/500 at tol1e-5 on identical capsules, with one independent FP64 reference and common loss/readout. This isolates immediate finite-solve differences without training confounding. High-cap training additionally tests whether those differences change learning. Report failed/censored solves, per-sample residual tails and the common local-equation residual, not only mean stopping status. Large iteration counts can reflect stagnation or roundoff. Never describe an unconverged cap500 result as a full solution.

No unconditional overnight ETA is justified. Per GPU, nine priority jobs require approximately t_Newton+t_A25+t_A500 plus evaluation/telemetry overhead. All twelve require another t_A100_tight. Measure these times in the server pilot. A500 may exit early, but can use up to20 times the native forward iteration budget of A25. This is not a20x prediction for total training time. Keep the scientific horizon fixed and defer lower-priority complete runs if needed.
