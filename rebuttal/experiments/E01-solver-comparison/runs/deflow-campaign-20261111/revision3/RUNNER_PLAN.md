# Revised strict-FP32 campaign plan

Status: revision3 implementation prepared, final peer review and fresh preflight required before scientific admission. User authorization covers preserving/stopping the old campaign and restarting the same twelve runs from scratch under uniform strict FP32. This document changes technical preflight and orchestration only. Review artifacts are not execution receipts.

## 1. Freeze the exact scope

Create a new immutable revision/output root. Keep revision2 and its TF32 artifacts intact and explicitly superseded for the strict comparison. No resume, scientific retry, seed replacement, horizon reduction, extra method, changed packing, solver optimization, new training hyperparameter or FP64 training.

All runs remain MultiGoalDockTight-v0, 200000 real transitions, seeds 1/2/3, original batch1024, T4 and original backward/density safeguards. N is Newton rtol1e-5/atol1e-7/cap25. A is Anderson tol1e-5/cap25. H is Anderson tol1e-5/cap500. T is Anderson tol1e-6/cap100.

| Physical slot | Seed | Sequential queue |
| --- | --- | --- |
| 0 | 1 | N1, A1, H1, T1 |
| 1 | 2 | A2, N2, H2, T2 |
| 2 | 3 | N3, A3, H3, T3 |

Retain recorded GPU UUID mapping and verify it against the server. Exactly one owned compute task per assigned GPU. Each 1000-episode evaluation, seed26152, completes and validates before the next trainer in that slot. Nine paired audits follow the completed training/evaluation queues. No hard rental deadline exists. Do not stop started jobs or defer H/T merely because a night ends.

Success: new contract reproduces original job argv and order except new paths and uniform precision implementation, and has explicit authorization/provenance for the new revision.

## 2. Preserve and stop the old campaign

First latch no-new-launch for the old coordinator and disable only its future scheduler launch. Parent reports an identity-verified SIGSTOP containment is being applied during review. This is a pause, not proof of exit. Inventory coordinator, monitor, trainers/evaluators/audits and sync workers with PID/start/boot/command identities. Save current state and a compact inventory. Signal only verified owned processes, with bounded graceful wait and identity recheck before escalation. For a stopped child, arrange delivery of graceful termination with continuation while keeping coordinator admission contained. Account for descendants and verify GPU PIDs and owned processes have exited. Preserve all files and record STOPPED/INTERRUPTED rather than success for incomplete attempts. Finish an old-revision artifact sync without letting its receipt authorize the new campaign.

Success: timestamped stop receipt identifies each affected process and exit, no old owned compute remains, no unknown/user process was killed, and old results remain separately available.

Parent execution update: identity-safe termination completed, and all three old GPU compute PIDs were verified absent. A2 stopped at181054, N1 at8604 and N3 at8611, so none completed200k. Initial coordinator SIGSTOP state was not confirmed. Do not report it as a verified pause. Exit codes were not observed and remain null rather than invented successful exits. A full5.8GiB preservation archive was still transferring at the time of this update. Final archive verification remains separate from process-stop verification.

## 3. Implement and pin the new condition

Apply strict policy in the actual training path after any legacy precision initialization: matmul precision highest, CUDA matmul TF32 false, cuDNN TF32 false, float32 training tensors, no AMP/autocast substitution. Confirm actual state at actor execution, not just before engine.main. Store requested and actual precision with runtime versions, device/UUID and source hashes in manifest, checkpoint/capsule metadata and terminal evidence. Evaluation restores the checkpoint's recorded policy and fails on absent/conflicting policy for this new campaign. Audit records and applies the same policy for FP32 candidates, with FP64 reference explicitly separate. Restore any temporary diagnostic precision changes on scope exit, including exceptions.

Preserve the original source hashes and authorized TensorBoard patch provenance. Add exact precision/operations diffs and independently pinned new hashes. Pin all imported runtime files, baseline manifests, contract, numerical preflight and local sync code. Do not rewrite old expected hashes to disguise a change.

Add the surgical operational fixes in operations.md. An explicit admission predicate must require current campaign identity, unchanged sources, healthy monitor, no stop/failure latch, correct preflight evidence, exclusive slot claim, resource availability and sufficiently recent verified sync receipt. A blocked predicate waits or records BLOCKED with a reason, never falls through. No implicit retries or coordinator reconstruction of scientific execution.

Use concrete campaign-local STOP.json and PAUSE.json latches. Check both before plan, claim, child launch, evaluation, audit and queue advancement, and on every wait iteration. PAUSE means no new work until explicitly cleared under authorization, without marking unstarted runs failed. STOP permanently closes admission for that revision and drives identity-safe child cleanup. Recheck under the slot claim immediately before Popen to close managed admission races. The preflight receipt records campaign ID, contract SHA256, source-manifest SHA256, precision-policy ID and exact evidence paths/hashes. The sync receipt records these identities plus index SHA256 and verified terminal artifact hashes. These are small JSON predicates, not a new scheduler framework. User authorization already covers the reviewed necessary fixes and the same twelve fresh strict runs after these gates pass.

Success: focused local checks pass for precision application/restoration and fail-closed lifecycle/sync behavior, followed by final diff review. New policy is not credited from environment variables alone.

Implementation lives under revision3/ops. DEPLOYMENT.md defines the concrete v3 contract and launchd configuration. External DEFLOW_CONTRACT_SHA256 anchors the contract, which pins runtime files and baseline manifests. Local lifecycle tests exercise stop/pause, FAILED-over-DONE, missing anchors, wrong-campaign and incomplete final-sync receipts, monitor failure and unknown-process termination refusal. Parent generates final hashes only after peer review.

## 4. Establish numerical and runtime preflight

Inventory Python/PyTorch/CUDA, GPU UUIDs, free memory/disk and precision-affecting environment. Verify pinned sources and isolated environment. Run the original sixteen solver/audit tests with no skips plus new targeted regression checks. Test count expansion must not hide disappearance of an original test.

Replay the three already inventoried full-batch capsules, with their saved hashes and selection rule, through unchanged Newton in strict CUDA FP32. Keep all1024 rows and all four local equations. Record native statuses/iterations/backtracks and independently recomputed fresh residuals, including full-batch versus sliced layouts. Run FP64 diagnostics/reference on known and newly discrepant rows. Save raw per-row evidence, tolerance normalization, reference-valid masks and exact rejected-row IDs. Old capsules are fixed diagnostic fixtures, not new strict-training controls.

Acceptance is an engineering consistency decision, not a claim of universal solver convergence. Native and fresh success counts must be separate. The known pilot borderline FP32 row may remain explicitly borderline if its residual is at the rounding boundary and FP64 confirms the explanation. The known late A row473 must remain a reported hard row when strict FP32/FP64 and invalid independent reference corroborate it. Do not relabel either as converged or exclude it. Unexpected nonfinite values, broad native/fresh disagreement, renewed TF32-scale layout drift, or unexplained new failures block production pending diagnosis. Freeze any quantitative regression envelopes from this evidence before production, never relax scientific tolerances to pass. Anderson budget failures are outcomes, not automatic engineering gate failures.

Then run four actual strict CUDA smoke trainings, N/A/H/T, seed0,256 transitions, small smoke batch, each with actor updates, finite checkpoint, telemetry and four-episode solver-aware evaluation. Verify effective precision and CUDA in both child processes. A smoke does not substitute for full-batch replay.

Run each smoke and pilot stage across all assigned cards: N on slot0, A then T on slot1, H on slot2. All tests and fixed-bank numerical checks precede smoke, all four smoke evaluations precede pilots, and all four pilots precede fresh-bank numerical validation. A shared PREFLIGHT_FAILED latch prevents any new child or dependent phase after failure. Already active independent preflight work may finish and preserve evidence. All three GPUs therefore execute real actor/critic updates and precision-boundary checks. Pin server Torch2.5.1+cu124/CUDA12.4 explicitly and run operational/numerical regression tests there as well as original16 and precision tests.

Run four separate timing pilots, N/A/H/T, seed0,8000 transitions, batch1024, no smoke. Inspect post-learning_start4000 data, full-batch actor-gradient capsules, native/fresh diagnostics, finite parameters/losses and policy persistence. Include newly encountered hard or borderline rows in diagnostics rather than silently passing on exit0. Do not apply an all-sample Anderson-convergence gate. Produce initial strict-mode training estimates using post-learning throughput, with at least the existing 50% planning margin and an explicit late cap500 caveat. Estimate evaluation/audit time separately and revise from actual observations.

Success: one provenance-bound preflight manifest enumerates original tests, numerical replay findings, four smoke runs, four smoke evaluations and four pilots. Any contradictory FAILED marker or missing artifact prevents success. Old TF32 smoke/pilot output cannot satisfy this gate.

## 5. Execute the twelve fresh runs

Validate --plan against the frozen contract for each job. Use new output directories, external logs, immutable attempt claims, actual process identities, timestamps and exit codes. Keep checkpoints/capsules and all raw telemetry on the server. Monitor progress, native solver health, fresh sampled diagnostics, disk, co-runners, heartbeat and sync. Unknown source changes or monitor failure block future launches immediately. Scientific failure stops its slot and preserves all evidence, while independent healthy slots may continue. A global precision/provenance failure closes all future admission.

Validate training exit and DONE together, checkpoint global_step200000..200007 and batch1024, finite model tensors, actor updates, manifests, solver telemetry and capsules. Read checkpoint paths from DONE. Validate each evaluation has1000 episodes, seed26152, matching checkpoint/source/solver/precision, actual CUDA and finite U4/success/return. Mark TRAINING_FINISHED separately from EVALUATION_FINISHED. Sync receipt for advancement must cover the completed train/eval artifacts, not merely EXIT0.

Success: twelve explicit terminal run rows, ideally twelve accepted training/evaluation pairs, with zero silent retries or omitted failures.

## 6. Run and account for all nine audits

Prepopulate nine audit rows for A seeds1/2/3 × target25k/100k/200k. Select nearest eligible full1024 training/actor_grad capsules with actor, critics and loss_config. Tie by smaller actual step then lexical path. Persist candidate eligibility, selected hashes, actual steps and duplicate selections if any. Missing capsules remain MISSING, and unavailable later work remains BLOCKED. Do not substitute a checkpoint or another condition.

Use the same capsule as --checkpoint and --inputs, caps25/100/500, tol1e-5, warmup2, repeats10. Preserve original Newton settings and strict FP32 candidate precision, independent FP64 reference and validity exclusions. No audit shares a GPU with training/evaluation. Validate summary/raw artifacts, batch/loss/provenance and reference-validity counts. Report action/logp/SAC-gradient differences, residual curves/tails and invalid references.

Success: nine rows exist regardless of outcome. COMPLETE requires nine accepted audit results in addition to all twelve evaluated runs.

## 7. Mirror, finish and report honestly

Continue compact Mac sync every120 seconds. Retain all originals remotely and selected audit capsules locally. Check actual selected size, projected growth, remote reserve and local reserve. The freed >150GB is useful headroom but does not itself authorize deleting or indiscriminately copying all raw files. If the frozen compact cap cannot fit required artifacts, record blocked sync and revise that operational capacity constraint explicitly before advancing.

Receipts include campaign/contract/index identity, verified terminal artifact hashes, local capacity and required attempt IDs. Use coherent snapshots for mutable metadata. After all terminal statuses and final reports are published, obtain a receipt covering those exact final artifacts before stopping the new sync supervisor. Remote scientifically complete with local final sync pending is a distinct state.

Final report gives counts and statuses for12 runs,12 production evaluations and9 audits, plus separate4 smokes/4 smoke evals/4 pilots. Preserve failed and interrupted old TF32 rows in a separate section, never pool them with strict controls. Report per-seed U4/success/return and durations, actual policy/source hashes, separate training/evaluation denominators, cap-reaching versus nonconvergence, residual tails and reference-invalid cases. ETA is updated from each slot's remaining sequential workload, with evaluation/audit/sync overhead and observed late-cost uncertainty. Do not claim exact finite-sampler likelihood, guaranteed high-cap convergence or equal-wall-clock comparison.


Canonical implementation notes: the shared failure latch is separate from PREFLIGHT_TERMINAL_FAILED, which is published only after sibling futures join and owned children finish. Local sync must not stop on the initial failure latch. Any CLEANUP_FAILED evidence keeps sync alive for recovery. Only the exact historical late-A2 row473/step2 shared failure is preclassified as an implementation-gate exception. Every new FP32/FP64 shared failure requires numerical review. Original scientific tolerances remain unchanged.
