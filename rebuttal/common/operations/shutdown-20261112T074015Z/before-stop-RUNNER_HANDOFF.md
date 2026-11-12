# DEFlow rebuttal runner handoff

Continue the approved solver campaign and analyse its evidence for Reviewer 2,
**dkt6**: sequential versus joint solving, numerical accuracy during training,
and computational cost. Reuse relevant evidence for VrBX and agWk.

## Find the current state

Work in the DEFlow repository. Read its agent instructions and the rebuttal
[README](README.md). Follow their links to the concern register, experiment,
and active profile, also available as `.agents/remote-experiment.json` at the
repository root. Obtain paths, SSH access, and the status command from the profile.

Use the frozen contract for settings and acceptance criteria, the plan for
context, and remote receipts for completed stages. Historical status text is
not live state. Do not replay setup simply because an old plan lists it.

The approved strict-FP32 study has **12 fresh runs**: Newton, Anderson cap25,
Anderson cap500, and tighter Anderson, each with seeds 1/2/3 and 200,000 real
transitions. Each successful training is followed by a 1,000-episode evaluation,
then nine paired audits follow the queues. Keep old TF32 results separate.

## Continue the work

1. Query the server. Verify identities, steps, terminal markers, checkpoints,
   GPU placement, source/precision records, and sync freshness.
2. Let the existing coordinator advance approved queues. Never duplicate it or
   replay completed setup. Maintain monitoring and follow its stop rules.
3. Analyse learning and cost by seed, phase, call role, and batch size. Use
   step/time deltas and solver work/backtracking to explain runtime. Reuse the speed report.
4. Validate terminal receipts, horizon, and artifacts before accepting training.
   Validate evaluation separately. Audits use prescribed actor-gradient capsules
   (batch size 1,024), identical inputs, provenance, selection rules, and retained
   invalid-reference masks.
5. Save reproducible code, compact tables, sources, and limitations tied to concern
   IDs. Report all 12 run and nine audit statuses, accepted metrics, artifact
   locations, alerts, and next jobs. Require final-sync acknowledgment for completion.

## User-facing run reports

When the user asks about runs, query live state and lead with a Markdown table,
one run per row. For "all runs", include all twelve production trainings,
including planned, failed and completed ones. Do not mix pilots or old TF32 runs
into this table. Show physical GPU index, state, real transitions / horizon,
latest training episode return, latest deterministic evaluation return with its
actual measurement step, recent delta SPS, elapsed training time and remaining
training ETA. For accepted final evaluations, also show U4 and success over the
prescribed 1,000 episodes. Use `—` for unavailable data, never zero or an invented
measurement step. If needed, keep U4/success in a compact additional column.

State the observation time and timezone. ETA comes from a named measured window,
excludes the following evaluation, queued work and audits, and may change as
solver cost evolves. Label intermediate evaluation separately from the accepted
final 1,000-episode evaluation. Mention stale metrics, errors, sync freshness and
co-running jobs when relevant. Shared-GPU timing must be labeled with its
co-runner and must not be pooled with exclusive-GPU cost measurements.

Illustrative format only, not a live status snapshot:

| Run | State | Physical GPU | Steps / target | Train return | Eval return @ step | U4 / success | Recent SPS | Elapsed | Training ETA |
| --- | --- | --- | --- | ---: | --- | --- | ---: | --- | --- |
| newton_s1 | RUNNING | 0 | 80,000 / 200,000 | -10.0 | -5.0 @ 75,000, intermediate | — | 3.0 | 4 h | ~11 h |
| anderson_s2 | EVALUATION_FINISHED | 1, historical placement | 200,005 / 200,000 | — | -8.11, final 1,000 episodes | 0.6615 / 94.9% | — | 2 h 23 m | complete |
| anderson_s1 | PLANNED | 0, planned placement | — / 200,000 | — | — | — | — | — | not running |

Immediately after the table, give the **entire queue as text**, not another
table. Distinguish the currently active queue from a proposed schedule. Include
all remaining training jobs, their evaluations, all nine paired audits and final
verified sync. For example, for the original exclusive-GPU schedule:

- GPU 0: N1 (running) → eval N1 → A1 → eval A1 → H1 → eval H1 → T1 → eval T1.
- GPU 1: A2 + eval (accepted) → N2 (running) → eval N2 → H2 → eval H2 → T2 → eval T2.
- GPU 2: N3 (running) → eval N3 → A3 → eval A3 → H3 → eval H3 → T3 → eval T3.
- Then A-seed 1/2/3 capsules near 25k/100k/200k: nine paired audits,
  followed by final reports and verified FINAL_SYNC_ACK.

Explain the experiment briefly below the queue: N = sequential Newton,
A = Anderson tol1e-5/cap25, H = Anderson tol1e-5/cap500,
T = Anderson tol1e-6/cap100. All train for 200k transitions in strict FP32 with
seeds 1/2/3. State what the comparison tests, where its artifacts go, and that it
supports E01 and reviewer dkt6 concerns about joint/sequential solving, achieved
accuracy and cost, with relevant evidence reused for VrBX/agWk.

The user approved two processes per physical GPU on 2026-11-11. The canonical
profile now selects `ops/idle-audits-v1`, which reuses the pinned `packing-v1`
implementation while retaining all live parents and real exit observers.
The separately
pinned manifest preserves all twelve scientific jobs, seed/GPU mapping, zero
retries/resume and frozen science. The original coordinator remains the parent
of current Newton processes, retaining real exits and training acceptance.
Its future admission is administratively blocked. Do not restart or signal it.
The current controller owns future evaluation and audits. Both older controllers
are administratively blocked from new admissions. Do not restart or signal
their live parents. N2 is a scientific FAILED attempt at step 131908 due to
non-finite actor loss. Its original failure evidence and last checkpoint are
verified locally. No retry/resume is authorized.

On 2026-11-12 the user approved running the original nine audits on idle GPU 1
while other physical GPUs train. Queue all three A seeds near 25k, then all
three near 100k, then near 200k, serially and exclusively on GPU 1. Scientific
settings and the frozen audit executable are unchanged. See the active
operational PROTOCOL.md, selected actual steps and capsule hashes.
The audit table covers A25/A100/A500 at tol1e-5 and the current Newton,
recorded full SAC actor objective, synchronized forward time, density errors,
full gradients and loss differences against the validated FP64 reference.
Never publish full-batch gradient errors if any reference row is invalid.
Retain all invalid masks, failures and raw.pt. The nine raw outputs and final
tables are included in verified synchronization. Existing training retains its
original GPUs and completes its prescribed final evaluations independently.
Report old administrative scheduling errors separately from scientific failures.

Audit completion receipt, 2026-11-12: all nine original paired audits finished
with exit 0 on exclusive GPU 1. All raw.pt, summaries, selected capsules and the
36-row table are SHA-256 verified locally. Do not relaunch them. Read
`mirror/ops/idle-audits-v1/analysis/REPORT.md`, `metrics.json` and `per_audit.csv`.
Full-batch reference comparisons cover five cases and 5120 trajectories only.
All three approximately-100k cases and A3 near 200k have invalid reference rows
(2/1/1/1). Preserve their N/A gradient/loss/reference-density comparisons and
their raw masks. Timing covers all nine cases. As of 02:34 UTC, N3 is evaluated,
N2 failed, and N1/T1/T3 continue. Query live state before acting.

## Preserve the experiment

- Do not change frozen sources or scientific settings. This campaign permits
  no scientific retries or resume. New conditions require approval.
- The active packed coordinator enforces at most two compute tasks per GPU.
  No pausing live jobs or competing GPU diagnostics without authorization.
  Never interfere with unknown processes.
- Preserve artifacts and compatibility aliases. Keep sync and task-owned
  keepawake working. The compact mirror is not a full raw backup.
- Distinguish native/fresh convergence, cap hits/failures, preliminary/final
  evaluation, CPU/GPU timings, and observations/controlled tests. Retain failures
  and reference-invalid rows. Do not overstate density or efficiency claims.
- Keep new analysis outside frozen files, do not reorganize during takeover,
  and keep credentials out of reports.

## Additional authorized campaign: E02 sequential Anderson

On 2026-11-12 the user separately authorized implementation, verification and immediate launch of three sequential Anderson docking seeds on `workspace.featurize.cn:13534`. This is not an alteration of the twelve E01 jobs above. Read [E02 README](experiments/E02-sequential-anderson/README.md) and its [handoff](experiments/E02-sequential-anderson/runs/sa-20261112/HANDOFF.md). Keep the two coordinators, remote roots, manifests, runtime differences and sync services separate. The original active profile and status command still refer to E01.


## Full raw transfer cancelled 2026-11-12

The user cancelled the full raw transfer and requested server-side analysis paths. Both full-backup downloaders and transport children are stopped. The E01 preservation worker and its actual exit observer exited, with recorded EXIT -15. The E02 preservation capture had already exited 0. Both temporary tmux sessions are absent. No scientific run or source data was removed or modified. Do not resume full packaging or downloading without a new user request.

Read [ANALYSIS_HANDOFF.md](common/operations/ANALYSIS_HANDOFF.md) for exact hosts, server roots, run IDs, checkpoints, raw solver_calls/capsules, final evaluations, audit tables/masks, scientific caveats and local compact mirrors. Scientific raw remains on the two servers. Existing compact sync and task keepawake remain active. Compact E02 sync freshness was restored after removing the task-owned archive from its remote synchronization root.

Cancellation and stop verification receipts remain in each campaign's `full-backups/<timestamp>/`. These interrupted local fragments have no LOCAL_VERIFIED.json and are not complete backups. Full-copy request history was archived before this update.
