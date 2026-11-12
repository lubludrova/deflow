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
profile now selects the `ops/packing-v1` operational overlay. Its separately
pinned manifest preserves all twelve scientific jobs, seed/GPU mapping, zero
retries/resume and frozen science. The original coordinator remains the parent
of current Newton processes, retaining real exits and training acceptance.
Its future admission is administratively blocked. Do not restart or signal it.
The packed controller owns future training, evaluation and audits. First A1
shares GPU 0 with N1. A3 on GPU 2 and H2 on GPU 1 follow only after the recorded
ten-minute canary passes. Read actual active state rather than assuming six
processes. Remaining second-lane queues are A1/H1/T1, H2/T2, A3/H3/T3, each with
its final evaluation. Audits follow all lanes with exclusive per-GPU timing.
Report old administrative scheduling errors separately from scientific failures.

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
