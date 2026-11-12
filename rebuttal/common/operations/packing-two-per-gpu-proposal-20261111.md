# Proposed two-training placement for the existing strict-FP32 campaign

Status: proposal only. No second training has been launched. The user requested
at most two runs per GPU on 2026-11-11, prioritizing early usable results and
completion of the entire existing campaign. This authorizes considering a new
placement policy, not changing any scientific setting or silently weakening the
current controller's checks.

## Suggested second lane

- Physical GPU 0: A1 → evaluation A1 → H1 → evaluation H1 → T1 → evaluation T1.
- Physical GPU 1: H2 → evaluation H2 → T2 → evaluation T2. A2 is already accepted.
- Physical GPU 2: A3 → evaluation A3 → H3 → evaluation H3 → T3 → evaluation T3.

The existing lane contains N1/N2/N3 and their prescribed final evaluations.
The total bound is two owned compute processes per physical GPU, including
any simultaneous evaluation. Do not expand beyond the original 12 trainings,
12 final evaluations and nine prescribed audits. Keep seed-to-GPU mapping.
Run paired audits after the training/evaluation queues, with exclusive timing
conditions and the original capsule selection rules. Final completion still
requires verified final synchronization acknowledgment.

A1 and A3 complete the fast A controls for all three seeds. H2 supplies the first
large-cap control while the other A seeds train. The accepted A2 training took
8571.4 seconds (about 2 h 23 m) with exclusive placement. This is an observed
single-seed duration, not a two-process ETA. Early single-process pilot rates
were A=23.07, H=15.87 and T=12.19 real transitions/s. Late solver cost may differ
substantially. These pilots do not prove two-process throughput or optimality.

## Required implementation before launching

The current frozen controller is incompatible with packing:

- `common.capacity` requires an empty GPU.
- `common.execute` takes an exclusive per-GPU lock.
- `campaign.monitor` treats every co-runner as an admission violation.
- Queue order is frozen and checked against the package manifest.
- The current local sync supervisor stops on the original final acknowledgment.

Do not edit these frozen files, change the active contract hash, or start a
sidecar trainer and ignore the resulting admission failure. A new reviewed
operational controller and explicit handover are required before new launches.
The implementation must preserve the already-running Newton processes and their
original supervisor until original attempt-scoped exit and acceptance receipts
are durable. Never SIGTERM the old coordinator casually: its cleanup terminates
owned children. Do not duplicate queues or infer process exit codes after
removing their supervisor.

A viable handover must first prevent old future admission while preserving
current children and receipts, then transfer ownership of each not-yet-started
training/evaluation exactly once. Preserve original controller terminal statuses
as historical operational evidence. Administrative handover failures must not be
presented as scientific training failures. Maintain one canonical current report
that links original and new operational evidence. The local sync lifecycle must
cover the complete new controller, required artifacts and its final hashes,
without stopping when the superseded controller finishes.

## Review and verification criteria

Before execution, pin the new launcher's exact source hash and persist the exact
queue, identities of adopted attempts, allowed GPU UUIDs, maximum concurrency,
zero retry/resume budget, reserve limits, sync target and no-deletion policy.
Test atomic claims, duplicate admission, unknown-process exclusion, old/new
handover races, failures, separate training/evaluation acceptance and final-sync
shutdown. Do not launch until this implementation exists and passes review.

Then start A1 as the first co-located scientific run. Verify its claim, exact
PID/start identity, assigned GPU, actual strict-FP32 state and progress. Observe
both jobs through post-learning-start operation for at least ten minutes before
expanding to A3 and H2. Report each job's rate and slowdown, GPU utilization and
memory, and aggregate normalized progress relative to suitable solo evidence.
Raw SPS summed across different solvers does not establish a fair speedup.
Do not interrupt a started scientific run just to undo a placement test. If
contention is unacceptable, block new admissions and report the evidence.

Tag training cost intervals before and after co-location. Shared-GPU timings are
not directly comparable to the original exclusive-GPU timings. Retain both
rather than mixing them into a clean solver-cost estimate.
