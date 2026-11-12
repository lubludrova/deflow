# Current handoff — completed cap10/cap100

E04/E05 restart1 and E06 are complete. Final evidence is in the E04/E05 `completion-snapshot-20261112/` folders and E06 `analysis/`. No training, evaluation or audit is queued. All task-owned monitoring and transfer processes have ended. Server preservation is verified, subject to the recorded full-raw exclusions. The sections below are dated operational history.

## Earlier handoff — 12 November 2026

E01 is stopped. E02 sequential Anderson has three completed 200k trainings and
three accepted 1000-episode evaluations. E03 has nine completed numerical audits.
The only active training campaigns are E04/E05 `restart1`, three seeds each,
explicitly reauthorized after the global stop. No other restart is authorized.

Current immutable evidence and publication limits: [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md).
Responses: [reviews](reviews/). Shared notes and all 35 concerns: [common/ISSUES.md](common/ISSUES.md).

| Active campaign | Frozen handoff | Status command from repository root |
| --- | --- | --- |
| E04 cap100 | [handoff](experiments/E04-joint-anderson-cap100/runs/a100-20261112-restart1/HANDOFF.md) | `python3 rebuttal/experiments/E04-joint-anderson-cap100/runs/a100-20261112-restart1/ops/status.py` |
| E05 cap10 | [handoff](experiments/E05-joint-anderson-cap10/runs/a10-20261112-restart1/HANDOFF.md) | `python3 rebuttal/experiments/E05-joint-anderson-cap10/runs/a10-20261112-restart1/ops/status.py` |

The dated publication snapshots are fixed. Live mirrors and synchronization
continue separately at their existing local and remote paths. Do not change
scientific settings, restart original campaigns, or bypass the E04 final-ACK /
E05 evaluation gates. Publication did not stop jobs or alter launch services.

The ignored local archive holds old attempts, diagnostics and checkpoints.
Ignored compatibility symlinks preserve historical absolute/local paths.
Do not remove those aliases while any frozen command or sync still uses them.

## E04/E05 completion — 2026-11-12T13:45:25.630771+00:00

Both restart1 campaigns are complete: six trainings at 200k real transitions and six 1000-episode stochastic evaluations, eval seed 26152, all child exit codes 0. All 48 checkpoints and six evaluation files match local/server SHA-256. Final authority is each restart1 `ops/COMPLETED.json` and `analysis/REPORT.md`. E05 seeds 1/3 used the recorded first-evaluation administrative handoff, preserving original failure markers. The original E05 status reader can therefore show historical FAILED labels. No further training or evaluation is queued. Publication snapshots remain unchanged. A final live GPU query was unavailable due to SSH transport failure.

Final follow-up at 13:44:46 UTC: the existing SSH control connection succeeded. All three GPUs were idle, with no compute processes. Verified completion allowed unloading the five restart1-only monitoring, sync and keepawake services. No files were deleted. See each restart1 `ops/FINAL_OBSERVATION_AND_CLEANUP.json`.

## E06 table audit and server preservation — 2026-11-12T14:19:08.252723+00:00

E06 completed 18 fixed-input audits on exclusive RTX5090 GPU1 for the five requested settings. No training or retries. All per-case exits, input hashes, local copies, reference masks and two independent table reductions passed. Table: `experiments/E06-anderson-table-audit/analysis/TABLE.md`. All numerical columns were remeasured on RTX5090 to avoid mixed-hardware timing. Only the table in `reviews/dkt6.md` was updated, with its prior text archived under E06 analysis.

Server 13534 preservation is in progress at `common/operations/server-retirement-13534-20261112/`. Do not infer readiness until its `VERIFIED.json` and `READY.md` exist. This preserves all selected scientific artifacts, excluding full per-call solver traces and the full raw archive under the earlier user constraint. No server deletion is performed.

## Server 13534 preservation completed — 2026-11-12T14:43:37.379455+00:00

All3435 selected files (32,476,921,072 bytes) match server SHA-256 locally. All95 checkpoint files load with finite model tensors. The18 E06 audits and independent table reductions passed. All GPUs are idle and task-owned local transfer/watch processes have exited. Read `common/operations/server-retirement-13534-20261112/READY.md` for the explicit full-raw exclusions and final retirement evidence. No server or file was deleted.
