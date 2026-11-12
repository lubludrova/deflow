# DEFlow rebuttal evidence

Curated on 12 November 2026 for [dkt6](reviews/dkt6.md), including the completed
RTX 5090 joint Anderson cap100 and cap10 campaigns. This is a research evidence
snapshot, not a final submitted rebuttal or a checkpoint distribution.

| Material | Location |
| --- | --- |
| Response drafts | [reviews/dkt6.md](reviews/dkt6.md), [VrBX.md](reviews/VrBX.md), [agWk.md](reviews/agWk.md) |
| All 35 concerns and shared notes | [common/ISSUES.md](common/ISSUES.md), [response-notes.md](common/response-notes.md) |
| Paper and original reviews | [common/sources](common/sources/) |
| Claim-to-file map and limits | [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md) |
| Publication checks and file hashes | [publication](common/publication/) |
| Operations and current campaign pointers | [common/operations](common/operations/README.md) |

## Experiments retained

| Experiment | Evidence |
| --- | --- |
| [E01](experiments/E01-solver-comparison/README.md) | Joint Anderson cap25, cap500 and tighter cap100, three seeds each. Nine paired numerical audits, including the cap100 solver setting on A25-trained inputs. |
| [E02](experiments/E02-sequential-anderson/README.md) | Sequential Anderson, three completed seeds. Matched-input joint/sequential timing and residual measurements. |
| [E03](experiments/E03-tighter-anderson-audit/README.md) | Nine tighter-setting audits on the same E01 inputs, raw tensors and validity masks. |
| [E04](experiments/E04-joint-anderson-cap100/README.md) | Joint Anderson tol1e-5/cap100, three completed `restart1` seeds with 1000-episode evaluations and verified final evidence. |
| [E05](experiments/E05-joint-anderson-cap10/README.md) | Joint Anderson tol1e-5/cap10, three completed `restart1` seeds with 1000-episode evaluations and verified final evidence. |
| [E06](experiments/E06-anderson-table-audit/README.md) | All five solver settings on nine matched A25 inputs, RTX 5090 timings, raw numerical evidence and early-stopping analysis. |

The bundle retains training/evaluation logs, TensorBoard events, episode records,
telemetry summaries, source/configuration hashes, frozen source, attempt receipts,
36 raw audits, nine audit input capsules and two timing input capsules. Selected
`.pt` files contain numerical evidence or replay inputs, not training checkpoints.
Checkpoint paths/hashes in historical receipts remain provenance references.

## Archive and active work

`archive/` is local and excluded from Git. It holds superseded revisions,
Newton-only training cohorts and investigations, old diagnostics, incomplete full
backups, stopped pre-restart E04/E05 attempts, unused tensor artifacts and inactive
checkpoints. Nothing was deleted. The exact move receipt is local at
`archive/publication-20261112/moves.json`. Ignored compatibility aliases preserve
old local paths. They are not needed to read the published evidence.

Live E04/E05 mirrors, checkpoints and monitoring state remain at their original
paths, ignored where mutable. Their existing synchronization and evaluation gates
continue unchanged. Git contains fixed `publication-snapshot-20261112/` copies,
not a live dashboard. No checkpoint, virtual environment, partial backup or full
per-call solver trace was downloaded for this publication. Full traces and
unselected capsules remain on their original servers.

Historical documents contain old status labels and absolute paths. Use the dated
snapshot and [current handoff](RUNNER_HANDOFF.md) to distinguish completed results,
user-stopped attempts and the six explicitly reauthorized `restart1` jobs.

## Completed cap10/cap100 update

E04/E05 final reports and `completion-snapshot-20261112/` supersede their earlier running-state snapshots. E06 supplies all numerical columns of the current five-setting table using RTX5090/PyTorch2.7.1+cu128. Success comes from the independent training cohorts. Checkpoints and the 32.48GB verified server copy remain local-only. [Current publication verification](common/publication/completed-cap10-cap100/README.md) describes the evidence and checks.
