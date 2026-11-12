# E01 — joint Anderson training and paired audits

This curated export retains the evidence used by the dkt6 draft. The original
strict-FP32 revision3 science, source files and internal evidence paths are unchanged.

| Condition | Run IDs | Final evaluations |
| --- | --- | --- |
| Joint Anderson tol1e-5/cap25 | `anderson_s1/s2/s3` | 3 × 1000 stochastic episodes |
| Joint Anderson tol1e-5/cap500 | `anderson_highcap_s1/s2/s3` | 3 × 1000 stochastic episodes |
| Joint Anderson tol1e-6/cap100 | `anderson_tighter_s1/s2/s3` | 3 × 1000 stochastic episodes |

All retained trainings reached 200000 real transitions. Sources and contract:
[revision3](runs/deflow-campaign-20261111/revision3/),
[execution contract](runs/deflow-campaign-20261111/revision3/ops/execution_contract.json).
Scientific source is in `revision3/package/` and `revision3/DEFlow/`.

Results and TensorBoard curves: [mirror/results](runs/deflow-campaign-20261111/revision3/mirror/results/).
Final episode records: [mirror/evaluations](runs/deflow-campaign-20261111/revision3/mirror/evaluations/).
Training/evaluation logs and attempt receipts: [mirror/ops](runs/deflow-campaign-20261111/revision3/mirror/ops/).
Numerical report: [REPORT.md](runs/deflow-campaign-20261111/revision3/mirror/ops/idle-audits-v1/analysis/REPORT.md).
Raw tensors and masks: [nine audits](runs/deflow-campaign-20261111/revision3/mirror/audits/).

Nine A25 actor-gradient capsules near 25k/100k/200k are retained under the three
`mirror/results/anderson_s*` directories. Actual steps and hashes are in the
[audit manifest](runs/deflow-campaign-20261111/revision3/ops/idle-audits-v1/manifest.json).
There are five fully valid batches and four reference-invalid batches. Preserve
all masks and N/A gradient rows. The original raw audit comparison includes Newton.

E01 is stopped. Do not restart its historical coordinator. Obsolete revisions,
Newton-only training/evaluation cohorts, diagnostics, unused preflight tensors,
incomplete full backups and checkpoints are in the ignored local archive.
Frozen contracts still describe the broader original campaign. Failed/stopped
Newton outcomes remain documented in shared preservation records. External/local
compatibility aliases remain available, but are excluded from Git.

RTX4090/torch2.5.1+cu124 and shared-GPU intervals differ from E02/E04/E05.
Full-run elapsed is not a controlled cross-campaign solver speed comparison.
See [evidence index](../../EVIDENCE_INDEX.md) for the exact claim mapping.
