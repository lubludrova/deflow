# Evidence used by the dkt6 draft

The three response drafts were moved into `reviews/` without editing their text.
This index identifies evidence, not approval of every interpretation in a draft.
Paths below are relative to this directory. Original-paper claims in W2 and
Appendix C.1 are sourced by [the paper](common/sources/DEFlow_main.pdf) and shared
[paper/code checks](common/evidence/). Published `src/` and `results/` were not changed.

## Joint versus sequential Anderson

- Joint A25 quality: [E01 evaluations](experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/mirror/evaluations/), `anderson_s1`, `anderson_s2`, `anderson_s3`.
- Sequential quality: [E02 final report](experiments/E02-sequential-anderson/analysis/final-results-20261112/report.md), [episode records and receipts](experiments/E02-sequential-anderson/runs/sa-20261112/mirror/).
- Same-input timing: [measurement report](experiments/E02-sequential-anderson/runs/sa-20261112/mirror/validation/fixed-capsules-v3/report.json), two adjacent `*_raw.pt` files, [two saved input capsules](experiments/E02-sequential-anderson/runs/sa-20261112/validation/inputs/), [protocol and limitations](experiments/E02-sequential-anderson/analysis/table-update-20261112-1233/report.md).
- Joint U4 = 0.6652 ± 0.0033 and success = 96.17 ± 1.10%. Sequential U4 = 0.6197 ± 0.0568 and success = 92.13 ± 5.41%. Each is mean ± sample SD over three completed seeds, 1000 stochastic evaluation episodes per seed.
- Joint forward latency = 19.55/19.81 ms and sequential = 66.82/66.63 ms on A2/N3 inputs. The 3.36–3.42× ratio is forward latency on two frozen batches. Achieved residuals differ. E01/E02 hardware/runtime and GPU sharing differ, so the draft's approximately 3.5× full-training speed claim is not established by this benchmark.

## Numerical reliability and training solver settings

- [E01 audit report](experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/mirror/ops/idle-audits-v1/analysis/REPORT.md), adjacent `metrics.json` and `per_audit.csv`.
- [Nine E01 raw audits](experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/mirror/audits/), including invalid reference masks and the original Newton comparison. Nine selected input capsules remain under `mirror/results/anderson_s{1,2,3}/`.
- [E03 tighter-setting report](experiments/E03-tighter-anderson-audit/analysis/REPORT.md), [metrics](experiments/E03-tighter-anderson-audit/analysis/metrics.json), [nine raw audits and exit receipts](experiments/E03-tighter-anderson-audit/runs/tol1e-6-cap100-20261112/mirror/).
- Training success for 1e-5/cap25, 1e-5/cap500 and 1e-6/cap100 comes respectively from `anderson_s*`, `anderson_highcap_s*` and `anderson_tighter_s*` in the E01 evaluations directory. All nine final evaluations and learning curves are retained.
- Numerical errors compare settings on A25-trained capsules. Five of nine batches are fully reference-valid, giving 5120 trajectories for pooled errors. Four batches retain five invalid rows in total. Timing covers all nine batches. Do not pool invalid full-batch gradient comparisons into the error aggregate.
- The E01 cap100 audit is not a completed independent cap100 training experiment. Cap10 audit measurements are not available in this bundle. Empty draft cells remain empty.

## Ongoing RTX 5090 campaigns

| Condition | Canonical attempt | Published snapshot |
| --- | --- | --- |
| 1e-5/cap100 | E04 `a100-20261112-restart1` | [snapshot](experiments/E04-joint-anderson-cap100/runs/a100-20261112-restart1/publication-snapshot-20261112/) |
| 1e-5/cap10 | E05 `a10-20261112-restart1` | [snapshot](experiments/E05-joint-anderson-cap10/runs/a10-20261112-restart1/publication-snapshot-20261112/) |

Each snapshot has a capture interval, SHA256 inventory and observed process
identities. All six jobs were training when captured. These are fresh replacement
attempts, not checkpoint continuation of the stopped originals. Each GPU hosts
matching E04/E05 seeds. No final success/U4 aggregate can yet be reported.
Original stopped attempts are local archive provenance, not final samples.

## Verification and reuse

[Publication verification](common/publication/verification.json) recomputes the
completed evaluation metrics from episode records and checks required evidence,
source hashes, byte-preserved response drafts and Git exclusions.
[File inventory](common/publication/files.sha256) records the committed evidence
snapshot. Checkpoint hashes are retained for provenance, but checkpoints themselves
are excluded. Frozen scripts can contain original server or local paths. This
bundle is sufficient to inspect the reported evidence without those machines,
not a promise that operational launch scripts are portable or safe to rerun.
