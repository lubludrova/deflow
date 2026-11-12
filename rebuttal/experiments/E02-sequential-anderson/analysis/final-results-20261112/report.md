# Sequential Anderson: final results, 12 November 2026

All three seeds completed training and the final stochastic evaluation (1,000 episodes each, evaluation seed 26152). Training and evaluation exits are zero. Accepted checkpoints reached the contracted 200,000–200,007 transitions.

| Seed | Transitions | Training hours | U4 | Success (%) | Mean return |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 200002 | 4.7040 | 0.5570 | 86.90 | -8.1934 |
| 2 | 200006 | 4.6873 | 0.6677 | 97.70 | -7.8261 |
| 3 | 200003 | 4.7124 | 0.6344 | 91.80 | -9.0495 |

Across three seeds, mean ± sample SD: U4 **0.6197 ± 0.0568**, success **92.13 ± 5.41%**.

Seed 1 reached the fourth goal successfully in only 2/1,000 evaluation episodes, explaining its lower goal coverage. Three seeds show substantial variation, so this does not establish statistical significance.

## Verification and local preservation

Verified 90 local result, evaluation, event and attempt files against remote SHA-256, plus all 30 frozen files and the manifest. Recomputed all evaluation summary metrics from the episode records. The final sync separately verified 34 checkpoint/model/validation .pt files. See [verification.json](verification.json) for hashes, sizes and terminal receipts.

Local files: [mirror](../../runs/sa-20261112/mirror). The frozen compact-mirror policy excludes raw solver_calls and capsules, the mutable ckpt.pt alias, the remote environment and redundant code. Raw solver traces and capsules total 14.48 GB across the remote campaign and remain on the server. No files were deleted.

## Interpretation limits

Training took 4.69–4.71 hours per seed on separate RTX 5090 GPUs. These elapsed times are campaign costs. E01 used different hardware/runtime and GPU co-location, so dividing E01/E02 training durations is not a controlled solver speed comparison. The earlier fixed-input forward benchmark remains the timing evidence, with its recorded residual and accuracy limitations. Final training and evaluation completion does not imply completion of additional paired numerical audits.

The documented seed-2 process-start identity race remains preserved in the original receipts and separate identity reconciliation. Completion here is established from supervisor exit codes, accepted markers, horizon, checkpoint linkage and independently verified output files.
