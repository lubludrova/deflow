# E06 — complete Anderson table

User-authorized paired analysis on all nine fixed E01 A25 capsules. All five forward timings are measured on the same exclusive RTX5090, replacing older RTX4090 measurements. Frozen E01 scientific code, strict FP32, E02 runtime. Independent final success from E01/E04/E05 training evaluations. No new training, retries or resume. Error denominators and validity masks are retained.

Run: `runs/rtx5090-20261112/manifest.json`. Final table and source/hash verification will be in `analysis/`.

## Completed measurements

All 18 audit commands exited 0 and every case was synchronized and SHA-256 verified before the next case. All five settings have nine timing batches and five complete valid-reference error batches (5,120 trajectories). The 18 validity masks match E01. Maximum valid FP64 reference-readout difference versus E01 is 3.3751e-14 across runtime/hardware. Both independent PyTorch and NumPy reductions agree.

[Table](analysis/TABLE.md), [method and limitations](analysis/METHOD.md), [exact metrics](analysis/metrics.json), [independent verification](analysis/INDEPENDENT_VERIFICATION.json).

The entire numerical table is newly measured on RTX5090 with torch2.7.1+cu128. Original RTX4090 values are preserved in `analysis/PREVIOUS_TABLE.md` and E01/E03. Candidate FP32 errors can change with runtime and GPU, particularly the tighter-setting gradient tail. These are not mixed with old RTX4090 numerical results. Success remains the independent training evaluation statistic for each setting.
