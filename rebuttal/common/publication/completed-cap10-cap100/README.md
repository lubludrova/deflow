# Completed cap10/cap100 publication

This update publishes E04/E05 restart1 results, six final episode-level evaluations, training curves and terminal receipts, and the E06 fixed-input audit of all five table settings. Checkpoints and the 32.48GB local server-retirement data copy are excluded. Compact retirement verification receipts are retained.

E04/E05 `completion-snapshot-20261112/` directories are complete immutable copies of selected terminal artifacts. Their older `publication-snapshot-20261112/` directories are unchanged running-state history. For E05 seeds1/3, original administrative admission failures and the separate successful first-evaluation receipts are both retained.

E06 includes all eighteen raw audit outputs, masks, nine unchanged A25 input capsules, scientific source, exact commands, and two independent reductions. The entire numerical table now uses RTX5090/PyTorch2.7.1+cu128. Success is measured separately on the corresponding trained policies. Full-batch gradient errors use five fully valid reference batches, log-density errors pool5120 trajectories, and timing includes all nine batches. See E06 `analysis/METHOD.md`, `NOTES.md`, and `EARLY_STOPPING.md`.

Run `python3 rebuttal/common/publication/completed-cap10-cap100/verify.py` from any checkout location. This check requires only the Python standard library. It recomputes the six evaluation results and aggregates, checks final receipts and snapshots, and verifies E06 evidence hashes and reported table/stopping statistics. The prior publication verifier describes the earlier snapshot and is not the validator for this completed update.

`files.sha256` records the newly published staged file contents, excluding this manifest and its verification receipt. Local absolute paths in historical receipts are provenance. Scientific tensor inputs/outputs are retained, but training checkpoint binaries are not part of this Git repository.
