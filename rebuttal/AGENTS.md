# Rebuttal working contract

Work in the DEFlow repository. Keep new rebuttal files in this area.

## Entry points

- `README.md`: the four-folder workflow.
- `common/ISSUES.md`: the single concern register, working notes and next checks.
- `reviews/VrBX.md`, `reviews/dkt6.md`, `reviews/agWk.md`: editable responses.
- `experiments/E01-solver-comparison/README.md`: current experiment and evidence.
- `common/operations/active_campaign.json`: canonical local/remote paths and status command.
- The repository's `.agents/remote-experiment.json` points to that same profile.

Reviewer mapping is stable: R1=VrBX, R2=dkt6, R3=agWk. Preserve all 35 concern IDs.
One experiment may address multiple reviewers. Keep one canonical evidence set
and reference it from the board and responses. New studies use the next unused `E<NN>-<question>`
with one README, code, runs and analysis. Do not create another parallel tracker.

## Running campaign

Discover the current frozen plan and execution contract through the profile and
experiment README. The existing coordinator owns approved queue advancement.
Never duplicate it or replay already-completed setup stages.

Do not modify frozen sources, precision, hyperparameters, seeds, horizons or
failure thresholds. No scientific retries or checkpoint resume. Do not interrupt
live jobs or add competing GPU diagnostics without authorization. Follow the
approved failure policy. New scientific conditions need separate approval.

Preserve run IDs, failed attempts, checkpoints, capsules, raw telemetry,
reference-invalid rows, and native/fresh residual masks. Separate training
completion, 1000-episode evaluation, and paired audit completion. Pilots and drafts
are not final scientific evidence.

## Preservation and reporting

The active campaign's original internal layout is retained under `runs/`.
Its historical documents are dated evidence, not current launch instructions.
Do not remove the external compatibility aliases required by its frozen contract.
Do not change remote active paths. Keep sync and task-owned keepawake working.

`archive/` contains previous workspace documents, handovers and migration
receipts. Do not execute old installers or migration scripts. Never prune
artifacts merely to simplify navigation. Do not mix new results into the
repository's published `results/` cohorts. Keep credentials out of files.

For every conclusion retain source/hash, measured interval, units, denominator,
validity masks and limitations. Distinguish historical claims, new measurements,
observations and controlled interventions. Archive originals before replacing
working documents. Historical status labels do not establish current completion.
