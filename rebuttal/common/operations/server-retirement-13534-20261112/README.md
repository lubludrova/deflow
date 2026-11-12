# Server retirement verification — port 13534

Scope: DEFlow E02 sequential Anderson, E04 cap100, E05 cap10, original stopped attempts, completed restart1 campaigns, and E06 fixed-input table audits. This operation does not delete or stop the server.

The transfer retains every selected checkpoint and capsule, episode-level evaluations, TensorBoard events, source/configuration/command records, training and evaluation logs, validity masks, raw numerical audit tensors, environment package versions, and unique non-trace historical archive files. Each selected file must match its server SHA-256 and byte size locally before readiness is recorded.

The original user prohibition on full raw downloads remains in effect. Full `solver_calls.jsonl` per-call traces and the redundant full `results.tar.gz` archive are excluded. Their paths, sizes, and SHA-256 are recorded. Unique non-trace members of the historical archive are extracted separately and retained. Recreating every historical solver call is outside the preserved scope. Runtime binaries, Python caches, macOS metadata, and credentials are not needed or copied.

`INVENTORY.json` is the source manifest. `VERIFIED.json` is produced only after successful transfer and verification of every selected file. `READY.md` is produced only after the table analysis and final checkpoint/evaluation validation also pass. Until these final records exist, this document is not a readiness certificate.

The full server is not an independently audited backup of unrelated user directories. Readiness applies to the DEFlow scientific evidence named above.
