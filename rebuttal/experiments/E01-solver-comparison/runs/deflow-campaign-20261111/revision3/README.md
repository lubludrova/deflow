**Current authoritative plan: [RUNNER_PLAN.md](RUNNER_PLAN.md).**

This is a fresh common strict-FP32 condition for the original twelve N/A/H/T runs. The user approved stopping revision2, retaining its artifacts, independent Astra review, the necessary fixes, fresh preflight and restart from scratch. No resume or pooling of old TF32 results with these controls is allowed.

`package/RUNNER_PROMPT.md`, `package/README.md`, and `package/VALIDATION_CONVERGENCE.json` retain the historical handoff and validation context. They do not supersede RUNNER_PLAN.md or authorize skipping the new numerical/runtime gates. `package/engine_sha256_historical.json` and `package/REVIEWED_SOURCE_MAP.json` preserve the original-to-new source provenance. The new engine changes numerical-policy initialization only. Sequential Newton source remains byte-identical.

The externally anchored `ops/execution_contract.json` pins all runtime code, expected source maps, the revised plan and numerical bank. Runtime receipts, preflight verdicts and terminal markers are the evidence. Review statements are not substitutes for actual CUDA validation.

Per-slot queue: GPU0 N1→A1→H1→T1, GPU1 A2→N2→H2→T2, GPU2 N3→A3→H3→T3. Every run is fresh, seed1/2/3, 200000 real transitions, one task per GPU. Each training is followed by1000-episode evaluation before the next trainer in its slot. The nine prescribed capsule audits follow the queues.

STOP.json closes this revision and requests owned-process cleanup. PAUSE.json holds new admission and permits active jobs to finish. Never remove failure/stop markers to force a retry. Scientific retries and resume remain disabled.

No credentials are stored in this revision. Local synchronization uses the previously configured task-specific SSH wrapper and macOS Keychain item. All source artifacts remain remotely available, and the declared compact mirror is bounded by the contract.
