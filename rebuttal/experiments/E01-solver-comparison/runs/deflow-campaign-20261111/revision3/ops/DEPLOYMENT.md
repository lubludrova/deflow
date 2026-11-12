# Frozen revision3 operations

No remote action is performed by this document. Parent generates the final execution_contract.json after all scientific, validation and ops files pass peer review. Do not mutate a deployed revision after its preflight begins.

The required contract schema is:

```json
{
  "schema_version": 3,
  "campaign_id": "deflow-campaign-20261111-revision3",
  "precision_policy": "strict_fp32",
  "remote_root": "/home/featurize/deflow-campaign-20261111/revision3",
  "python": "/home/featurize/deflow-campaign-20261111/revision3/venv/bin/python",
  "server": {"host": "workspace.featurize.cn", "port": 18290, "user": "featurize"},
  "gpu_by_slot": {
    "0": "GPU-25aeeb0b-2f1a-6068-db50-71c18174ccba",
    "1": "GPU-8c5841b8-9c0d-e867-7e29-7714a772fa58",
    "2": "GPU-911aa272-5dd1-019f-9636-accc89c876da"
  },
  "retry_budget": 0,
  "resume": false,
  "jobs": [],
  "queue_by_gpu_slot": {},
  "frozen_files": {},
  "sync": {
    "local_root": "/Users/ruslanageev/dev/mbrl-world-models/output/deflow-campaign-20261111/revision3/mirror",
    "ssh_wrapper": "/Users/ruslanageev/dev/mbrl-world-models/output/deflow-campaign-20261111/ops/ssh.sh",
    "interval_seconds": 120,
    "max_mirror_gib": 22,
    "min_local_free_gib": 8,
    "launchd_label": "com.codex.deflow-20261111-v3-sync"
  }
}
```

Populate jobs and queue_by_gpu_slot verbatim from package/night_campaign.json. The runtime appends `--precision-policy strict_fp32` without changing original scientific argv. Populate frozen_files as relative path to SHA256 for every runtime scientific .py, all required DEFlow files, original engine_sha256.json and VALIDATION_CONVERGENCE.json, prior patch/provenance manifests, night_campaign.json, validation scripts/fixed bank, all five operational launchers, tests, RUNNER_PLAN.md and deployment instructions. Exclude the contract itself and generated evidence. The exact final contract SHA256 is the external anchor, not a field trusted inside the contract.

Run each supervisor with `DEFLOW_CONTRACT_SHA256=<literal final SHA256>` in its inherited environment. An externally supplied anchor is mandatory for preflight, campaign and sync_index. No fallback derives it from the mutable contract file. Use dedicated tmux names for preflight and campaign. A new supervisor entry cannot resume a prior coordinator attempt. Existing run/attempt roots fail closed.

Run `python -m unittest discover -s ops -p test_lifecycle.py -v` locally and remotely before preflight. Run preflight.py under the isolated interpreter. It runs original16 tests, precision tests, fixed-bank CUDA numerical gate, staged four smoke/evaluations across3 cards, four8000-transition pilots across3 cards, and fresh-bank gate. The campaign may wait for preflight but cannot admit work before a matching evidence-bound success receipt. FAILED always overrides DONE.

Local launchd template fields:

- Label: contract sync.launchd_label.
- ProgramArguments: absolute local Python3 interpreter, absolute revision3/ops/sync_local.py.
- EnvironmentVariables: DEFLOW_CONTRACT_SHA256 with literal reviewed hash.
- StartInterval:120, RunAtLoad:true.
- StandardOutPath/StandardErrorPath: new revision3/ops sync log paths.
- KeepAlive: omit. The timer owns retries of transfer only, never scientific retries.

Use the existing task-owned SSH wrapper only after verifying it targets the authorized host/port and does not expose credentials. Preserve its hash in deployment provenance outside the remote relative-file map if its bytes cannot be mapped into this revision. Local sync verifies its own code and common.py against the anchored contract. Place no password in the plist, argv, contract or logs.

STOP.json permanently blocks all future work in this revision. A supervisor SIGINT/SIGTERM writes STOP and requests owned child cleanup. PAUSE.json blocks each new plan, claim, launch, evaluation, audit and queue advance while allowing active jobs to finish. Clearing PAUSE is an explicit operator action. Clear neither STOP nor failure markers to force continuation. New scientific work after an interrupted revision requires a separately authorized new root, not deleting receipts.

Remote sync_receipt.json binds campaign/contract/precision, exact index SHA256, verified immutable path hashes and ACCEPTED attempt IDs. The exact remote index bytes are retained locally in ops/sync_index.json. Accepted evaluation receipts include its final checkpoint and train/eval evidence. A future trainer requires the predecessor evaluation ID in a verified recent receipt. A stale/error receipt blocks admission while active training is preserved. Mutable live telemetry is copied opportunistically and does not count as immutable acceptance evidence.

Campaign writes its terminal report and marker before waiting for their exact hashes to be acknowledged. Then it writes FINAL_SYNC_ACK.json. Local sync continues until it verifies that ACK and the covered final hashes, or verifies a quiescent terminal failure. The initial PREFLIGHT_FAILED latch cannot stop sync while sibling workers still execute. PREFLIGHT_TERMINAL_FAILED is required after those workers join, along with COORDINATOR_FAILED if that supervisor started. CLEANUP_FAILED always vetoes automatic sync shutdown. Only then does it stop its own launchd label. `CAMPAIGN_TERMINAL.local_sync=PENDING` is immutable history, and FINAL_SYNC_ACK plus local SYNC_TERMINAL provide the separate verified-final-sync status.

The compact cap remains22GiB unless the parent explicitly records a revised operational capacity budget before freezing. Measure current selected/checkpoint growth and free space first. No remote deletion and no automatic broadening to all raw artifacts.
