"""Enumerate compact mirror and hash immutable acceptance artifacts."""
import json
import time
from common import *


def build_index():
    contract = contract_check()
    files = set()
    hashes = {}
    for base in ('ops', 'results', 'evaluations', 'preflight', 'audits', 'validation'):
        for path in (ROOT / base).rglob('*'):
            if not path.is_file() or path.name.endswith(('.tmp', '.lock')) or '__pycache__' in path.parts:
                continue
            if path.name.startswith('capsule_') or path.name in ('solver_calls.jsonl', 'raw.pt', 'sync_receipt.json'):
                continue
            if path.suffix == '.pt' and time.time() - path.stat().st_mtime < 30:
                continue
            if path.suffix not in ('.json', '.md', '.log', '.txt', '.py', '.sh', '.pt', '.exit') and 'tfevents' not in path.name and path.name != 'resource_history.jsonl':
                continue
            files.add(str(path.relative_to(ROOT)))
    selection = OPS / 'audit_selection.json'
    if selection.exists():
        for row in read(selection):
            if 'path' in row:
                files.add(str(Path(row['path']).relative_to(ROOT)))
    accepted_attempts = {}
    for path in (OPS / 'attempts').glob('*/ACCEPTED.json'):
        row = read(path)
        if any(row.get(k) != v for k, v in binding().items()):
            raise RuntimeError('Acceptance receipt from another campaign')
        if (path.parent / 'FAILED.json').exists():
            raise RuntimeError('Contradictory accepted and failed attempt')
        artifacts = dict(row['artifacts_sha256'])
        artifacts[str(path.relative_to(ROOT))] = digest(path)
        if not all(digest(ROOT / name) == expected for name, expected in artifacts.items()):
            raise RuntimeError('Accepted immutable artifact changed')
        files.update(artifacts)
        hashes.update(artifacts)
        accepted_attempts[row['run_id']] = artifacts
    for name in ('campaign_report.json', 'campaign_report.md'):
        if (ROOT / name).exists():
            files.add(name)
    immutable_names = {'EXIT.json', 'DONE.json', 'FAILED.json', 'PREFLIGHT_DONE.json',
        'PREFLIGHT_FAILED.json', 'PREFLIGHT_TERMINAL_FAILED.json', 'CAMPAIGN_TERMINAL.json', 'FINAL_SYNC_ACK.json', 'COORDINATOR_FAILED.json'}
    for name in sorted(files):
        path = ROOT / name
        if path.suffix == '.pt' or path.name in immutable_names or name in contract['frozen_files']:
            hashes[name] = digest(path)
    if any((OPS / name).exists() for name in ('CAMPAIGN_TERMINAL.json', 'COORDINATOR_FAILED.json')):
        for name in ('campaign_report.json', 'campaign_report.md'):
            if (ROOT / name).exists():
                hashes[name] = digest(ROOT / name)
    return dict(binding(), at=time.time(), files=sorted(files), sha256=hashes,
        accepted_attempts=accepted_attempts,
        selected_bytes=sum((ROOT / name).stat().st_size for name in files))


if __name__ == '__main__':
    print(json.dumps(build_index(), sort_keys=True))
