"""Read the active remote campaign through the canonical rebuttal profile."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print the complete observation as JSON.')
    args = parser.parse_args()
    operations = Path(__file__).resolve().parent
    profile = json.loads((operations / 'active_campaign.json').read_text())
    revision = Path(profile['revision_root'])
    contract = revision / 'ops/execution_contract.json'
    if hashlib.sha256(contract.read_bytes()).hexdigest() != profile['contract_sha256']:
        raise RuntimeError('Local campaign contract hash differs from the recorded anchor')
    script = f'''from pathlib import Path
import datetime, json, sys, time
root = Path({profile['remote_root']!r})
sys.path.insert(0, str(root / 'ops'))
from common import contract_check, same_process, progress
contract_check()
overlay = {profile.get('operations_overlay')!r}
operations = root / 'ops' / overlay if overlay else root / 'ops'
if overlay:
    sys.path.insert(0, str(operations))
    from controller import packing_check
    packing_check()
live = json.loads((operations / 'campaign_live.json').read_text())
now = time.time()
runs = {{}}
recent_terminals = {{}}
for name, row in live['active'].items():
    verified = same_process(row['identity'])
    terminal = root / 'ops/attempts' / name / 'EXIT.json'
    if not verified and terminal.exists():
        marker = json.loads(terminal.read_text())
        if marker.get('run_id') == name and marker.get('identity') == row['identity']:
            recent_terminals[name] = dict(exit_code=marker['exit_code'], marker=str(terminal.relative_to(root)))
            continue
    telemetry = row['telemetry']
    current = progress(Path(row['run_path'])) if row.get('run_path') else {{}}
    runs[name] = dict(pid=row['pid'], identity_verified=verified,
        gpu_slot=row['gpu_slot'], step=current.get('step', telemetry.get('step')),
        sps=telemetry.get('delta_sps'), eta_seconds=telemetry.get('eta_seconds'),
        telemetry_age_seconds=current.get('last_telemetry_age_seconds'),
        co_runners=telemetry.get('co_runners'), returns=telemetry.get('returns'),
        scalars=telemetry.get('scalars'), gpu_utilization=telemetry.get('gpu_utilization_window'),
        elapsed_seconds=now-row.get('started_at', now),
        checkpoint_age_seconds=telemetry.get('checkpoint_age_seconds'))
report = json.loads(((operations if overlay else root) / 'campaign_report.json').read_text())
receipt = json.loads((root / 'ops/sync_receipt.json').read_text())
alerts = [str(p.relative_to(root)) for p in operations.glob('*FAILED*')]
alerts += [str(p.relative_to(root)) for p in (operations / 'failures').glob('*.json')]
alerts += [str(p.relative_to(root)) for p in operations.glob('ADMISSION_BLOCKED.json')]
alerts += [str(p.relative_to(root)) for p in operations.glob('CANARY_BLOCKED.json')]
alerts += [str(p.relative_to(root)) for p in operations.glob('STALE_*.json')]
alerts += [name + '=' + row['state'] for name, row in report['audits'].items()
           if row['state'] in ('FAILED', 'MISSING', 'BLOCKED')]
alerts += [name + '=' + row['state'] for name, row in report['runs'].items()
           if row['state'] in ('FAILED', 'INTERRUPTED', 'BLOCKED')]
if overlay:
    alerts += [str(p.relative_to(root)) for name in ('STOP.json', 'CLEANUP_FAILED.json', 'PREFLIGHT_FAILED.json', 'MONITOR_FAILED.json')
               if (p := root / 'ops' / name).exists()]
if now-live['observed_at'] > 120:
    alerts.append('MONITOR_STALE: ' + str(round(now-live['observed_at'])) + 's')
if not same_process(live['coordinator']):
    alerts.append('COORDINATOR_IDENTITY_UNVERIFIED')
alerts += [name + ': exit ' + str(row['exit_code']) for name,row in recent_terminals.items() if row['exit_code']]
alerts += [name + ': active identity unverified' for name,row in runs.items() if not row['identity_verified']]
print(json.dumps(dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    campaign_id=live['campaign_id'], contract_sha256=live['contract_sha256'],
    source_hashes_verified=True, coordinator_identity_verified=same_process(live['coordinator']),
    live_age_seconds=now-live['observed_at'], active=runs,
    jobs={{k:v['state'] for k,v in report['runs'].items()}}, counts=report['counts'],
    recent_terminals=recent_terminals,
    audits={{k:v['state'] for k,v in report['audits'].items()}},
    audit_details={{k:dict(state=v['state'], seed=v['seed'], target_step=v['target_step'],
        actual_step=v.get('actual_step'), gpu_slot=v.get('gpu_slot'),
        reference_valid=v.get('summary',{{}}).get('reference_valid'),
        reference_invalid=v.get('summary',{{}}).get('reference_invalid'),
        elapsed_seconds=v.get('receipt',{{}}).get('elapsed_seconds'))
        for k,v in report['audits'].items()}},
    run_results=report['runs'], operations_overlay=overlay,
    canary=json.loads((operations / 'CANARY_PASSED.json').read_text()) if (operations / 'CANARY_PASSED.json').exists() else None,
    remote_free_bytes=live['disk_free_bytes'], alerts=alerts,
    sync_result=receipt.get('result'), sync_age_seconds=now-receipt['at']), indent=2))
'''
    environment = ['env', 'DEFLOW_CONTRACT_SHA256=' + profile['contract_sha256']]
    if profile.get('packing_sha256'):
        environment.append('DEFLOW_PACKING_SHA256=' + profile['packing_sha256'])
    command = shlex.join(environment + [json.loads(contract.read_text())['python'], '-'])
    server = profile['server']
    output = subprocess.check_output([profile['ssh_wrapper'], server['user'] + '@' + server['host'],
                                      command], input=script, text=True, timeout=90)
    result = json.loads(output)
    local_sync = json.loads((revision / 'ops/sync_status.json').read_text())
    result['local_sync_result'] = local_sync.get('result')
    if local_sync.get('result') != 'ok':
        result['alerts'].append('LOCAL_SYNC_ERROR')
    result['canonical_mirror'] = profile['mirror']
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(result['observed_at'], result['campaign_id'])
    print('GPU  Run                    Steps        SPS     Training ETA')
    for name, row in sorted(result['active'].items(), key=lambda item: item[1]['gpu_slot']):
        sps = f"{row['sps']:.2f}" if row['sps'] is not None else '?'
        eta = f"{row['eta_seconds']/3600:.1f} h" if row['eta_seconds'] is not None else '?'
        print(f"{row['gpu_slot']:<4} {name:<22} {row['step']:>7}     {sps:>6}  {eta}")
    print('States:', ', '.join(f'{name}={state}' for name, state in result['jobs'].items()))
    print('Counts:', result['counts'])
    print('Verified coordinator identity:', result['coordinator_identity_verified'],
          'trainer identities:', all(row['identity_verified'] for row in result['active'].values()))
    print(f"Sync: {result['sync_result']}, age {result['sync_age_seconds']:.0f}s. Alerts: {result['alerts']}")
    print('ETA excludes evaluation and later queued work. Remote failures are never retried here.')


if __name__ == '__main__':
    main()
