"""Retain the frozen transfer verifier, extending shutdown to the packed campaign."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(ROOT / 'ops'))
import common as c
import sync_local as frozen


def terminal_ready(mirror, verified):
    prefix = 'ops/packing-v1/'
    ack_path = mirror / (prefix + 'FINAL_SYNC_ACK.json')
    if not ack_path.exists() or prefix + 'FINAL_SYNC_ACK.json' not in verified:
        return False
    ack = json.loads(ack_path.read_text())
    if ack.get('packing_sha256') != os.environ['DEFLOW_PACKING_SHA256']:
        raise RuntimeError('Packing final acknowledgment binding mismatch')
    if not all(verified.get(k) == v for k, v in ack['final_sha256'].items()):
        raise RuntimeError('Packed final artifacts not verified locally')
    if (mirror / 'ops/CLEANUP_FAILED.json').exists():
        return False
    return True


def main():
    config = c.read(BASE / 'manifest.json')
    if c.digest(BASE / 'manifest.json') != os.environ['DEFLOW_PACKING_SHA256']:
        raise RuntimeError('Local packing manifest anchor mismatch')
    for name, sha in config['files_sha256'].items():
        if c.digest(BASE / name) != sha:
            raise RuntimeError('Local packing source changed: ' + name)
    frozen.terminal_ready = terminal_ready
    frozen.main()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        c.write(ROOT / 'ops/sync_status.json', dict(at=time.time(), result='error', traceback=traceback.format_exc()))
        raise
