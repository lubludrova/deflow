"""Frozen transfer verification, shutdown gated by this operational revision."""
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
    prefix = 'ops/idle-audits-v1/'
    path = mirror / (prefix + 'FINAL_SYNC_ACK.json')
    if not path.exists() or prefix + 'FINAL_SYNC_ACK.json' not in verified:
        return False
    ack = c.read(path)
    if ack.get('packing_sha256') != os.environ['DEFLOW_PACKING_SHA256']:
        raise RuntimeError('Idle-audit final acknowledgment binding mismatch')
    if not all(verified.get(k) == sha for k, sha in ack['final_sha256'].items()):
        raise RuntimeError('Final idle-audit artifacts are not verified')
    return not (mirror / 'ops/CLEANUP_FAILED.json').exists()


def main():
    from controller import packing_check
    packing_check()
    frozen.terminal_ready = terminal_ready
    frozen.main()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        c.write(ROOT / 'ops/sync_status.json', dict(at=time.time(), result='error', traceback=traceback.format_exc()))
        raise
