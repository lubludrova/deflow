"""First evaluation after a verified administrative pre-admission failure."""
import json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import runner
run_id=sys.argv[1]
m=runner.contract()
c=json.loads((HERE/'contract.json').read_text())
assert runner.digest(ROOT/'ops/manifest.json')==c['original_manifest_sha256']
assert runner.digest(Path(__file__))==c['script_sha256']
assert run_id in c['allowed_run_ids']
j=next(x for x in m['jobs'] if x['run_id']==run_id)
original=ROOT/'attempts'/run_id/'0000'
assert json.loads((original/'training_exit.json').read_text())['exit_code']==0
failure=json.loads((original/'FAILED.json').read_text())
assert failure['traceback'].rstrip().endswith('AssertionError: Peer identity changed')
assert not (original/'evaluation_launch.json').exists()
assert not (ROOT/'evaluations'/run_id).exists()
ident=json.loads((original/'claim.json').read_text())['supervisor']
assert not Path(f"/proc/{ident['pid']}").exists(), 'Original supervisor still present'
a=HERE/'attempts'/run_id/'0000'
a.mkdir(parents=True,exist_ok=False)
runner.write(a/'claim.json',dict(time=runner.now(),identity=runner.identity(os.getpid()),original_failure_sha256=runner.digest(original/'FAILED.json'),contract_sha256=runner.digest(HERE/'contract.json'),scope='First evaluation only, no training or scientific retry'))
try:
    accepted=json.loads((original/'TRAINING_ACCEPTED.json').read_text())
    checkpoint=ROOT/accepted['checkpoint']
    assert runner.digest(checkpoint)==accepted['sha256']
    runner.wait_sync(a,accepted['checkpoint'])
    peer=Path(m['peer']['remote_root'])
    assert runner.digest(peer/'ops/manifest.json')==m['peer']['manifest_sha256']
    pm=json.loads((peer/'ops/manifest.json').read_text())
    pj=next(x for x in pm['jobs'] if x['gpu']==j['gpu'])
    pa=peer/'attempts'/pj['run_id']/'0000'
    while True:
        terminal=(pa/'EVALUATION_ACCEPTED.json').exists() or (pa/'FAILED.json').exists()
        free=not any(j['gpu_uuid'] in row for row in runner.gpu_snapshot()['processes'])
        if terminal and free:break
        runner.write(a/'waiting_peer.json',dict(time=runner.now(),peer_terminal=terminal,gpu_free=free))
        time.sleep(15)
    argv=[v.replace('{CHECKPOINT}',str(checkpoint)) for v in j['eval_argv']]
    runner.run_child(j,'evaluation',argv,a)
    p=ROOT/'evaluations'/run_id/'evaluation.json'
    ev=json.loads(p.read_text())
    assert len(ev['episodes'])==1000 and ev['eval_seed']==26152
    assert ev['solver']=='anderson' and ev['checkpoint_sha256']==accepted['sha256']
    runner.write(a/'EVALUATION_ACCEPTED.json',dict(time=runner.now(),sha256=runner.digest(p),summary=ev['summary'],original_training_checkpoint_sha256=accepted['sha256']))
except BaseException:
    runner.write(a/'FAILED.json',dict(time=runner.now(),traceback=traceback.format_exc()))
    raise
