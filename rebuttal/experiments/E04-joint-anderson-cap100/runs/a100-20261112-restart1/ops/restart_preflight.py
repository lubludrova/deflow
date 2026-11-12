import json,os,subprocess,sys,hashlib,time
from pathlib import Path
root=Path(sys.argv[1]);m=json.loads((root/'ops/manifest.json').read_text());v=root/'validation'
env=os.environ.copy();env.update(m['environment'])
prior=Path(m['lineage']['previous_campaign_root'])
old=json.loads((prior/'ops/manifest.json').read_text())
for rel,h in m['frozen_files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
plans=[]
for j in m['jobs']:
 p=json.loads(subprocess.check_output(j['train_argv'][:-1]+['--plan'],env=env,text=True))
 baseline=json.loads((prior/'results'/j['run_id']/'manifest.json').read_text())
 diffs={k:(baseline['configuration'].get(k),p['configuration'].get(k)) for k in baseline['configuration'].keys()|p['configuration'].keys() if baseline['configuration'].get(k)!=p['configuration'].get(k)}
 assert set(diffs)<={'rebuttal_actor_kwargs_sha256'},diffs
 assert p['source_sha256']==baseline['source_sha256'] and p['package_sha256']==baseline['package_sha256']
 assert p['anderson']==dict(tol=1e-5,cap=m['science']['cap'])
 assert p['solver']=='anderson' and p['precision_policy']=='strict_fp32'
 (v/f"plan-s{j['seed']}.json").write_text(json.dumps(p,indent=2)+'\n')
 plans.append(dict(seed=j['seed'],configuration_differences=diffs))
for pattern in (['test_runner.py','test_shared.py'] if m['science']['cap']==10 else ['test_runner.py']):
 with (v/(pattern+'.log')).open('x') as f:r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'ops'),'-p',pattern],env=env,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,pattern
import torch
assert torch.__version__=='2.7.1+cu128'
assert torch.cuda.device_count()==3
sys.path.insert(0,str(root/'code/package'))
from deflow_rebuttal.precision import apply_policy
apply_policy('strict_fp32')
for i in range(3):
 with torch.cuda.device(i):
  x=torch.randn(4,3,device='cuda',requires_grad=True);(x@x.T).sum().backward()
  assert bool(torch.isfinite(x.grad).all());torch.cuda.synchronize()
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
assert freeze==(prior/'validation/runtime-pip-freeze.txt').read_text()
(v/'runtime-pip-freeze.txt').write_text(freeze)
assert json.loads((v/'PRIOR_RELEASE_READY.json').read_text())['passed']
receipt=dict(passed=True,at=time.time(),fresh_restart=True,prior_smoke_reused='Same verified scientific files and runtime. No new scientific smoke repetition.',configurations=plans,torch=torch.__version__,gpus=[torch.cuda.get_device_name(i) for i in range(3)],source_hashes_verified=len(m['frozen_files']))
(v/'RESTART_READY.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
