"""Independent NumPy reduction of the published table from raw audit tensors."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import torch


root=Path(sys.argv[1])
manifest=json.loads((root/'manifest.json').read_text())
result=json.loads((root/'analysis/metrics.json').read_text())
checks={}
for setting, expected in result['metrics'].items():
    tol, cap=setting.split('_');cap=int(cap)
    times=[];log_errors=[];gradient_errors=[]
    for job in manifest['jobs']:
        if job['tol']!=float(tol) or cap not in job['caps']:continue
        p=root/'audits'/job['run_id']
        summary=json.loads((p/'summary.json').read_text())
        raw=torch.load(p/'raw.pt',map_location='cpu',weights_only=False)
        label='anderson' if tol=='1e-6' else f'anderson_cap{cap}'
        # The frozen audit uses torch.median (the lower middle of 10 repetitions).
        repetitions=np.asarray(summary[label]['forward_seconds_repetitions'],dtype=np.float32)
        times.append(float(np.sort(repetitions)[4])*1000)
        if 'reference_readout' not in raw:continue
        ref=raw['reference_readout'];current=raw[label]['readout']
        log_errors.extend(np.abs(current['logp'].numpy().astype(np.float64)-ref['logp'].numpy()).reshape(-1).tolist())
        g=current['gradient'].numpy().astype(np.float64);reference=ref['gradient'].numpy()
        gradient_errors.append(float(100*np.linalg.norm(g-reference)/np.linalg.norm(reference)))
    success=[]
    for seed in (1,2,3):
        ev=json.loads((root/'evaluations'/f'{setting}_s{seed}.json').read_text())
        flags=[x['success'] for x in ev['episodes']]
        assert len(flags)==1000 and all(x in (0,1,False,True) for x in flags)
        success.append(sum(flags)/10)
    actual={'forward_ms':float(np.median(times)),
            'log_density_median':float(np.percentile(log_errors,50)),
            'log_density_p95':float(np.percentile(log_errors,95)),
            'gradient_percent_median':float(np.percentile(gradient_errors,50)),
            'gradient_percent_p95':float(np.percentile(gradient_errors,95)),
            'success_mean':float(np.mean(success)),'success_sample_sd':float(np.std(success,ddof=1))}
    for key,value in actual.items():assert math.isclose(value,expected[key],rel_tol=1e-10,abs_tol=1e-12),(setting,key,value,expected[key])
    checks[setting]=actual
out={'passed':True,'settings':checks,'method':'Independent NumPy reduction, integer success counts, sample SD ddof=1',
     'metrics_sha256':hashlib.sha256((root/'analysis/metrics.json').read_bytes()).hexdigest(),
     'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (root/'analysis/INDEPENDENT_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
