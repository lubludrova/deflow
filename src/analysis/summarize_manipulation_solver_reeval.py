"""Recompute paired tables without replacing historical benchmark artifacts."""

import argparse
import json
import hashlib
from pathlib import Path
from scipy.stats import t

import numpy as np
from published_results import solver_records


def mean_sd(values):
    return {'mean':float(np.mean(values)), 'sd':float(np.std(values,ddof=1)) if len(values)>1 else None, 'n':len(values)}


def quantiles(values):
    if not values:
        return dict(p50=None, p95=None, p99=None, max=None, n=0)
    array=np.asarray(values,dtype=float)
    return dict(zip(['p50','p95','p99','max'],map(float,np.quantile(array,[.5,.95,.99,1]))), n=len(array))


def summarize_results(root, output):
    groups = {}
    identities = set()
    for record in solver_records(root):
        identity = (record["task"], record.get("seed", record.get("replicate")))
        if identity in identities:
            raise ValueError(f"Duplicate replicate: {identity}")
        identities.add(identity)
        evaluations = {arm: json.loads((root / path).read_text())
                       for arm, path in record["evaluations"].items()}
        for arm, evaluation in evaluations.items():
            assert evaluation["arm"] == arm and evaluation["seed"] == record["seed"]
        episode_keys = lambda data: [(e["channel"], e["env_seed"], e["policy_seed"])
                                     for e in data["episodes"]]
        assert episode_keys(evaluations["original"]) == episode_keys(evaluations["tight100"])
        accuracy = json.loads((root / record["accuracy"]).read_text())
        groups.setdefault(record["task"], []).append((evaluations, accuracy))
    tasks = []
    for task, records in groups.items():
        row = dict(task=task, conditions={}, numerical={})
        for arm in ("original", "tight100"):
            row["conditions"][arm] = {}
            for channel in ("stoch", "det"):
                row["conditions"][arm][channel] = {}
                for metric in ("native_ever_success", "success_at_horizon", "native_dense_return"):
                    values = [float(np.mean([e[metric] for e in ev[arm]["episodes"]
                                             if e["channel"] == channel])) for ev, _ in records]
                    row["conditions"][arm][channel][metric] = mean_sd(values)
            values = [r for _, audit in records for r in audit["rows"] if r["arm"] == arm]
            valid = [r for r in values if r["reference_ok"]]
            gradients = [g for _, audit in records for g in audit["gradients"]
                         if g.get("arm") == arm and g["status"] == "ok"]
            expected = sum(len({g["start"] for g in audit["gradients"]}) for _, audit in records)
            row["numerical"][arm] = {
                **{metric: quantiles([r[metric] for r in valid])
                   for metric in ("action_linf", "logp_abs_nats")},
                "gradient_relative_percent": quantiles([100*g["relative_error"] for g in gradients]),
                "failed_input_count": len(values) - len(valid),
                "failed_gradient_count": expected - len(gradients),
            }
        tasks.append(row)
    result = dict(tasks=tasks, reference_failures=sum(a["root_failures"]
                  for records in groups.values() for _, a in records))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--root', type=Path)
    inputs.add_argument('--results-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.results_root is not None:
        if args.output is None:
            parser.error('--output is required with --results-root')
        summarize_results(args.results_root, args.output)
        return
    root = args.root
    manifest=json.loads((root/'bundle/manifest.json').read_text())
    total_episodes = total_steps = 0
    seeds=[]
    numerical=[]
    aliases=json.loads((root/'artifact_paths.json').read_text())
    for job in manifest['jobs']:
        key=job['key']
        results={}
        banks={}
        for arm in ['original','tight100']:
            folder=aliases.get(f'{key}_{arm}_cpu',f'{key}_{arm}_cpu')
            path=root/'remote'/folder
            assert (path/'sync_receipt.json').exists()
            receipt=json.loads((path/'sync_receipt.json').read_text())
            for relative, expected in receipt['hashes'].items():
                assert hashlib.sha256((path/relative).read_bytes()).hexdigest()==expected, (folder,relative)
            result=json.loads((path/'data/result.json').read_text())
            terminal=json.loads((path/'data/terminal.json').read_text())
            assert terminal['exit_code']==0 and terminal['episodes']==len(result['episodes'])==manifest['episodes_per_job_per_arm']
            assert sum(e['steps'] for e in result['episodes']) == terminal['steps']
            total_episodes += terminal['episodes']
            total_steps += terminal['steps']
            assert not result['pilot']
            results[arm]=result
            banks[arm]=np.load(path/'data/probes.npz')
        old=json.loads((root/'bundle/inputs'/key/'historical.json').read_text())
        record=dict(key=key,task=job['task'],seed=job['seed'], arms={}, historical={}, paired={})
        for channel in ['stoch','det']:
            episode_lists={arm:[e for e in r['episodes'] if e['channel']==channel] for arm,r in results.items()}
            keys=lambda e:(e['env_seed'],e['policy_seed'])
            assert list(map(keys,episode_lists['original']))==list(map(keys,episode_lists['tight100']))
            historic=[e for e in old['episodes'] if e['channel']==channel]
            assert set(map(keys,historic))==set(map(keys,episode_lists['original']))
            for arm,episodes in episode_lists.items():
                record['arms'].setdefault(arm,{})[channel]={k:float(np.mean([e[k] for e in episodes])) for k in
                    ['native_ever_success','success_at_horizon','native_dense_return']}
            record['historical'][channel]={k:float(np.mean([e[k] for e in historic])) for k in
                ['native_ever_success','success_at_horizon','native_dense_return']}
            record['paired'][channel]={}
            for metric in ['native_ever_success','success_at_horizon','native_dense_return']:
                difference=[float(b[metric])-float(a[metric]) for a,b in
                            zip(episode_lists['original'],episode_lists['tight100'])]
                record['paired'][channel][metric]={'mean':float(np.mean(difference)),
                    'positive':sum(x>0 for x in difference),'negative':sum(x<0 for x in difference),
                    'zero':sum(x==0 for x in difference)}
        record['noise_identical']=bool(np.array_equal(banks['original']['noise'],banks['tight100']['noise']))
        initial={arm: np.cumsum([0] + [e['steps'] for e in results[arm]['episodes'][:-1]]) for arm in results}
        record['reset_observation_linf']=float(np.max(np.abs(
            banks['original']['observations'][initial['original']]-banks['tight100']['observations'][initial['tight100']])))
        assert record['noise_identical'] and record['reset_observation_linf']==0., key
        for arm,result in results.items():
            record['arms'][arm]['solver']={k:result[k] for k in ['solver_cap_fraction','solver_tolerance_fail_fraction',
                'solver_relative_residual_quantiles','inference_seconds','wall_seconds','inference_ms_quantiles']}
        seeds.append(record)
        audit_result=json.loads((root/'numerical'/key/'result.json').read_text())
        recovered_path=root/'numerical'/key/'local_branch_validation.json'
        audit_result['initial_root_failures']=audit_result['root_failures']
        audit_result['recovered_reference_samples']=[]
        if recovered_path.exists():
            recovered=json.loads(recovered_path.read_text())
            if recovered['accepted']:
                for replacement in recovered['rows']:
                    old_row=next(r for r in audit_result['rows'] if
                        r['arm']==replacement['arm'] and r['sample']==replacement['sample'])
                    old_row.update(replacement)
                audit_result['gradients'].extend(recovered['gradients'])
                audit_result['root_failures']-=len(recovered['recovered_samples'])
                audit_result['recovered_reference_samples']=recovered['recovered_samples']
        numerical.append(audit_result)
    tasks=[]
    for task in dict.fromkeys(j['task'] for j in manifest['jobs']):
        selected=[r for r in seeds if r['task']==task]
        row=dict(task=task,conditions={},paired={},numerical={})
        for condition in ['historical','original','tight100']:
            row['conditions'][condition]={}
            for channel in ['stoch','det']:
                row['conditions'][condition][channel]={}
                for metric in ['native_ever_success','success_at_horizon','native_dense_return']:
                    values=[(r['historical'] if condition=='historical' else r['arms'][condition])[channel][metric] for r in selected]
                    row['conditions'][condition][channel][metric]=mean_sd(values)
        for channel in ['stoch','det']:
            row['paired'][channel]={}
            for metric in ['native_ever_success','success_at_horizon','native_dense_return']:
                values=[r['paired'][channel][metric]['mean'] for r in selected]
                d=mean_sd(values)
                half=t.ppf(.975, d['n']-1)*d['sd']/np.sqrt(d['n']) if d['n']>1 else None
                d['training_seed_t95_ci']=[d['mean']-half,d['mean']+half] if half is not None else None
                row['paired'][channel][metric]=d
        audits=[r for r in numerical if r['job']['task']==task]
        for arm in ['original','tight100']:
            values=[r for a in audits for r in a['rows'] if r['arm']==arm]
            valid=[r for r in values if r['reference_ok']]
            gradients=[g for a in audits for g in a['gradients'] if g.get('arm')==arm and g['status']=='ok']
            row['numerical'][arm]={metric:quantiles([r[metric] for r in valid]) for metric in ['action_linf','logp_abs_nats']}
            row['numerical'][arm]['gradient_relative_percent']=quantiles([100*g['relative_error'] for g in gradients])
            row['numerical'][arm]['failed_input_count']=len(values)-len(valid)
            expected = sum(len({g['start'] for g in a['gradients']}) for a in audits)
            row['numerical'][arm]['failed_gradient_count']=expected-len(gradients)
        tasks.append(row)
    output=dict(tasks=tasks,seeds=seeds,total_episodes=total_episodes,total_steps=total_steps,
        reference_failures=sum(a['root_failures'] for a in numerical),
        initial_reference_failures=sum(a['initial_root_failures'] for a in numerical),
        locally_validated_references=sum(len(a['recovered_reference_samples']) for a in numerical),

        all_noise_identical=all(r['noise_identical'] for r in seeds),
        reset_observation_max_difference=max(r['reset_observation_linf'] for r in seeds),
        inference_seconds={arm:sum(r['arms'][arm]['solver']['inference_seconds'] for r in seeds) for arm in ['original','tight100']},
        scope='Paired t95 intervals summarize the supplied independent training replicates; they are not equivalence tests.')
    (root/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'episodes':total_episodes,'steps':output['total_steps'],'noise_identical':output['all_noise_identical']}))


if __name__=='__main__':
    main()
