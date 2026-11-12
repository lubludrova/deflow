import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


controller = load('idle_controller', 'controller.py')
report = load('idle_report', 'report.py')
sync = load('idle_sync', 'sync_packed.py')


class AuditTests(unittest.TestCase):
    def test_handover_blocks_only_future_admission_and_retains_live_identity(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,DEFLOW_PACKING_SHA256='new'):
            root=Path(folder);previous=root/'ops/packing-v1';previous.mkdir(parents=True)
            identity={'pid':123,'process_start_ticks':456}
            snapshot={'runs':{'run':{'state':'RUNNING'}}}
            active={'run':{'identity':identity}}
            live={'coordinator':identity,'observed_at':time.time(),'active':active}
            (previous/'campaign_report.json').write_text(json.dumps(snapshot))
            (previous/'campaign_live.json').write_text(json.dumps(live))
            with patch.object(controller,'ROOT',root),patch.object(controller.c,'latch_check'), \
                 patch.object(controller.c,'same_process',return_value=True):
                returned,adopted,_=controller.choose_snapshot({'previous_overlay':'packing-v1'},True)
            self.assertEqual(returned,snapshot)
            self.assertEqual(adopted,active)
            self.assertTrue((previous/'ADMISSION_BLOCKED.json').exists())
            self.assertEqual(json.loads((previous/'campaign_live.json').read_text()),live)

    def summary(self, invalid=0):
        row = {'forward_seconds_median': .025, 'loss': 5.,
               'logp_abs_difference': {'p95': .2}, 'gradient_relative_difference': .3,
               'reference_gradient_available': not invalid}
        return {'batch':1024,'loss_kind':'configured_SAC_actor_loss','precision_policy':'strict_fp32',
            'input_kind':'campaign_same_capsule','runtime_precision':{'matmul_allow_tf32':False},
            'anderson_sweep_caps':[25,100,500],'hashes':{'checkpoint':'same','inputs':'same'},
            'reference_valid':1024-invalid,'reference_invalid':invalid,
            **{label:dict(row) for label in report.LABELS}}

    def test_invalid_reference_never_leaks_full_batch_errors(self):
        row = report.metric_row(self.summary(1), 'newton', False, 1.)
        self.assertIsNone(row['gradient_relative_error'])
        self.assertIsNone(row['p95_logp_error'])
        self.assertIsNone(row['delta_actor_loss'])
        self.assertEqual(row['forward_ms'],25.)

    def test_signed_loss_and_finite_errors(self):
        row = report.metric_row(self.summary(), 'newton', True, 8.)
        self.assertEqual(row['delta_actor_loss'],-3.)
        self.assertEqual(row['gradient_relative_error'],.3)
        self.assertEqual(row['p95_logp_error'],.2)

    def test_nonfinite_values_are_not_reported_as_metrics(self):
        summary=self.summary();summary['newton']['gradient_relative_difference']=float('inf')
        row=report.metric_row(summary,'newton',True,1.)
        self.assertIsNone(row['gradient_relative_error'])

    def test_empty_denominator_yields_na(self):
        self.assertIsNone(report.median([]))
        self.assertIsNone(report.median([None,float('nan')]))

    def test_audit_acceptance_checks_masks_and_same_capsule(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.pt';path.write_bytes(b'evidence')
            controller.validate_audit(self.summary(),path,{'sha256':'same'})
            invalid=self.summary(1)
            with self.assertRaisesRegex(RuntimeError,'Invalid full-batch'):
                controller.validate_audit(invalid,path,{'sha256':'same'})
            for label in report.LABELS: invalid[label].pop('gradient_relative_difference')
            controller.validate_audit(invalid,path,{'sha256':'same'})
            invalid['hashes']['inputs']='different'
            with self.assertRaisesRegex(RuntimeError,'identical capsule'):
                controller.validate_audit(invalid,path,{'sha256':'same'})

    def test_previous_final_ack_cannot_stop_sync(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ack=root/'ops/packing-v1/FINAL_SYNC_ACK.json'
            ack.parent.mkdir(parents=True);ack.write_text('{}')
            self.assertFalse(sync.terminal_ready(root,{'ops/packing-v1/FINAL_SYNC_ACK.json':'old'}))

    def test_exact_final_hash_coverage_required(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,DEFLOW_PACKING_SHA256='new'):
            root=Path(folder);ack=root/'ops/idle-audits-v1/FINAL_SYNC_ACK.json'
            ack.parent.mkdir(parents=True)
            ack.write_text(json.dumps({'packing_sha256':'new','final_sha256':{'report':'correct'}}))
            verified={'ops/idle-audits-v1/FINAL_SYNC_ACK.json':'ack','report':'wrong'}
            with self.assertRaises(RuntimeError): sync.terminal_ready(root,verified)
            verified['report']='correct';self.assertTrue(sync.terminal_ready(root,verified))


if __name__ == '__main__':
    unittest.main()
