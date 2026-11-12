"""Operational regression checks. No scientific workload or GPU calls."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, BASE / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


m = module('packed_controller', 'controller.py')
s = module('packed_sync', 'sync_packed.py')


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {'gpu_by_slot': {'0': 'gpu0'}}
        self.sample = {'gpus': [['0', 'gpu0', '1024', '22000', '20']],
                       'processes': [['gpu0', '10', '900']]}
        self.running = {'n1': {'pid': 10, 'gpu_uuid': 'gpu0'}}

    def test_one_known_job_allows_second(self):
        with patch.object(m.c, 'read', return_value=self.mapping):
            m.check_occupancy(self.sample, self.running, 0, launching=True)

    def test_third_job_rejected_even_before_gpu_initialization(self):
        running = dict(self.running, a1={'pid': 11, 'gpu_uuid': 'gpu0'})
        with patch.object(m.c, 'read', return_value=self.mapping):
            with self.assertRaisesRegex(RuntimeError, 'concurrency'):
                m.check_occupancy(self.sample, running, 0, launching=True)
            m.check_occupancy(self.sample, running, 0)

    def test_unknown_job_rejected(self):
        self.sample['processes'].append(['gpu0', '99', '100'])
        with patch.object(m.c, 'read', return_value=self.mapping):
            with self.assertRaisesRegex(RuntimeError, 'Unknown'):
                m.check_occupancy(self.sample, self.running, 0, launching=True)

    def test_memory_reserve(self):
        self.sample['gpus'][0][3] = '4000'
        with patch.object(m.c, 'read', return_value=self.mapping):
            with self.assertRaisesRegex(RuntimeError, '4 GiB'):
                m.check_occupancy(self.sample, self.running, 0, launching=True)

    def test_short_windows_do_not_invent_rates(self):
        self.assertIsNone(m.rate([(10, 4000), (20, 9000)]))
        self.assertEqual(m.rate([(10, 5000), (610, 11000)]), 10)
        self.assertEqual(m.canary_score({'n': 2, 'a': 10}, {'n': 4, 'a': 20}), 1)

    def test_old_terminal_cannot_stop_new_sync(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'ops').mkdir()
            (root / 'ops/FINAL_SYNC_ACK.json').write_text('{}')
            self.assertFalse(s.terminal_ready(root, {'ops/FINAL_SYNC_ACK.json': 'sha'}))

    def test_final_ack_must_cover_exact_new_artifacts(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, DEFLOW_PACKING_SHA256='anchor'):
            root = Path(folder)
            ack = root / 'ops/packing-v1/FINAL_SYNC_ACK.json'
            ack.parent.mkdir(parents=True)
            ack.write_text(json.dumps({'packing_sha256': 'anchor', 'final_sha256': {'report': 'good'}}))
            verified = {'ops/packing-v1/FINAL_SYNC_ACK.json': 'ack', 'report': 'bad'}
            with self.assertRaisesRegex(RuntimeError, 'not verified'):
                s.terminal_ready(root, verified)
            verified['report'] = 'good'
            self.assertTrue(s.terminal_ready(root, verified))
            (root / 'ops/CLEANUP_FAILED.json').write_text('{}')
            self.assertFalse(s.terminal_ready(root, verified))

    def test_duplicate_claim_does_not_corrupt_existing_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / 'attempts/job').mkdir(parents=True)
            existing = base / 'attempts/job/START.json'
            existing.write_text('{"owned_by":"another"}')
            m.LOCAL.lane = 1
            with patch.object(m, 'BASE', base), patch.object(m.c, 'OPS', base), \
                 patch.object(m.c, 'latch_check'), patch.object(m, 'packing_check'), \
                 patch.object(m, 'health_gate'), patch.object(m, 'capacity', return_value={}):
                with self.assertRaises(FileExistsError):
                    m.execute('job', ['never-launched'], 0)
            self.assertEqual(existing.read_text(), '{"owned_by":"another"}')
            self.assertFalse((base / 'attempts/job/FAILED.json').exists())

    def test_real_exit_and_failed_exit_keep_separate_terminal_evidence(self):
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                (base / 'attempts').mkdir()
                (base / 'logs').mkdir()
                m.LOCAL.lane = 1
                identity = {'pid': 100, 'process_start_ticks': 200}
                with patch.object(m, 'BASE', base), patch.object(m.c, 'OPS', base), \
                     patch.object(m.c, 'PACKAGE', base), patch.object(m.c, 'latch_check'), \
                     patch.object(m, 'packing_check'), patch.object(m, 'health_gate'), \
                     patch.object(m, 'capacity', return_value={}), \
                     patch.object(m.c, 'binding', return_value={}), \
                     patch.object(m.c, 'env', return_value={'CUDA_VISIBLE_DEVICES': 'gpu0', 'PYTHONPATH': '', 'DEFLOW_ENGINE': ''}), \
                     patch.object(m.c, 'identity', return_value=identity), \
                     patch.object(m.c, 'owned_group_members', return_value=[]), \
                     patch.object(m.c, 'stop_requested', return_value=False), \
                     patch.dict(os.environ, DEFLOW_PACKING_SHA256='anchor'):
                    argv = [sys.executable, '-c', 'raise SystemExit(' + str(code) + ')']
                    if code:
                        with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                            m.execute('job', argv, 0)
                    else:
                        result = m.execute('job', argv, 0)
                        self.assertEqual(result['exit_code'], 0)
                terminal = json.loads((base / 'attempts/job/EXIT.json').read_text())
                self.assertEqual(terminal['exit_code'], code)
                self.assertEqual(terminal['identity'], identity)
                self.assertEqual((base / 'attempts/job/FAILED.json').exists(), bool(code))
                self.assertNotIn('job', m.c.ACTIVE)

    def test_adopted_attempt_requires_real_success_receipt_and_matching_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            attempt = base / 'attempts/job'
            attempt.mkdir(parents=True)
            identity = {'pid': 10, 'process_start_ticks': 20}
            config = {'adopted': {'job': {'identity': identity}}}
            with patch.object(m.c, 'OPS', base), patch.object(m.c, 'ROOT', base), \
                 patch.object(m, 'CONFIG', config), patch.object(m.c, 'latch_check'):
                (attempt / 'EXIT.json').write_text(json.dumps({'identity': identity, 'exit_code': 9}))
                with self.assertRaisesRegex(RuntimeError, 'terminal mismatch'):
                    m.adopted_training('job')
                (attempt / 'EXIT.json').write_text(json.dumps({'identity': identity, 'exit_code': 0}))
                (attempt / 'ACCEPTED.json').write_text(json.dumps({'artifacts_sha256': {}}))
                self.assertEqual(m.adopted_training('job')['exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
