"""CPU-only lifecycle regressions. No subprocesses or GPUs are launched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import common


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ops = self.root / 'ops'
        self.ops.mkdir()
        self.patchers = [patch.object(common, 'ROOT', self.root), patch.object(common, 'OPS', self.ops)]
        for p in self.patchers:
            p.start()
        common.SHUTDOWN.clear()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        common.SHUTDOWN.clear()
        self.temp.cleanup()

    def test_failed_preflight_wins_over_done(self):
        common.write(self.ops / 'PREFLIGHT_DONE.json', {})
        common.write(self.ops / 'PREFLIGHT_FAILED.json', {})
        with patch.object(common, 'binding', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'FAILED'):
                common.preflight_gate()

    def test_stop_interrupts_sync_wait(self):
        common.write(self.ops / 'STOP.json', {})
        with patch.object(common, 'binding', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                common.sync_gate()

    def test_pause_blocks_admission(self):
        common.write(self.ops / 'PAUSE.json', {})
        with self.assertRaisesRegex(RuntimeError, 'PAUSE'):
            common.latch_check(wait_pause=False)

    def test_monitor_failure_wins_over_recent_heartbeat(self):
        common.write(self.ops / 'MONITOR_FAILED.json', {})
        with self.assertRaisesRegex(RuntimeError, 'Monitor'):
            common.health_gate()

    def test_wrong_campaign_receipt_not_accepted(self):
        common.write(self.ops / 'sync_receipt.json', dict(campaign_id='old', result='ok', at=__import__('time').time(), index_sha256='x', accepted_attempts=[]))
        with patch.object(common, 'binding', return_value={'campaign_id': 'new'}), patch.object(common, 'health_gate'), patch.object(common.SHUTDOWN, 'wait', side_effect=lambda _: common.SHUTDOWN.set()):
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                common.sync_gate()

    def test_final_sync_requires_exact_artifact_hashes(self):
        row = dict(campaign_id='new', result='ok', at=__import__('time').time(), index_sha256='x', accepted_attempts=[], verified_sha256={'report': 'wrong'})
        common.write(self.ops / 'sync_receipt.json', row)
        with patch.object(common, 'binding', return_value={'campaign_id': 'new'}), patch.object(common.SHUTDOWN, 'wait', side_effect=lambda _: common.SHUTDOWN.set()):
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                common.sync_gate(final_hashes={'report': 'correct'})

    def test_matching_final_receipt_works_without_running_monitor(self):
        common.write(self.ops / 'sync_receipt.json', dict(campaign_id='new', result='ok', at=__import__('time').time(), index_sha256='x', accepted_attempts=[], verified_sha256={'report': 'correct'}))
        with patch.object(common, 'binding', return_value={'campaign_id': 'new'}), patch.object(common, 'contract_check'), patch.object(common, 'health_gate', side_effect=AssertionError('Monitor should be stopped')):
            common.sync_gate(final_hashes={'report': 'correct'})

    def test_refuses_unknown_process_termination(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        with patch.object(common.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'Refusing'):
                common.terminate_owned(process, None)
            kill.assert_not_called()

    def test_atomic_write_no_shared_temp(self):
        path = self.ops / 'state.json'
        common.write(path, {'status': 'one'})
        common.write(path, {'status': 'two'})
        self.assertEqual(common.read(path), {'status': 'two'})
        self.assertEqual(list(self.ops.glob('*.tmp')), [])

    def test_fast_child_exit_has_terminal_receipt(self):
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = 0
        process.wait.return_value = 0
        environment = {'PYTHONPATH': '/package', 'DEFLOW_ENGINE': '/engine', 'CUDA_VISIBLE_DEVICES': 'gpu'}
        with patch.object(common, 'check_sources'), patch.object(common, 'binding', return_value={}), patch.object(common, 'capacity', return_value={}), patch.object(common, 'env', return_value=environment), patch.object(common.subprocess, 'Popen', return_value=process), patch.object(common, 'identity', side_effect=FileNotFoundError):
            result = common.execute('fast', ['python', '-c', 'pass'], 0)
        self.assertEqual(result['exit_code'], 0)
        self.assertTrue((self.ops / 'attempts/fast/EXIT.json').exists())
        self.assertFalse(common.ACTIVE)

    def test_launch_exception_preserves_failure_claim(self):
        environment = {'PYTHONPATH': '/package', 'DEFLOW_ENGINE': '/engine', 'CUDA_VISIBLE_DEVICES': 'gpu'}
        with patch.object(common, 'check_sources'), patch.object(common, 'binding', return_value={}), patch.object(common, 'capacity', return_value={}), patch.object(common, 'env', return_value=environment), patch.object(common.subprocess, 'Popen', side_effect=OSError('launch error')):
            with self.assertRaisesRegex(OSError, 'launch error'):
                common.execute('bad_launch', ['python'], 0)
        self.assertTrue((self.ops / 'attempts/bad_launch/CLAIM.json').exists())
        self.assertTrue((self.ops / 'attempts/bad_launch/FAILED.json').exists())

    def test_failed_audit_retains_all_nine_rows(self):
        import campaign
        original = campaign.AUDITS
        campaign.AUDITS = {f'audit_a{seed}_{target}': dict(state='PLANNED', seed=seed, target_step=target)
            for seed in (1, 2, 3) for target in (25000, 100000, 200000)}
        selected = [dict(seed=1, target_step=t, state='PLANNED', path='/missing', sha256='hash')
                    for t in (25000, 100000, 200000)]
        try:
            with patch.object(campaign, 'sync_gate'), patch.object(campaign, 'contract_check'), patch.object(campaign, 'digest', return_value='hash'), patch.object(campaign, 'execute', side_effect=RuntimeError('audit failure')), patch.object(campaign, 'report'):
                campaign.audit_seed(1, selected)
            self.assertEqual(len(campaign.AUDITS), 9)
            self.assertEqual(campaign.AUDITS['audit_a1_25000']['state'], 'FAILED')
            self.assertEqual(campaign.AUDITS['audit_a1_100000']['state'], 'BLOCKED')
            self.assertEqual(campaign.AUDITS['audit_a1_200000']['state'], 'BLOCKED')
        finally:
            campaign.AUDITS = original

    def test_failure_latch_does_not_stop_sync_before_siblings_join(self):
        import sync_local
        verified = {'ops/PREFLIGHT_FAILED.json': 'a'}
        self.assertFalse(sync_local.terminal_ready(self.root, verified))
        common.write(self.ops / 'COORDINATOR_STARTED.json', {})
        verified['ops/PREFLIGHT_TERMINAL_FAILED.json'] = 'b'
        self.assertFalse(sync_local.terminal_ready(self.root, verified))
        verified['ops/COORDINATOR_FAILED.json'] = 'c'
        self.assertTrue(sync_local.terminal_ready(self.root, verified))
        common.write(self.ops / 'CLEANUP_FAILED.json', {})
        self.assertFalse(sync_local.terminal_ready(self.root, verified))

    def test_missing_contract_anchor_fails(self):
        with patch.dict(common.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'anchor'):
                common.contract_check()


if __name__ == '__main__':
    unittest.main()
