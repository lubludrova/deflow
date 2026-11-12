"""Checks for shared placement and the original E04 evaluation admission gate."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import runner

class SharedTests(unittest.TestCase):
    def test_only_identified_peer_on_selected_gpu_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'ops').mkdir()
            (root/'ops/manifest.json').write_text(json.dumps({'jobs':[dict(gpu=0,gpu_uuid='GPU0',run_id='peer')]}))
            attempt=root/'attempts/peer/0000'
            attempt.mkdir(parents=True)
            ident=runner.identity(__import__('os').getpid())
            (attempt/'training_launch.json').write_text(json.dumps(dict(identity=ident)))
            contract=dict(peer=dict(remote_root=tmp,manifest_sha256=runner.digest(root/'ops/manifest.json')),gpu_by_slot={'0':'GPU0'})
            with patch.object(runner,'contract',return_value=contract):
                runner.check_peers({'processes':[f"{ident['pid']}, GPU0, 1024"]},0)
                with self.assertRaisesRegex(AssertionError,'Unknown'):
                    runner.check_peers({'processes':['987654, GPU0, 1024']},0)
                runner.check_peers({'processes':['987654, GPU1, 1024']},0)
                with patch.object(runner,'identity',return_value={**ident,'start_ticks':'different'}):
                    with self.assertRaisesRegex(AssertionError,'identity'):
                        runner.check_peers({'processes':[f"{ident['pid']}, GPU0, 1024"]},0)

    def test_final_ack_is_held_until_real_training_exit(self):
        tree=ast.parse((Path(__file__).parent/'peer_sync_e04.py').read_text())
        program=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='guard_program' for t in n.targets))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'ops').mkdir()
            def held():return json.loads(subprocess.check_output([sys.executable,'-c',program,tmp],text=True))
            self.assertEqual(held(),[])
            (root/'ops/STARTED.json').write_text('{}')
            self.assertEqual(len(held()),3)
            attempt=root/'attempts/joint_anderson_a10_s1/0000'
            attempt.mkdir(parents=True)
            (attempt/'training_exit.json').write_text('{"exit_code":0}')
            self.assertEqual(held(),['joint_anderson_a100_s2','joint_anderson_a100_s3'])

if __name__=='__main__':unittest.main()
