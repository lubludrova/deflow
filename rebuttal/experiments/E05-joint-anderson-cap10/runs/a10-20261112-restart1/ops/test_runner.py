import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import runner


class RunnerTests(unittest.TestCase):
    def test_identity_contains_start_time_and_command(self):
        record=runner.identity(os.getpid())
        self.assertEqual(record['pid'],os.getpid())
        self.assertTrue(record['start_ticks'].isdigit())
        self.assertEqual(len(record['command_sha256']),64)
        self.assertTrue(record['boot_id'])

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'ops').mkdir()
            source=root/'source.py'
            source.write_text('original')
            data=dict(remote_root=str(root),frozen_files={'source.py':runner.digest(source)})
            (root/'ops/manifest.json').write_text(json.dumps(data))
            with patch.object(runner,'ROOT',root):
                self.assertEqual(runner.contract(),data)
                source.write_text('changed')
                with self.assertRaises(AssertionError): runner.contract()

    def test_duplicate_attempt_cannot_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'attempts/test/0000').mkdir(parents=True)
            with patch.object(runner,'ROOT',root), patch.object(runner,'contract',return_value={'jobs':[{'run_id':'test'}]}), patch.object(runner,'run_child') as child:
                with self.assertRaises(FileExistsError): runner.worker('test')
                child.assert_not_called()

    def test_real_nonzero_exit_is_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'ops').mkdir()
            (root/'ops/manifest.json').write_text('{}')
            with patch.object(runner,'ROOT',root), patch.object(runner,'contract',return_value={'environment':{}}), patch.object(runner,'gpu_snapshot',return_value={}), patch.object(runner.time,'sleep'):
                with self.assertRaisesRegex(RuntimeError,'exited 7'):
                    runner.run_child(dict(gpu=0,gpu_uuid='test'),'training',[sys.executable,'-c','raise SystemExit(7)'],root)
            receipt=json.loads((root/'training_exit.json').read_text())
            self.assertEqual(receipt['exit_code'],7)
            self.assertTrue(receipt['identity']['start_ticks'])
            self.assertFalse((root/'evaluation_launch.json').exists())


if __name__=='__main__': unittest.main()
