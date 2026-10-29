from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import zipfile
from scripts import restore_artifacts

class RecoveryTests(unittest.TestCase):
    def test_restore_preserves_original_bundle_before_replacing(self):
        repository=Path(__file__).resolve().parents[1]
        blobs={name:(repository/'artifacts'/name).read_bytes() for name in restore_artifacts.NAMES}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'artifacts').mkdir()
            for name in restore_artifacts.NAMES:
                (root/'artifacts'/name).write_bytes(b'original local '+name.encode())
            def git_result(args,**kwargs):
                return SimpleNamespace(stdout=blobs[args[-1].split('/')[-1]])
            with patch.object(restore_artifacts,'ROOT',root),patch.object(restore_artifacts.subprocess,'run',side_effect=git_result):
                restore_artifacts.restore()
            backups=list((root/'.local_backups').glob('*.zip'))
            self.assertEqual(len(backups),1)
            with zipfile.ZipFile(backups[0]) as backup:
                for name in restore_artifacts.NAMES:
                    self.assertEqual(backup.read('artifacts/'+name),b'original local '+name.encode())
                    self.assertEqual((root/'artifacts'/name).read_bytes(),blobs[name])
