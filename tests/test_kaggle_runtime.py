import json
from pathlib import Path
import tempfile
import unittest

from src.kaggle_runtime import mount_checkpoint,detach_inputs
from src.cloud_store import signature,model_signature


class KaggleRuntimeTests(unittest.TestCase):
    def test_checkpoint_overlay_detaches_only_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'work'; old=Path(tmp)/'old'; new=Path(tmp)/'new'
            for folder in (root,old,new):
                (folder/'cache').mkdir(parents=True)
            (old/'cache/checkpoint.txt').write_bytes(b'original')
            (new/'cache/checkpoint.txt').write_bytes(b'updated')
            (new/'cache/incomplete.tmp.parquet').write_bytes(b'partial')
            try:
                mount_checkpoint(root,old)
            except OSError as error:
                self.skipTest(f'OS does not allow symlinks: {error}')
            mount_checkpoint(root,new)
            self.assertEqual((root/'cache/checkpoint.txt').read_bytes(),b'updated')
            self.assertFalse((root/'cache/incomplete.tmp.parquet').exists())
            (root/'cache/fresh.parquet').write_bytes(b'fresh')
            self.assertEqual(detach_inputs(root),1)
            self.assertEqual((old/'cache/checkpoint.txt').read_bytes(),b'original')
            self.assertEqual((new/'cache/checkpoint.txt').read_bytes(),b'updated')
            self.assertEqual((root/'cache/fresh.parquet').read_bytes(),b'fresh')

    def test_incompatible_partition_overlay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'work'; old=Path(tmp)/'old'; new=Path(tmp)/'new'
            for folder in (root,old,new):
                (folder/'cache').mkdir(parents=True)
            (old/'cache/config.json').write_text('{"fingerprint":"one"}')
            (new/'cache/config.json').write_text('{"fingerprint":"two"}')
            try:
                mount_checkpoint(root,old)
            except OSError as error:
                self.skipTest(f'OS does not allow symlinks: {error}')
            with self.assertRaisesRegex(ValueError,'Conflicting immutable'):
                mount_checkpoint(root,new)

    def test_model_change_preserves_feature_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'src').mkdir()
            for name in ('blocking','gpu_sparse','normalize','features','cloud_store','cloud_features',
                         'streaming','cloud_model','cloud_validation'):
                (root/'src'/f'{name}.py').write_text('# v1')
            before=signature(root,{'policy':'full6'})
            model_before=model_signature(root,before)
            (root/'src/cloud_model.py').write_text('# model experiment')
            self.assertEqual(signature(root,{'policy':'full6'}),before)
            self.assertNotEqual(model_signature(root,before),model_before)
