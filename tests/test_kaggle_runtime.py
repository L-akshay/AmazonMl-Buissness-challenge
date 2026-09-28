import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import importlib.util
import io
import zipfile

from src.kaggle_runtime import mount_checkpoint,detach_inputs,partial_checkpoint
from src.cloud_store import signature,model_signature
from src.handoff import source_files


class KaggleRuntimeTests(unittest.TestCase):
    def test_restored_aggregate_archive_verifies_hashes_and_rejects_escape(self):
        spec=importlib.util.spec_from_file_location('job_builder',Path(__file__).resolve().parents[1]/'scripts/build_kaggle_job.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);name='cache/cloud/audit/chunk_00000.json'
            path=source/name;path.parent.mkdir(parents=True);path.write_bytes(b'{"queries":100000}')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            payload=io.BytesIO()
            with zipfile.ZipFile(payload,'w') as archive:
                self.assertEqual(builder.restore_files(archive,source,{'files':{name:sha}}),{name:sha})
                with self.assertRaisesRegex(ValueError,'hash differs'):
                    builder.restore_files(archive,source,{'files':{name:'0'*64}})
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    builder.restore_files(archive,source,{'files':{'../outside':sha}})
            with zipfile.ZipFile(io.BytesIO(payload.getvalue())) as archive:
                self.assertEqual(archive.read(name),path.read_bytes())

    def test_partial_checkpoint_requires_unique_exact_hashes_and_safe_paths(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            inputs=Path(tmp); source=inputs/'cancelled/project'
            relative='cache/cloud/audit/config.json'
            path=source/relative; path.parent.mkdir(parents=True);path.write_bytes(b'approved')
            sha=hashlib.sha256(b'approved').hexdigest()
            proof={'sentinel':relative,'files':{relative:sha}}
            self.assertEqual(partial_checkpoint(inputs,proof),source.resolve())
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                partial_checkpoint(inputs,{'sentinel':'../secret','files':{'../secret':sha}})
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'No unique'):
                partial_checkpoint(inputs,proof)
            path.write_bytes(b'approved')
            duplicate=inputs/'other/project'/relative
            duplicate.parent.mkdir(parents=True);duplicate.write_bytes(b'approved')
            with self.assertRaisesRegex(ValueError,'No unique'):
                partial_checkpoint(inputs,proof)

    def test_tracked_report_overlay_remains_packageable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'work'; old=Path(tmp)/'old'; new=Path(tmp)/'new'
            for folder in (root,old,new):
                (folder/'reports').mkdir(parents=True)
            names=['reports/data_audit.json']
            (root/'HANDOFF_FILES.json').write_text(json.dumps(names))
            (root/names[0]).write_text('historical')
            (old/names[0]).write_text('first measured report')
            (new/names[0]).write_text('latest measured report')
            mount_checkpoint(root,old)
            mount_checkpoint(root,new)
            self.assertEqual(source_files(root),names)
            self.assertFalse((root/names[0]).is_symlink())
            self.assertEqual((root/names[0]).read_text(),'latest measured report')
            self.assertEqual(detach_inputs(root),0)
            self.assertEqual((old/names[0]).read_text(),'first measured report')
            self.assertEqual((new/names[0]).read_text(),'latest measured report')
            (root/'HANDOFF_FILES.json').write_text(json.dumps(['../new/reports/data_audit.json']))
            with self.assertRaisesRegex(ValueError,'Invalid code manifest path'):
                source_files(root)

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
