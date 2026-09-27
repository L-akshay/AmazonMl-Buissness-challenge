import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('delivery',Path(__file__).resolve().parents[1]/'scripts/download_kaggle_results.py')
delivery=importlib.util.module_from_spec(spec); spec.loader.exec_module(delivery)


class DeliveryTests(unittest.TestCase):
    def test_remote_paths_cannot_escape_delivery_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):
                delivery.destination(root,'../credentials.txt')
            with self.assertRaises(ValueError):
                delivery.destination(root,str(root.parent/'outside.txt'))
            result=delivery.destination(root,'business_entity_resolution/output/cloud/oof_best/matching_results.tsv')
            self.assertTrue(result.is_relative_to(root.resolve()))

    def test_download_resumes_and_checks_remote_hash(self):
        payload=b'entity_id\tmatched_ids\nS1_1\tS2_1\n'
        class Response:
            status_code=206
            headers={'Content-Range':f'bytes 7-{len(payload)-1}/{len(payload)}',
                     'Content-Length':str(len(payload)-7)}
            def raise_for_status(self): pass
            def iter_content(self,chunk_size):
                yield payload[7:15]
                yield payload[15:]
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'matching_results.tsv'
            target.with_suffix('.tsv.part').write_bytes(payload[:7])
            def get(url,**kwargs):
                self.assertTrue(kwargs['stream'])
                self.assertEqual(kwargs['headers'],{'Range':'bytes=7-'})
                return Response()
            with patch.dict(sys.modules,{'requests':SimpleNamespace(get=get)}):
                result=delivery.stream_file('https://example.invalid/output',target,hashlib.sha256(payload).hexdigest())
            self.assertEqual(target.read_bytes(),payload)
            self.assertEqual(result['bytes'],len(payload))

    def test_wrong_hash_never_publishes_submission(self):
        class Response:
            status_code=200
            headers={'Content-Length':'3'}
            def raise_for_status(self): pass
            def iter_content(self,chunk_size): yield b'bad'
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'matching_results.tsv'
            with patch.dict(sys.modules,{'requests':SimpleNamespace(get=lambda *a,**k:Response())}):
                with self.assertRaisesRegex(ValueError,'validated hash'):
                    delivery.stream_file('https://example.invalid/output',target,'0'*64)
            self.assertFalse(target.exists())
