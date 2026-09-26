from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.blocking import build_index,Retriever


class IndexResumeTests(unittest.TestCase):
    def test_failed_binary_write_is_not_a_completed_index(self):
        records=[(f'S1-{i}',f'uniquename{i}',f'uniqueaddress{i}','us') for i in range(250)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cache').mkdir()
            folder=root/'cache/sparse_v1_train'
            with patch('src.blocking.reference_records',return_value=records):
                with patch('src.blocking.pickle.dump',side_effect=OSError('simulated interruption')):
                    with self.assertRaisesRegex(OSError,'simulated interruption'):
                        build_index(root,'train',channels=('token',))
                self.assertFalse((folder/'token.json').exists())
                self.assertFalse((folder/'token.pkl').exists())
                build_index(root,'train',channels=('token',))
            found=Retriever(folder,channels=('token',)).search(records[:1])['token']
            self.assertEqual(found.indices[0],0)
            self.assertTrue((folder/'token.json').is_file())
            with patch('src.blocking.reference_records',side_effect=AssertionError('completed index was recomputed')):
                self.assertEqual(build_index(root,'train',channels=('token',)),folder)
