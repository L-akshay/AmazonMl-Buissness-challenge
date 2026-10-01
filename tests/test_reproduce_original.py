import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.reproduce_original import install_original_model, publish, main


class OriginalReproductionTests(unittest.TestCase):
    def test_original_model_identity_and_conflicting_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); resources = root / 'resources/original_submission'
            model = resources / 'model/model.txt'; model.parent.mkdir(parents=True)
            model.write_bytes(b'synthetic model')
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            (model.parent / 'complete.json').write_text(json.dumps({'kind': 'gbdt', 'sha256': checksum}))
            (resources / 'manifest.json').write_text(json.dumps({'model_sha256': checksum,
                                                               'policy': {'threshold': .625, 'relative': 0}}))
            install_original_model(root); install_original_model(root)
            final = root / 'cache/cloud/final.json'
            self.assertEqual(json.loads(final.read_text())['variants'][0]['threshold'], .625)
            final.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'fresh workspace'):
                install_original_model(root)
            model.write_bytes(b'changed model')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                install_original_model(root)

    def test_published_files_must_match_original_not_later_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); resources = root / 'resources/original_submission'; resources.mkdir(parents=True)
            out = root / 'output/cloud'; (out / 'oof_best').mkdir(parents=True)
            outputs = {}
            for name, relative in [('matching_results.tsv', 'oof_best/matching_results.tsv'),
                                   ('candidate_pairs.tsv', 'candidate_pairs.tsv')]:
                content = (name + '\n').encode(); (out / relative).write_bytes(content)
                outputs[name] = {'sha256': hashlib.sha256(content).hexdigest()}
            (resources / 'manifest.json').write_text(json.dumps({'outputs': outputs}))
            publish(root)
            (out / 'oof_best/matching_results.tsv').write_text('later submission')
            with self.assertRaisesRegex(ValueError, 'differs from submitted'):
                publish(root)

    def test_remote_guard_blocks_accidental_local_full_run(self):
        with patch.dict('os.environ', {}, clear=True), patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'remote compute'):
                main([])
            run.assert_not_called()
