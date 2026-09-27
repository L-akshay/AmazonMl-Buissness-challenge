import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.cloud_country_final import (load_policy, fit_country, score_country, merge_batch, assemble_country_scores)
from src.cloud_country_audit import mask_digest
from src.cloud_export import write_variants
from src.cloud_store import atomic_json, digest, read_parquet, write_parquet


def fixture(root):
    audit = {'supports_new_submission': True, 'confirmation_macro_delta': .007,
             'confirmation_country_deltas': {'india': .009, 'us': .005},
             'selected_routes': {'india': {'source': 'specialist', 'threshold': .625, 'relative': 0},
                                 'us': {'source': 'specialist', 'threshold': float(np.float32(.65)), 'relative': 0}},
             'unknown_country_fallback': {'source': 'baseline', 'threshold': .625, 'relative': 0}}
    audit_path = root / 'reports/cloud_country_route_audit.json'
    atomic_json(audit_path, audit)
    train = {'ri': np.arange(6, dtype=np.int32), 'fold': np.array([0, 0, 2, 2, 4, 4]),
             'country_norm': np.array(['india', 'us'] * 3)}
    test = {**train, 'country_norm': np.array(['india', 'us', 'france', 'india', 'us', 'unknown'])}
    for split, ref in [('train', train), ('test', test)]:
        write_parquet(root / f'cache/cloud/references_{split}.parquet', ref)
    names = [f'batch_{i:05d}.parquet' for i in range(2)]
    atomic_json(root / 'cache/cloud/features_test/complete.json', {'files': names, 'rows': 12})
    for country in ('baseline', 'india', 'us'):
        score_name = 'test_gbdt' if country == 'baseline' else f'country_test/{country}'
        model_name = 'full_gbdt' if country == 'baseline' else f'country_full/{country}'
        model = root / 'cache/cloud/models' / model_name
        scores = root / 'cache/cloud/scores' / score_name
        model.mkdir(parents=True)
        (model / 'model.txt').write_text(country)
        sha = digest(model / 'model.txt')
        fitting = np.ones(6, dtype=bool) if country == 'baseline' else train['country_norm'] == country
        scoring = np.ones(6, dtype=bool) if country == 'baseline' else test['country_norm'] == country
        atomic_json(model / 'config.json', {'allowed_sha256': mask_digest(fitting)})
        atomic_json(model / 'complete.json', {'sha256': sha})
        atomic_json(model / 'routing.json', {'audit_sha256': digest(audit_path), 'country': country})
        atomic_json(scores / 'config.json', {'scope': mask_digest(scoring), 'split': 'test', 'model_sha256': sha})
        atomic_json(scores / 'complete.json', {'files': names, 'model_sha256': sha})
        for i, name in enumerate(names):
            write_parquet(scores / name, {'ri': test['ri'][scoring],
                         'tid': ((2 << 32) + np.arange(6, dtype=np.uint64) + i * 6)[scoring],
                         'y': np.zeros(6, dtype=np.uint8)[scoring],
                         'p': np.full(scoring.sum(), .7 if country == 'baseline' else .9, dtype=np.float32)})
    return audit, train, test


class CountryFinalTests(unittest.TestCase):
    def test_final_masks_include_every_country_fold_and_only_country_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, train, test = fixture(root)
            for country in ('india', 'us'):
                model = root / f'cache/cloud/models/country_full/{country}/model.txt'
                with patch('src.cloud_country_final.train_one', return_value=model) as fitting:
                    fit_country(root, country, {}, 'fingerprint')
                mask = fitting.call_args.args[2]
                np.testing.assert_array_equal(mask, train['country_norm'] == country)
                self.assertTrue(np.any(mask & (train['fold'] == 4)))
                with patch('src.cloud_country_final.score_model', return_value=[]) as scoring:
                    score_country(root, country, {})
                np.testing.assert_array_equal(scoring.call_args.args[3], test['country_norm'] == country)

    def test_assembly_preserves_all_candidates_unseen_scores_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, _, ref = fixture(root)
            real_write = write_parquet
            calls = []
            def interrupted(path, batch):
                calls.append(path.name)
                if len(calls) == 2:
                    raise RuntimeError('synthetic interruption')
                real_write(path, batch)
            with patch('src.cloud_country_final.write_parquet', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    assemble_country_scores(root, {})
            with patch('src.cloud_country_final.write_parquet', wraps=real_write) as writing:
                final = assemble_country_scores(root, {})
                self.assertEqual(writing.call_count, 1)
            self.assertEqual(final['candidate_pairs'], 12)
            out = root / 'cache/cloud/scores/country_routed'
            for path in sorted(out.glob('*.parquet')):
                batch = read_parquet(path)
                np.testing.assert_array_equal(batch['ri'], ref['ri'])
                np.testing.assert_allclose(batch['p'], [.9, .9, .7, .9, .9, .7])
            with patch('src.cloud_country_final.merge_batch', side_effect=AssertionError('must reuse shards')):
                self.assertEqual(assemble_country_scores(root, {}), final)
            marker = out / 'batch_00000.parquet.json'
            proof = json.loads(marker.read_text()); proof['sha256'] = 'corrupted'
            atomic_json(marker, proof)
            with self.assertRaisesRegex(ValueError, 'shard changed'):
                assemble_country_scores(root, {})

    def test_gate_and_wrong_fitting_scope_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); audit, train, _ = fixture(root)
            path = root / 'reports/cloud_country_route_audit.json'
            atomic_json(path, {**audit, 'supports_new_submission': False})
            with self.assertRaisesRegex(ValueError, 'confirmation gate'):
                load_policy(root)
            atomic_json(path, audit)
            atomic_json(root / 'cache/cloud/models/country_full/india/config.json',
                        {'allowed_sha256': mask_digest((train['country_norm'] == 'india') & (train['fold'] < 4))})
            with self.assertRaisesRegex(ValueError, 'fitting or test scoring scope'):
                assemble_country_scores(root, {})

    def test_missing_misaligned_and_duplicate_candidates_are_rejected(self):
        ref = {'ri': np.arange(3), 'country_norm': np.array(['india', 'us', 'france'])}
        base = {'ri': np.arange(3), 'tid': np.arange(3), 'y': np.zeros(3), 'p': np.full(3, .7)}
        bad = {k: v[:0] for k, v in base.items()}
        with self.assertRaisesRegex(ValueError, 'exactly the baseline'):
            merge_batch(base, {'india': bad}, ref)
        bad = {k: v[:1].copy() for k, v in base.items()}; bad['tid'][0] = 50
        with self.assertRaisesRegex(ValueError, 'exactly the baseline'):
            merge_batch(base, {'india': bad}, ref)
        duplicate = {k: np.repeat(v[:1], 2) for k, v in base.items()}
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            merge_batch(duplicate, {}, ref)

    def test_export_frozen_country_thresholds_and_unknown_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = {'name': 'routed', 'threshold': .625, 'relative': 0,
                      'country_routes': {'us': {'threshold': float(np.float32(.65)), 'relative': 0},
                                         'india': {'threshold': .625, 'relative': 0}}}
            ids = [f'S1-{i}' for i in range(5)]
            countries = ['india', 'us', 'france', 'us', 'unknown']
            edges = [(i, (2 << 32) + i, p) for i, p in enumerate([.64, .64, .64, float(np.float32(.65)), .624])]
            counts = write_variants(ids, edges, np.ones(5), [policy], root, countries)
            self.assertEqual(counts['routed']['candidate_pairs'], 5)
            self.assertEqual(counts['routed']['matched_pairs'], 3)
            with (root / 'routed/matching_results.tsv').open(newline='') as f:
                rows = list(csv.reader(f, delimiter='\t'))
            self.assertEqual([r[1] for r in rows[1:]], ['S2-0', '', 'S2-2', 'S2-3', ''])
            with self.assertRaisesRegex(ValueError, 'one country per reference'):
                write_variants(ids, edges, np.ones(5), [policy], root)
