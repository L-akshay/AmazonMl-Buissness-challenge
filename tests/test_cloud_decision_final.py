import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.cloud_decision_final import transform, apply_decisions
from src.cloud_store import atomic_json, digest, read_parquet, write_parquet


class DecisionFinalTests(unittest.TestCase):
    def test_competition_preserves_candidates_and_unknown_country_scores(self):
        countries = np.array(['india', 'us', 'france', 'india'])
        b = {'ri': np.array([0, 1, 2, 0, 3]), 'tid': np.array([11, 11, 11, 12, 12]),
             'y': np.zeros(5, dtype=np.uint8), 'p': np.array([.8, .9, .7, .8, .8], dtype=np.float32)}
        rules = {'india': {'source': 'exclusive', 'threshold': .6, 'relative': 0},
                 'us': {'source': 'soft_025', 'threshold': .6, 'relative': 0}}
        result = transform(b, countries, rules)
        for key in ('ri', 'tid', 'y'):
            np.testing.assert_array_equal(result[key], b[key])
        self.assertEqual(result['p'][2], b['p'][2])
        np.testing.assert_array_equal(result['p'][[0, 3, 4]], [0, 0, 0])
        self.assertAlmostEqual(result['p'][1], .9 * .2 ** .25, places=6)
        np.testing.assert_array_equal(b['p'], np.array([.8, .9, .7, .8, .8], dtype=np.float32))

    def test_full_transform_resume_and_changed_source_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root / 'cache/cloud'; source = cache / 'scores/country_routed'
            write_parquet(cache / 'references_test.parquet', {'country_norm': np.array(['india', 'us', 'france'])})
            rules = {'india': {'source': 'exclusive', 'threshold': .6, 'relative': 0},
                     'us': {'source': 'soft_025', 'threshold': .6, 'relative': 0}}
            audit = {'supports_new_submission': True, 'confirmation_macro_delta': .002,
                     'confirmation_country_deltas': {'india': .002, 'us': .002}, 'selected_decisions': rules}
            atomic_json(root / 'reports/cloud_routed_decision_audit.json', audit)
            atomic_json(cache / 'routed_decision_audit/config.json', {'country_report': 'route-sha'})
            policy = {'name': 'country_routed', 'source': 'baseline', 'threshold': .625, 'relative': 0,
                      'country_routes': {}}
            prior = {'kind': 'country_routed_gbdt', 'score_name': 'country_routed',
                     'audit_sha256': 'route-sha', 'candidate_pairs': 3, 'policy': policy}
            atomic_json(cache / 'final.json', prior)
            b = {'ri': np.arange(3, dtype=np.int32), 'tid': np.full(3, (2 << 32) + 1000, dtype=np.uint64),
                 'y': np.zeros(3, dtype=np.uint8), 'p': np.array([.9, .8, .7], dtype=np.float32)}
            path = source / 'batch_00000.parquet'; write_parquet(path, b)
            atomic_json(source / (path.name + '.json'), {'sha256': digest(path)})
            complete = {'files': [path.name], 'rows': 3}
            atomic_json(source / 'complete.json', complete)
            atomic_json(cache / 'features_test/complete.json', complete)
            with patch('src.cloud_decision_final.check_headroom'):
                final = apply_decisions(root, {})
                self.assertEqual(final['policy']['threshold'], .625)
                self.assertEqual(final['candidate_pairs'], 3)
                out = read_parquet(cache / 'scores/country_decisions' / path.name)
                np.testing.assert_allclose(out['p'], [.9, 0, .7])
                self.assertEqual(out['p'][2], b['p'][2])
                self.assertEqual(out['p'][1], 0)
                with patch('src.cloud_decision_final.transform', side_effect=AssertionError('recomputed')):
                    self.assertEqual(apply_decisions(root, {}), final)
                changed = copy.deepcopy(b); changed['p'][0] = .1; write_parquet(path, changed)
                with self.assertRaisesRegex(ValueError, 'Original V4 score shard changed'):
                    apply_decisions(root, {})
