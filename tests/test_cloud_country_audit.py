import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.cloud_country_audit import (COUNTRIES, audit_country_routes, checked_scores,
                                     grid_from_histograms, mask_digest, paired_scores,
                                     routed_metrics, select_routes)
from src.cloud_model import entity_metrics
from src.cloud_store import atomic_json, digest, write_parquet


def fixture(root):
    cache = root / 'cache/cloud'
    ref = {'ri': np.arange(10, dtype=np.int32), 'fold': np.repeat(np.arange(5), 2),
           'truth_count': np.tile([3, 0], 5), 'country_norm': np.array(['india', 'us'] * 5)}
    write_parquet(cache / 'references_train.parquet', ref)
    for fold in range(4):
        for source in ('baseline',) + COUNTRIES:
            name = f'gbdt_fold_{fold}' if source == 'baseline' else f'research/country-{source}/fold_{fold}'
            model, scores = cache / 'models' / name, cache / 'scores' / name
            model.mkdir(parents=True)
            (model / 'model.txt').write_text(name)
            model_hash = digest(model / 'model.txt')
            fit, score = (ref['fold'] < 4) & (ref['fold'] != fold), ref['fold'] == fold
            if source != 'baseline':
                fit &= ref['country_norm'] == source
                score &= ref['country_norm'] == source
            atomic_json(model / 'config.json', {'allowed_sha256': mask_digest(fit)})
            atomic_json(model / 'complete.json', {'sha256': model_hash})
            atomic_json(scores / 'config.json', {'scope': mask_digest(score), 'split': 'train',
                                                'model_sha256': model_hash})
            names = []
            for shard in range(2):
                ri = np.array([2 * fold, 2 * fold + 1] + ([2 * fold] if shard else []), dtype=np.int32)
                y = np.array([1, 0] + ([0] if shard else []), dtype=np.uint8)
                p = np.where(y, .6, .7) if source == 'baseline' else np.where(y, .9, .1)
                b = {'ri': ri, 'tid': np.arange(len(ri), dtype=np.uint64) + 10 * (2 * fold + shard),
                     'y': y, 'p': p.astype(np.float32)}
                b = {k: val[score[ri]] for k, val in b.items()}
                filename = f'batch_{shard:05d}.parquet'
                write_parquet(scores / filename, b)
                names.append(filename)
            atomic_json(scores / 'complete.json', {'files': names, 'model_sha256': model_hash})
    count = np.tile([1, 2], 5)
    expected = entity_metrics(ref['truth_count'], count, np.zeros(10, dtype=np.int64), ref['fold'] < 4)
    atomic_json(root / 'reports/cloud_model_validation.json', {'comparisons': {'gbdt': {'oof': {'overall': expected}}}})
    return ref


class CountryAuditTests(unittest.TestCase):
    def test_histogram_and_country_routing_match_direct_counts(self):
        ref = {'ri': np.arange(10), 'fold': np.repeat(np.arange(5), 2),
               'truth_count': np.array([0, 1, 2, 1, 1, 0, 2, 1, 1, 1]),
               'country_norm': np.array(['india', 'us'] * 5)}
        ri = np.repeat(np.arange(8), 2)
        p = np.tile([.9, .5, .3, .8], 4).astype(np.float32)
        y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0])
        thresholds = np.array([.5, .7, .9], dtype=np.float32)
        counts = np.zeros((10, 4), dtype=np.int32)
        hits = np.zeros_like(counts)
        bins = np.searchsorted(thresholds, p, side='right')
        np.add.at(counts, (ri, bins), 1)
        np.add.at(hits, (ri[y > 0], bins[y > 0]), 1)
        rows = grid_from_histograms(counts, hits, ref, thresholds)
        for row in rows:
            keep = p >= row['threshold']
            expected = entity_metrics(ref['truth_count'], np.bincount(ri[keep], minlength=10),
                                      np.bincount(ri[keep & (y > 0)], minlength=10), ref['fold'] < 4)
            self.assertEqual(row['development']['overall'], expected)
        grids = {'baseline': rows, 'specialist': copy.deepcopy(rows)}
        routes = select_routes(grids)
        # Identical scores prefer the baseline; confirmation labels never select routes.
        self.assertTrue(all(route['source'] == 'baseline' for route in routes.values()))
        changed = copy.deepcopy(grids)
        for row in changed['specialist']:
            for country in COUNTRIES:
                row['confirmation']['countries'][country]['macro_f05'] = 1
        self.assertEqual(select_routes(changed), routes)
        actual = routed_metrics(grids, routes, ref, 'development')['overall']
        keep = np.array([p[i] >= routes[ref['country_norm'][r]]['threshold'] for i, r in enumerate(ri)])
        expected = entity_metrics(ref['truth_count'], np.bincount(ri[keep], minlength=10),
                                  np.bincount(ri[keep & (y > 0)], minlength=10), ref['fold'] < 4)
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value)

    def test_complete_pair_coverage_fold_scope_and_probabilities_are_enforced(self):
        ref = {'ri': np.arange(3), 'fold': np.array([0, 0, 4]),
               'country_norm': np.array(['india', 'us', 'india'])}
        b = {'ri': np.array([0, 1]), 'tid': np.array([10, 11]),
             'y': np.array([1, 0]), 'p': np.array([.9, .1])}
        parts = {country: {key: val[[i]] for key, val in b.items()} for i, country in enumerate(COUNTRIES)}
        paired_scores(b, parts, ref, 0)
        wrong = copy.deepcopy(parts)
        wrong['india']['tid'][0] = 12
        with self.assertRaisesRegex(ValueError, 'same ordered candidate'):
            paired_scores(b, wrong, ref, 0)
        for changed, message in [('scope', 'fitted or reserved'), ('probability', 'Invalid probability'),
                                 ('country', 'wrong country'), ('duplicate', 'Duplicate candidate')]:
            item = copy.deepcopy(parts['india'])
            if changed == 'scope': item['ri'][0] = 2
            elif changed == 'probability': item['p'][0] = np.nan
            elif changed == 'country': item['ri'][0] = 1
            else: item = {key: np.repeat(val, 2) for key, val in item.items()}
            with self.assertRaisesRegex(ValueError, message):
                checked_scores(item, ref, 0, 'india')

    def test_interrupted_histogram_resume_reuses_processed_shards_and_complete_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            calls = 0
            def interrupt():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise TimeoutError('synthetic interruption after first atomic checkpoint')
            with patch('src.cloud_country_audit.check_headroom', side_effect=lambda root: interrupt()):
                with self.assertRaises(TimeoutError):
                    audit_country_routes(root, {'country_audit_checkpoint_shards': 1})
            checkpoint = root / 'cache/cloud/country_route_audit/histograms.npz'
            with np.load(checkpoint) as saved:
                self.assertEqual(int(saved['next_shard']), 1)
            with patch('src.cloud_country_audit.check_headroom'), patch(
                    'src.cloud_country_audit.paired_scores', wraps=paired_scores) as paired:
                result = audit_country_routes(root, {'country_audit_checkpoint_shards': 1})
                self.assertEqual(paired.call_count, 7)
                self.assertEqual(result['score_rows_per_model'], 20)
                self.assertTrue(result['supports_new_submission'])
                self.assertAlmostEqual(result['routed']['confirmation']['overall']['macro_f05'], (10 / 11 + 1) / 2)
                with patch('src.cloud_country_audit.paired_scores', side_effect=AssertionError('recomputed')):
                    self.assertEqual(audit_country_routes(root, {}), result)

    def test_fitted_scope_mismatch_is_rejected_before_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = fixture(root)
            atomic_json(root / 'cache/cloud/models/gbdt_fold_0/config.json',
                        {'allowed_sha256': mask_digest(ref['fold'] < 4)})
            with self.assertRaisesRegex(ValueError, 'expected grouped OOF scope'):
                audit_country_routes(root, {})
