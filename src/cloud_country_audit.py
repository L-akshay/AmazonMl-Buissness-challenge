"""Confirm country routing using complete cached OOF scores, without refitting.

Policy selection uses folds 0/1 and confirmation uses folds 2/3. The models are
cross-fitted, so confirmation is not a new untouched or nested model holdout.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
from src.cloud_decisions import THRESHOLDS, summarize
from src.cloud_features import references
from src.cloud_store import atomic_json, check_headroom, claim_config, digest, parquet_files, read_parquet

COUNTRIES = ('india', 'us')


def mask_digest(mask):
    return hashlib.sha256(mask.tobytes()).hexdigest()


def checked_scores(batch, ref, fold, country=None):
    ri = batch['ri']
    if (not np.issubdtype(ri.dtype, np.integer) or np.any(ri < 0)
            or np.any(ri >= len(ref['ri'])) or np.any(ref['fold'][ri] != fold)):
        raise ValueError('Score shard contains fitted or reserved references')
    if country is not None and np.any(ref['country_norm'][ri] != country):
        raise ValueError('Specialist shard contains the wrong country')
    if not np.isfinite(batch['p']).all() or np.any((batch['p'] < 0) | (batch['p'] > 1)):
        raise ValueError('Invalid probability in score shard')
    if not np.isin(batch['y'], [0, 1]).all():
        raise ValueError('Invalid label in score shard')
    order = np.lexsort((batch['tid'], ri))
    if np.any((ri[order][1:] == ri[order][:-1])
              & (batch['tid'][order][1:] == batch['tid'][order][:-1])):
        raise ValueError('Duplicate candidate pair in score shard')


def paired_scores(baseline, specialists, ref, fold):
    checked_scores(baseline, ref, fold)
    for country in COUNTRIES:
        part = specialists[country]
        checked_scores(part, ref, fold, country)
        scope = ref['country_norm'][baseline['ri']] == country
        if any(not np.array_equal(baseline[key][scope], part[key]) for key in ('ri', 'tid', 'y')):
            raise ValueError('Specialist scores do not cover the same ordered candidate pairs')
    return {key: np.concatenate([specialists[c][key] for c in COUNTRIES])
            for key in ('ri', 'tid', 'y', 'p')}


def grid_from_histograms(counts, hits, ref, thresholds):
    count = np.zeros(len(ref['ri']), dtype=np.int64)
    tp = np.zeros_like(count)
    scopes = {'selection': np.isin(ref['fold'], [0, 1]),
              'confirmation': np.isin(ref['fold'], [2, 3]), 'development': ref['fold'] < 4}
    rows = []
    for index in range(len(thresholds) - 1, -1, -1):
        count += counts[:, index + 1]
        tp += hits[:, index + 1]
        rows.append({'threshold': float(thresholds[index]),
                     **{name: summarize(ref, count, tp, scope) for name, scope in scopes.items()}})
    return rows


def select_routes(grids):
    routes = {}
    for country in COUNTRIES:
        candidates = [(source, row) for source, rows in grids.items() for row in rows]
        source, row = max(candidates, key=lambda item: (
            item[1]['selection']['countries'][country]['macro_f05'],
            item[1]['selection']['countries'][country]['link_precision'],
            item[0] == 'baseline', item[1]['threshold']))
        routes[country] = {'source': source, 'threshold': row['threshold'], 'relative': 0,
                           'selection': row['selection']['countries'][country]}
    return routes


def routed_metrics(grids, routes, ref, split):
    countries = {}
    for country, route in routes.items():
        row = next(r for r in grids[route['source']] if r['threshold'] == route['threshold'])
        countries[country] = row[split]['countries'][country]
    n = sum(row['entities'] for row in countries.values())
    truth = sum(row['true_links'] for row in countries.values())
    count = sum(row['predicted_links'] for row in countries.values())
    tp = sum(row['true_positive_links'] for row in countries.values())
    scope = (np.isin(ref['fold'], [0, 1]) if split == 'selection' else
             np.isin(ref['fold'], [2, 3]) if split == 'confirmation' else ref['fold'] < 4)
    singleton_counts = {c: int(np.sum(scope & (ref['country_norm'] == c) & (ref['truth_count'] == 0)))
                        for c in countries}
    singletons = sum(singleton_counts.values())
    overall = {'entities': n, 'macro_f05': sum(r['entities'] * r['macro_f05'] for r in countries.values()) / n,
               'link_precision': tp / max(count, 1), 'link_recall': tp / max(truth, 1),
               'singleton_fp_rate': sum((countries[c]['singleton_fp_rate'] or 0) * size
                                        for c, size in singleton_counts.items()) / singletons if singletons else None,
               'true_links': truth, 'predicted_links': count, 'true_positive_links': tp}
    return {'overall': overall, 'countries': countries}


def audit_country_routes(root, config):
    refpath = references(root, 'train')
    ref = read_parquet(refpath, 'ri,fold,truth_count,country_norm')
    if not np.array_equal(ref['ri'], np.arange(len(ref['ri']))):
        raise ValueError('Reference indices are not contiguous')
    if set(ref['country_norm'][ref['fold'] < 4]) != set(COUNTRIES):
        raise ValueError('Country-routing audit requires the declared full development countries')
    if any(not np.any((ref['fold'] == fold) & (ref['country_norm'] == country))
           for fold in range(4) for country in COUNTRIES):
        raise ValueError('Every development fold must contain both countries')
    interval = config.get('country_audit_checkpoint_shards', 128)
    if not isinstance(interval, int) or interval < 1:
        raise ValueError('country_audit_checkpoint_shards must be a positive integer')
    tasks, identities = [], []
    for fold in range(4):
        inputs = {}
        for source in ('baseline',) + COUNTRIES:
            name = f'gbdt_fold_{fold}' if source == 'baseline' else f'research/country-{source}/fold_{fold}'
            scores = root / 'cache/cloud/scores' / name
            model = root / 'cache/cloud/models' / name
            scoring = ref['fold'] == fold
            fitting = (ref['fold'] < 4) & (ref['fold'] != fold)
            if source != 'baseline':
                scoring &= ref['country_norm'] == source
                fitting &= ref['country_norm'] == source
            score_config = json.loads((scores / 'config.json').read_text())
            model_config = json.loads((model / 'config.json').read_text())
            score_complete = json.loads((scores / 'complete.json').read_text())
            model_complete = json.loads((model / 'complete.json').read_text())
            if (score_config['split'] != 'train' or score_config['scope'] != mask_digest(scoring)
                    or model_config['allowed_sha256'] != mask_digest(fitting)):
                raise ValueError('Cached model/scoring scope is not the expected grouped OOF scope')
            if not (score_config['model_sha256'] == score_complete['model_sha256']
                    == model_complete['sha256'] == digest(model / 'model.txt')):
                raise ValueError('Cached model and score identities disagree')
            inputs[source] = parquet_files(scores)
            names = [p.name for p in inputs[source]]
            if not names or len(names) != len(set(names)) or names != sorted(names):
                raise ValueError('Duplicate or unordered score shard manifest')
            identities.append({'name': name, 'score_config': digest(scores / 'config.json'),
                               'score_complete': digest(scores / 'complete.json'),
                               'model_config': digest(model / 'config.json'),
                               'model_complete': digest(model / 'complete.json')})
        if any([p.name for p in inputs[c]] != [p.name for p in inputs['baseline']] for c in COUNTRIES):
            raise ValueError('Specialist and baseline shard manifests differ')
        tasks.extend((fold, {source: paths[i] for source, paths in inputs.items()})
                     for i in range(len(inputs['baseline'])))
    out = root / 'cache/cloud/country_route_audit'
    baseline_report = root / 'reports/cloud_model_validation.json'
    claim_config(out, {'code': digest(Path(__file__)), 'metrics_code': digest(Path(__file__).with_name('cloud_model.py')),
                       'summary_code': digest(Path(__file__).with_name('cloud_decisions.py')),
                       'references': digest(refpath), 'sources': identities,
                       'baseline_report': digest(baseline_report), 'thresholds': THRESHOLDS.tolist()})
    report_path = out / 'report.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    shape = (2, len(ref['ri']), len(THRESHOLDS) + 1)
    checkpoint = out / 'histograms.npz'
    next_shard, rows = 0, 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            counts, hits = saved['counts'], saved['hits']
            next_shard, rows = int(saved['next_shard']), int(saved['rows'])
        if counts.shape != shape or hits.shape != shape or counts.dtype != np.int32 or hits.dtype != np.int32:
            raise ValueError('Invalid country audit histogram checkpoint')
        if not 0 <= next_shard <= len(tasks):
            raise ValueError('Invalid resumed score shard offset')
    else:
        counts = np.zeros(shape, dtype=np.int32)
        hits = np.zeros_like(counts)
    for index in range(next_shard, len(tasks)):
        check_headroom(root)
        fold, paths = tasks[index]
        baseline = read_parquet(paths['baseline'], 'ri,tid,y,p')
        specialist = paired_scores(baseline, {c: read_parquet(paths[c], 'ri,tid,y,p') for c in COUNTRIES}, ref, fold)
        for source_index, batch in enumerate((baseline, specialist)):
            bins = np.searchsorted(THRESHOLDS, batch['p'], side='right')
            np.add.at(counts[source_index], (batch['ri'], bins), 1)
            positive = batch['y'] > 0
            np.add.at(hits[source_index], (batch['ri'][positive], bins[positive]), 1)
        rows += len(baseline['ri'])
        if (index + 1) % interval == 0 or index + 1 == len(tasks):
            temporary = checkpoint.with_suffix('.tmp.npz')
            np.savez(temporary, counts=counts, hits=hits, next_shard=index + 1, rows=rows)
            temporary.replace(checkpoint)
            print(f'Country route audit: {index + 1}/{len(tasks)} OOF shards checkpointed', flush=True)
    if np.any(hits.sum(axis=2) > ref['truth_count'][None, :]):
        raise ValueError('Retrieved positives exceed reference truth counts')
    grids = {name: grid_from_histograms(counts[i], hits[i], ref, THRESHOLDS)
             for i, name in enumerate(('baseline', 'specialist'))}
    baseline = next(row for row in grids['baseline'] if row['threshold'] == .625)
    expected = json.loads(baseline_report.read_text())['comparisons']['gbdt']['oof']['overall']
    if baseline['development']['overall']['entities'] != expected['entities'] or abs(
            baseline['development']['overall']['macro_f05'] - expected['macro_f05']) > 1e-10:
        raise ValueError('Cached scores do not reproduce the deployed-model OOF baseline')
    routes = select_routes(grids)
    atomic_json(out / 'selected_before_confirmation.json', routes)
    routed = {split: routed_metrics(grids, routes, ref, split)
              for split in ('selection', 'confirmation', 'development')}
    delta = routed['confirmation']['overall']['macro_f05'] - baseline['confirmation']['overall']['macro_f05']
    country_deltas = {c: routed['confirmation']['countries'][c]['macro_f05']
                     - baseline['confirmation']['countries'][c]['macro_f05'] for c in COUNTRIES}
    result = {'scope': 'Complete unchanged candidate pairs, development folds 0-3; no new reserved evaluation',
              'score_rows_per_model': rows, 'selected_routes': routes, 'baseline': baseline,
              'routed': routed, 'confirmation_macro_delta': delta, 'confirmation_country_deltas': country_deltas,
              'supports_new_submission': bool(delta > .0005 and min(country_deltas.values()) >= -.002),
              'unknown_country_fallback': {'source': 'baseline', 'threshold': .625, 'relative': 0},
              'grids': grids,
              'limits': ['Exploratory follow-up proposed after viewing all-development specialist results.',
                         'Policy selection uses folds 0/1 only; confirmation is cross-fitted, not untouched or nested.',
                         'Only India/US labels exist; no conclusion about France quality is supported.',
                         'No candidate, feature, model, or submission is regenerated by this audit.']}
    atomic_json(report_path, result)
    atomic_json(root / 'reports/cloud_country_route_audit.json', result)
    return result
