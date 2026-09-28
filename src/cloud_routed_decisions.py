"""Evaluate secondary competition on the deployed country routes using cached OOF scores."""

import json
from pathlib import Path
import numpy as np
from src.cloud_country_audit import (COUNTRIES, audit_country_routes, paired_scores,
                                     routed_metrics, select_routes)
from src.cloud_decisions import MODES, competition, scan_grid, summarize
from src.cloud_features import references
from src.cloud_store import (atomic_json, check_headroom, claim_config, digest,
                             parquet_files, read_parquet, write_parquet)
from src.predict import decode_secondary


def audit_routed_decisions(root, config):
    # Rechecks model/fold hashes and returns the existing country audit cache.
    # Missing prior evidence must not trigger an unexpected full audit here.
    prior_path = root / 'cache/cloud/country_route_audit/report.json'
    if not prior_path.exists():
        raise ValueError('Attach the completed country-routing audit checkpoint')
    prior = audit_country_routes(root, config)
    refpath = references(root, 'train')
    ref = read_parquet(refpath, 'ri,fold,truth_count,country_norm')
    routes = prior['selected_routes']
    sources = []
    for fold in range(4):
        sources.append({source: parquet_files(root / 'cache/cloud/scores' / (
            f'gbdt_fold_{fold}' if source == 'baseline' else f'research/country-{source}/fold_{fold}'))
                        for source in ('baseline',) + COUNTRIES})
    names = [p.name for p in sources[0]['baseline']]
    if any([p.name for p in paths] != names for fold in sources for paths in fold.values()):
        raise ValueError('Cross-fold score shard boundaries differ')
    out = root / 'cache/cloud/routed_decision_audit'
    claim_config(out, {'code': digest(Path(__file__)),
                      'decision_code': digest(Path(__file__).with_name('cloud_decisions.py')),
                      'country_config': digest(prior_path.parent / 'config.json'),
                      'country_report': digest(prior_path), 'references': digest(refpath)})
    if (out / 'report.json').exists():
        return json.loads((out / 'report.json').read_text())
    n = len(ref['ri'])
    count = np.zeros(n, dtype=np.int64); hits = np.zeros_like(count)
    candidate_hits = np.zeros_like(count)
    files = []; previous = None; rows = 0
    for index, name in enumerate(names):
        check_headroom(root)
        target = out / 'scores' / name; marker = target.with_suffix('.json')
        if target.exists() and marker.exists():
            b = read_parquet(target); info = json.loads(marker.read_text())
        else:
            parts = []
            for fold, paths in enumerate(sources):
                baseline = read_parquet(paths['baseline'][index], 'ri,tid,y,p')
                specialists = {c: read_parquet(paths[c][index], 'ri,tid,y,p') for c in COUNTRIES}
                paired_scores(baseline, specialists, ref, fold)
                for c in COUNTRIES:
                    part = specialists[c] if routes[c]['source'] == 'specialist' else {
                        k: v[ref['country_norm'][baseline['ri']] == c] for k, v in baseline.items()}
                    parts.append(part)
            b = {key: np.concatenate([p[key] for p in parts]) for key in ('ri', 'tid', 'y', 'p')}
            b['winner'], b['second'] = competition(b)
            boundaries = [decode_secondary(int(t)) for t in np.unique(b['tid'])]
            info = {'first': min(boundaries) if boundaries else None,
                    'last': max(boundaries) if boundaries else None, 'rows': len(b['ri'])}
            write_parquet(target, b); atomic_json(marker, info)
        if info['first'] is not None:
            if previous is not None and info['first'] <= previous:
                raise ValueError('Secondary IDs cross shard boundaries')
            previous = info['last']
        if len(b['ri']) != info['rows']:
            raise ValueError('Incomplete routed score shard')
        rows += len(b['ri']); files.append(target)
        np.add.at(candidate_hits, b['ri'][b['y'] > 0], 1)
        thresholds = np.zeros(len(b['ri']), dtype=np.float32)
        for c in COUNTRIES:
            thresholds[ref['country_norm'][b['ri']] == c] = routes[c]['threshold']
        chosen = b['p'] >= thresholds
        np.add.at(count, b['ri'][chosen], 1)
        np.add.at(hits, b['ri'][chosen & (b['y'] > 0)], 1)
        if index % 50 == 0:
            print(f'Routed decision audit: joined {index + 1}/{len(names)} shards', flush=True)
    if rows != prior['score_rows_per_model'] or np.any(candidate_hits > ref['truth_count']):
        raise ValueError('Routed candidate population differs')
    scopes = {'selection': np.isin(ref['fold'], [0, 1]),
              'confirmation': np.isin(ref['fold'], [2, 3]), 'development': ref['fold'] < 4}
    baseline = {split: summarize(ref, count, hits, mask) for split, mask in scopes.items()}
    for split in scopes:
        for key, expected in prior['routed'][split]['overall'].items():
            actual = baseline[split]['overall'][key]
            if (actual is None) != (expected is None) or (actual is not None and abs(actual - expected) > 1e-10):
                raise ValueError('Scores do not reproduce the deployed country baseline')
    grids = {}
    for mode in MODES:
        path = out / f'grid_{mode}.json'
        if not path.exists():
            atomic_json(path, scan_grid(files, ref, mode, root=root))
        # Existing selection helper prefers the unchanged independent rule on ties.
        grids['baseline' if mode == 'independent' else mode] = json.loads(path.read_text())
    selected = select_routes(grids)
    atomic_json(out / 'selected_before_confirmation.json', selected)
    evaluated = {split: routed_metrics(grids, selected, ref, split) for split in scopes}
    delta = evaluated['confirmation']['overall']['macro_f05'] - baseline['confirmation']['overall']['macro_f05']
    changes = {c: evaluated['confirmation']['countries'][c]['macro_f05']
               - baseline['confirmation']['countries'][c]['macro_f05'] for c in COUNTRIES}
    result = {'scope': 'Complete routed OOF pairs, folds 0-3; no reserved/test labels',
              'rows': rows, 'baseline': baseline, 'selected_decisions': selected,
              'evaluated': evaluated, 'confirmation_macro_delta': delta,
              'confirmation_country_deltas': changes,
              'supports_new_submission': bool(delta > .0005 and min(changes.values()) >= -.002),
              'candidate_oracle': summarize(ref, candidate_hits, candidate_hits, ref['fold'] < 4),
              'grids': grids,
              'limits': ['Exploratory follow-up; folds 2/3 are cross-fitted confirmation, not untouched holdout.',
                         'Competition covers development references only; full test competition can differ.',
                         'No France labels; unknown-country policy remains unchanged.',
                         'No submission changed by this audit.']}
    atomic_json(out / 'report.json', result)
    atomic_json(root / 'reports/cloud_routed_decision_audit.json', result)
    return result
