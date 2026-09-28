"""Apply confirmed country decision rules to every cached test candidate."""

import json
from pathlib import Path
import numpy as np
from src.cloud_decisions import competition, probabilities, MODES
from src.cloud_features import references
from src.cloud_store import (atomic_json, check_headroom, claim_config, digest,
                             parquet_files, read_parquet, write_parquet)
from src.predict import decode_secondary


def transform(batch, countries, rules):
    ri = batch['ri']
    if not np.issubdtype(ri.dtype, np.integer) or np.any(ri < 0) or np.any(ri >= len(countries)):
        raise ValueError('Invalid test reference indices')
    winner, second = competition(batch)
    enriched = {**batch, 'winner': winner, 'second': second}
    result = {**batch, 'p': batch['p'].copy()}
    for country, rule in rules.items():
        mode = rule['source']
        if mode == 'baseline':
            mode = 'independent'
        if mode not in MODES or rule['relative'] != 0 or not 0 < rule['threshold'] <= 1:
            raise ValueError('Invalid frozen decision policy')
        selected = countries[ri] == country
        result['p'][selected] = probabilities(enriched, mode)[selected]
    # Unlabeled countries keep exactly their original probabilities. All pairs,
    # including rejected competitors with zero score, remain in the output.
    return result


def apply_decisions(root, config):
    evidence = root / 'reports/cloud_routed_decision_audit.json'
    audit = json.loads(evidence.read_text())
    rules = audit['selected_decisions']
    if (not audit['supports_new_submission'] or audit['confirmation_macro_delta'] <= .0005
            or min(audit['confirmation_country_deltas'].values()) < -.002
            or set(rules) != {'india', 'us'}):
        raise ValueError('Decision policy has not passed confirmation against V4')
    prior_path = root / 'cache/cloud/final.json'
    prior = json.loads(prior_path.read_text())
    # Resuming a completed transform reads its preserved original manifest.
    out = root / 'cache/cloud/scores/country_decisions'
    original = out / 'original_final.json'
    if original.exists():
        prior = json.loads(original.read_text())
    if prior['kind'] != 'country_routed_gbdt' or prior['score_name'] != 'country_routed':
        raise ValueError('Expected unchanged V4 country score manifest')
    audit_identity = json.loads((root / 'cache/cloud/routed_decision_audit/config.json').read_text())
    if audit_identity['country_report'] != prior['audit_sha256']:
        raise ValueError('Decision audit and V4 scores use different country routes')
    source = root / 'cache/cloud/scores/country_routed'
    feature_manifest = root / 'cache/cloud/features_test/complete.json'
    expected = json.loads(feature_manifest.read_text())
    files = parquet_files(source)
    if [p.name for p in files] != expected['files']:
        raise ValueError('Test score shard manifest differs from full candidates')
    refpath = references(root, 'test')
    countries = read_parquet(refpath, 'country_norm')['country_norm']
    claim_config(out, {'code': digest(Path(__file__)),
                      'decision_code': digest(Path(__file__).with_name('cloud_decisions.py')),
                      'audit': digest(evidence), 'prior_final': prior,
                      'source': digest(source / 'complete.json'),
                      'features': digest(feature_manifest), 'references': digest(refpath)})
    if not original.exists():
        atomic_json(original, prior)
    rows = 0; previous = None; names = []
    for index, path in enumerate(files):
        check_headroom(root)
        source_proof = json.loads((source / (path.name + '.json')).read_text())
        if digest(path) != source_proof['sha256']:
            raise ValueError('Original V4 score shard changed')
        target = out / path.name; marker = target.with_suffix('.json')
        if target.exists() and marker.exists():
            info = json.loads(marker.read_text())
            if digest(target) != info['sha256']:
                raise ValueError('Transformed checkpoint changed')
        else:
            batch = read_parquet(path, 'ri,tid,y,p')
            transformed = transform(batch, countries, rules)
            boundaries = [decode_secondary(int(t)) for t in np.unique(batch['tid'])]
            info = {'rows': len(batch['ri']), 'first': min(boundaries) if boundaries else None,
                    'last': max(boundaries) if boundaries else None}
            write_parquet(target, transformed)
            info['sha256'] = digest(target); atomic_json(marker, info)
        if info['first'] is not None:
            if previous is not None and info['first'] <= previous:
                raise ValueError('Secondary competitors cross score shard boundaries')
            previous = info['last']
        rows += info['rows']; names.append(path.name)
        if index % 50 == 0:
            print(f'Applied frozen decisions: {index + 1}/{len(files)} shards', flush=True)
    if rows != expected['rows'] or rows != prior['candidate_pairs']:
        raise ValueError('Transformed scores do not preserve all test candidates')
    atomic_json(out / 'complete.json', {'files': names, 'rows': rows})
    policy = {**prior['policy'], 'name': 'country_decisions',
              'country_routes': {c: {k: rule[k] for k in ('source', 'threshold', 'relative')}
                                 for c, rule in rules.items()}}
    final = {**prior, 'kind': 'country_decision_gbdt', 'score_name': 'country_decisions',
             'policy': policy, 'variants': [policy], 'decision_audit_sha256': digest(evidence)}
    if prior_path.is_symlink():
        prior_path.unlink()
    atomic_json(prior_path, final)
    atomic_json(root / 'reports/cloud_decision_final.json', final)
    return final
