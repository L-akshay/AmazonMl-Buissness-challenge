"""Fit and export frozen country routes using complete cached candidate populations."""

import json
from pathlib import Path
import numpy as np
from src.cloud_country_audit import COUNTRIES, mask_digest
from src.cloud_features import references
from src.cloud_model import train_one, score_model
from src.cloud_store import atomic_json, claim_config, digest, parquet_files, read_parquet, write_parquet, check_headroom


def load_policy(root):
    path = root / 'reports/cloud_country_route_audit.json'
    report = json.loads(path.read_text())
    if (report['supports_new_submission'] is not True
            or report['confirmation_macro_delta'] <= .0005
            or min(report['confirmation_country_deltas'].values()) < -.002):
        raise ValueError('Country routing did not pass its confirmation gate')
    routes = report['selected_routes']
    fallback = report['unknown_country_fallback']
    if set(routes) != set(COUNTRIES) or fallback['source'] != 'baseline':
        raise ValueError('Unexpected country-routing policy')
    for rule in [fallback, *routes.values()]:
        if rule['source'] not in ('baseline', 'specialist') or not 0 <= rule['threshold'] <= 1 or rule['relative'] != 0:
            raise ValueError('Unsupported frozen routing rule')
    return path, routes, fallback


def reference_scope(root, split, country=None):
    ref = read_parquet(references(root, split), 'ri,country_norm')
    if not np.array_equal(ref['ri'], np.arange(len(ref['ri']))):
        raise ValueError('Reference indices are not contiguous')
    return ref, (np.ones(len(ref['ri']), dtype=bool) if country is None else ref['country_norm'] == country)


def fit_country(root, country, config, fingerprint):
    policy_path, routes, _ = load_policy(root)
    if country not in COUNTRIES or routes[country]['source'] != 'specialist':
        raise ValueError('No specialist was selected for this country')
    _, allowed = reference_scope(root, 'train', country)
    # Final production fit includes every fold of this country. No new metric is
    # computed from these fitted labels, and no candidate negatives are sampled.
    model = train_one(root, f'country_full/{country}', allowed, 'gbdt', config, fingerprint)
    atomic_json(model.parent / 'routing.json', {'audit_sha256': digest(policy_path), 'country': country,
                'entities': int(allowed.sum()), 'scope': 'All training folds and all candidate pairs in this country',
                'policy': routes[country]})
    return model


def score_country(root, country, config):
    policy_path, routes, _ = load_policy(root)
    model = root / f'cache/cloud/models/country_full/{country}/model.txt'
    proof = json.loads((model.parent / 'routing.json').read_text())
    if proof['audit_sha256'] != digest(policy_path) or proof['country'] != country or routes[country]['source'] != 'specialist':
        raise ValueError('Final country model belongs to another routing audit')
    _, allowed = reference_scope(root, 'test', country)
    return score_model(root, model, 'test', allowed, f'country_test/{country}', config)


def checked_batch(batch, ref):
    ri = batch['ri']
    if (not np.issubdtype(ri.dtype, np.integer) or np.any(ri < 0) or np.any(ri >= len(ref['ri']))
            or any(len(batch[k]) != len(ri) for k in ('tid', 'y', 'p'))):
        raise ValueError('Invalid reference index or score columns')
    if not np.isfinite(batch['p']).all() or np.any((batch['p'] < 0) | (batch['p'] > 1)):
        raise ValueError('Invalid routed probability')
    order = np.lexsort((batch['tid'], ri))
    if np.any((ri[order][1:] == ri[order][:-1]) & (batch['tid'][order][1:] == batch['tid'][order][:-1])):
        raise ValueError('Duplicate candidate pair')


def merge_batch(baseline, specialists, ref):
    checked_batch(baseline, ref)
    result = {**baseline, 'p': baseline['p'].copy()}
    for country, batch in specialists.items():
        checked_batch(batch, ref)
        selected = ref['country_norm'][baseline['ri']] == country
        if any(not np.array_equal(baseline[k][selected], batch[k]) for k in ('ri', 'tid', 'y')):
            raise ValueError('Country scores do not cover exactly the baseline candidate pairs')
        result['p'][selected] = batch['p']
    return result


def assemble_country_scores(root, config):
    policy_path, routes, fallback = load_policy(root)
    ref, all_test = reference_scope(root, 'test')
    train, all_train = reference_scope(root, 'train')
    sources = {'baseline': ('test_gbdt', 'full_gbdt', all_test, all_train)}
    for country, rule in routes.items():
        if rule['source'] == 'specialist':
            sources[country] = (f'country_test/{country}', f'country_full/{country}',
                                ref['country_norm'] == country, train['country_norm'] == country)
    inputs, identities = {}, {}
    for source, (score_name, model_name, scoring, fitting) in sources.items():
        folder = root / 'cache/cloud/scores' / score_name
        model = root / 'cache/cloud/models' / model_name
        score_config = json.loads((folder / 'config.json').read_text())
        complete = json.loads((folder / 'complete.json').read_text())
        model_config = json.loads((model / 'config.json').read_text())
        model_complete = json.loads((model / 'complete.json').read_text())
        if (score_config['split'] != 'test' or score_config['scope'] != mask_digest(scoring)
                or model_config['allowed_sha256'] != mask_digest(fitting)):
            raise ValueError('Incorrect final fitting or test scoring scope')
        if not (score_config['model_sha256'] == complete['model_sha256'] == model_complete['sha256'] == digest(model / 'model.txt')):
            raise ValueError('Final model and cached score identities disagree')
        if source != 'baseline':
            proof = json.loads((model / 'routing.json').read_text())
            if proof['audit_sha256'] != digest(policy_path) or proof['country'] != source:
                raise ValueError('Specialist belongs to another routing audit')
        inputs[source] = parquet_files(folder)
        names = [p.name for p in inputs[source]]
        if not names or names != sorted(set(names)):
            raise ValueError('Invalid score shard manifest')
        identities[source] = {'score_config': digest(folder / 'config.json'),
                              'score_manifest': digest(folder / 'complete.json'),
                              'model_config': digest(model / 'config.json'), 'model_sha256': model_complete['sha256']}
    names = [p.name for p in inputs['baseline']]
    if any([p.name for p in paths] != names for paths in inputs.values()):
        raise ValueError('Specialist shard manifest differs from baseline')
    features = root / 'cache/cloud/features_test/complete.json'
    expected = json.loads(features.read_text())
    if expected['files'] != names:
        raise ValueError('Score shards differ from the full test feature manifest')
    out = root / 'cache/cloud/scores/country_routed'
    claim_config(out, {'code_sha256': digest(Path(__file__)), 'audit_sha256': digest(policy_path),
                      'references_sha256': digest(references(root, 'test')), 'sources': identities,
                      'features_manifest_sha256': digest(features)})
    rows = 0
    for index, name in enumerate(names):
        check_headroom(root)
        path = out / name
        proof_path = out / (name + '.json')
        if path.exists() and proof_path.exists():
            proof = json.loads(proof_path.read_text())
            if digest(path) != proof['sha256']:
                raise ValueError('Completed routed score shard changed')
            rows += proof['rows']
            continue
        baseline = read_parquet(inputs['baseline'][index], 'ri,tid,y,p')
        specialists = {c: read_parquet(paths[index], 'ri,tid,y,p') for c, paths in inputs.items() if c != 'baseline'}
        combined = merge_batch(baseline, specialists, ref)
        write_parquet(path, combined)
        atomic_json(proof_path, {'rows': len(combined['ri']), 'sha256': digest(path)})
        rows += len(combined['ri'])
        if (index + 1) % 50 == 0:
            print(f'Routed {index + 1}/{len(names)} complete test score shards', flush=True)
    if rows != expected['rows']:
        raise ValueError('Routed scores changed the complete candidate count')
    atomic_json(out / 'complete.json', {'files': names, 'rows': rows, 'sources': identities})
    policy = {'name': 'country_routed', **fallback,
              'country_routes': {c: {k: rule[k] for k in ('source', 'threshold', 'relative')} for c, rule in routes.items()}}
    final = {'kind': 'country_routed_gbdt', 'score_name': 'country_routed', 'policy': policy, 'variants': [policy],
             'audit_sha256': digest(policy_path), 'models': identities, 'candidate_pairs': rows,
             'unknown_country_behavior': 'Unchanged mixed-model scores and policy; no unseen-country labels used'}
    # Refuse attaching an old final/export checkpoint accidentally. Original
    # submissions stay immutable in their own remote job and local delivery path.
    destination = root / 'cache/cloud/final.json'
    if destination.exists() and json.loads(destination.read_text()) != final:
        raise ValueError('Existing final manifest differs; use a separate routed-export job')
    atomic_json(destination, final)
    atomic_json(root / 'reports/cloud_country_final.json', final)
    return final
