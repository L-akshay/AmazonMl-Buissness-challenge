"""Prepare country-specialist production jobs without changing the live controller."""

import argparse
import json
from pathlib import Path
from build_kaggle_job import build
from manage_kaggle import validate_plan


def create(root, source, destination):
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Choose a new proposed plan path; never overwrite the active plan')
    plan = json.loads(source.read_text())
    by_id = {j['id']: j for j in plan['jobs']}
    prefix = 'lakshaytechai/'
    old = lambda slug: prefix + 'amazon-er-v3-' + slug
    audit = prefix + 'amazon-er-v4-country-routing'
    overrides = json.loads((root / 'configs/country_final.json').read_text())
    train_inputs = list(by_id[old('full-gbdt')]['depends'])
    test_inputs = [name for name in by_id[old('test-score-gbdt')]['depends'] if name != old('full-gbdt')]
    fits, scores, added = [], [], []

    def add(slug, stages, inputs, priority):
        identifier = prefix + slug
        if identifier in by_id:
            raise ValueError('Production job already appears in the controller plan')
        folder = root / 'cache/kaggle_jobs' / slug
        if folder.exists():
            raise ValueError('Job folder already exists; inspect its committed revision before reusing')
        build(root, folder, slug, stages, inputs, overrides)
        job = {'id': identifier, 'folder': str(folder), 'depends': inputs, 'priority': priority, 'timeout': 36000}
        plan['jobs'].append(job); by_id[identifier] = job; added.append(identifier)
        return identifier

    # Start the larger fit first; the single controller allocates free slots.
    for country in ('us', 'india'):
        fit = add(f'amazon-er-v4-fit-{country}', [f'country-fit-{country}'], [*train_inputs, audit], -10)
        fits.append(fit)
        scores.append(add(f'amazon-er-v4-score-{country}', [f'country-score-{country}'], [*test_inputs, fit, audit], -9))
    assembled = add('amazon-er-v4-routed-scores', ['country-assemble'],
                    [old('prepare'), old('test-assembled'), old('full-gbdt'), old('test-score-gbdt'),
                     *fits, *scores, audit], -8)
    export = add('amazon-er-v4-routed-export', ['export'], [old('prepare'), old('test-assembled'), assembled], -8)
    by_id[export]['download'] = str(root / 'output/kaggle_delivery/country_routed')
    validate_plan(plan)
    destination.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'proposed_plan': str(destination), 'new_jobs': added,
                      'activation': 'Not activated. Preserve the single controller and its journal when reloading.'}, indent=2))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    create(Path(__file__).resolve().parents[1], args.source, args.output)
