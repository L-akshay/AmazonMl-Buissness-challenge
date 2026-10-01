"""Reproduce the original leaderboard submission from the final review package."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

INFERENCE = ('prepare', 'retrieve-test', 'features-test', 'score', 'export', 'publish')
RETRAIN = ('prepare', 'retrieve-train', 'features-train', 'validate', 'final',
           'retrieve-test', 'features-test', 'score', 'export', 'publish')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def install_original_model(root):
    resources = root / 'resources/original_submission'
    manifest = json.loads((resources / 'manifest.json').read_text())
    model = resources / 'model/model.txt'
    metadata = json.loads((model.parent / 'complete.json').read_text())
    if sha256(model) != manifest['model_sha256'] or metadata['sha256'] != manifest['model_sha256']:
        raise ValueError('Bundled original model checksum differs')
    if metadata['kind'] != 'gbdt' or manifest['policy'] != {'threshold': .625, 'relative': 0}:
        raise ValueError('This package must reproduce the original mixed-country GBDT')
    final = {'model': model.relative_to(root).as_posix(), 'kind': 'gbdt',
             'score_name': 'original_submission', 'policy': manifest['policy'],
             'variants': [{'name': 'oof_best', **manifest['policy']}]}
    destination = root / 'cache/cloud/final.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and json.loads(destination.read_text()) != final:
        raise ValueError('Use a fresh workspace: a different final model manifest exists')
    destination.write_text(json.dumps(final, indent=2))


def publish(root, exact=True):
    manifest = json.loads((root / 'resources/original_submission/manifest.json').read_text())
    for name, relative in [('matching_results.tsv', 'oof_best/matching_results.tsv'),
                           ('candidate_pairs.tsv', 'candidate_pairs.tsv')]:
        source = root / 'output/cloud' / relative
        actual = sha256(source)
        if exact and actual != manifest['outputs'][name]['sha256']:
            raise ValueError(f'{name} differs from submitted output; retain logs and investigate runtime/cache provenance')
        destination = root / 'output' / name
        shutil.copyfile(source, destination)
        print(json.dumps({'file': str(destination), 'sha256': actual,
                          'matches_submitted_file': actual == manifest['outputs'][name]['sha256']}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inference', 'retrain'), default='inference')
    parser.add_argument('--stage', choices=('all',) + RETRAIN, default='all')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not (os.environ.get('ER_REMOTE_COMPUTE') == '1' or os.environ.get('KAGGLE_KERNEL_RUN_TYPE')):
        raise RuntimeError('Run full reproduction on remote compute; set ER_REMOTE_COMPUTE=1 there')
    steps = INFERENCE if args.mode == 'inference' else RETRAIN
    if args.stage != 'all' and args.stage not in steps:
        raise ValueError('Requested stage does not belong to this reproduction mode')
    for stage in steps if args.stage == 'all' else (args.stage,):
        if stage == 'publish':
            publish(root, exact=args.mode == 'inference')
        else:
            if stage == 'score' and args.mode == 'inference':
                install_original_model(root)
            subprocess.run([sys.executable, '-u', '-m', 'src.cloud_pipeline', '--root', str(root),
                            '--config', str(root / 'configs/submission.json'), '--stage', stage],
                           cwd=root, check=True)


if __name__ == '__main__':
    main()
