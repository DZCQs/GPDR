"""Compare public-core runs with the independent, full notebook reference runs.

Unlike a verbatim cell port, a reusable optimizer does not leak its loop locals
into the caller. Exclude only those declared implementation-local scratch values
and unused GPyTorch parameters. All reference data, trained variational values,
training traces, prediction/evaluation arrays, and figures remain required.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_support import expected_metrics, read_exports, verify_source


OPTIMIZER_LOCALS = {
    'F', 'info', 'loss', 'step', 'iteration', 'num_steps', 'num_iterations',
    'number_of_steps', 'averaging_window', 'start_averaging', 'averaged_params',
    'num_averaged', 'initial_lr', 'max_grad_norm', 'print_interval', 'step_verbose',
    'w', 'mu_norm', 's_mean', 'param',
}


def required(key):
    if key.startswith('figure/') or '/trace/' in key:
        return True
    if key.startswith('train/model/'):
        return not key.startswith('train/model/kern.base.')
    return key.split('/')[1] not in OPTIMIZER_LOCALS


def compare(root, name, source_sha256):
    ref_dir = root / name / 'reference'
    pkg_dir = root / 'shared_core' / name / 'package'
    ref = json.loads((ref_dir/'audit.json').read_text())
    pkg = json.loads((pkg_dir/'audit.json').read_text())
    keys = sorted(k for k in ref if required(k))
    missing = [k for k in keys if k not in pkg]
    different = [k for k in keys if k in pkg and pkg[k] != ref[k]]
    excluded = sorted(k for k in ref if not required(k))
    figure_ids = sorted({k.split('/')[1] for k in keys if k.startswith('figure/')})
    figure_names = [f'figure_{i}.png' for i in figure_ids]
    figures = {name: (ref_dir/name).is_file() and (pkg_dir/name).is_file()
                       and (ref_dir/name).read_bytes() == (pkg_dir/name).read_bytes()
               for name in figure_names}
    extra_figures = {mode: sorted(p.name for p in folder.glob('figure_*.png') if p.name not in figures)
                     for mode, folder in [('reference',ref_dir),('package',pkg_dir)]}
    exports = read_exports(ref_dir) == read_exports(pkg_dir) == expected_metrics(name, ref)
    completion = {}
    for mode, folder in [('reference',ref_dir),('package',pkg_dir)]:
        marker = json.loads((folder/'completion.json').read_text())
        completion[mode] = (marker['complete'] and marker['exports_verified']
                            and marker['source_sha256'] == source_sha256
                            and marker['figures'] == len(figures)
                            and figure_ids == [f'{i:02d}' for i in range(marker['figures'])]
                            and marker['numeric_objects'] == len(ref if mode == 'reference' else pkg)
                            and marker['audit_sha256'] == hashlib.sha256((folder/'audit.json').read_bytes()).hexdigest())
    return dict(required_records=len(keys), different=different, missing=missing,
                excluded_optimizer_locals=excluded, figures=figures, extra_figures=extra_figures,
                metrics_exact=exports, completion=completion,
                exact_match=not missing and not different and bool(figures) and all(figures.values())
                            and not any(extra_figures.values()) and exports and all(completion.values()))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reproduction'))
    parser.add_argument('--example',choices=['toy','weather','gini'])
    parser.add_argument('--project-root',type=Path,default=Path('..'))
    args=parser.parse_args()
    names=[args.example] if args.example else ['toy','weather','gini']
    results={}
    for name in names:
        _, _, manifest = verify_source(args.project_root,name)
        results[name]=compare(args.root,name,manifest['sha256'])
        print(name,json.dumps({k:v for k,v in results[name].items() if k!='excluded_optimizer_locals'},indent=2))
    target=args.root/'shared_core'/('comparison_'+args.example+'.json' if args.example else 'comparison.json')
    target.write_text(json.dumps(results,indent=2))
    if not all(r['exact_match'] for r in results.values()):
        raise SystemExit(1)
