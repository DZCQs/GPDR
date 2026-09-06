"""Independent provenance and exported-table checks for full reproduction runs."""
import csv
import hashlib
import json
from pathlib import Path


def verify_source(project_root, example):
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'gpdr/paper/sources.json').read_text())[example]
    path = Path(project_root) / manifest['source']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise ValueError(f'Source notebook changed since this package was ported: {path}')
    nb = json.loads(raw)
    for i, expected in manifest['cells'].items():
        actual = hashlib.sha256(''.join(nb['cells'][int(i)]['source']).encode()).hexdigest()
        if actual != expected:
            raise ValueError(f'Source cell {i} does not match its provenance hash')
    return path, nb, manifest


def expected_metrics(example, audit):
    """Read original notebook scalars, independently of the package table mapper."""
    def scalar(name):
        return audit['diagnostics/' + name]['value']

    n = audit['diagnostics/y_test']['shape'][0]
    names = {
        'toy': [
            ['base_normal_log_density_vals', 'rmse_normal', 'coverage_normal', 'length_normal'],
            ['estimated_log_density_vals', 'rmse_updated', 'coverage_updated', 'length_updated']],
        'weather': [
            ['base_tilde_log_density_vals', 'rmse_g_tilde_orig', 'coverage_g_tilde', 'avg_length_g_tilde'],
            ['estimated_log_density_vals', 'rmse_updated_orig', 'coverage_updated', 'avg_length_updated']],
        'gini': [
            ['loglik_base', 'rmse_base', 'coverage_base', 'avg_length_base'],
            ['loglik_updated', 'rmse_updated', 'coverage_updated', 'avg_length_updated']],
    }[example]
    rows = []
    for method, fields in zip(['parametric model', 'GPDR'], names):
        score, error, coverage, width = map(scalar, fields)
        rows.append(dict(method=method, log_score=score, mean_log_score=score/n,
                         RMSE=error, coverage_95=coverage, average_width_95=width))
    return rows


def read_exports(directory):
    directory = Path(directory)
    rows = json.loads((directory / 'metrics.json').read_text())
    with (directory / 'metrics.csv').open(newline='') as stream:
        csv_rows = [{k: v if k == 'method' else float(v) for k, v in row.items()}
                    for row in csv.DictReader(stream)]
    if rows != csv_rows:
        raise ValueError('CSV and JSON metrics disagree')
    return rows


def export_and_validate(example, directory, audit, *, mode, state=None, source_sha256):
    expected = expected_metrics(example, audit)
    if mode == 'package':
        from gpdr.paper.results import metric_table
        if state is None:
            # Finish exporting a completed audit without rerunning its training.
            state = {k.removeprefix('diagnostics/'): v['value'] for k, v in audit.items()
                     if k.startswith('diagnostics/') and 'value' in v}
            n = audit['diagnostics/y_test']['shape'][0]
            state['x_test'] = range(n)
            state['df_test'] = range(n)
        actual = metric_table(example, state).to_dict(orient='records')
        if actual != expected:
            raise ValueError('Package metric mapping differs from notebook scalar results')
    else:
        actual = expected
    directory = Path(directory)
    (directory / 'metrics.json').write_text(json.dumps(actual, indent=2, allow_nan=False))
    with (directory / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(actual[0]))
        writer.writeheader()
        writer.writerows(actual)
    assert read_exports(directory) == expected
    marker = dict(complete=True, source_sha256=source_sha256,
                  numeric_objects=len(audit), figures=len(list(directory.glob('figure_*.png'))),
                  audit_sha256=hashlib.sha256((directory/'audit.json').read_bytes()).hexdigest(),
                  exports_verified=True)
    (directory / 'completion.json').write_text(json.dumps(marker, indent=2))
