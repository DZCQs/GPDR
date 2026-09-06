import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_support
from compare_paper import compare


def small_audit():
    values = dict(loglik_base=3.0, loglik_updated=6.0, rmse_base=0.2,
                  rmse_updated=0.1, coverage_base=0.9, coverage_updated=0.95,
                  avg_length_base=0.4, avg_length_updated=0.3)
    audit = {'diagnostics/' + k: {'value': v} for k, v in values.items()}
    audit['diagnostics/y_test'] = {'shape': [3]}
    return audit


def completed_pair(tmp_path):
    audit = small_audit()
    for mode in ('package', 'reference'):
        folder = tmp_path / 'gini' / mode
        folder.mkdir(parents=True)
        (folder / 'audit.json').write_text(json.dumps(audit))
        (folder / 'figure_00.png').write_bytes(b'fixture-pixels')
        audit_support.export_and_validate('gini', folder, audit, mode=mode, source_sha256='fixture')
    return tmp_path / 'gini'


def test_audit_requires_verified_exports_and_completion(tmp_path):
    folder = completed_pair(tmp_path)
    assert compare(tmp_path, 'gini')['exact_match']
    (folder / 'package' / 'metrics.json').unlink()
    assert not compare(tmp_path, 'gini')['exact_match']


def test_audit_rejects_changed_results(tmp_path):
    folder = completed_pair(tmp_path)
    path = folder / 'package' / 'audit.json'
    content = json.loads(path.read_text())
    content['diagnostics/loglik_updated']['value'] = 7.0
    path.write_text(json.dumps(content))
    assert not compare(tmp_path, 'gini')['exact_match']


def test_source_hash_checked_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_support, '__file__', str(tmp_path/'scripts/audit_support.py'))
    manifest_path = tmp_path/'gpdr/paper/sources.json'
    manifest_path.parent.mkdir(parents=True)
    source = 'print(1)'
    notebook = tmp_path/'source.ipynb'
    notebook.write_text(json.dumps({'cells': [{'source': [source]}]}))
    manifest_path.write_text(json.dumps({'toy': {
        'source': 'source.ipynb',
        'sha256': hashlib.sha256(notebook.read_bytes()).hexdigest(),
        'cells': {'0': hashlib.sha256(source.encode()).hexdigest()}}}))
    audit_support.verify_source(tmp_path, 'toy')
    notebook.write_text(notebook.read_text() + '\n')
    with pytest.raises(ValueError, match='Source notebook changed'):
        audit_support.verify_source(tmp_path, 'toy')
