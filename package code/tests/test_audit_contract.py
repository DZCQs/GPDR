"""The reproduction checker must fail closed on missing figures/provenance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def checker():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('compare_shared_core',scripts/'compare_shared_core.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_full_audit_contract(tmp_path,monkeypatch):
    module = checker()
    monkeypatch.setattr(module,'read_exports',lambda folder: [])
    monkeypatch.setattr(module,'expected_metrics',lambda name,audit: [])
    dirs = [tmp_path/'gini'/'reference',tmp_path/'shared_core'/'gini'/'package']
    for folder in dirs:
        folder.mkdir(parents=True)
        audit = {'figure/00/axis0/line0': {'shape':[2,2],'sha256':'line'}}
        raw = json.dumps(audit).encode()
        (folder/'audit.json').write_bytes(raw)
        (folder/'figure_00.png').write_bytes(b'rendered figure')
        marker = dict(complete=True,exports_verified=True,source_sha256='source',
                      numeric_objects=1,figures=1,audit_sha256=hashlib.sha256(raw).hexdigest())
        (folder/'completion.json').write_text(json.dumps(marker))
    assert module.compare(tmp_path,'gini','source')['exact_match']
    assert not module.compare(tmp_path,'gini','other source')['exact_match']
    for folder in dirs:
        (folder/'figure_00.png').unlink()
    result = module.compare(tmp_path,'gini','source')
    assert not result['exact_match'] and not result['figures']['figure_00.png']
