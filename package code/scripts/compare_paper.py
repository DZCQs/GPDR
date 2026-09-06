#!/usr/bin/env python3
"""Compare independent notebook/package numeric and rendered-figure audits."""
import argparse
import hashlib
import json
from pathlib import Path
from audit_support import expected_metrics, read_exports


def compare(root, example):
    directory = root / example
    package = json.loads((directory / "package/audit.json").read_text())
    reference = json.loads((directory / "reference/audit.json").read_text())
    missing = sorted(package.keys() ^ reference.keys())
    different = sorted(k for k in package.keys() & reference.keys() if package[k] != reference[k])
    pngs = {}
    for path in sorted((directory / "reference").glob("figure_*.png")):
        other = directory / "package" / path.name
        pngs[path.name] = other.exists() and other.read_bytes() == path.read_bytes()
    extra_pngs = sorted(p.name for p in (directory/"package").glob("figure_*.png") if p.name not in pngs)
    exports = {}
    completion = {}
    for mode, audit in [('package', package), ('reference', reference)]:
        folder = directory / mode
        try:
            exports[mode] = read_exports(folder) == expected_metrics(example, audit)
            marker = json.loads((folder / 'completion.json').read_text())
            completion[mode] = (marker['complete'] and marker['exports_verified']
                                and marker['numeric_objects'] == len(audit)
                                and marker['figures'] == len(list(folder.glob('figure_*.png')))
                                and marker['audit_sha256'] == hashlib.sha256((folder/'audit.json').read_bytes()).hexdigest())
        except (FileNotFoundError, KeyError, ValueError):
            exports[mode] = False
            completion[mode] = False
    complete = all(completion.values()) and all(exports.values())
    return {"complete": complete, "numeric_objects": len(reference), "different": different,
            "missing": missing, "figures": pngs, "extra_figures": extra_pngs, "metric_exports": exports,
            "exact_match": complete and not missing and not different and not extra_pngs and bool(pngs) and all(pngs.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reproduction"))
    parser.add_argument("--example", choices=["toy","gini","weather"])
    args = parser.parse_args()
    names = [args.example] if args.example else ["toy","weather","gini"]
    results = {}
    for name in names:
        if not (args.root/name/"package/audit.json").exists() or not (args.root/name/"reference/audit.json").exists():
            results[name] = {"exact_match": False, "error": "Both complete audit runs are required"}
        else:
            results[name] = compare(args.root,name)
        print(name, json.dumps(results[name],indent=2))
    (args.root/"comparison.json").write_text(json.dumps(results,indent=2))
    if not all(result['exact_match'] for result in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
