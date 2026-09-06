"""Validate/export completed audit snapshots produced before export checks existed.

Requires a successful completion line in the actual run log; never trains a model
or substitutes stored notebook outputs for computed audit results.
"""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_support import verify_source, export_and_validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', required=True, choices=['toy','gini','weather'])
    parser.add_argument('--mode', required=True, choices=['package','reference'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    _, _, manifest = verify_source(args.project_root, args.example)
    log = args.log.read_text()
    completed = re.search(r'AUDIT COMPLETE: (\d+) numerical objects; (\d+) figures\s*$', log)
    if completed is None:
        raise ValueError('Run log does not confirm successful completion')
    audit = json.loads((args.output/'audit.json').read_text())
    if len(audit) != int(completed[1]) or len(list(args.output.glob('figure_*.png'))) != int(completed[2]):
        raise ValueError('Saved artifacts do not match the completed run log')
    export_and_validate(args.example, args.output, audit, mode=args.mode, source_sha256=manifest['sha256'])
    print(f'{args.example}/{args.mode}: source and metric exports verified')


if __name__ == '__main__':
    main()
