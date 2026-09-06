#!/usr/bin/env python3
"""Independent full notebook/package reproduction audit (no source edits).

Reference mode reads only GPDR cells from the user's source notebook. Package mode
uses native importable Python modules and never reads executable notebook source.
Both export numeric values and figure pixels for exact comparison.
"""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from gpdr.paper.runner import new_state, output_directory
from audit_support import verify_source, export_and_validate

CELL_IDS = {
    "toy": [0, 1, 2, 5, 6, 12],
    "weather": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 29],
    "gini": [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 23, 24, 25, 26, 33],
}
EXTRA_DEFS = {"gini": {20: [18, 19]}}


class Audit:
    def __init__(self, output):
        self.output = Path(output)
        self.rows = {}
        self.stage = "start"
        self.figures = 0

    def value(self, key, value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            try:
                value = np.asarray(value)
            except (ValueError, TypeError):
                return
        if isinstance(value, dict):
            for k, v in value.items():
                self.value(f"{key}/{k}", v)
            return
        if isinstance(value, (bool, int, float, np.number)):
            value = np.asarray(value)
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "biufc":
            return
        a = np.ascontiguousarray(value)
        row = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        if value.size == 1:
            row["value"] = value.item()
        self.rows[key] = row

    def show(self, *args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            prefix = f"figure/{self.figures:02d}"
            for i, ax in enumerate(fig.axes):
                for j, line in enumerate(ax.lines):
                    self.value(f"{prefix}/axis{i}/line{j}", line.get_xydata())
                for j, coll in enumerate(ax.collections):
                    self.value(f"{prefix}/axis{i}/collection{j}/offsets", np.asarray(coll.get_offsets()))
                    self.value(f"{prefix}/axis{i}/collection{j}/values", coll.get_array())
                    for k, path in enumerate(coll.get_paths()):
                        self.value(f"{prefix}/axis{i}/collection{j}/path{k}", path.vertices)
                for j, im in enumerate(ax.images):
                    self.value(f"{prefix}/axis{i}/image{j}", np.asarray(im.get_array()))
                self.value(f"{prefix}/axis{i}/xlim", ax.get_xlim())
                self.value(f"{prefix}/axis{i}/ylim", ax.get_ylim())
            fig.savefig(self.output / f"figure_{self.figures:02d}.png", dpi=90)
            self.figures += 1
            plt.close(fig)

    def snapshot(self, stage, state):
        # Wall-clock readings and presentation counters are not numerical results.
        excluded = {"gpdr_logscore_start", "gpdr_metric_start", "weather_logscore_start", "weather_metric_start", "gini_logscore_start", "gini_metric_start", "current_lr"}
        for key, value in state.items():
            if key.startswith('_') or key in excluded or 'time_seconds' in key or key.endswith('_start'):
                continue
            self.value(f"{stage}/{key}", value)
        if stage == "train":
            model = state["model"]
            for name, param in model.named_parameters():
                self.value(f"train/model/{name}", param)
            for name in ("x", "y", "Vu", "g", "gy", "gyy", "K11uu", "K11uu_inv", "A", "B", "A2"):
                self.value(f"train/model/{name}", getattr(model, name))
        self.write()

    def write(self):
        (self.output / "audit.json").write_text(json.dumps(self.rows, indent=2, allow_nan=True))


class ReferenceIO(ast.NodeTransformer):
    """Redirect I/O only. Numerical statements remain the original notebook's."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def visit_With(self, node):
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "pickle" and n.func.attr == "dump" for n in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("read_csv", "read_file") and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            node.args[0] = ast.Constant(str(self.data_dir / node.args[0].value))
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == 'SHAPE_RESTORE_SHX' for t in node.targets):
            node.value = ast.Constant('NO')
        if any(isinstance(t, ast.Name) and t.id == "csv_path" for t in node.targets):
            node.value = ast.Constant(str(self.data_dir / "reg_df.csv"))
        return node


def execute_reference(source, index, ns, data_dir):
    src = ''.join(source["cells"][index]["source"])
    src = '\n'.join(l for l in src.splitlines() if not l.startswith('%'))
    node = ReferenceIO(data_dir).visit(ast.parse(src))
    exec(compile(ast.fix_missing_locations(node), f"reference_cell_{index}", "exec"), ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=CELL_IDS, required=True)
    parser.add_argument("--mode", choices=["package", "reference"], required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-before", default=None)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    path, nb, manifest = verify_source(args.project_root, args.example)
    executed = set(CELL_IDS[args.example])
    for extras in EXTRA_DEFS.get(args.example, {}).values():
        executed.update(extras)
    if executed != set(map(int, manifest['cells'])):
        raise ValueError('Executed source cells must exactly match the manifest')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    module, state = new_state(args.example, args.project_root)
    if len(module.STAGES) != len(CELL_IDS[args.example]):
        raise ValueError('Package stages no longer match the audited notebook cells')
    audit = Audit(args.output)
    plt.switch_backend("Agg")
    plt.show = lambda *a, **kw: audit.show(*a, **kw)
    version = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(), "mps_available": torch.backends.mps.is_available(), "mode": args.mode}
    version['workflow_sha256'] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    (args.output / "environment.json").write_text(json.dumps(version, indent=2))
    if args.mode == "reference":
        ns_module = types.ModuleType("gpdr_notebook_reference")
        sys.modules[ns_module.__name__] = ns_module
        state = ns_module.__dict__
        state.update(__name__="__main__", display=display)
        # dataclasses consult sys.modules while resolving annotations.
        sys.modules["__main__"] = ns_module
    with output_directory(args.output):
        for i, stage in zip(CELL_IDS[args.example], module.STAGES):
            if stage.__name__ == args.stop_before:
                break
            print(f"[{args.mode}/{args.example}] {stage.__name__}", flush=True)
            if args.mode == "package":
                stage(state)
            else:
                for extra in EXTRA_DEFS.get(args.example, {}).get(i, []):
                    execute_reference(nb, extra, state, path.parent)
                execute_reference(nb, i, state, path.parent)
            audit.snapshot(stage.__name__, state)
            if stage.__name__ == 'train' and args.mode == 'package':
                torch.save(state['model'].state_dict(), args.output / 'model_state.pt')
    audit.write()
    if args.stop_before is None:
        export_and_validate(args.example, args.output, audit.rows, mode=args.mode,
                            state=state, source_sha256=manifest['sha256'])
    print(f"AUDIT COMPLETE: {len(audit.rows)} numerical objects; {audit.figures} figures", flush=True)


if __name__ == "__main__":
    main()
