"""Run the paper workflows with explicit input and output locations."""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
from .results import metric_table

FOLDERS = {"toy": "toyexample_thickertail", "weather": "weatherdata_thickertail", "gini": "giniindex_thickertail"}


def new_state(example, project_root):
    module = importlib.import_module(f"gpdr.paper.{example}")
    state = {}
    state["data_dir"] = Path(project_root).expanduser().resolve() / FOLDERS[example]
    return module, state


@contextmanager
def output_directory(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(old)


def prepare_example(example, project_root, output_dir):
    """Prepare the exact notebook data/base fit, stopping before GPDR training.

    Returns the state mapping consumed by the example's train/evaluation functions.
    Global RNGs are seeded as in the notebook. Run examples sequentially in a
    process, or in separate processes, since PyTorch and Matplotlib use globals.
    """
    module, state = new_state(example, project_root)
    with output_directory(output_dir):
        for stage in module.STAGES:
            if stage.__name__ == "train":
                break
            stage(state)
    return state


def run_example(example, project_root, output_dir, *, on_stage=None):
    """Train and reproduce all active GPDR figures and metrics, in notebook order.

    Input notebooks, data, and existing model files are never written. Cross-
    validation and non-GPDR benchmark cells are not executed. The selected paper
    hyperparameters are already included in these example modules.
    """
    module, state = new_state(example, project_root)
    with output_directory(output_dir):
        for stage in module.STAGES:
            print(f"[{example}] {stage.__name__}", flush=True)
            stage(state)
            if stage.__name__ == "train":
                import torch
                torch.save(state["model"].state_dict(), "model_state.pt")
            if on_stage is not None:
                on_stage(stage.__name__, state)
        state["metric_table"] = metric_table(example, state)
        state["metric_table"].to_csv("metrics.csv", index=False)
        Path("metrics.json").write_text(json.dumps(state["metric_table"].to_dict(orient="records"), indent=2, allow_nan=False))
    return state
