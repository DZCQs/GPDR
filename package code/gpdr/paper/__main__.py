"""Command line interface: python -m gpdr.paper --example toy ..."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reproduce the current GPDR paper example, including every active GPDR figure and metric.")
    parser.add_argument("--example", choices=["toy", "gini", "weather"], required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .runner import run_example
    plt.switch_backend("Agg")
    figure_index = 0
    def close_figures(*unused_args, **kwargs):
        nonlocal figure_index
        for num in plt.get_fignums():
            plt.figure(num).savefig(f"figure_{figure_index:02d}.png", dpi=150)
            figure_index += 1
        plt.close("all")
    plt.show = close_figures
    result = run_example(args.example, args.project_root, args.output)
    print(result["metric_table"].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
