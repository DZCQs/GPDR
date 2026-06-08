#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpdr.examples import load_gini_saved, load_toy_saved, load_weather_saved


LOADERS = {
    "toy": load_toy_saved,
    "gini": load_gini_saved,
    "weather": load_weather_saved,
}


def main():
    parser = argparse.ArgumentParser(description="Run GPDR example checks.")
    parser.add_argument("--project-root", default="/Users/zichuanchen/Desktop/gp_density_regression")
    parser.add_argument("--example", choices=sorted(LOADERS), required=True)
    parser.add_argument("--mode", choices=["saved"], default="saved")
    args = parser.parse_args()

    bundle = LOADERS[args.example](Path(args.project_root))
    F, info = bundle.model.forward_objective()
    bits = [f"example={args.example}", f"objective={float(F.detach()):.12g}"]
    bits.extend(f"{k}={float(v.detach()):.12g}" for k, v in info.items() if k != "w")
    print(" | ".join(bits))


if __name__ == "__main__":
    main()
