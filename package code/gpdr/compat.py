import pickle
from pathlib import Path

import torch

from .kernels import KernelParams, RBFSEARD, chol_inv
from .models import BetaGPMixtureDiag, LogisticGPMixtureDiag


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            mapping = {
                "KernelParams": KernelParams,
                "RBF_SE_ARD": RBFSEARD,
                "RBFSEARD": RBFSEARD,
                "LogisticGPMixtureDiag": LogisticGPMixtureDiag,
                "BetaGPMixtureDiag": BetaGPMixtureDiag,
            }
            if name in mapping:
                return mapping[name]
        return super().find_class(module, name)


def load_legacy_model(path):
    """Load a notebook-created pickle whose classes were saved from `__main__`."""
    path = Path(path)
    original_torch_load = torch.load

    def _cpu_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = _cpu_torch_load
        with path.open("rb") as f:
            model = _LegacyUnpickler(f).load()
    finally:
        torch.load = original_torch_load
    return model


__all__ = ["load_legacy_model", "KernelParams", "RBFSEARD", "chol_inv"]
