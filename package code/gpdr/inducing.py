"""Inducing-point construction, independent of the data example and base model."""
import torch


def tensor_grid(*axes):
    """Cartesian product of explicit covariate/PIT axes, last coordinate is z."""
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=1)


def sample_inducing(V, count, seed=2):
    """Select training (x, PIT) rows without changing the global RNG."""
    generator = torch.Generator(device=V.device).manual_seed(seed)
    indices = torch.randperm(V.shape[0], generator=generator, device=V.device)
    return V[indices[:count]].clone()


def kmeans_inducing(V, count):
    """Cross covariate k-means centers with a uniform PIT grid.

    Count is a target: rounding the number per dimension changes the actual
    count. kmeans-pytorch uses the current RNG state; seed it before this call.
    """
    from kmeans_pytorch import kmeans
    x = V[:, :-1]
    d = x.shape[1]
    per_axis = int(round(count ** (1.0 / (d + 1))))
    _, centers = kmeans(X=x, num_clusters=per_axis**d,
                        distance="euclidean", device=x.device)
    centers = torch.as_tensor(centers, device=x.device, dtype=x.dtype)
    z = torch.linspace(1e-5, 1 - 1e-5, per_axis,
                       device=x.device, dtype=x.dtype)
    return torch.cat([
        centers[:, None, :].expand(centers.shape[0], per_axis, d),
        z[None, :, None].expand(centers.shape[0], per_axis, 1),
    ], dim=2).reshape(-1, d + 1)
