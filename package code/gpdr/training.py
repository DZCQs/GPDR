"""Shared optimizer for every GPDR model and dataset."""
import torch


def fit_adam(model, *, steps, lr=0.5, average_last=0, verbose_every=0,
             averaging="sum", trace_at="after"):
    """Return (model, trace), with explicit iterate averaging and trace timing.

    ``before`` records parameters before each update; ``after`` records them
    after it. Objective values always describe the pre-update model. Averaging
    is applied only after optimization finishes.
    """
    if averaging not in {"sum", "running"} or trace_at not in {"before", "after"}:
        raise ValueError("invalid averaging or trace_at")
    if steps <= 0 or not 0 <= average_last <= steps:
        raise ValueError("require steps > 0 and 0 <= average_last <= steps")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace = {key: [] for key in ("F", "KL", "beta_EH", "w", "mu_norm", "s_mean",
                                 "s_component_mean", "mu_samples", "s_samples")}
    averaged, n_avg = None, 0
    start_avg = steps - average_last

    def record(F, info):
        with torch.no_grad():
            trace["F"].append(float(F.detach()))
            trace["KL"].append(float(info.get("KL", 0.0)))
            trace["beta_EH"].append(float(model.beta * info["EH"]) if "EH" in info else 0.0)
            w = model._weights() if hasattr(model, "_weights") else info["w"]
            trace["w"].append(w.detach().cpu().numpy().copy())
            trace["s_mean"].append(float(torch.mean(model.s_params).detach().cpu()))
            if hasattr(model, "mus"):
                trace["mu_norm"].append(torch.linalg.norm(model.mus, dim=1).detach().cpu().numpy().copy())
                trace["s_component_mean"].append(torch.mean(model.s_params, dim=1).detach().cpu().numpy().copy())
                trace["mu_samples"].append(model.mus[0, :3].detach().cpu().numpy().copy())
                trace["s_samples"].append(model.s_params[0, :3].detach().cpu().numpy().copy())

    for step in range(steps):
        optimizer.zero_grad()
        F, info = model.forward_objective()
        if trace_at == "before":
            record(F, info)
        F.backward()
        optimizer.step()
        if average_last and step >= start_avg:
            with torch.no_grad():
                if averaged is None:
                    averaged = {name: (param.detach().clone() if averaging == "running" else torch.zeros_like(param))
                                for name, param in model.named_parameters()}
                    if averaging == "running":
                        n_avg = 1
                elif averaging == "running":
                    n_avg += 1
                    for name, param in model.named_parameters():
                        averaged[name] = (averaged[name] * (n_avg - 1) + param) / n_avg
                if averaging == "sum":
                    for name, param in model.named_parameters():
                        averaged[name] += param.data
                    n_avg += 1
        if trace_at == "after":
            record(F, info)
        if verbose_every and (step == 0 or (step + 1) % verbose_every == 0):
            print(f"Step {step + 1:6d}/{steps} | F = {float(F.detach()):.6f}", flush=True)
    if averaged is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(averaged[name] if averaging == "running" else averaged[name] / n_avg)
    return model, trace
