import torch


def fit_adam(model, *, steps, lr=0.5, average_last=0, verbose_every=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace = {"F": [], "w": [], "s_mean": []}
    averaged = None
    n_avg = 0
    start_avg = max(steps - average_last, steps + 1)
    for step in range(steps):
        optimizer.zero_grad()
        F, info = model.forward_objective()
        F.backward()
        optimizer.step()
        with torch.no_grad():
            trace["F"].append(float(F.detach()))
            trace["w"].append(info["w"].detach().cpu().numpy().copy())
            trace["s_mean"].append(float(torch.mean(model.s_params).detach().cpu()))
            if step >= start_avg:
                if averaged is None:
                    averaged = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}
                for name, param in model.named_parameters():
                    averaged[name] += param.data
                n_avg += 1
        if verbose_every and (step == 0 or (step + 1) % verbose_every == 0):
            print(f"Step {step + 1:6d}/{steps} | F = {float(F.detach()):.6f}")
    if averaged and n_avg:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(averaged[name] / n_avg)
    return model, trace

