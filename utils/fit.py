import torch
import numpy as np
import cvxpy as cp
# import pandas as pd
import time
import matplotlib.pyplot as plt
import warnings

torch.set_default_dtype(torch.double)


def fit(X, A=None, M=1, H=1, alpha=.1, kappa=.01, rank=20, niter=20, verbose=True, cuda=None,
        seed=0, opt_kwargs={}, U0=None, V0=None, l2_regularization=0.0):
    """Fits a 
    Arguments:
        - X
        - A
        - M
        - H
        - alpha
        - kappa
        - rank
        - niter
        - verbose
        - cuda
        - seed
        - opt_kwargs
        - l2_regularization

    Returns:
        - z
        - U
        - V
        - P
        - F
    """
    cuda_available = torch.cuda.is_available()
    if cuda is None and cuda_available:
        cuda = True
    if cuda is None and not cuda_available:
        cuda = False

    if cuda and not cuda_available:
        warnings.warn("Cuda is not available. Setting cuda=False.")
        cuda = False
    if not cuda and cuda_available:
        warnings.warn("Cuda is available. To use it, set cuda=True.")

    T, n = X.shape

    P = []
    F = []

    for t in range(M, T - H + 1):
        inputs = torch.from_numpy(X[t - M:t]).flatten()
        outputs = torch.from_numpy(X[t:t + H]).flatten()
        if A is not None:
            inputs = torch.cat([inputs, torch.from_numpy(A[t]).flatten()])
        P.append(inputs[None, :])
        F.append(outputs[None, :])
    P = torch.cat(P, axis=0)
    F = torch.cat(F, axis=0)
    if cuda:
        P = P.cuda()
        F = F.cuda()

    N, _ = P.shape

    lam_max = 1 / N * torch.svd(P.T @ F).S[0]
    lam = alpha * lam_max

    if verbose:
        print("Setting lam to %.3f." % lam)

    torch.manual_seed(seed)
    U = 1e-3 * torch.randn(P.shape[1],
                           rank) if U0 is None else torch.from_numpy(U0)
    V = 1e-3 * torch.randn(rank, F.shape[1]
                           ) if V0 is None else torch.from_numpy(V0)
    if cuda:
        U = U.cuda()
        V = V.cuda()
    U.requires_grad_(True)
    V.requires_grad_(True)

    if verbose:
        print("Setting up forecaster consistency.")

    row_indices = []
    col_indices = []
    for tau in range(M + 1, T + 1):
        t = torch.arange(max(tau - H, M), min(tau - 1, T - H) + 1)
        i = t - M
        j = tau - t - 1
        row_index = i.unsqueeze(1).repeat(1, n)
        col_index = torch.cat([torch.arange(s, e)[None, :]
                               for s, e in zip(j * n, (j + 1) * n)], dim=0)
        row_indices.append(row_index)
        col_indices.append(col_index)
    row_indices = torch.cat(row_indices)
    col_indices = torch.cat(col_indices)

    def forecaster_consistency(P, U, V):
        Fhat = P @ U @ V
        Proj = torch.zeros_like(Fhat)
        expanded = Fhat[row_indices, col_indices]
        up = []
        down = []
        i, ct = 0, 1
        for _ in range(H - 1):
            up.append(
                expanded[i:i + ct].mean(dim=0).unsqueeze(0).repeat_interleave(ct, dim=0))
            i += ct
            ct += 1
        mid = expanded[
            i:-i].reshape(-1, H, n).mean(dim=1).repeat_interleave(H, dim=0)
        i, ct = expanded.shape[0] - i, H - 1
        for _ in range(H - 1):
            down.append(
                expanded[i:i + ct].mean(dim=0).unsqueeze(0).repeat_interleave(ct, dim=0))
            i += ct
            ct -= 1
        Proj[row_indices, col_indices] = torch.cat(up + [mid] + down)
        return (Fhat - Proj).pow(2).sum()

    def eval_objective():
        loss_term = .5 / N * (P @ U @ V - F).pow(2).sum()
        nuc_term = lam / 2 * (U.pow(2).sum() + V.pow(2).sum())
        l2_term = l2_regularization * (U @ V).pow(2).sum()
        consistency_term = forecaster_consistency(P, U, V)
        return loss_term + nuc_term + kappa * consistency_term

    objective = float("inf")
    tic = time.time()
    for k in range(niter):
        opt = torch.optim.LBFGS(
            [U], line_search_fn='strong_wolfe', **opt_kwargs)

        def closure():
            opt.zero_grad()
            l = eval_objective()
            l.backward()
            return l
        opt.step(closure)
        opt = torch.optim.LBFGS(
            [V], line_search_fn='strong_wolfe', **opt_kwargs)

        def closure():
            opt.zero_grad()
            l = eval_objective()
            l.backward()
            return l
        opt.step(closure)

        l = eval_objective()
        if verbose:
            print(k, "|", l.item())

    fc = forecaster_consistency(P, U, V).item()

    toc = time.time()
    P = P.cpu().detach().numpy()
    F = F.cpu().detach().numpy()
    U = U.cpu().detach().numpy()
    V = V.cpu().detach().numpy()
    UU, SU, VUT = np.linalg.svd(U, full_matrices=False)

    idx = SU > 1e-2
    rank = idx.sum()
    U = UU[:, idx]
    V = np.diag(SU[idx]) @ VUT[idx, :] @ V

    if verbose:
        print("Theta has rank %d" % rank)
        print("Time:", toc - tic)

    info = {
        "forecaster_consistency": fc,
        "time": toc - tic,
        "fc": forecaster_consistency
    }
    return P @ U, U, V, P, F, info
