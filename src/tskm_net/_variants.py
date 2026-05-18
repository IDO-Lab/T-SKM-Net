from typing import Callable, Dict

import torch


@torch.jit.script
def skm(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    beta: int,
    delta: float,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> torch.Tensor:
    """Sampling Kaczmarz-Motzkin iteration for batched inequalities."""
    x = x0.clone()
    norm2 = (A**2).sum(dim=1)

    for _ in range(max_iter):
        res = x @ A.T - b
        if torch.all(res <= tol):
            break

        indices = torch.randint(0, A.shape[0], (x.shape[0], beta), device=x.device)
        A_sampled = A[indices, :]
        res_sampled = torch.gather(res, 1, indices).relu()

        rows = torch.arange(x.shape[0], device=x.device)
        i_star = torch.argmax(res_sampled, dim=1)
        selected_indices = indices[rows, i_star]
        a_i_star = A_sampled[rows, i_star]
        r_i_star = res_sampled[rows, i_star]

        if not torch.all(r_i_star == 0):
            x = x - delta * r_i_star.unsqueeze(1) * a_i_star / norm2[selected_indices].unsqueeze(1)

    return x


@torch.jit.script
def heavy_ball_skm(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    beta: int,
    delta: float,
    tol: float = 1e-3,
    max_iter: int = 100,
    momentum: float = 0.25,
) -> torch.Tensor:
    """Sampling Kaczmarz-Motzkin iteration with heavy-ball momentum."""
    x = x0.clone()
    v = torch.zeros_like(x)
    norm2 = (A**2).sum(dim=1)

    for _ in range(max_iter):
        res = x @ A.T - b
        if torch.all(res <= tol):
            break

        indices = torch.randint(0, A.shape[0], (x.shape[0], beta), device=x.device)
        A_sampled = A[indices, :]
        res_sampled = torch.gather(res, 1, indices).relu()

        rows = torch.arange(x.shape[0], device=x.device)
        i_star = torch.argmax(res_sampled, dim=1)
        selected_indices = indices[rows, i_star]
        a_i_star = A_sampled[rows, i_star]
        r_i_star = res_sampled[rows, i_star]

        if not torch.all(r_i_star == 0):
            v = momentum * v - delta * r_i_star.unsqueeze(1) * a_i_star / norm2[selected_indices].unsqueeze(1)
        else:
            v = momentum * v
        x = x + v

    return x


@torch.jit.script
def nesterov_skm(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    beta: int,
    delta: float,
    tol: float = 1e-3,
    max_iter: int = 100,
    momentum: float = 0.25,
) -> torch.Tensor:
    """Sampling Kaczmarz-Motzkin iteration with Nesterov look-ahead momentum."""
    x = x0.clone()
    v = torch.zeros_like(x)
    norm2 = (A**2).sum(dim=1)

    for _ in range(max_iter):
        res = x @ A.T - b
        if torch.all(res <= tol):
            break

        y = x + momentum * v
        indices = torch.randint(0, A.shape[0], (x.shape[0], beta), device=x.device)
        A_sampled = A[indices, :]
        b_sampled = torch.gather(b, 1, indices)
        res_sampled = ((y.unsqueeze(1) @ A_sampled.transpose(-1, -2)).squeeze(1) - b_sampled).relu()

        rows = torch.arange(x.shape[0], device=x.device)
        i_star = torch.argmax(res_sampled, dim=1)
        selected_indices = indices[rows, i_star]
        a_i_star = A_sampled[rows, i_star]
        r_i_star = res_sampled[rows, i_star]

        if not torch.all(r_i_star == 0):
            v = momentum * v - delta * r_i_star.unsqueeze(1) * a_i_star / norm2[selected_indices].unsqueeze(1)
        else:
            v = momentum * v
        x = x + v

    return x


@torch.jit.script
def gskm(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    beta: int,
    delta: float,
    tol: float = 1e-3,
    max_iter: int = 100,
    extrapolation: float = -0.25,
) -> torch.Tensor:
    """Generalized Sampling Kaczmarz-Motzkin iteration."""
    x = x0.clone()
    z_prev = x
    norm2 = (A**2).sum(dim=1)

    for _ in range(max_iter):
        res = x @ A.T - b
        if torch.all(res <= tol):
            break

        indices = torch.randint(0, A.shape[0], (x.shape[0], beta), device=x.device)
        A_sampled = A[indices, :]
        res_sampled = torch.gather(res, 1, indices).relu()

        rows = torch.arange(x.shape[0], device=x.device)
        i_star = torch.argmax(res_sampled, dim=1)
        selected_indices = indices[rows, i_star]
        a_i_star = A_sampled[rows, i_star]
        r_i_star = res_sampled[rows, i_star]

        if not torch.all(r_i_star == 0):
            z_curr = x - delta * r_i_star.unsqueeze(1) * a_i_star / norm2[selected_indices].unsqueeze(1)
        else:
            z_curr = x

        x = (1 - extrapolation) * z_curr + extrapolation * z_prev
        z_prev = z_curr

    return x


_VARIANTS: Dict[str, Callable[..., torch.Tensor]] = {
    "skm": skm,
    "heavy_ball_skm": heavy_ball_skm,
    "nesterov_skm": nesterov_skm,
    "gskm": gskm,
}
