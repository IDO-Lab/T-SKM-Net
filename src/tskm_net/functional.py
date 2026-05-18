from typing import Dict, Optional

import torch

from ._variants import _VARIANTS


_SVD_RANK_EPS = 10.0 * torch.finfo(torch.float64).eps
_NULLSPACE_CHECK_EPS = 100.0 * torch.finfo(torch.float64).eps


def _validate_inputs(
    A: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    d: torch.Tensor,
    x0: torch.Tensor,
) -> None:
    if A.dim() != 2:
        raise ValueError(f"Expected A to have shape (p, n), got {tuple(A.shape)}.")
    if b.dim() != 1:
        raise ValueError(f"Expected b to have shape (p,), got {tuple(b.shape)}.")
    if C.dim() != 2:
        raise ValueError(f"Expected C to have shape (q, n), got {tuple(C.shape)}.")
    if d.dim() != 2:
        raise ValueError(f"Expected d to have shape (B, q), got {tuple(d.shape)}.")
    if x0.dim() != 2:
        raise ValueError(f"Expected x0 to have shape (B, n), got {tuple(x0.shape)}.")

    tensors = {"A": A, "b": b, "C": C, "d": d, "x0": x0}
    for name, tensor in tensors.items():
        if not tensor.is_floating_point():
            raise TypeError(f"Expected {name} to be a floating-point tensor, got {tensor.dtype}.")

    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        device_summary = ", ".join(f"{name}={tensor.device}" for name, tensor in tensors.items())
        raise ValueError(f"Expected all inputs to be on the same device, got {device_summary}.")

    p, n = A.shape
    q, c_n = C.shape
    batch, x_n = x0.shape

    if b.shape[0] != p:
        raise ValueError(f"Expected b.shape[0] == A.shape[0], got {b.shape[0]} and {p}.")
    if c_n != n:
        raise ValueError(f"Expected C.shape[1] == A.shape[1], got {c_n} and {n}.")
    if x_n != n:
        raise ValueError(f"Expected x0.shape[1] == A.shape[1], got {x_n} and {n}.")
    if d.shape != (batch, q):
        raise ValueError(f"Expected d to have shape {(batch, q)}, got {tuple(d.shape)}.")


def _validate_svd(
    C: torch.Tensor,
    U: Optional[torch.Tensor],
    S: Optional[torch.Tensor],
    Vh: Optional[torch.Tensor],
) -> None:
    svd_parts = (U, S, Vh)
    if any(part is None for part in svd_parts) and not all(part is None for part in svd_parts):
        raise ValueError("Pass U, S, and Vh together, or leave all three as None.")

    if U is None or S is None or Vh is None:
        return

    q, n = C.shape
    expected_k = min(q, n)
    if U.shape != (q, q):
        raise ValueError(f"Expected U to have shape {(q, q)}, got {tuple(U.shape)}.")
    if S.shape != (expected_k,):
        raise ValueError(f"Expected S to have shape {(expected_k,)}, got {tuple(S.shape)}.")
    if Vh.shape != (n, n):
        raise ValueError(f"Expected Vh to have shape {(n, n)}, got {tuple(Vh.shape)}.")


def _svd_rank(S: torch.Tensor, matrix_shape: torch.Size) -> int:
    if S.numel() == 0:
        return 0
    cutoff = S.max() * max(matrix_shape) * _SVD_RANK_EPS
    return int(torch.sum(S > cutoff).item())


def _check_nullspace(C: torch.Tensor, nullspace_basis: torch.Tensor) -> None:
    if nullspace_basis.numel() == 0:
        return

    residual = C @ nullspace_basis
    residual_norm = torch.linalg.matrix_norm(residual, ord=2)
    scale = torch.linalg.matrix_norm(C, ord=2).clamp_min(1.0)
    threshold = scale * max(C.shape) * _NULLSPACE_CHECK_EPS
    if residual_norm > threshold:
        raise RuntimeError("Failed to construct a valid nullspace basis from C.")


def t_skm_project(
    A: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    d: torch.Tensor,
    x0: torch.Tensor,
    beta: int = 50,
    delta: float = 1.0,
    variant: str = "skm",
    variant_args: Optional[Dict[str, float]] = None,
    tol: float = 1e-3,
    max_iter: int = 100,
    U: Optional[torch.Tensor] = None,
    S: Optional[torch.Tensor] = None,
    Vh: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project batched inputs toward mixed linear constraints with T-SKM.

    The constraints are shared across the batch except for the equality
    right-hand side:

    - ``A`` has shape ``(p, n)`` and ``b`` has shape ``(p,)`` for ``A x <= b``.
    - ``C`` has shape ``(q, n)`` and ``d`` has shape ``(B, q)`` for ``C x = d``.
    - ``x0`` has shape ``(B, n)``.

    ``U``, ``S``, and ``Vh`` may be supplied from
    ``torch.linalg.svd(C.to(torch.float64), full_matrices=True)`` to reuse the
    SVD when ``C`` is fixed.
    """
    _validate_inputs(A, b, C, d, x0)
    _validate_svd(C, U, S, Vh)

    if beta <= 0:
        raise ValueError(f"Expected beta to be positive, got {beta}.")
    if max_iter < 0:
        raise ValueError(f"Expected max_iter to be non-negative, got {max_iter}.")
    if variant_args is None:
        variant_args = {}
    elif not isinstance(variant_args, dict):
        raise TypeError("Expected variant_args to be a dict or None.")

    variant_fn = _VARIANTS.get(variant)
    if variant_fn is None:
        expected = ", ".join(sorted(_VARIANTS))
        raise ValueError(f"Unknown variant {variant!r}. Expected one of: {expected}.")

    A64 = A.to(torch.float64)
    b64 = b.to(torch.float64)
    C64 = C.to(torch.float64)
    d64 = d.to(torch.float64)
    x064 = x0.to(torch.float64)

    if U is None or S is None or Vh is None:
        U64, S64, Vh64 = torch.linalg.svd(C64, full_matrices=True)
    else:
        U64 = U.to(dtype=torch.float64, device=C.device)
        S64 = S.to(dtype=torch.float64, device=C.device)
        Vh64 = Vh.to(dtype=torch.float64, device=C.device)

    V = Vh64.T
    rank = _svd_rank(S64, C64.shape)

    S_r_inv = torch.diag(S64[:rank].reciprocal())
    C_pinv = V[:, :rank] @ S_r_inv @ U64[:, :rank].T

    residual = x064 @ C64.T - d64
    x0_projected = x064 - residual @ C_pinv.T

    nullspace_basis = V[:, rank:]
    expected_null_dim = C.shape[1] - rank
    if nullspace_basis.shape[1] != expected_null_dim:
        raise RuntimeError(
            "Nullspace dimension mismatch: "
            f"expected {expected_null_dim}, got {nullspace_basis.shape[1]}."
        )

    _check_nullspace(C64, nullspace_basis)

    A_reduced = A64 @ nullspace_basis
    b_reduced = b64 - x0_projected @ A64.T
    z0 = torch.zeros(
        (x064.shape[0], nullspace_basis.shape[1]),
        dtype=x064.dtype,
        device=x064.device,
    )

    try:
        z = variant_fn(
            A_reduced,
            b_reduced,
            z0,
            beta=beta,
            delta=delta,
            tol=tol,
            max_iter=max_iter,
            **variant_args,
        )
    except TypeError as exc:
        raise TypeError(f"Invalid variant_args for variant {variant!r}: {exc}") from exc

    x = x0_projected + z @ nullspace_basis.T
    return x.to(dtype=x0.dtype)
