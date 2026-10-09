# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
The owner-compute math for orthogonalized optimization on local tensors.

Pure per-parameter steps — weight decay, momentum, Nesterov, the Newton-Schulz orthogonalization,
and the weight update — over local tensor shards, driven by a composed `emerging_optimizers` inner
optimizer. This module is the sole `emerging_optimizers` import site and needs neither FSDP nor
`torch.distributed`, so the helpers are CPU-testable against a plain reference.

Every ≥2D tensor is contracted to a 2D matrix `(shape[0], rest)` for orthogonalization, per Muon's
matrix view of weights. Like the upstream `OrthogonalizedOptimizer`, `<2D` tensors are not
contracted and error when `orthogonalize` is called; callers that route parameters by dimensionality
(e.g., `FsdpMuon`) must reject them earlier.
"""

from collections.abc import Sequence
from typing import Any, cast

import torch

try:
    from emerging_optimizers import utils as eo_utils
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer

    HAVE_EMERGING_OPTIMIZERS = True
except (ModuleNotFoundError, ImportError):
    eo_utils = cast(Any, None)
    OrthogonalizedOptimizer = cast(Any, object)
    HAVE_EMERGING_OPTIMIZERS = False


def _require_emerging_optimizers() -> None:
    """Raise if `emerging_optimizers` is not installed."""
    if not HAVE_EMERGING_OPTIMIZERS:
        raise ModuleNotFoundError(
            "Emerging-Optimizers is required for orthogonalized optimizer support. "
            "Please install the necessary dependencies with "
            "`pip install 'megatron_fsdp[emerging-optimizers]'`."
        )


def compute_orthogonalization_inputs(
    inner: OrthogonalizedOptimizer,
    param_local: torch.Tensor,
    momentum_local: torch.Tensor,
    grad_local: torch.Tensor,
    group: dict[str, Any],
    lr: float,
) -> torch.Tensor:
    """Apply weight decay and update momentum, then return the pre-orthogonalization input.

    The inner's `l2` weight-decay method mutates `grad_local` in place, so callers passing a
    persistent view (e.g., MFSDP's `main_grad`) must overwrite or zero it before the next
    accumulation.

    Args:
        inner: The composed inner optimizer providing the weight-decay method and the `nesterov`
            flag.
        param_local: The local parameter shard the decay reads.
        momentum_local: The local momentum buffer shard; updated in place.
        grad_local: The local gradient shard, cast and reshaped to the momentum buffer if needed.
        group: The `torch.optim` param group holding `weight_decay` and `momentum`.
        lr: The learning rate of this step.
    """
    if grad_local.dtype != momentum_local.dtype:
        grad_local = grad_local.to(dtype=momentum_local.dtype)
    if grad_local.shape != momentum_local.shape:
        grad_local = grad_local.reshape(momentum_local.shape)
    inner._apply_weight_decay_inplace(param_local, grad_local, lr, group["weight_decay"])
    momentum_local.lerp_(grad_local, 1 - group["momentum"])
    if inner.nesterov:
        return grad_local.lerp(momentum_local, group["momentum"])
    return momentum_local


def orthogonalize_owned(
    inner: OrthogonalizedOptimizer,
    this_rank: int,
    tensor_to_owner: dict[int, int],
    destination: dict[int, torch.Tensor],
    tensor_shapes: Sequence[torch.Size],
    group_kwargs: dict[str, Any],
) -> dict[int, torch.Tensor]:
    """Orthogonalize this rank's owned tensors into full updates.

    Each ≥2D input is contracted to a 2D matrix `(shape[0], rest)` per Muon's matrix view of
    weights. `<2D` inputs must be rejected by the caller rank-uniformly before any communication is
    posted (a per-rank kernel error would deadlock the P2P); the kernel-level check remains as a
    guard. The per-parameter loop is the seam for future batching: parameters of the same shape
    could be stacked into one batched kernel.

    Args:
        inner: The composed inner optimizer providing the `orthogonalize` kernel and its
            `fp32_matmul_prec` setting.
        this_rank: This rank's global rank.
        tensor_to_owner: Participating tensor index to owner global rank.
        destination: The gathered full tensor per owned tensor index.
        tensor_shapes: Full tensor shapes, indexable by tensor index (the buffer's layout's
            `tensor_shapes`).
        group_kwargs: The param group's hyperparameters, forwarded to the kernel.
    """
    updates: dict[int, torch.Tensor] = {}
    for i, owner in tensor_to_owner.items():
        if owner != this_rank:
            continue
        shape = tensor_shapes[i]
        matrix = destination[i].view(shape[0], -1) if len(shape) >= 2 else destination[i]
        with eo_utils.fp32_matmul_precision(inner.fp32_matmul_prec):
            # The kernel ignores the parameter argument, so `None` is fine.
            updates[i] = inner.orthogonalize(
                None, matrix.to(torch.float32), **group_kwargs  # ty: ignore[invalid-argument-type]
            )
    return updates


def apply_update(
    inner: OrthogonalizedOptimizer, param_local: torch.Tensor, update: torch.Tensor, lr: float
) -> None:
    """Apply an orthogonalization result shard to the local parameter, in place.

    The update is viewed and cast to the local parameter's shape and dtype, and the inner
    optimizer's weight-update hooks run around the in-place step.

    Args:
        inner: The composed inner optimizer providing the weight-update hooks.
        param_local: The local parameter shard to update in place.
        update: This rank's flat result shard for the parameter.
        lr: The learning rate of this step.
    """
    if update.shape != param_local.shape:
        update = update.view(param_local.shape)
    if update.dtype != param_local.dtype:
        update = update.to(dtype=param_local.dtype)
    inner.pre_weight_update_fn_inplace(param_local, update)
    param_local.add_(update, alpha=-lr)
    inner.post_weight_update_fn_inplace(param_local)
