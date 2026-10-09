# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The Muon variant of the owner-compute orthogonalized optimizer."""

import inspect
import warnings
from collections.abc import Sequence
from typing import override

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.optim.optimizer import ParamsT

from .optimizer import FsdpOrthogonalizedOptimizer
from .orthogonalization import Muon, _require_emerging_optimizers


class FsdpMuon(FsdpOrthogonalizedOptimizer):
    """Muon optimizer for all-`RowAtomic` MFSDP v2 parameters.

    Composes a `Muon` inner optimizer (an `OrthogonalizedOptimizer`) for the Newton-Schulz
    orthogonalization and update scaling. The inner `Muon` installs its own
    `scaled_orthogonalize_fn`, so the base `scaled_orthogonalize_fn` is ignored.

    Rejects `<2D` parameters at construction per Muon's convention — route them (biases,
    norms) to a separate optimizer.
    """

    def __init__(
        self,
        params: ParamsT,
        inner_optimizer: Muon,  # ty: ignore[invalid-type-form]
        dp_mesh: DeviceMesh,
    ) -> None:
        """Build the FSDP Muon optimizer; see `FsdpOrthogonalizedOptimizer` for the rest."""
        _require_emerging_optimizers()

        if hasattr(inner_optimizer, "num_ns_steps"):
            self._num_ns_steps = inner_optimizer.num_ns_steps
        else:
            # For older `emerging_optimizers` versions, we use introspection
            # techniques to get a sensible value for `num_ns_steps`.
            try:
                ortho_fn_vars = inspect.getclosurevars(inner_optimizer.scaled_orthogonalize_fn)
                self._num_ns_steps = ortho_fn_vars.nonlocals["num_ns_steps"]
            except KeyError:
                warnings.warn(
                    "Cannot access Muon closure non-locals; going with "
                    "`emerging_optimizers.orthogonalized_optimizer.Muon` "
                    "default `num_ns_steps` for compute cost estimation"
                )
                muon_sig = inspect.signature(inner_optimizer)
                self._num_ns_steps = muon_sig.parameters["num_ns_steps"].default
        super().__init__(params, inner_optimizer, dp_mesh=dp_mesh, num_ns_steps=self._num_ns_steps)

    @override
    def _validate_param_ndims(self, params: Sequence[torch.Tensor]) -> None:
        """Reject `<2D` parameters: Muon orthogonalizes matrices only."""
        offending_shapes = [tuple(p.shape) for p in params if p.ndim < 2]
        if offending_shapes:
            raise ValueError(
                "FsdpMuon only supports >=2D parameters (they are contracted to 2D "
                "matrices for orthogonalization). Route <2D parameters — biases, norms "
                f"— to a separate optimizer. Offending shapes: {offending_shapes}."
            )
