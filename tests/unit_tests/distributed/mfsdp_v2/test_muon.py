# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for `FsdpMuon`, the Muon variant of the owner-compute orthogonalized optimizer.

The construction and import-guard tests run without `torchrun`; the routing test runs
under `torchrun` and routes 1D parameters to a separate optimizer.
"""

import contextlib
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from emerging_optimizers.orthogonalized_optimizers.muon import Muon
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer import (
    FsdpMuon,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic

try:
    from tests.unit_tests.distributed.mfsdp_v2.conftest import DistributedSetup
except Exception:  # pragma: no cover
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class DistributedSetup:
        rank: int
        world_size: int
        device: torch.device


def _row_atomic_placements() -> Placements:
    """All-`RowAtomic` placements for the models in this file."""
    return Placements(
        dp_axes=[0], parameter=[RowAtomic()], gradient=[RowAtomic()], optimizer=[RowAtomic()]
    )


def _needs_p2p_ranks(distributed_setup):
    """Skip unless the config can run the owner-compute P2P step."""
    world_size = distributed_setup.world_size
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; a 1-GPU-multi-rank config cannot "
            "run the owner-compute P2P step."
        )
    return init_device_mesh(distributed_setup.device.type, (world_size,))


class Bias1d(nn.Module):
    """A single 1D parameter in its own module so it can be RowAtomic-sharded separately."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class MixedModel(nn.Module):
    """A 2D linear (routed to `FsdpMuon`) and a 1D bias (routed to a separate optimizer)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)
        self.relu = nn.ReLU()
        self.bias_mod = Bias1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias_mod(self.relu(self.fc(x)))


def _make_muon(params, **kwargs):
    """Build a `Muon` inner optimizer with standard test defaults.

    `cast(Any, ...)`: the upstream `@registry.register_optimizer` decorator's annotations
    erase the subclass (`Callable[[type[Optimizer]], type[Optimizer]]`), so `ty` only sees
    `Muon` as its base class and cannot check the constructor. Drop the cast once upstream
    ships the generic identity decorator.
    """
    return cast(Any, Muon)(
        params,
        lr=kwargs.get("lr", 0.05),
        momentum=kwargs.get("momentum", 0.9),
        weight_decay=kwargs.get("weight_decay", 0.0),
        nesterov=kwargs.get("nesterov", True),
        coefficient_type=kwargs.get("coefficient_type", "quintic"),
        num_ns_steps=kwargs.get("num_ns_steps", 5),
        scale_mode=kwargs.get("scale_mode", "spectral"),
        fp32_matmul_prec=kwargs.get("fp32_matmul_prec", "medium"),
        use_syrk=kwargs.get("use_syrk", False),
    )


def test_fsdp_muon_rejects_1d_params():
    """`FsdpMuon` rejects `<2D` parameters at construction, per Muon's convention.

    The rejection fires in `_validate_param_ndims` before the FSDP-sharded and mesh
    validation, so this test needs no distributed setup.
    """
    param = nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match=">=2D"):
        FsdpMuon([param], inner_optimizer=_make_muon([param]), dp_mesh=None)


def test_step_mixed_params_with_separate_optimizer(distributed_setup):
    """1D params are routed to a separate optimizer; 2D params to `FsdpMuon`."""
    mesh = _needs_p2p_ranks(distributed_setup)
    device = distributed_setup.device

    torch.manual_seed(1234)
    model = MixedModel().to(device)
    with fully_shard_context(device=device):
        fully_shard(model.fc, mesh=mesh, placements=_row_atomic_placements())
        fully_shard(model.bias_mod, mesh=mesh, placements=_row_atomic_placements())

    torch.manual_seed(1234)
    baseline = MixedModel().to(device)
    with torch.no_grad():
        baseline.fc.weight.copy_(_gather_full_param(model.fc.weight, mesh))
        baseline.bias_mod.bias.copy_(_gather_full_param(model.bias_mod.bias, mesh))

    # The 2D weight goes to `FsdpMuon`; the 1D bias goes to a separate optimizer.
    matrix_opt = FsdpMuon(
        [model.fc.weight], inner_optimizer=_make_muon([model.fc.weight]), dp_mesh=mesh
    )
    bias_opt = torch.optim.SGD([model.bias_mod.bias], lr=0.05, momentum=0.9)
    base_matrix_opt = _make_muon([baseline.fc.weight])
    base_bias_opt = torch.optim.SGD([baseline.bias_mod.bias], lr=0.05, momentum=0.9)

    x = torch.randn(4, 8, device=device)
    for _ in range(3):
        for opt in (matrix_opt, bias_opt, base_matrix_opt, base_bias_opt):
            opt.zero_grad(set_to_none=True)
        model(x).sum().backward()
        baseline(x).sum().backward()
        matrix_opt.step()
        bias_opt.step()
        base_matrix_opt.step()
        base_bias_opt.step()

    torch.testing.assert_close(
        _gather_full_param(model.fc.weight, mesh), baseline.fc.weight, atol=0, rtol=0
    )
    torch.testing.assert_close(
        _gather_full_param(model.bias_mod.bias, mesh), baseline.bias_mod.bias, atol=0, rtol=0
    )


def _gather_full_param(param, mesh):
    """All-gather a RowAtomic-sharded parameter into its full tensor."""
    local = param.to_local().contiguous().flatten()
    parts = [torch.empty_like(local) for _ in range(mesh.size())]
    dist.all_gather(parts, local, group=mesh.get_group())
    return torch.cat(parts)[: param.numel()].view(param.shape)


# ---------------------------------------------------------------------------
# Import guard test
# ---------------------------------------------------------------------------

_EO_PACKAGE = (
    "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer"
)
_EO_MODULES = (
    _EO_PACKAGE,
    f"{_EO_PACKAGE}.orthogonalization",
    f"{_EO_PACKAGE}.optimizer",
    f"{_EO_PACKAGE}.muon",
)


@contextlib.contextmanager
def _simulate_no_emerging_optimizers():
    """Reload the muon modules with `emerging_optimizers` blocked."""
    import importlib
    import sys

    class _BlockEO:
        @staticmethod
        def find_spec(name, _path, _target=None):
            if name == "emerging_optimizers" or name.startswith("emerging_optimizers."):
                raise ModuleNotFoundError(name)
            return None

    saved_eo = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "emerging_optimizers" or k.startswith("emerging_optimizers.")
    }
    saved_mods = {k: sys.modules.pop(k, None) for k in _EO_MODULES}

    sys.meta_path.insert(0, _BlockEO)
    try:
        yield importlib.import_module(f"{_EO_PACKAGE}.muon")
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(saved_eo)
        for name, module in saved_mods.items():
            if module is not None:
                sys.modules[name] = module
                # The blocked reload also rebound the parent's attribute; restore it,
                # since `import a.b.c as mod` binds through the parent's attributes.
                parent_name, _, leaf = name.rpartition(".")
                if parent_name and parent_name in sys.modules:
                    setattr(sys.modules[parent_name], leaf, module)
            else:
                sys.modules.pop(name, None)


def test_construction_error_without_emerging_optimizers():
    """Constructing without `emerging_optimizers` raises `ModuleNotFoundError`."""
    with _simulate_no_emerging_optimizers() as mod:
        with pytest.raises(ModuleNotFoundError, match="emerging-optimizers"):
            mod.FsdpMuon([], object(), dp_mesh=None)
