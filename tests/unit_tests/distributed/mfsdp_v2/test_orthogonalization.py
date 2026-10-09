# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for the owner-compute math of orthogonalized optimization."""

import contextlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer.orthogonalization import (
    OrthogonalizedOptimizer,
    _require_emerging_optimizers,
    apply_update,
    compute_orthogonalization_inputs,
    orthogonalize_owned,
)

_require_emerging_optimizers()


def _make_muon(params, **kwargs):
    """Build a `Muon` inner optimizer with standard test defaults.

    `cast(Any, ...)`: the upstream `@registry.register_optimizer` decorator's annotations
    erase the subclass (`Callable[[type[Optimizer]], type[Optimizer]]`), so `ty` only sees
    `Muon` as its base class and cannot check the constructor. Drop the cast once upstream
    ships the generic identity decorator.
    """
    from emerging_optimizers.orthogonalized_optimizers.muon import Muon

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


def _reference_pre_ns(
    param: torch.Tensor, grad: torch.Tensor, lr: float, group: dict
) -> torch.Tensor:
    """The same weight decay + momentum + Nesterov math on a plain `Muon` state."""
    ref_param = nn.Parameter(param.detach().clone())
    ref_opt = _make_muon([ref_param])
    with torch.no_grad():
        ref_opt._init_group(ref_opt.param_groups[0], skip_non_grad_params=False)
        ref_mom = ref_opt.state[ref_param]["momentum_buffer"]
        ref_grad = grad.clone()
        ref_opt._apply_weight_decay_inplace(ref_param, ref_grad, lr, group["weight_decay"])
        ref_mom.lerp_(ref_grad, 1 - group["momentum"])
        if ref_opt.nesterov:
            return ref_grad.lerp(ref_mom, group["momentum"])
        return ref_mom


def test_compute_orthogonalization_inputs_matches_reference():
    """Weight decay + momentum + Nesterov match a plain `Muon` reference."""
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(8, 4))
    inner = _make_muon([param])
    group = {"weight_decay": 0.1, "momentum": 0.9}
    lr = 0.05
    with torch.no_grad():
        inner._init_group(inner.param_groups[0], skip_non_grad_params=False)
    momentum = inner.state[param]["momentum_buffer"]
    grad = torch.randn(8, 4)

    pre = compute_orthogonalization_inputs(inner, param.detach(), momentum, grad, group, lr)

    expected = _reference_pre_ns(param.detach(), grad, lr, group)
    torch.testing.assert_close(pre, expected, atol=0, rtol=0)


def test_compute_orthogonalization_inputs_without_nesterov_returns_momentum():
    """Without Nesterov, the pre-orthogonalization input is the momentum buffer."""
    torch.manual_seed(1)
    param = nn.Parameter(torch.randn(8, 4))
    inner = _make_muon([param], nesterov=False)
    group = {"weight_decay": 0.0, "momentum": 0.9}
    lr = 0.05
    with torch.no_grad():
        inner._init_group(inner.param_groups[0], skip_non_grad_params=False)
    momentum = inner.state[param]["momentum_buffer"]
    grad = torch.randn(8, 4)

    pre = compute_orthogonalization_inputs(inner, param.detach(), momentum, grad, group, lr)

    torch.testing.assert_close(pre, momentum, atol=0, rtol=0)


def test_orthogonalize_owned_processes_only_owned_in_fp32():
    """Only owned tensors are orthogonalized, contracted to 2D and cast to fp32."""
    tensor_to_owner = {0: 0}
    destination = {0: torch.randn(32)}
    tensor_shapes = [torch.Size((8, 2, 2))]
    inner = cast(
        OrthogonalizedOptimizer,
        SimpleNamespace(
            fp32_matmul_prec="medium", orthogonalize=lambda _param, matrix, **_kwargs: matrix * 2.0
        ),
    )

    updates = orthogonalize_owned(
        inner,
        this_rank=0,
        tensor_to_owner=tensor_to_owner,
        destination=destination,
        tensor_shapes=tensor_shapes,
        group_kwargs={},
    )

    assert list(updates) == [0]
    expected = destination[0].view(8, 4).to(torch.float32) * 2.0
    torch.testing.assert_close(updates[0], expected, atol=0, rtol=0)


def _fail_orthogonalize(_param, _matrix, **_kwargs):
    raise AssertionError("must not be called for other ranks' tensors")


def test_orthogonalize_owned_skips_tensors_owned_by_other_ranks():
    """Tensors owned by other ranks never reach the kernel."""
    tensor_to_owner = {0: 1}
    destination = {0: torch.randn(32)}
    tensor_shapes = [torch.Size((8, 4))]
    inner = cast(
        OrthogonalizedOptimizer,
        SimpleNamespace(fp32_matmul_prec="medium", orthogonalize=_fail_orthogonalize),
    )

    updates = orthogonalize_owned(
        inner,
        this_rank=0,
        tensor_to_owner=tensor_to_owner,
        destination=destination,
        tensor_shapes=tensor_shapes,
        group_kwargs={},
    )

    assert updates == {}


def test_apply_update_views_casts_and_steps_between_hooks():
    """The update is viewed/cast to the local parameter and applied between the hooks."""
    hooks: list[str] = []
    inner = cast(
        OrthogonalizedOptimizer,
        SimpleNamespace(
            pre_weight_update_fn_inplace=lambda _param, update: hooks.append("pre"),
            post_weight_update_fn_inplace=lambda _param: hooks.append("post"),
        ),
    )
    param_local = torch.randn(2, 3)
    before = param_local.clone()
    update = torch.randn(6, dtype=torch.float64)

    apply_update(inner, param_local, update, lr=0.1)

    torch.testing.assert_close(param_local, before - 0.1 * update.to(torch.float32).view(2, 3))
    assert hooks == ["pre", "post"]


# ---------------------------------------------------------------------------
# Import guard tests
# ---------------------------------------------------------------------------

_EO_MOD = (
    "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer."
    "orthogonalization"
)


def test_import_guard_emerging_optimizers_available():
    """When emerging_optimizers is installed, HAVE_EMERGING_OPTIMIZERS is True."""
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer as RealEO

    import megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer.orthogonalization as mod

    assert mod.HAVE_EMERGING_OPTIMIZERS is True
    assert mod.OrthogonalizedOptimizer is RealEO


@contextlib.contextmanager
def _simulate_no_emerging_optimizers():
    """Reload `orthogonalization` with `emerging_optimizers` blocked."""
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
    saved_mod = sys.modules.pop(_EO_MOD, None)

    sys.meta_path.insert(0, _BlockEO)
    try:
        yield importlib.import_module(_EO_MOD)
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(saved_eo)
        if saved_mod is not None:
            sys.modules[_EO_MOD] = saved_mod
        else:
            sys.modules.pop(_EO_MOD, None)


def test_import_guard_without_emerging_optimizers():
    """When emerging_optimizers is not installed, the module falls back gracefully."""
    with _simulate_no_emerging_optimizers() as mod:
        assert mod.HAVE_EMERGING_OPTIMIZERS is False
        assert mod.OrthogonalizedOptimizer is object
