# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import (
    append_unique_process_group,
    get_dtensor_data_parallel_shard_groups,
    get_pg_size,
    log_single_rank,
)

from .optimizer_config import ParamKey, ParamPredicate

try:
    from torch.distributed.tensor import DTensor as _DTensor
    from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        _assert_chunks_cover_full_tensor,
        redistribute_uneven_dtensor_to_replicated,
        update_uneven_dtensor_chunk_metadata,
    )

    _HAVE_DTENSOR = True
except ImportError:
    _DTensor = None  # type: ignore[assignment,misc]
    Replicate = None  # type: ignore[assignment,misc]
    Shard = None  # type: ignore[assignment,misc]
    _StridedShard = None  # type: ignore[assignment,misc]
    _assert_chunks_cover_full_tensor = None  # type: ignore[assignment]
    redistribute_uneven_dtensor_to_replicated = None  # type: ignore[assignment]
    update_uneven_dtensor_chunk_metadata = None  # type: ignore[assignment]
    _HAVE_DTENSOR = False

try:
    from emerging_optimizers import registry, triton_kernels
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object


logger = logging.getLogger(__name__)


def _should_use_padded_all_gather(rank_total_numels: list[int], pad_factor: float) -> bool:
    total_numel = sum(rank_total_numels)
    if total_numel == 0:
        return False
    max_rank_numel = max(rank_total_numels)
    if max_rank_numel == 0 or pad_factor <= 0:
        return False
    return max_rank_numel * len(rank_total_numels) <= pad_factor * total_numel


def _chunk_infos_are_contiguous_full_order(
    full_shape: torch.Size, chunk_infos: list[dict[str, Any]]
) -> bool:
    """Return whether row-sharded chunks cover `full_shape` in row-major order."""
    full_shape = torch.Size(full_shape)
    if full_shape.numel() == 0:
        return all(chunk_info["numel"] == 0 for chunk_info in chunk_infos)

    expected_flat_start = 0
    trailing_numel = torch.Size(full_shape[1:]).numel() if len(full_shape) > 1 else 1
    for chunk_info in chunk_infos:
        chunk_numel = chunk_info["numel"]
        if chunk_numel == 0:
            continue

        offset = tuple(chunk_info["offset"])
        chunk_shape = torch.Size(chunk_info["shape"])
        if len(offset) != len(full_shape) or len(chunk_shape) != len(full_shape):
            return False
        if len(full_shape) > 1:
            for dim in range(1, len(full_shape)):
                if offset[dim] != 0 or chunk_shape[dim] != full_shape[dim]:
                    return False
        flat_start = offset[0] * trailing_numel
        if flat_start != expected_flat_start:
            return False
        expected_flat_start += chunk_numel

    return expected_flat_start == full_shape.numel()


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is not None:
        eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    else:
        eopt_kwargs = _default_adam_based_eopt_config_to_kwargs(
            eopt_name, config, model_chunks, pg_collection
        )
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _get_qkv_split_shapes(model_cfg) -> List[int]:
    """Compute QKV split shapes from model config."""
    return [
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]


def _is_named_qkv_param(param: torch.Tensor) -> bool:
    """Return true when FSDP metadata names this tensor as a fused QKV weight."""
    param_name = getattr(param, "megatron_fsdp_param_name", None)
    if param_name is None:
        param_name = getattr(getattr(param, "orig_param", None), "megatron_fsdp_param_name", None)
    return isinstance(param_name, str) and "linear_qkv.weight" in param_name


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        use_syrk: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        if use_syrk:
            sm_version = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            if not triton_kernels.HAS_TRITON_340:  # type: ignore[attr-defined]
                log_single_rank(
                    logger,
                    logging.ERROR,
                    "Triton 3.4.0 or higher is required for --muon-use-syrk; "
                    "falling back to torch matmul Newton-Schulz.",
                )
                use_syrk = False
            elif sm_version not in ((8, 0), (9, 0), (10, 0), (10, 3)):
                log_single_rank(
                    logger,
                    logging.ERROR,
                    f"Correctness of Triton SYRK kernels on SM {sm_version} is not "
                    "validated; falling back to torch matmul Newton-Schulz.",
                )
                use_syrk = False

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
                use_syrk=use_syrk,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            orth_grad.mul_(scale_factor * extra_scale_factor)
            return orth_grad

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.use_syrk = use_syrk

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        # Use explicit class call instead of super() so that subclasses with
        # multiple inheritance (e.g. TensorParallelAdaptiveMuon) don't route
        # through an intermediate class that doesn't accept scaled_orthogonalize_fn.
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    @staticmethod
    def _param_grad(p: torch.Tensor) -> torch.Tensor | None:
        """Return the gradient tensor, including precision-aware decoupled grads."""
        decoupled_grad = getattr(p, "decoupled_grad", None)
        if decoupled_grad is not None:
            return decoupled_grad
        return p.grad

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        split_dim = (
            self._qkv_split_dim_for_shape(grad.shape)
            if self.split_qkv and self.is_qkv_fn(p)  # type: ignore[misc]
            else None
        )
        if split_dim is not None:
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split_dim={split_dim}, '
                f'split shapes {self.qkv_split_shapes}',
            )
            qkv_total = sum(self.qkv_split_shapes)
            if split_dim == 0:
                num_query_groups = grad_shape[0] // qkv_total
                qkv_grads = torch.split(
                    grad.view(num_query_groups, qkv_total, -1), self.qkv_split_shapes, dim=1
                )
                qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]
            else:
                num_query_groups = grad_shape[1] // qkv_total
                qkv_grads = torch.split(
                    grad.view(grad_shape[0], num_query_groups, qkv_total),
                    self.qkv_split_shapes,
                    dim=2,
                )
                qkv_grads = [g.reshape(grad_shape[0], -1) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim) for g in qkv_grads
            ]
            if split_dim == 0:
                qkv_grads = [g.view(num_query_groups, -1, grad_shape[-1]) for g in qkv_grads]
                grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
            else:
                qkv_grads = [g.view(grad_shape[0], num_query_groups, -1) for g in qkv_grads]
                grad = torch.cat(qkv_grads, dim=2).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad

    def _qkv_split_dim_for_shape(self, shape: torch.Size | tuple[int, ...]) -> int | None:
        if self.qkv_split_shapes is None or len(shape) < 2:
            return None
        qkv_total = sum(self.qkv_split_shapes)
        if shape[0] % qkv_total == 0:
            return 0
        if shape[1] % qkv_total == 0:
            return 1
        return None


class FSDPTensorParallelMuon(TensorParallelMuon):
    """TensorParallelMuon for Megatron-FSDP ZeRO-1/2/3.

    M-FSDP shards parameters unevenly across DP ranks; params split at rank
    boundaries must be gathered before Newton-Schulz orthogonalization. Fully
    local params are orthogonalized without any collective.
    """

    def __init__(
        self,
        params: ParamsT,
        dp_group: torch.distributed.ProcessGroup | None = None,
        fsdp_batched_all_gather: bool = False,
        fsdp_flat_batched_all_gather: bool = False,
        fsdp_flat_batched_all_gather_nonempty_group: bool = False,
        fsdp_reuse_gather_scratch: bool = False,
        fsdp_padded_all_gather: bool = False,
        fsdp_padded_all_gather_pad_factor: float = 1.25,
        fsdp_batch_max_gather_bytes: int = 1024 * 1024 * 1024,
        fsdp_padded_all_gather_zero_pad: bool = True,
        fsdp_fused_async_gather_repack: bool = False,
        fsdp_boundary_pre_ns_into_gather_buffer: bool = False,
        fsdp_boundary_batch_sort_by_size: bool = False,
        fsdp_fast_reconstruct: bool = True,
        fsdp_boundary_gather_dtype: str = "fp32",
        fsdp_distributed_ns: bool = False,
        fsdp_distributed_ns_min_numel: int = 0,
        fsdp_distributed_ns_small_col_dim: int = 0,
        fsdp_distributed_ns_single_all_reduce: bool = False,
        fsdp_distributed_ns_gram_refresh_interval: int = 1,
        fsdp_distributed_ns_exclude_qkv: bool = False,
        fsdp_distributed_ns_nonempty_group: bool = False,
        fsdp_partial_distributed_ns: bool = False,
        fsdp_defer_distributed_ns_under_gather: bool = False,
        fsdp_defer_partial_distributed_ns_under_gather: bool = False,
        fsdp_async_partial_distributed_gather: bool = False,
        fsdp_prioritize_distributed_ns: bool = False,
        fsdp_approx_distributed_ns_update: bool = False,
        fsdp_approx_local_boundary_update: bool = False,
        fsdp_approx_local_boundary_full_shape_scale: bool = False,
        fsdp_approx_local_boundary_exclude_qkv: bool = False,
        fsdp_approx_local_boundary_max_local_numel: int = 0,
        fsdp_approx_local_boundary_global_norm_scale: bool = False,
        fsdp_approx_local_boundary_foreach_norm: bool = False,
        fsdp_approx_local_boundary_flat_norm_all_reduce: bool = False,
        fsdp_approx_local_boundary_async_norm_all_reduce: bool = False,
        fsdp_overlap_local_ns_first: bool = False,
        fsdp_overlap_comm_compute: bool = False,
        fsdp_overlap_boundary_ready_event: bool = False,
        fsdp_overlap_boundary_prefetch_batches: int = 1,
        fsdp_overlap_boundary_post_compute_prefetch_batches: int = 0,
        fsdp_overlap_boundary_progress_during_local: bool = False,
        fsdp_overlap_defer_boundary_batch_size: int = 1,
        fsdp_batched_newton_schulz: bool = False,
        fsdp_batched_newton_schulz_max_numel: int = 16 * 1024 * 1024,
        fsdp_batched_newton_schulz_max_batch_bytes: int = 2 * 1024 * 1024 * 1024,
        fsdp_batched_distributed_newton_schulz_max_batch_bytes: int = 0,
        fsdp_foreach_pre_ns: bool = False,
        fsdp_foreach_weight_update: bool = False,
        fsdp_foreach_gather_weight_update: bool = False,
        **kwargs: Any,
    ) -> None:
        assert _HAVE_DTENSOR, (
            "[Megatron-FSDP] torch.distributed.tensor.DTensor "
            f"is required to use {type(self).__name__}."
        )
        self.dp_group = dp_group
        self.fsdp_batched_all_gather = fsdp_batched_all_gather
        self.fsdp_flat_batched_all_gather = fsdp_flat_batched_all_gather
        self.fsdp_flat_batched_all_gather_nonempty_group = (
            fsdp_flat_batched_all_gather_nonempty_group
        )
        self.fsdp_reuse_gather_scratch = fsdp_reuse_gather_scratch
        self.fsdp_padded_all_gather = fsdp_padded_all_gather
        self.fsdp_padded_all_gather_pad_factor = fsdp_padded_all_gather_pad_factor
        self.fsdp_batch_max_gather_bytes = fsdp_batch_max_gather_bytes
        self.fsdp_padded_all_gather_zero_pad = fsdp_padded_all_gather_zero_pad
        self.fsdp_fused_async_gather_repack = fsdp_fused_async_gather_repack
        self.fsdp_boundary_pre_ns_into_gather_buffer = fsdp_boundary_pre_ns_into_gather_buffer
        self.fsdp_boundary_batch_sort_by_size = fsdp_boundary_batch_sort_by_size
        self.fsdp_fast_reconstruct = fsdp_fast_reconstruct
        supported_boundary_gather_dtypes = ("fp32", "bf16", "int8", "fp8_e4m3fn", "fp8_e5m2")
        if fsdp_boundary_gather_dtype not in supported_boundary_gather_dtypes:
            raise ValueError(
                "fsdp_boundary_gather_dtype must be one of "
                f"{supported_boundary_gather_dtypes}, "
                f"got {fsdp_boundary_gather_dtype!r}."
            )
        self.fsdp_boundary_gather_dtype = fsdp_boundary_gather_dtype
        self.fsdp_distributed_ns = fsdp_distributed_ns
        self.fsdp_distributed_ns_min_numel = fsdp_distributed_ns_min_numel
        self.fsdp_distributed_ns_small_col_dim = fsdp_distributed_ns_small_col_dim
        self.fsdp_distributed_ns_single_all_reduce = fsdp_distributed_ns_single_all_reduce
        self.fsdp_distributed_ns_gram_refresh_interval = max(
            1, fsdp_distributed_ns_gram_refresh_interval
        )
        self.fsdp_distributed_ns_exclude_qkv = fsdp_distributed_ns_exclude_qkv
        self.fsdp_distributed_ns_nonempty_group = fsdp_distributed_ns_nonempty_group
        self.fsdp_partial_distributed_ns = fsdp_partial_distributed_ns
        self.fsdp_defer_distributed_ns_under_gather = fsdp_defer_distributed_ns_under_gather
        self.fsdp_defer_partial_distributed_ns_under_gather = (
            fsdp_defer_partial_distributed_ns_under_gather
        )
        self.fsdp_async_partial_distributed_gather = fsdp_async_partial_distributed_gather
        self.fsdp_prioritize_distributed_ns = fsdp_prioritize_distributed_ns
        self.fsdp_approx_distributed_ns_update = fsdp_approx_distributed_ns_update
        self.fsdp_approx_local_boundary_update = fsdp_approx_local_boundary_update
        self.fsdp_approx_local_boundary_full_shape_scale = (
            fsdp_approx_local_boundary_full_shape_scale
        )
        self.fsdp_approx_local_boundary_exclude_qkv = fsdp_approx_local_boundary_exclude_qkv
        self.fsdp_approx_local_boundary_max_local_numel = max(
            0, fsdp_approx_local_boundary_max_local_numel
        )
        self.fsdp_approx_local_boundary_global_norm_scale = (
            fsdp_approx_local_boundary_global_norm_scale
        )
        self.fsdp_approx_local_boundary_foreach_norm = fsdp_approx_local_boundary_foreach_norm
        self.fsdp_approx_local_boundary_flat_norm_all_reduce = (
            fsdp_approx_local_boundary_flat_norm_all_reduce
        )
        self.fsdp_approx_local_boundary_async_norm_all_reduce = (
            fsdp_approx_local_boundary_async_norm_all_reduce
        )
        self.fsdp_overlap_local_ns_first = fsdp_overlap_local_ns_first
        self.fsdp_overlap_comm_compute = fsdp_overlap_comm_compute
        self.fsdp_overlap_boundary_ready_event = fsdp_overlap_boundary_ready_event
        self.fsdp_overlap_boundary_prefetch_batches = max(1, fsdp_overlap_boundary_prefetch_batches)
        self.fsdp_overlap_boundary_post_compute_prefetch_batches = max(
            0, fsdp_overlap_boundary_post_compute_prefetch_batches
        )
        self.fsdp_overlap_boundary_progress_during_local = (
            fsdp_overlap_boundary_progress_during_local
        )
        self.fsdp_overlap_defer_boundary_batch_size = max(1, fsdp_overlap_defer_boundary_batch_size)
        self.fsdp_batched_newton_schulz = fsdp_batched_newton_schulz
        self.fsdp_batched_newton_schulz_max_numel = fsdp_batched_newton_schulz_max_numel
        self.fsdp_batched_newton_schulz_max_batch_bytes = fsdp_batched_newton_schulz_max_batch_bytes
        self.fsdp_batched_distributed_newton_schulz_max_batch_bytes = (
            fsdp_batched_distributed_newton_schulz_max_batch_bytes
            or fsdp_batched_newton_schulz_max_batch_bytes
        )
        self.fsdp_foreach_pre_ns = fsdp_foreach_pre_ns
        self.fsdp_foreach_weight_update = fsdp_foreach_weight_update
        self.fsdp_foreach_gather_weight_update = fsdp_foreach_gather_weight_update
        self._boundary_gather_indices_cache: dict[tuple[int, ...], set[int]] = {}
        self._uneven_gather_plan_cache: dict[int, dict[str, Any]] = {}
        self._partial_uneven_gather_plan_cache: dict[
            tuple[int, tuple[int, ...]], dict[str, Any] | None
        ] = {}
        self._flat_uneven_gather_plan_cache: dict[tuple[int, int], dict[str, Any] | None] = {}
        self._flat_nonempty_group_cache: dict[tuple[int, ...], Any] = {}
        self._fsdp_nonempty_group_cache: dict[tuple[int, ...], Any] = {}
        self._fsdp_gather_scratch_cache: dict[tuple[Any, ...], Any] = {}
        self._fsdp_gather_scratch_scope: Any = None
        self._fsdp_comm_stream_cache: dict[torch.device, torch.cuda.Stream] = {}
        self._fsdp_gather_summary_logged = False
        self._fsdp_update_mode_summary_logged = False
        self._fsdp_boundary_layout_summary_logged = False
        self._fsdp_batched_ns_summary_logged_modes: set[str] = set()
        super().__init__(params, **kwargs)

    def _fsdp_diagnostic_rank_info(self) -> tuple[int | None, int | None]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None, None
        return torch.distributed.get_rank(), torch.distributed.get_world_size()

    def _should_print_fsdp_diagnostic(self) -> bool:
        rank, world_size = self._fsdp_diagnostic_rank_info()
        if rank is None or world_size is None:
            return True

        selected_ranks = {0, 1, world_size - 1}
        if world_size > 127:
            selected_ranks.add(127)

        rank_spec = os.environ.get("MUON_FSDP_DIAGNOSTIC_RANKS")
        if rank_spec:
            selected_ranks.clear()
            for item in rank_spec.split(","):
                item = item.strip().lower()
                if not item:
                    continue
                if item == "all":
                    return True
                if item == "last":
                    selected_ranks.add(world_size - 1)
                    continue
                try:
                    selected_ranks.add(int(item))
                except ValueError:
                    continue
        return rank in selected_ranks

    def _fsdp_diagnostic_prefix(self) -> str:
        rank, world_size = self._fsdp_diagnostic_rank_info()
        if rank is None or world_size is None:
            return ""
        return f"[rank {rank}/{world_size}] "

    def _maybe_print_fsdp_diagnostic(self, message) -> None:
        if self._should_print_fsdp_diagnostic():
            print(f"{self._fsdp_diagnostic_prefix()}{message}", flush=True)  # pylint: disable=W0141

    def _fsdp_gather_scratch_cache_bytes(self) -> int:
        total = 0
        for value in self._fsdp_gather_scratch_cache.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
            elif isinstance(value, list):
                total += sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in value
                    if isinstance(tensor, torch.Tensor)
                )
        return total

    def clear_fsdp_gather_scratch_cache(self) -> None:
        """Clear the gather scratch buffer."""
        self._fsdp_gather_scratch_cache.clear()

    def _get_fsdp_comm_stream(self, device: torch.device) -> torch.cuda.Stream:
        cached = self._fsdp_comm_stream_cache.get(device)
        if cached is None:
            with torch.cuda.device(device):
                cached = torch.cuda.Stream()
            self._fsdp_comm_stream_cache[device] = cached
        return cached

    def _get_fsdp_gather_scratch_tensor(
        self, key: tuple[Any, ...], numel: int, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if not self.fsdp_reuse_gather_scratch:
            return torch.empty(numel, dtype=dtype, device=device)

        if self._fsdp_gather_scratch_scope is not None:
            key = ("scope", self._fsdp_gather_scratch_scope, *key)
        cached = self._fsdp_gather_scratch_cache.get(key)
        if (
            not isinstance(cached, torch.Tensor)
            or cached.numel() < numel
            or cached.dtype != dtype
            or cached.device != device
        ):
            cached = torch.empty(numel, dtype=dtype, device=device)
            self._fsdp_gather_scratch_cache[key] = cached
        return cached[:numel]

    def _get_uneven_group_tensors(
        self,
        rank_numels: list[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
        shard_group: torch.distributed.ProcessGroup,
    ) -> list[torch.Tensor]:
        if not self.fsdp_reuse_gather_scratch:
            return [torch.empty(numel, dtype=dtype, device=device) for numel in rank_numels]

        key = ("uneven_group_tensors", id(shard_group), dtype, device, tuple(rank_numels))
        if self._fsdp_gather_scratch_scope is not None:
            key = ("scope", self._fsdp_gather_scratch_scope, *key)
        cached = self._fsdp_gather_scratch_cache.get(key)
        if not isinstance(cached, list):
            cached = [torch.empty(numel, dtype=dtype, device=device) for numel in rank_numels]
            self._fsdp_gather_scratch_cache[key] = cached
        return cached

    def _with_fsdp_gather_scratch_scope(self, scope: Any):
        optimizer = self

        class _ScratchScope:
            def __enter__(self):
                self.previous_scope = optimizer._fsdp_gather_scratch_scope
                optimizer._fsdp_gather_scratch_scope = scope

            def __exit__(self, exc_type, exc, tb):
                optimizer._fsdp_gather_scratch_scope = self.previous_scope

        return _ScratchScope()

    def _candidate_batch_gather_bytes(
        self, stage_rank_total_numels: list[list[int]], plan: dict[str, Any], *, element_size: int
    ) -> int:
        max_stage_bytes = 0
        for stage_idx, stage in enumerate(plan["stages"]):
            candidate_rank_total_numels = [
                stage_rank_total_numels[stage_idx][rank] + stage["rank_numels"][rank]
                for rank in range(len(stage["rank_numels"]))
            ]
            if self.fsdp_padded_all_gather:
                gathered_numel = max(candidate_rank_total_numels) * len(candidate_rank_total_numels)
            else:
                gathered_numel = sum(candidate_rank_total_numels)
            max_stage_bytes = max(max_stage_bytes, gathered_numel * element_size)
        return max_stage_bytes

    def _candidate_rank_batch_gather_bytes(
        self, rank_total_numels: list[int], candidate_rank_numels: list[int], *, element_size: int
    ) -> int:
        candidate_rank_total_numels = [
            total_numel + candidate_numel
            for total_numel, candidate_numel in zip(rank_total_numels, candidate_rank_numels)
        ]
        if self.fsdp_padded_all_gather:
            gathered_numel = max(candidate_rank_total_numels) * len(candidate_rank_total_numels)
        else:
            gathered_numel = sum(candidate_rank_total_numels)
        return gathered_numel * element_size

    def _rank_batch_gather_bytes(
        self, rank_total_numels: list[int], *, element_size: int, use_padded: bool
    ) -> int:
        if use_padded:
            gathered_numel = max(rank_total_numels) * len(rank_total_numels)
        else:
            gathered_numel = sum(rank_total_numels)
        return gathered_numel * element_size

    def _planned_gather_batch_bytes(self, batch: dict[str, Any]) -> tuple[int, int]:
        element_size = batch.get("element_size")
        if element_size is None:
            element_size = torch.empty((), dtype=batch["dtype"]).element_size()

        if batch.get("is_flat", False):
            rank_total_numels = batch.get(
                "flat_collective_rank_total_numels", batch["flat_rank_total_numels"]
            )
            gather_bytes = self._rank_batch_gather_bytes(
                rank_total_numels,
                element_size=element_size,
                use_padded=self._batch_uses_padded_all_gather(rank_total_numels),
            )
            return gather_bytes, gather_bytes

        max_gather_bytes = 0
        total_gather_bytes = 0
        for rank_total_numels in batch["stage_rank_total_numels"]:
            gather_bytes = self._rank_batch_gather_bytes(
                rank_total_numels,
                element_size=element_size,
                use_padded=self._batch_uses_padded_all_gather(rank_total_numels),
            )
            max_gather_bytes = max(max_gather_bytes, gather_bytes)
            total_gather_bytes += gather_bytes
        return max_gather_bytes, total_gather_bytes

    def _sort_gather_batches_by_size(self, key_batches: list[dict[str, Any]]) -> None:
        if not self.fsdp_boundary_batch_sort_by_size:
            return
        key_batches.sort(
            key=lambda batch: tuple(-value for value in self._planned_gather_batch_bytes(batch))
        )

    def _boundary_gather_wire_element_size(self, fallback_dtype: torch.dtype) -> int:
        if self.fsdp_boundary_gather_dtype in ("int8", "fp8_e4m3fn", "fp8_e5m2"):
            return 1
        if self.fsdp_boundary_gather_dtype == "bf16":
            return torch.empty((), dtype=torch.bfloat16).element_size()
        return torch.empty((), dtype=fallback_dtype).element_size()

    def _batch_uses_padded_all_gather(self, rank_total_numels: list[int]) -> bool:
        return (
            self.fsdp_padded_all_gather
            and hasattr(torch.distributed, "all_gather_into_tensor")
            and _should_use_padded_all_gather(
                rank_total_numels, self.fsdp_padded_all_gather_pad_factor
            )
        )

    def _maybe_log_fsdp_gather_batch_summary(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        batches: list[dict[str, Any]],
        *,
        completed_without_batch: int,
    ) -> None:
        if self._fsdp_gather_summary_logged:
            return
        self._fsdp_gather_summary_logged = True

        flat_batches = sum(1 for batch in batches if batch.get("is_flat", False))
        item_counts = [len(batch["item_indices"]) for batch in batches]
        stage_collectives = 0
        padded_collectives = 0
        total_gather_bytes = 0
        max_gather_bytes = 0
        for batch in batches:
            element_size = batch.get("element_size")
            if element_size is None:
                element_size = torch.empty((), dtype=batch["dtype"]).element_size()
            if batch.get("is_flat", False):
                rank_total_numels = batch.get(
                    "flat_collective_rank_total_numels", batch["flat_rank_total_numels"]
                )
                use_padded = self._batch_uses_padded_all_gather(rank_total_numels)
                gather_bytes = self._rank_batch_gather_bytes(
                    rank_total_numels, element_size=element_size, use_padded=use_padded
                )
                stage_collectives += 1
                padded_collectives += int(use_padded)
                total_gather_bytes += gather_bytes
                max_gather_bytes = max(max_gather_bytes, gather_bytes)
                continue

            for rank_total_numels in batch["stage_rank_total_numels"]:
                use_padded = self._batch_uses_padded_all_gather(rank_total_numels)
                gather_bytes = self._rank_batch_gather_bytes(
                    rank_total_numels, element_size=element_size, use_padded=use_padded
                )
                stage_collectives += 1
                padded_collectives += int(use_padded)
                total_gather_bytes += gather_bytes
                max_gather_bytes = max(max_gather_bytes, gather_bytes)

        if item_counts:
            min_items = min(item_counts)
            max_items = max(item_counts)
            avg_items = sum(item_counts) / len(item_counts)
        else:
            min_items = max_items = 0
            avg_items = 0.0
        wire_element_size = (
            self._boundary_gather_wire_element_size(items[0][1].dtype) if items else 0
        )

        boundary_shape_counts: dict[tuple[tuple[int, ...], tuple[int, ...], torch.dtype], int] = {}
        for param, local_tensor in items:
            shape_key = (tuple(param.shape), tuple(local_tensor.shape), local_tensor.dtype)
            boundary_shape_counts[shape_key] = boundary_shape_counts.get(shape_key, 0) + 1
        top_boundary_shapes = sorted(
            boundary_shape_counts.items(), key=lambda item: item[1], reverse=True
        )[:8]
        top_boundary_shapes_text = ", ".join(
            f"full={full_shape} local={local_shape} dtype={dtype}: {count}"
            for (full_shape, local_shape, dtype), count in top_boundary_shapes
        )

        message = (
            "Muon+M-FSDP gather summary: "
            f"boundary_items={len(items)}, completed_without_batch={completed_without_batch}, "
            f"batches={len(batches)} (flat={flat_batches}, staged={len(batches) - flat_batches}), "
            f"batch_items_min/avg/max={min_items}/{avg_items:.1f}/{max_items}, "
            f"collectives={stage_collectives}, padded_collectives={padded_collectives}, "
            f"total_planned_gather_gib={total_gather_bytes / (1024 ** 3):.2f}, "
            f"max_collective_mib={max_gather_bytes / (1024 ** 2):.1f}, "
            f"max_gather_gib={self.fsdp_batch_max_gather_bytes / (1024 ** 3):.2f}, "
            f"boundary_gather_dtype={self.fsdp_boundary_gather_dtype}, "
            f"wire_element_size={wire_element_size}, "
            f"flat={self.fsdp_flat_batched_all_gather}, "
            f"flat_nonempty_group={self.fsdp_flat_batched_all_gather_nonempty_group}, "
            f"fused_async_repack={self.fsdp_fused_async_gather_repack}, "
            "boundary_pre_ns_into_gather_buffer="
            f"{self.fsdp_boundary_pre_ns_into_gather_buffer}, "
            f"batch_sort_by_size={self.fsdp_boundary_batch_sort_by_size}, "
            f"overlap={self.fsdp_overlap_comm_compute}, "
            f"overlap_local_ns_first={self.fsdp_overlap_local_ns_first}, "
            f"overlap_boundary_ready_event={self.fsdp_overlap_boundary_ready_event}, "
            f"prefetch_batches={self.fsdp_overlap_boundary_prefetch_batches}, "
            "post_compute_prefetch_batches="
            f"{self.fsdp_overlap_boundary_post_compute_prefetch_batches}, "
            "progress_during_local="
            f"{self.fsdp_overlap_boundary_progress_during_local}, "
            f"defer_boundary_batch_size={self.fsdp_overlap_defer_boundary_batch_size}, "
            f"top_boundary_shapes=[{top_boundary_shapes_text}].",
        )
        log_single_rank(logger, logging.INFO, message)
        self._maybe_print_fsdp_diagnostic(message)

    def _gather_collective_nvtx_label(self, stage_state: dict[str, Any]) -> str:
        batch = stage_state["batch"]
        kind = "flat" if batch.get("is_flat", False) else f"stage{stage_state['stage_idx']}"
        mode = "padded" if stage_state["use_padded_all_gather"] else "uneven"
        rank_total_numels = stage_state.get(
            "collective_rank_total_numels", stage_state["rank_total_numels"]
        )
        gather_bytes = self._rank_batch_gather_bytes(
            rank_total_numels,
            element_size=stage_state["local_buffer"].element_size(),
            use_padded=stage_state["use_padded_all_gather"],
        )
        active_text = ""
        if "collective_rank_indices" in stage_state:
            active_text = (
                f" active={len(stage_state['collective_rank_indices'])}/"
                f"{stage_state['group_size']}"
            )
        return (
            f"Muon-FSDP gather collective launch {kind} {mode} "
            f"items={len(batch['item_indices'])}{active_text} "
            f"MiB={gather_bytes / (1024 ** 2):.1f}"
        )

    def _maybe_log_fsdp_update_mode_summary(self, all_updates: list) -> None:
        if self._fsdp_update_mode_summary_logged:
            return
        self._fsdp_update_mode_summary_logged = True
        mode_counts = {"local": 0, "gather": 0, "distributed": 0, "partial_distributed": 0}
        mode_numels = {"local": 0, "gather": 0, "distributed": 0, "partial_distributed": 0}
        qkv_counts = {"local": 0, "gather": 0, "distributed": 0, "partial_distributed": 0}
        named_qkv_counts = {"local": 0, "gather": 0, "distributed": 0, "partial_distributed": 0}
        attr_qkv_counts = {"local": 0, "gather": 0, "distributed": 0, "partial_distributed": 0}
        update_shape_counts: dict[tuple[str, tuple[int, ...]], int] = {}
        partial_distributed_candidate_count = 0
        partial_distributed_candidate_numels = 0
        partial_distributed_shape_counts: dict[
            tuple[tuple[int, ...], tuple[int, ...], int, tuple[tuple[int, int, int], ...]], int
        ] = {}
        boundary_full_shape_counts: dict[tuple[str, tuple[int, ...], tuple[int, ...]], int] = {}
        for param, pre_ns_grad, update_mode, _, _ in all_updates:
            mode_counts[update_mode] = mode_counts.get(update_mode, 0) + 1
            tensor = pre_ns_grad if pre_ns_grad is not None else param._local_tensor
            mode_numels[update_mode] = mode_numels.get(update_mode, 0) + tensor.numel()
            shape_key = (update_mode, tuple(tensor.shape))
            update_shape_counts[shape_key] = update_shape_counts.get(shape_key, 0) + 1
            if update_mode in ("gather", "local_boundary", "partial_distributed"):
                boundary_key = (update_mode, tuple(param.shape), tuple(tensor.shape))
                boundary_full_shape_counts[boundary_key] = (
                    boundary_full_shape_counts.get(boundary_key, 0) + 1
                )
            if getattr(param, "is_qkv", False) or getattr(
                getattr(param, "orig_param", None), "is_qkv", False
            ):
                attr_qkv_counts[update_mode] = attr_qkv_counts.get(update_mode, 0) + 1
            if _is_named_qkv_param(param):
                named_qkv_counts[update_mode] = named_qkv_counts.get(update_mode, 0) + 1
            if self._is_split_qkv_param(param):
                qkv_counts[update_mode] = qkv_counts.get(update_mode, 0) + 1
            candidate_key = self._partial_distributed_ns_candidate_key(param, tensor, update_mode)
            if candidate_key is not None:
                partial_distributed_candidate_count += 1
                partial_distributed_candidate_numels += tensor.numel()
                partial_distributed_shape_counts[candidate_key] = (
                    partial_distributed_shape_counts.get(candidate_key, 0) + 1
                )

        top_update_shapes = sorted(
            update_shape_counts.items(), key=lambda item: item[1], reverse=True
        )[:8]
        top_update_shapes_text = ", ".join(
            f"{mode}{shape}: {count}" for (mode, shape), count in top_update_shapes
        )
        top_boundary_shapes = sorted(
            boundary_full_shape_counts.items(), key=lambda item: item[1], reverse=True
        )[:8]
        top_boundary_shapes_text = ", ".join(
            f"{mode} full={full_shape} local={local_shape}: {count}"
            for (mode, full_shape, local_shape), count in top_boundary_shapes
        )
        top_partial_distributed_shapes = sorted(
            partial_distributed_shape_counts.items(), key=lambda item: item[1], reverse=True
        )[:8]
        top_partial_distributed_shapes_text = ", ".join(
            "full="
            f"{full_shape} local={local_shape} partition_dim={partition_dim} "
            f"placements={placement_signature}: {count}"
            for (
                full_shape,
                local_shape,
                partition_dim,
                placement_signature,
            ), count in top_partial_distributed_shapes
        )

        message = (
            "Muon+M-FSDP update mode summary: "
            f"local={mode_counts.get('local', 0)} "
            f"({mode_numels.get('local', 0) / 1e9:.3f}B local elems), "
            f"gather={mode_counts.get('gather', 0)} "
            f"({mode_numels.get('gather', 0) / 1e9:.3f}B local elems), "
            f"distributed_ns={mode_counts.get('distributed', 0)} "
            f"({mode_numels.get('distributed', 0) / 1e9:.3f}B local elems), "
            f"partial_distributed_ns={mode_counts.get('partial_distributed', 0)} "
            f"({mode_numels.get('partial_distributed', 0) / 1e9:.3f}B local elems), "
            f"approx_local_boundary={mode_counts.get('local_boundary', 0)} "
            f"({mode_numels.get('local_boundary', 0) / 1e9:.3f}B local elems), "
            f"qkv_local/gather/distributed/partial={qkv_counts.get('local', 0)}/"
            f"{qkv_counts.get('gather', 0)}/{qkv_counts.get('distributed', 0)}/"
            f"{qkv_counts.get('partial_distributed', 0)}, "
            f"qkv_attr_local/gather/distributed/partial={attr_qkv_counts.get('local', 0)}/"
            f"{attr_qkv_counts.get('gather', 0)}/{attr_qkv_counts.get('distributed', 0)}/"
            f"{attr_qkv_counts.get('partial_distributed', 0)}, "
            f"qkv_named_local/gather/distributed/partial={named_qkv_counts.get('local', 0)}/"
            f"{named_qkv_counts.get('gather', 0)}/{named_qkv_counts.get('distributed', 0)}/"
            f"{named_qkv_counts.get('partial_distributed', 0)}, "
            f"distributed_ns_enabled={self.fsdp_distributed_ns}, "
            f"partial_distributed_ns_enabled={self.fsdp_partial_distributed_ns}, "
            f"distributed_ns_single_all_reduce={self.fsdp_distributed_ns_single_all_reduce}, "
            "distributed_ns_gram_refresh_interval="
            f"{self.fsdp_distributed_ns_gram_refresh_interval}, "
            f"distributed_ns_exclude_qkv={self.fsdp_distributed_ns_exclude_qkv}, "
            f"distributed_ns_nonempty_group={self.fsdp_distributed_ns_nonempty_group}, "
            f"defer_distributed_ns_under_gather={self.fsdp_defer_distributed_ns_under_gather}, "
            "defer_partial_distributed_ns_under_gather="
            f"{self.fsdp_defer_partial_distributed_ns_under_gather}, "
            f"prioritize_distributed_ns={self.fsdp_prioritize_distributed_ns}, "
            f"approx_distributed_ns_update={self.fsdp_approx_distributed_ns_update}, "
            f"approx_local_boundary_update={self.fsdp_approx_local_boundary_update}, "
            "approx_local_boundary_full_shape_scale="
            f"{self.fsdp_approx_local_boundary_full_shape_scale}, "
            "approx_local_boundary_exclude_qkv="
            f"{self.fsdp_approx_local_boundary_exclude_qkv}, "
            "approx_local_boundary_max_local_numel="
            f"{self.fsdp_approx_local_boundary_max_local_numel}, "
            "approx_local_boundary_global_norm_scale="
            f"{self.fsdp_approx_local_boundary_global_norm_scale}, "
            "approx_local_boundary_foreach_norm="
            f"{self.fsdp_approx_local_boundary_foreach_norm}, "
            "approx_local_boundary_flat_norm_all_reduce="
            f"{self.fsdp_approx_local_boundary_flat_norm_all_reduce}, "
            "approx_local_boundary_async_norm_all_reduce="
            f"{self.fsdp_approx_local_boundary_async_norm_all_reduce}, "
            f"foreach_pre_ns={self.fsdp_foreach_pre_ns}, "
            f"distributed_ns_small_col_dim={self.fsdp_distributed_ns_small_col_dim}, "
            f"distributed_ns_min_numel={self.fsdp_distributed_ns_min_numel}, "
            f"partial_distributed_candidates={partial_distributed_candidate_count} "
            f"({partial_distributed_candidate_numels / 1e9:.3f}B local elems), "
            f"top_partial_distributed_candidates=[{top_partial_distributed_shapes_text}], "
            f"top_boundary_shapes=[{top_boundary_shapes_text}], "
            f"top_update_shapes=[{top_update_shapes_text}]."
        )
        log_single_rank(logger, logging.INFO, message)
        self._maybe_print_fsdp_diagnostic(message)

    def _partial_distributed_ns_candidate_key(
        self, param: torch.Tensor, local_tensor: torch.Tensor, update_mode: str
    ) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[tuple[int, int, int], ...]] | None:
        if update_mode != "gather" or not self.fsdp_distributed_ns:
            return None
        if _DTensor is None or not isinstance(param, _DTensor):
            return None
        if len(param.shape) != 2:
            return None

        rows, cols = int(param.shape[-2]), int(param.shape[-1])
        partition_dim = 0 if rows > cols else 1
        placement_signature = []
        has_partition_shard = False
        has_other_matrix_shard = False
        shard_placement_types = (Shard, _StridedShard)
        for mesh_dim, placement in enumerate(param.placements):
            if isinstance(placement, Replicate):
                continue
            if not isinstance(placement, shard_placement_types):
                return None
            shard_dim = getattr(placement, "dim", None)
            if shard_dim not in (0, 1):
                return None
            try:
                group_size = get_pg_size(param.device_mesh.get_group(mesh_dim))
            except (RuntimeError, ValueError, TypeError, AttributeError):
                group_size = -1
            placement_signature.append((int(mesh_dim), int(shard_dim), int(group_size)))
            if shard_dim == partition_dim:
                has_partition_shard = True
            else:
                has_other_matrix_shard = True

        if not has_partition_shard or not has_other_matrix_shard:
            return None

        return (
            tuple(int(dim) for dim in param.shape),
            tuple(int(dim) for dim in local_tensor.shape),
            partition_dim,
            tuple(placement_signature),
        )

    def _maybe_log_mfsdp_boundary_layout_summary(
        self, params: list[torch.Tensor], boundary_indices: set[int]
    ) -> None:
        if self._fsdp_boundary_layout_summary_logged:
            return
        self._fsdp_boundary_layout_summary_logged = True

        layout_count = 0
        distributed_count = 0
        selected_count = 0
        flat_cross_count = 0
        selected_flat_cross_count = 0
        selected_local_partial_count = 0
        selected_item_le_shard_count = 0
        selected_item_gt_shard_count = 0
        selected_aligned_start_count = 0
        max_span_shards = 0
        max_item_mib = 0.0
        shape_counts: dict[
            tuple[tuple[int, ...], tuple[int, ...], bool, bool, bool], dict[str, Any]
        ] = {}

        for idx, param in enumerate(params):
            layout = self._get_mfsdp_param_layout(param, idx)
            if layout is None:
                continue
            layout_count += 1

            gbuf, item_index, bucket_index, shard_bucket_index = layout
            if not getattr(gbuf, "is_data_distributed", True):
                continue
            distributed_count += 1

            item_size = int(item_index.size)
            if item_size <= 0:
                continue
            bucket_start = int(bucket_index.global_data_index)
            bucket_size = int(bucket_index.size)
            shard_size = int(shard_bucket_index.size)
            if shard_size <= 0:
                continue

            item_start = int(item_index.global_data_index)
            item_end = item_start + item_size
            first_shard = (item_start - bucket_start) // shard_size
            last_shard = (item_end - 1 - bucket_start) // shard_size
            span_shards = int(last_shard - first_shard + 1)
            crosses_boundary = first_shard != last_shard
            selected = idx in boundary_indices
            local_tensor = param._local_tensor
            local_partial = local_tensor.numel() > 0 and tuple(param.shape) != tuple(
                local_tensor.shape
            )
            item_le_shard = item_size <= shard_size
            aligned_start = (item_start - bucket_start) % shard_size == 0

            flat_cross_count += int(crosses_boundary)
            selected_count += int(selected)
            if selected:
                selected_flat_cross_count += int(crosses_boundary)
                selected_local_partial_count += int(local_partial)
                selected_item_le_shard_count += int(item_le_shard)
                selected_item_gt_shard_count += int(not item_le_shard)
                selected_aligned_start_count += int(aligned_start)
                max_span_shards = max(max_span_shards, span_shards)
                max_item_mib = max(
                    max_item_mib, item_size * local_tensor.element_size() / (1024**2)
                )

                key = (
                    tuple(param.shape),
                    tuple(local_tensor.shape),
                    crosses_boundary,
                    item_le_shard,
                    local_partial,
                )
                stat = shape_counts.setdefault(
                    key,
                    {
                        "count": 0,
                        "max_span": 0,
                        "min_item_mib": float("inf"),
                        "max_item_mib": 0.0,
                        "shard_mib": shard_size * local_tensor.element_size() / (1024**2),
                    },
                )
                stat["count"] += 1
                stat["max_span"] = max(stat["max_span"], span_shards)
                item_mib = item_size * local_tensor.element_size() / (1024**2)
                stat["min_item_mib"] = min(stat["min_item_mib"], item_mib)
                stat["max_item_mib"] = max(stat["max_item_mib"], item_mib)

        if layout_count == 0:
            return

        top_shapes = sorted(shape_counts.items(), key=lambda item: item[1]["count"], reverse=True)[
            :8
        ]
        top_shapes_text = ", ".join(
            (
                f"full={full_shape} local={local_shape} crosses={crosses} "
                f"item_le_shard={item_le_shard} local_partial={local_partial}: "
                f"count={stat['count']} item_mib={stat['min_item_mib']:.1f}-"
                f"{stat['max_item_mib']:.1f} shard_mib={stat['shard_mib']:.1f} "
                f"max_span={stat['max_span']}"
            )
            for (full_shape, local_shape, crosses, item_le_shard, local_partial), stat in top_shapes
        )

        message = (
            "Muon+M-FSDP boundary layout summary: "
            f"mfsdp_params={layout_count}, distributed_params={distributed_count}, "
            f"selected_boundary={selected_count}, flat_cross={flat_cross_count}, "
            f"selected_flat_cross={selected_flat_cross_count}, "
            f"selected_local_partial={selected_local_partial_count}, "
            f"selected_item_le_shard={selected_item_le_shard_count}, "
            f"selected_item_gt_shard={selected_item_gt_shard_count}, "
            f"selected_aligned_start={selected_aligned_start_count}, "
            f"max_selected_span_shards={max_span_shards}, "
            f"max_selected_item_mib={max_item_mib:.1f}, "
            f"top_selected_shapes=[{top_shapes_text}]."
        )
        log_single_rank(logger, logging.INFO, message)
        self._maybe_print_fsdp_diagnostic(message)

    def _process_group_size_if_member(
        self, group: torch.distributed.ProcessGroup | None
    ) -> int | None:
        if group is None:
            return None
        try:
            torch.distributed.get_rank(group)
            return get_pg_size(group)
        except (RuntimeError, ValueError):
            return None

    def _get_existing_flat_uneven_gather_group(
        self, dtensor_ref, plan: dict[str, Any]
    ) -> torch.distributed.ProcessGroup | None:
        product_size = 1
        for stage in plan["stages"]:
            product_size *= len(stage["rank_numels"])

        if self._process_group_size_if_member(self.dp_group) == product_size:
            return self.dp_group
        try:
            if torch.distributed.get_world_size() == product_size:
                return torch.distributed.group.WORLD
        except RuntimeError:
            pass

        mesh_dim_names = getattr(dtensor_ref.device_mesh, "mesh_dim_names", None)
        if mesh_dim_names is not None:
            shard_names = tuple(mesh_dim_names[dim] for dim in plan["shard_mesh_dims"])
            try:
                flat_mesh = dtensor_ref.device_mesh[shard_names]
                flat_group = flat_mesh.get_group()
            except (KeyError, RuntimeError, ValueError, TypeError, AttributeError):
                flat_group = None
            if self._process_group_size_if_member(flat_group) == product_size:
                return flat_group
        return None

    def _get_flat_nonempty_gather_group(
        self, flat_group: torch.distributed.ProcessGroup, active_rank_indices: tuple[int, ...]
    ) -> torch.distributed.ProcessGroup | None:
        """Return a cached subgroup containing only active flat-gather ranks.

        All ranks enter the WORLD all-gather and create the same sorted set of
        requested subgroups. This avoids rank-divergent NCCL communicator
        creation when different HSDP replicas need different active spans.
        """
        flat_group_size = get_pg_size(flat_group)
        group_ranks = tuple(torch.distributed.get_process_group_ranks(flat_group))
        if len(active_rank_indices) == flat_group_size:
            local_request: tuple[int, ...] = ()
        elif active_rank_indices:
            local_request = tuple(group_ranks[rank] for rank in active_rank_indices)
        else:
            local_request = ()

        world_size = torch.distributed.get_world_size()
        requests: list[tuple[int, ...] | None] = [None] * world_size
        torch.distributed.all_gather_object(
            requests, local_request, group=torch.distributed.group.WORLD
        )
        for requested_ranks in sorted({request for request in requests if request}):
            if requested_ranks not in self._flat_nonempty_group_cache:
                self._flat_nonempty_group_cache[requested_ranks] = torch.distributed.new_group(
                    ranks=list(requested_ranks)
                )

        if len(active_rank_indices) == flat_group_size:
            return flat_group
        if not active_rank_indices:
            return None

        return self._flat_nonempty_group_cache[local_request]

    def _get_dtensor_mesh_dims_group(
        self, dtensor_ref, mesh_dims: tuple[int, ...]
    ) -> torch.distributed.ProcessGroup | None:
        """Return the process group for a DTensor submesh, if this rank is a member."""
        if not mesh_dims:
            return None
        if len(mesh_dims) == 1:
            try:
                return dtensor_ref.device_mesh.get_group(mesh_dims[0])
            except (RuntimeError, ValueError, TypeError, AttributeError):
                return None

        product_size = 1
        for mesh_dim in mesh_dims:
            try:
                product_size *= get_pg_size(dtensor_ref.device_mesh.get_group(mesh_dim))
            except (RuntimeError, ValueError, TypeError, AttributeError):
                return None

        mesh_dim_names = getattr(dtensor_ref.device_mesh, "mesh_dim_names", None)
        if mesh_dim_names is None:
            return None
        dim_names = tuple(mesh_dim_names[dim] for dim in mesh_dims)
        try:
            mesh = dtensor_ref.device_mesh[dim_names]
            group = mesh.get_group()
        except (KeyError, RuntimeError, ValueError, TypeError, AttributeError):
            return None
        if self._process_group_size_if_member(group) != product_size:
            return None
        return group

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Callable | None = None) -> float | None:
        """Muon step for Megatron-FSDP ZeRO-1/2/3.

        Separates collective (AG) and local (NS) work into three phases so that
        no rank is blocked waiting on another rank computing NS to reach AG:
          1. Compute momentum updates locally for all params.
          2. All-gather boundary params — all collectives, no NS interleaved.
          3. Newton-Schulz + weight update locally for all params.
        """
        loss = None if closure is None else closure()

        if self.dp_group is None or get_pg_size(self.dp_group) == 1:
            with torch.autograd.profiler.record_function("Muon-FSDP local-only step"):
                for group in self.param_groups:
                    self._init_group(group, skip_non_grad_params=False)
                    for p in group["params"]:
                        grad = self._param_grad(p)
                        if grad is None:
                            continue
                        self._local_muon_update(p, grad, group)
            return loss

        overlap_enabled = self.fsdp_overlap_comm_compute and self.fsdp_batched_all_gather

        # Track all parameters to update ordered by parameter group index.
        # (param, pre_ns_grad, update_mode, lr, group_kwargs), where
        # update_mode is "local", "gather", or "distributed".
        all_updates: list = []

        group_contexts = []
        with torch.autograd.profiler.record_function("Muon-FSDP phase 0 boundary planning"):
            for group in self.param_groups:
                self._init_group(group, skip_non_grad_params=False)
                group_contexts.append(
                    (
                        group,
                        self._get_boundary_gather_param_indices(group),
                        group["lr"],
                        {k: v for k, v in group.items() if k != "params"},
                    )
                )

        early_gather_state = None
        boundary_update_indices = []
        direct_boundary_pre_ns = (
            self.fsdp_boundary_pre_ns_into_gather_buffer
            and self.fsdp_boundary_gather_dtype not in ("int8", "fp8_e4m3fn", "fp8_e5m2")
            and not self.fsdp_flat_batched_all_gather
        )
        if overlap_enabled:
            # Boundary pre-NS tensors are the only values needed by the
            # all-gather. Compute them first and launch communication before
            # spending time on non-boundary momentum and NS work.
            with torch.autograd.profiler.record_function("Muon-FSDP phase 1a boundary pre-NS"):
                for group, gather_param_indices, lr, group_kwargs in group_contexts:
                    for param_idx, p in enumerate(group["params"]):
                        if param_idx not in gather_param_indices:
                            continue
                        if self._fsdp_boundary_update_mode(p) != "gather":
                            continue
                        pre_ns_grad = (
                            None
                            if direct_boundary_pre_ns
                            else self._compute_local_pre_ns_grad(p, group, lr)
                        )
                        boundary_update_indices.append(len(all_updates))
                        all_updates.append((p, pre_ns_grad, "gather", lr, group_kwargs))

            if boundary_update_indices:
                with torch.autograd.profiler.record_function(
                    "Muon-FSDP phase 2a start overlapped boundary gather"
                ):
                    early_gather_state = self._start_overlap_boundary_gathers(
                        all_updates, boundary_update_indices, direct_pre_ns=direct_boundary_pre_ns
                    )

        from emerging_optimizers import utils

        early_applied_update_indices: set[int] = set()

        def append_pre_ns_updates(*, local_only: bool | None) -> list[int]:
            appended_indices = []
            foreach_candidates: list[dict[str, Any]] = []

            def flush_foreach_candidates() -> None:
                nonlocal foreach_candidates
                if not foreach_candidates:
                    return
                if not self._append_foreach_local_pre_ns_updates(
                    foreach_candidates, all_updates, appended_indices
                ):
                    for candidate in foreach_candidates:
                        pre_ns_grad = self._compute_local_pre_ns_grad(
                            candidate["param"], candidate["group"], candidate["lr"]
                        )
                        appended_indices.append(len(all_updates))
                        all_updates.append(
                            (
                                candidate["param"],
                                pre_ns_grad,
                                candidate["update_mode"],
                                candidate["lr"],
                                candidate["group_kwargs"],
                            )
                        )
                foreach_candidates = []

            for group, gather_param_indices, lr, group_kwargs in group_contexts:
                for param_idx, p in enumerate(group["params"]):
                    update_mode = (
                        self._fsdp_boundary_update_mode(p)
                        if param_idx in gather_param_indices
                        else "local"
                    )
                    if overlap_enabled and update_mode == "gather":
                        continue
                    is_local_update = update_mode in ("local", "local_boundary")
                    if local_only is True and not is_local_update:
                        continue
                    if local_only is False and is_local_update:
                        continue
                    if p._local_tensor.numel() == 0 and is_local_update:
                        # If this parameter is not split by Megatron-FSDP,
                        # and is empty on this DP rank, then we can skip this
                        # update for all TP ranks, as tensor parallelism uses
                        # even sharding, so empty implies that FSDP did not
                        # assign any fraction of the parameter to this DP rank.
                        if not (
                            update_mode == "local_boundary"
                            and self.fsdp_approx_local_boundary_global_norm_scale
                        ):
                            continue

                    grad = self._param_grad(p)
                    if self.fsdp_foreach_pre_ns and grad is not None:
                        foreach_candidates.append(
                            {
                                "param": p,
                                "grad": grad,
                                "group": group,
                                "group_kwargs": group_kwargs,
                                "update_mode": update_mode,
                                "lr": lr,
                                "momentum": group["momentum"],
                                "weight_decay": group["weight_decay"],
                            }
                        )
                    else:
                        flush_foreach_candidates()
                        pre_ns_grad = self._compute_local_pre_ns_grad(p, group, lr)
                        appended_indices.append(len(all_updates))
                        all_updates.append((p, pre_ns_grad, update_mode, lr, group_kwargs))
                flush_foreach_candidates()
            return appended_indices

        # Phase 1: Compute remaining momentum updates.  With local-first
        # overlap, run fully local NS/update immediately after launching the
        # boundary gather so it can cover the first communication window.
        if overlap_enabled and self.fsdp_overlap_local_ns_first and boundary_update_indices:
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 1b local-only pre-NS first"
            ):
                early_local_update_indices = append_pre_ns_updates(local_only=True)
            if early_local_update_indices:
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    with torch.autograd.profiler.record_function(
                        "Muon-FSDP phase 3a early local-only NS/update under gather"
                    ):
                        self._apply_precomputed_muon_updates(
                            [all_updates[i] for i in early_local_update_indices]
                        )
                early_applied_update_indices.update(early_local_update_indices)

            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 1c nonlocal pre-NS after early local"
            ):
                append_pre_ns_updates(local_only=False)
        else:
            with torch.autograd.profiler.record_function("Muon-FSDP phase 1b local pre-NS"):
                append_pre_ns_updates(local_only=None)

        self._maybe_log_fsdp_update_mode_summary(all_updates)

        # Phase 2: AG all boundary gradients.
        if not overlap_enabled:
            boundary_update_indices = [
                i
                for i, (_, _, update_mode, _, _) in enumerate(all_updates)
                if update_mode == "gather"
            ]

        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            with torch.autograd.profiler.record_function("Muon-FSDP phase 3 NS/update"):
                if overlap_enabled and boundary_update_indices:
                    self._overlap_boundary_gather_and_update(
                        all_updates,
                        boundary_update_indices,
                        early_gather_state,
                        early_applied_update_indices,
                    )
                else:
                    if boundary_update_indices:
                        boundary_items = [
                            (
                                all_updates[i][0],
                                self._prepare_boundary_gather_tensor(all_updates[i][1]),
                            )
                            for i in boundary_update_indices
                        ]
                        with torch.autograd.profiler.record_function(
                            "Muon-FSDP phase 2 boundary gather"
                        ):
                            if self.fsdp_batched_all_gather:
                                gathered_boundary_updates = (
                                    self._gather_full_uneven_local_tensors_like(boundary_items)
                                )
                            else:
                                gathered_boundary_updates = [
                                    self._gather_full_uneven_local_tensor_like(p, local_tensor)
                                    for p, local_tensor in boundary_items
                                ]
                        for i, full_pre_ns_grad in zip(
                            boundary_update_indices, gathered_boundary_updates
                        ):
                            p, local_pre_ns_grad, _, lr, group_kwargs = all_updates[i]
                            full_pre_ns_grad = self._restore_boundary_gather_tensor(
                                full_pre_ns_grad, local_pre_ns_grad
                            )
                            all_updates[i] = (p, full_pre_ns_grad, "gather", lr, group_kwargs)

                    boundary_update_index_set = set(boundary_update_indices)
                    with torch.autograd.profiler.record_function(
                        "Muon-FSDP phase 3c local NS/update"
                    ):
                        local_updates = [
                            update
                            for update_idx, update in enumerate(all_updates)
                            if update_idx not in boundary_update_index_set
                        ]
                        self._apply_precomputed_muon_updates(local_updates)

                    with torch.autograd.profiler.record_function(
                        "Muon-FSDP phase 3d boundary NS/update"
                    ):
                        boundary_updates = [
                            all_updates[update_idx] for update_idx in boundary_update_indices
                        ]
                        self._apply_precomputed_muon_updates(boundary_updates)
                        for update_idx in boundary_update_indices:
                            p, _, update_mode, lr, group_kwargs = all_updates[update_idx]
                            all_updates[update_idx] = (p, None, update_mode, lr, group_kwargs)

        return loss

    def _compute_local_pre_ns_grad(
        self, p: torch.Tensor, group: dict[str, Any], lr: float, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        p_local = p._local_tensor
        state = self.state[p]
        mom_local = state["momentum_buffer"]._local_tensor
        if out is not None:
            if tuple(out.shape) != tuple(mom_local.shape):
                raise AssertionError(
                    "Direct Muon+M-FSDP boundary pre-NS output shape mismatch: "
                    f"got {tuple(out.shape)}, expected {tuple(mom_local.shape)}."
                )
            if out.dtype != mom_local.dtype:
                raise AssertionError(
                    "Direct Muon+M-FSDP boundary pre-NS requires the gather buffer "
                    "dtype to match the local momentum dtype to preserve exact math: "
                    f"buffer={out.dtype}, momentum={mom_local.dtype}."
                )

        grad = self._param_grad(p)
        local_grad = grad._local_tensor if grad is not None else torch.zeros_like(mom_local)
        if local_grad.dtype != mom_local.dtype:
            local_grad = local_grad.to(dtype=mom_local.dtype)

        self._apply_weight_decay_inplace(p_local, local_grad, lr, group["weight_decay"])
        mom_local.lerp_(local_grad, 1 - group["momentum"])
        if out is not None:
            if self.nesterov:
                out.copy_(local_grad)
                out.lerp_(mom_local, group["momentum"])
            else:
                out.copy_(mom_local)
            return out
        if self.nesterov:
            return local_grad.lerp(mom_local, group["momentum"])
        return mom_local

    def _append_foreach_local_pre_ns_updates(
        self, candidates: list[dict[str, Any]], all_updates: list, appended_indices: list[int]
    ) -> bool:
        """Append pre-NS updates after batched elementwise momentum work."""
        if len(candidates) < 2:
            return False
        required_ops = ("_foreach_mul_", "_foreach_add_", "_foreach_mul")
        if any(not hasattr(torch, op_name) for op_name in required_ops):
            return False

        weight_decay_method = getattr(self, "weight_decay_method", "l2")
        weight_decay_values = {candidate["weight_decay"] for candidate in candidates}
        if weight_decay_values != {0.0} and weight_decay_method != "decoupled":
            return False

        buckets: dict[
            tuple[Any, ...],
            list[tuple[int, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]],
        ] = {}
        for candidate_idx, candidate in enumerate(candidates):
            p = candidate["param"]
            grad = candidate["grad"]
            if grad is None:
                return False
            p_local = p._local_tensor
            mom_local = self.state[p]["momentum_buffer"]._local_tensor
            local_grad = grad._local_tensor
            if local_grad.dtype != mom_local.dtype:
                local_grad = local_grad.to(dtype=mom_local.dtype)
            key = (
                p_local.device,
                p_local.dtype,
                mom_local.device,
                mom_local.dtype,
                local_grad.device,
                local_grad.dtype,
                candidate["lr"],
                candidate["momentum"],
                candidate["weight_decay"],
            )
            buckets.setdefault(key, []).append(
                (candidate_idx, candidate, p_local, mom_local, local_grad)
            )

        pre_ns_results: list[torch.Tensor | None] = [None] * len(candidates)
        for bucket_items in buckets.values():
            _, first_candidate, _, _, _ = bucket_items[0]
            lr = first_candidate["lr"]
            momentum = first_candidate["momentum"]
            weight_decay = first_candidate["weight_decay"]
            p_tensors = [item[2] for item in bucket_items]
            mom_tensors = [item[3] for item in bucket_items]
            grad_tensors = [item[4] for item in bucket_items]
            with torch.autograd.profiler.record_function(
                f"Muon-FSDP foreach pre-NS count={len(bucket_items)}"
            ):
                if weight_decay != 0.0:
                    torch._foreach_add_(p_tensors, p_tensors, alpha=-(weight_decay * lr))
                if hasattr(torch, "_foreach_lerp_"):
                    torch._foreach_lerp_(mom_tensors, grad_tensors, 1 - momentum)
                else:
                    torch._foreach_mul_(mom_tensors, momentum)
                    torch._foreach_add_(mom_tensors, grad_tensors, alpha=1 - momentum)
                if self.nesterov:
                    if hasattr(torch, "_foreach_lerp"):
                        pre_ns_tensors = torch._foreach_lerp(grad_tensors, mom_tensors, momentum)
                    else:
                        pre_ns_tensors = torch._foreach_mul(grad_tensors, 1 - momentum)
                        torch._foreach_add_(pre_ns_tensors, mom_tensors, alpha=momentum)
                else:
                    pre_ns_tensors = mom_tensors

            for (candidate_idx, _, _, _, _), pre_ns_grad in zip(bucket_items, pre_ns_tensors):
                pre_ns_results[candidate_idx] = pre_ns_grad

        for candidate, pre_ns_grad in zip(candidates, pre_ns_results):
            if pre_ns_grad is None:
                return False
            appended_indices.append(len(all_updates))
            all_updates.append(
                (
                    candidate["param"],
                    pre_ns_grad,
                    candidate["update_mode"],
                    candidate["lr"],
                    candidate["group_kwargs"],
                )
            )
        return True

    def _prepare_boundary_gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.fsdp_boundary_gather_dtype == "bf16" and tensor.dtype != torch.bfloat16:
            return tensor.to(dtype=torch.bfloat16)
        return tensor

    def _boundary_gather_wire_dtype(self, fallback_dtype: torch.dtype) -> torch.dtype:
        if self.fsdp_boundary_gather_dtype == "bf16":
            return torch.bfloat16
        return fallback_dtype

    def _restore_boundary_gather_tensor(
        self, gathered_tensor: torch.Tensor | None, reference_tensor: torch.Tensor | None
    ) -> torch.Tensor | None:
        if gathered_tensor is None or reference_tensor is None:
            return gathered_tensor
        if gathered_tensor.dtype != reference_tensor.dtype:
            return gathered_tensor.to(dtype=reference_tensor.dtype)
        return gathered_tensor

    def _compute_int8_boundary_gather_scales(
        self,
        current_buffers: list[torch.Tensor],
        gather_groups: list[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        device = current_buffers[0].device
        with torch.autograd.profiler.record_function("Muon-FSDP int8 gather absmax"):
            absmaxes = torch.empty(len(current_buffers), dtype=torch.float32, device=device)
            for idx, buffer in enumerate(current_buffers):
                if buffer.numel() == 0:
                    absmaxes[idx] = 0.0
                else:
                    absmaxes[idx] = buffer.detach().abs().amax().to(dtype=torch.float32)
        with torch.autograd.profiler.record_function("Muon-FSDP int8 gather scale all-reduce"):
            for group in gather_groups:
                torch.distributed.all_reduce(
                    absmaxes, op=torch.distributed.ReduceOp.MAX, group=group
                )
        return (absmaxes / 127.0).clamp_min(torch.finfo(torch.float32).tiny)

    def _fp8_boundary_gather_dtype(self) -> torch.dtype | None:
        if self.fsdp_boundary_gather_dtype == "fp8_e4m3fn":
            return getattr(torch, "float8_e4m3fn", None)
        if self.fsdp_boundary_gather_dtype == "fp8_e5m2":
            return getattr(torch, "float8_e5m2", None)
        return None

    def _compute_fp8_boundary_gather_scales(
        self,
        current_buffers: list[torch.Tensor],
        gather_groups: list[torch.distributed.ProcessGroup],
        fp8_dtype: torch.dtype,
    ) -> torch.Tensor:
        device = current_buffers[0].device
        with torch.autograd.profiler.record_function("Muon-FSDP fp8 gather absmax"):
            absmaxes = torch.empty(len(current_buffers), dtype=torch.float32, device=device)
            for idx, buffer in enumerate(current_buffers):
                if buffer.numel() == 0:
                    absmaxes[idx] = 0.0
                else:
                    absmaxes[idx] = buffer.detach().abs().amax().to(dtype=torch.float32)
        with torch.autograd.profiler.record_function("Muon-FSDP fp8 gather scale all-reduce"):
            for group in gather_groups:
                torch.distributed.all_reduce(
                    absmaxes, op=torch.distributed.ReduceOp.MAX, group=group
                )
        return (absmaxes / torch.finfo(fp8_dtype).max).clamp_min(torch.finfo(torch.float32).tiny)

    def _maybe_quantize_boundary_gather_buffer(
        self,
        local_buffer: torch.Tensor,
        batch: dict[str, Any],
        *,
        stage_idx: int,
        item_numels: list[int],
    ) -> torch.Tensor:
        if self.fsdp_boundary_gather_dtype not in ("int8", "fp8_e4m3fn", "fp8_e5m2"):
            return local_buffer
        fp8_dtype = self._fp8_boundary_gather_dtype()
        if fp8_dtype is not None:
            if stage_idx == 0:
                scales = batch.get("_fp8_gather_scales")
                if scales is None:
                    raise AssertionError("Muon+M-FSDP fp8 boundary gather has no item scales.")
                with torch.autograd.profiler.record_function("Muon-FSDP fp8 gather quantize"):
                    quantized = torch.empty(
                        local_buffer.numel(), dtype=fp8_dtype, device=local_buffer.device
                    )
                    offset = 0
                    for item_idx, item_numel in enumerate(item_numels):
                        if item_numel == 0:
                            continue
                        item_slice = slice(offset, offset + item_numel)
                        quantized[item_slice].copy_(
                            (local_buffer[item_slice].to(dtype=torch.float32) / scales[item_idx])
                            .clamp(min=-torch.finfo(fp8_dtype).max, max=torch.finfo(fp8_dtype).max)
                            .to(dtype=fp8_dtype)
                        )
                        offset += item_numel
                    if offset != local_buffer.numel():
                        raise AssertionError(
                            "Muon+M-FSDP fp8 boundary gather quantized an unexpected size: "
                            f"quantized={offset}, expected={local_buffer.numel()}."
                        )
                    return quantized

            if local_buffer.dtype != fp8_dtype:
                raise AssertionError(
                    "Muon+M-FSDP fp8 boundary gather expected later stages to repack fp8 buffers, "
                    f"got {local_buffer.dtype} at stage {stage_idx}."
                )
            if "_fp8_gather_scales" not in batch:
                raise AssertionError("Muon+M-FSDP fp8 boundary gather lost its batch scale.")
            return local_buffer

        if stage_idx == 0:
            scales = batch.get("_int8_gather_scales")
            if scales is None:
                raise AssertionError("Muon+M-FSDP int8 boundary gather has no item scales.")
            with torch.autograd.profiler.record_function("Muon-FSDP int8 gather quantize"):
                quantized = torch.empty(
                    local_buffer.numel(), dtype=torch.int8, device=local_buffer.device
                )
                offset = 0
                for item_idx, item_numel in enumerate(item_numels):
                    if item_numel == 0:
                        continue
                    item_slice = slice(offset, offset + item_numel)
                    quantized[item_slice].copy_(
                        torch.clamp(
                            torch.round(
                                local_buffer[item_slice].to(dtype=torch.float32) / scales[item_idx]
                            ),
                            -127,
                            127,
                        ).to(dtype=torch.int8)
                    )
                    offset += item_numel
                if offset != local_buffer.numel():
                    raise AssertionError(
                        "Muon+M-FSDP int8 boundary gather quantized an unexpected size: "
                        f"quantized={offset}, expected={local_buffer.numel()}."
                    )
                return quantized

        if local_buffer.dtype != torch.int8:
            raise AssertionError(
                "Muon+M-FSDP int8 boundary gather expected later stages to repack int8 buffers, "
                f"got {local_buffer.dtype} at stage {stage_idx}."
            )
        if "_int8_gather_scales" not in batch:
            raise AssertionError("Muon+M-FSDP int8 boundary gather lost its batch scale.")
        return local_buffer

    def _maybe_dequantize_boundary_gather_result(
        self,
        full_tensor: torch.Tensor,
        reference_tensor: torch.Tensor,
        batch: dict[str, Any],
        batch_item_idx: int,
    ) -> torch.Tensor:
        if self.fsdp_boundary_gather_dtype not in ("int8", "fp8_e4m3fn", "fp8_e5m2"):
            return full_tensor
        fp8_dtype = self._fp8_boundary_gather_dtype()
        if fp8_dtype is not None:
            scales = batch.get("_fp8_gather_scales")
            if scales is None:
                raise AssertionError("Muon+M-FSDP fp8 boundary gather result has no scale.")
            scale = scales[batch_item_idx]
            with torch.autograd.profiler.record_function("Muon-FSDP fp8 gather dequantize"):
                return (full_tensor.to(dtype=torch.float32) * scale).to(
                    dtype=reference_tensor.dtype
                )

        scales = batch.get("_int8_gather_scales")
        if scales is None:
            raise AssertionError("Muon+M-FSDP int8 boundary gather result has no scale.")
        scale = scales[batch_item_idx]
        with torch.autograd.profiler.record_function("Muon-FSDP int8 gather dequantize"):
            return (full_tensor.to(dtype=torch.float32) * scale).to(dtype=reference_tensor.dtype)

    def _tp_partition_dim_for_param(self, p: torch.Tensor) -> int | None:
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            return None
        if self.pg_collection is not None and get_pg_size(self.pg_collection.tp) == 1:
            return None
        return partition_dim

    def _fsdp_batched_ns_key(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor | None, update_mode: str
    ) -> tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype, torch.device] | None:
        if not self.fsdp_batched_newton_schulz:
            return None
        if update_mode in ("distributed", "partial_distributed") or pre_ns_grad is None:
            return None
        if pre_ns_grad.ndim != 2 or pre_ns_grad.numel() == 0:
            return None
        if (
            update_mode == "local_boundary"
            and self.fsdp_approx_local_boundary_global_norm_scale
            and self.fsdp_approx_local_boundary_full_shape_scale
        ):
            return None
        if pre_ns_grad.numel() > self.fsdp_batched_newton_schulz_max_numel:
            return None
        if self._is_split_qkv_param(p) or self._tp_partition_dim_for_param(p) is not None:
            return None
        scale_shape: tuple[int, ...] = ()
        if update_mode == "local_boundary" and self.fsdp_approx_local_boundary_full_shape_scale:
            scale_shape = tuple(int(dim) for dim in p.shape)
        return (
            update_mode,
            tuple(pre_ns_grad.shape),
            scale_shape,
            pre_ns_grad.dtype,
            pre_ns_grad.device,
        )

    def _fsdp_batched_distributed_ns_key(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor | None, update_mode: str
    ) -> tuple[str, tuple[int, ...], torch.dtype, torch.device, int] | None:
        if not self.fsdp_batched_newton_schulz:
            return None
        if update_mode != "distributed" or pre_ns_grad is None:
            return None
        if pre_ns_grad.ndim != 2:
            return None
        if self._is_split_qkv_param(p) or self._tp_partition_dim_for_param(p) is not None:
            return None
        fsdp_group = self._get_fsdp_distributed_ns_group(p)
        if fsdp_group is None:
            return None
        return (update_mode, tuple(p.shape), pre_ns_grad.dtype, pre_ns_grad.device, id(fsdp_group))

    def _fsdp_batched_partial_distributed_ns_key(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor | None, update_mode: str
    ) -> tuple[str, tuple[int, ...], torch.dtype, torch.device, int, int] | None:
        if not self.fsdp_batched_newton_schulz:
            return None
        if update_mode != "partial_distributed" or pre_ns_grad is None:
            return None
        if pre_ns_grad.ndim != 2:
            return None
        plan = self._get_fsdp_partial_distributed_ns_plan(p)
        if plan is None:
            return None
        return (
            update_mode,
            tuple(int(dim) for dim in p.shape),
            pre_ns_grad.dtype,
            pre_ns_grad.device,
            id(plan["partition_group"]),
            int(plan["partition_dim"]),
        )

    def _fsdp_batched_qkv_ns_key(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor | None, update_mode: str
    ) -> tuple[str, tuple[int, ...], torch.dtype, torch.device, int] | None:
        if not self.fsdp_batched_newton_schulz:
            return None
        if update_mode in ("distributed", "partial_distributed") or pre_ns_grad is None:
            return None
        if update_mode == "local_boundary" and self.fsdp_approx_local_boundary_global_norm_scale:
            return None
        if not self._is_split_qkv_param(p) or self.qkv_split_shapes is None:
            return None
        if self._tp_partition_dim_for_param(p) is not None:
            return None
        if pre_ns_grad.ndim != 2 or pre_ns_grad.numel() == 0:
            return None
        if pre_ns_grad.numel() > self.fsdp_batched_newton_schulz_max_numel:
            return None
        split_dim = self._qkv_split_dim_for_shape(pre_ns_grad.shape)
        if split_dim is None:
            return None
        return (
            update_mode,
            tuple(pre_ns_grad.shape),
            pre_ns_grad.dtype,
            pre_ns_grad.device,
            split_dim,
        )

    def _iter_batched_ns_chunks(self, updates: list) -> list[list]:
        if not updates:
            return []
        pre_ns_grad = updates[0][1]
        assert pre_ns_grad is not None
        tensor_bytes = pre_ns_grad.numel() * pre_ns_grad.element_size()
        max_items = max(1, self.fsdp_batched_newton_schulz_max_batch_bytes // max(1, tensor_bytes))
        return [updates[start : start + max_items] for start in range(0, len(updates), max_items)]

    def _iter_batched_distributed_ns_chunks(self, updates: list) -> list[list]:
        if not updates:
            return []
        p = updates[0][0]
        # Distributed row-sharded NS all-reduces a Gram matrix of shape
        # (hidden, hidden) per update. Keep chunks bounded by the configured
        # batch memory target, based on the BF16 Gram tensor used by NS.
        gram_numel = int(p.shape[-1]) * int(p.shape[-1])
        gram_bytes = gram_numel * 2
        max_items = max(
            1, self.fsdp_batched_distributed_newton_schulz_max_batch_bytes // max(1, gram_bytes)
        )
        return [updates[start : start + max_items] for start in range(0, len(updates), max_items)]

    def _iter_batched_partial_distributed_ns_chunks(self, updates: list) -> list[list]:
        if not updates:
            return []
        p, pre_ns_grad, *_ = updates[0]
        plan = self._get_fsdp_partial_distributed_ns_plan(p)
        if plan is None or pre_ns_grad is None:
            return [[update] for update in updates]

        partition_dim = int(plan["partition_dim"])
        if partition_dim == 0:
            partial_numel = int(pre_ns_grad.shape[0]) * int(p.shape[-1])
            gram_dim = int(p.shape[-1])
        else:
            partial_numel = int(p.shape[-2]) * int(pre_ns_grad.shape[1])
            gram_dim = int(p.shape[-2])
        partial_bytes = partial_numel * pre_ns_grad.element_size()
        gram_bytes = gram_dim * gram_dim * 2
        max_items_by_partial = max(
            1, self.fsdp_batched_newton_schulz_max_batch_bytes // max(1, partial_bytes)
        )
        max_items_by_gram = max(
            1, self.fsdp_batched_distributed_newton_schulz_max_batch_bytes // max(1, gram_bytes)
        )
        max_items = max(1, min(max_items_by_partial, max_items_by_gram))
        return [updates[start : start + max_items] for start in range(0, len(updates), max_items)]

    def _apply_batched_distributed_muon_updates(self, chunk: list) -> None:
        p0 = chunk[0][0]
        fsdp_group, participates = self._get_fsdp_distributed_ns_runtime_group(p0, chunk[0][1])
        if not participates:
            return
        if fsdp_group is None and not self.fsdp_distributed_ns_nonempty_group:
            raise AssertionError("Batched distributed NS received an ineligible parameter.")

        row_counts = [int(update[1].shape[0]) for update in chunk]
        max_rows = max(row_counts)
        cols = int(chunk[0][1].shape[1])
        if all(rows == max_rows for rows in row_counts):
            stacked_pre_ns = torch.stack([update[1] for update in chunk], dim=0)
            padded_local_rows = False
        else:
            stacked_pre_ns = chunk[0][1].new_zeros((len(chunk), max_rows, cols))
            for batch_idx, (_, pre_ns_grad, _, _, _) in enumerate(chunk):
                rows = row_counts[batch_idx]
                if rows:
                    stacked_pre_ns[batch_idx, :rows].copy_(pre_ns_grad)
            padded_local_rows = True

        if fsdp_group is None:
            orth_updates = self.scaled_orthogonalize_fn(stacked_pre_ns, None, None)
        else:
            orth_updates = newton_schulz_tp(
                stacked_pre_ns,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,
                tp_group=fsdp_group,
                partition_dim=0,
                tp_mode="distributed",
                use_syrk=self.use_syrk,
                distributed_gram_recurrence=self.fsdp_distributed_ns_single_all_reduce,
                distributed_gram_refresh_interval=self.fsdp_distributed_ns_gram_refresh_interval,
            )
            scale_factor = get_muon_scale_factor(p0.shape[-2], p0.shape[-1], mode=self.scale_mode)
            orth_updates.mul_(scale_factor * self.extra_scale_factor)
        if not padded_local_rows and self._try_apply_orthogonal_muon_update_foreach(
            chunk, orth_updates
        ):
            return
        for batch_idx, (p, _, update_mode, lr, _) in enumerate(chunk):
            if padded_local_rows:
                orth_update = orth_updates[batch_idx, : row_counts[batch_idx]]
            else:
                orth_update = orth_updates[batch_idx]
            self._apply_orthogonal_muon_update(p, orth_update, update_mode, lr)

    def _apply_batched_partial_distributed_muon_updates(self, chunk: list) -> None:
        p0 = chunk[0][0]
        plan = self._get_fsdp_partial_distributed_ns_plan(p0)
        if plan is None:
            raise AssertionError("Batched partial distributed NS received an ineligible parameter.")

        partial_plans = []
        gather_items = []
        reference_tensors = []
        with torch.autograd.profiler.record_function(
            f"Muon-FSDP partial distributed gather count={len(chunk)}"
        ):
            for p, pre_ns_grad, _, _, _ in chunk:
                item_plan = self._get_fsdp_partial_distributed_ns_plan(p)
                if item_plan is None:
                    raise AssertionError(
                        "Batched partial distributed NS chunk contains an ineligible parameter."
                    )
                partial_plans.append(item_plan["gather_plan"])
                gather_items.append((p, self._prepare_boundary_gather_tensor(pre_ns_grad)))
                reference_tensors.append(pre_ns_grad)

            gathered_partials = self._gather_partial_uneven_local_tensors_like(
                gather_items, partial_plans
            )

        partial_pre_ns_updates = [
            self._restore_boundary_gather_tensor(gathered, reference)
            for gathered, reference in zip(gathered_partials, reference_tensors)
        ]
        self._apply_batched_partial_distributed_muon_updates_from_partials(
            chunk, partial_pre_ns_updates
        )

    def _apply_batched_partial_distributed_muon_updates_from_partials(
        self, chunk: list, partial_pre_ns_updates: list[torch.Tensor]
    ) -> None:
        p0 = chunk[0][0]
        plan = self._get_fsdp_partial_distributed_ns_plan(p0)
        if plan is None:
            raise AssertionError("Partial distributed NS received an ineligible parameter.")

        partial_shapes = [tuple(partial.shape) for partial in partial_pre_ns_updates]

        if any(shape != partial_shapes[0] for shape in partial_shapes):
            raise AssertionError(
                "Batched partial distributed NS expected equal partial tensor shapes, "
                f"got {partial_shapes}."
            )

        stacked_pre_ns = torch.stack(partial_pre_ns_updates, dim=0)
        orth_updates = newton_schulz_tp(
            stacked_pre_ns,
            steps=self.num_ns_steps,
            coefficient_type=self.coefficient_type,
            tp_group=plan["partition_group"],
            partition_dim=int(plan["partition_dim"]),
            tp_mode="distributed",
            use_syrk=self.use_syrk,
            distributed_gram_recurrence=self.fsdp_distributed_ns_single_all_reduce,
            distributed_gram_refresh_interval=self.fsdp_distributed_ns_gram_refresh_interval,
        )
        scale_factor = get_muon_scale_factor(p0.shape[-2], p0.shape[-1], mode=self.scale_mode)
        orth_updates.mul_(scale_factor * self.extra_scale_factor)
        for orth_update, (p, _, update_mode, lr, _) in zip(orth_updates.unbind(0), chunk):
            local_update = self._local_shard_from_partial_update_like(p, orth_update)
            self._apply_orthogonal_muon_update(p, local_update, update_mode, lr)

    def _start_async_partial_distributed_gathers(
        self, indexed_updates: list[tuple[int, tuple]]
    ) -> tuple[list[dict[str, Any]], list[tuple[int, tuple]]]:
        partial_distributed_batch_map: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.device, int, int], list
        ] = {}
        fallback_updates: list[tuple[int, tuple]] = []

        for idx, update in indexed_updates:
            p, pre_ns_grad, update_mode, _, _ = update
            partial_key = self._fsdp_batched_partial_distributed_ns_key(p, pre_ns_grad, update_mode)
            if partial_key is None:
                fallback_updates.append((idx, update))
                continue
            partial_distributed_batch_map.setdefault(partial_key, []).append((idx, update))

        pending_chunks: list[dict[str, Any]] = []
        for _, candidate_updates in partial_distributed_batch_map.items():
            update_chunk_candidates = [update for _, update in candidate_updates]
            chunk_start = 0
            for chunk in self._iter_batched_partial_distributed_ns_chunks(update_chunk_candidates):
                indexed_chunk = candidate_updates[chunk_start : chunk_start + len(chunk)]
                chunk_start += len(chunk)
                if len(chunk) < 2:
                    fallback_updates.extend(indexed_chunk)
                    continue

                partial_plans = []
                gather_items = []
                reference_tensors = []
                for p, pre_ns_grad, _, _, _ in chunk:
                    item_plan = self._get_fsdp_partial_distributed_ns_plan(p)
                    if item_plan is None:
                        raise AssertionError(
                            "Async partial distributed NS chunk contains an ineligible "
                            "parameter."
                        )
                    partial_plans.append(item_plan["gather_plan"])
                    gather_items.append((p, self._prepare_boundary_gather_tensor(pre_ns_grad)))
                    reference_tensors.append(pre_ns_grad)

                pending_gather = self._start_gather_partial_uneven_local_tensors_like_async(
                    gather_items, partial_plans
                )
                pending_chunks.append(
                    {
                        "chunk": chunk,
                        "reference_tensors": reference_tensors,
                        "pending_gather": pending_gather,
                    }
                )

        return pending_chunks, fallback_updates

    def _finish_async_partial_distributed_gathers(
        self, pending_chunks: list[dict[str, Any]]
    ) -> None:
        for pending_chunk in pending_chunks:
            chunk = pending_chunk["chunk"]
            with torch.autograd.profiler.record_function(
                f"Muon-FSDP async partial distributed finish/update count={len(chunk)}"
            ):
                gathered_partials = self._finish_gather_partial_uneven_local_tensors_like_async(
                    pending_chunk["pending_gather"]
                )
                partial_pre_ns_updates = [
                    self._restore_boundary_gather_tensor(gathered, reference)
                    for gathered, reference in zip(
                        gathered_partials, pending_chunk["reference_tensors"]
                    )
                ]
                self._apply_batched_partial_distributed_muon_updates_from_partials(
                    chunk, partial_pre_ns_updates
                )

    def _maybe_log_fsdp_batched_ns_summary(
        self,
        *,
        candidate_updates: int,
        batched_chunks: int,
        batched_updates: int,
        fallback_updates: int,
        shape_counts: dict[tuple[str, tuple[int, ...]], int],
        fallback_shape_counts: dict[tuple[str, str, tuple[int, ...]], int],
    ) -> None:
        summary_modes = ",".join(sorted({mode for mode, _ in shape_counts})) or "none"
        if summary_modes in self._fsdp_batched_ns_summary_logged_modes:
            return
        self._fsdp_batched_ns_summary_logged_modes.add(summary_modes)

        top_shapes = sorted(shape_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        top_shapes_text = ", ".join(
            f"{mode}{shape}: {count}" for (mode, shape), count in top_shapes
        )
        top_fallback_shapes = sorted(
            fallback_shape_counts.items(), key=lambda item: item[1], reverse=True
        )[:8]
        top_fallback_shapes_text = ", ".join(
            f"{reason}:{mode}{shape}: {count}"
            for (reason, mode, shape), count in top_fallback_shapes
        )
        distributed_max_batch_mib = self.fsdp_batched_distributed_newton_schulz_max_batch_bytes / (
            1024**2
        )
        message = (
            "Muon+M-FSDP batched Newton-Schulz summary: "
            f"enabled={self.fsdp_batched_newton_schulz}, "
            f"candidate_updates={candidate_updates}, batched_updates={batched_updates}, "
            f"batched_chunks={batched_chunks}, fallback_updates={fallback_updates}, "
            f"modes={summary_modes}, "
            f"max_numel={self.fsdp_batched_newton_schulz_max_numel}, "
            f"max_batch_mib={self.fsdp_batched_newton_schulz_max_batch_bytes / (1024 ** 2):.1f}, "
            f"distributed_max_batch_mib={distributed_max_batch_mib:.1f}, "
            f"top_shapes=[{top_shapes_text}], "
            f"top_fallback_shapes=[{top_fallback_shapes_text}]."
        )
        log_single_rank(logger, logging.INFO, message)
        self._maybe_print_fsdp_diagnostic(message)

    def _approx_local_boundary_orthogonalize(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor
    ) -> torch.Tensor:
        norm_scale = None
        if self.fsdp_approx_local_boundary_global_norm_scale:
            if pre_ns_grad.numel() == 0:
                return torch.empty_like(pre_ns_grad)
            norm_scale = self._get_approx_local_boundary_global_norm_scale(p, pre_ns_grad)

        orth_update = newton_schulz_tp(
            pre_ns_grad,
            steps=self.num_ns_steps,
            coefficient_type=self.coefficient_type,
            tp_group=None,
            partition_dim=None,
            tp_mode="duplicated" if self.tp_mode == "blockwise" else self.tp_mode,
            use_syrk=self.use_syrk,
        )
        scale_factor = get_muon_scale_factor(
            int(p.shape[-2]), int(p.shape[-1]), mode=self.scale_mode
        )
        orth_update.mul_(scale_factor * self.extra_scale_factor)
        if norm_scale is not None:
            orth_update.mul_(norm_scale)
        return orth_update

    def _uses_approx_local_boundary_full_shape_scale(
        self, p: torch.Tensor, update_mode: str
    ) -> bool:
        return (
            update_mode == "local_boundary"
            and self.fsdp_approx_local_boundary_full_shape_scale
            and not self._is_split_qkv_param(p)
            and self._tp_partition_dim_for_param(p) is None
        )

    def _get_approx_local_boundary_global_norm_scale(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor
    ) -> torch.Tensor:
        """Return local/global Frobenius norm ratio for a boundary shard.

        This is an approximation knob: local Newton-Schulz remains block-local,
        but the final update magnitude is shrunk according to the full parameter
        pre-NS norm so local shards do not use an overly large local-only scale.
        """
        local_sq = pre_ns_grad.float().square().sum()
        global_sq = local_sq.clone()
        plan = self._get_uneven_gather_plan(p)
        if plan is not None:
            for stage in plan["stages"]:
                torch.distributed.all_reduce(
                    global_sq, op=torch.distributed.ReduceOp.SUM, group=stage["shard_group"]
                )
        local_norm = local_sq.sqrt()
        global_norm = global_sq.sqrt().clamp_min(1e-7)
        return (local_norm / global_norm).to(device=pre_ns_grad.device, dtype=torch.float32)

    def _begin_precompute_approx_local_boundary_global_norm_scales(
        self, updates: list
    ) -> dict[str, Any] | None:
        """Start batched local/global norm-ratio computation for boundary updates."""
        if not self.fsdp_approx_local_boundary_global_norm_scale:
            return None

        entries: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]] = []
        for p, pre_ns_grad, update_mode, _, _ in updates:
            if update_mode != "local_boundary" or pre_ns_grad is None:
                continue
            if self._uses_approx_local_boundary_full_shape_scale(p, update_mode):
                continue
            plan = self._get_uneven_gather_plan(p)
            entries.append((p, pre_ns_grad, plan))

        if not entries:
            return None

        def compute_local_sqs() -> torch.Tensor:
            device = entries[0][1].device
            nonempty_entries = [
                (entry_idx, pre_ns_grad)
                for entry_idx, (_, pre_ns_grad, _) in enumerate(entries)
                if pre_ns_grad.numel() > 0
            ]
            if not nonempty_entries:
                return torch.zeros(len(entries), dtype=torch.float32, device=device)

            tensors = [pre_ns_grad for _, pre_ns_grad in nonempty_entries]
            if self.fsdp_approx_local_boundary_foreach_norm and hasattr(torch, "_foreach_norm"):
                try:
                    norms = torch._foreach_norm(tensors, 2.0)
                    nonempty_sqs = torch.stack(
                        [norm.to(dtype=torch.float32).square() for norm in norms]
                    )
                    if len(nonempty_entries) == len(entries):
                        return nonempty_sqs
                    local_sqs = torch.zeros(
                        len(entries), dtype=torch.float32, device=nonempty_sqs.device
                    )
                    nonempty_indices = torch.tensor(
                        [entry_idx for entry_idx, _ in nonempty_entries],
                        dtype=torch.long,
                        device=nonempty_sqs.device,
                    )
                    local_sqs.index_copy_(0, nonempty_indices, nonempty_sqs)
                    return local_sqs
                except RuntimeError as exc:
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        "Falling back from Muon-FSDP foreach norm scale path: %s",
                        exc,
                    )
            nonempty_sqs = torch.stack(
                [pre_ns_grad.float().square().sum() for _, pre_ns_grad in nonempty_entries]
            )
            if len(nonempty_entries) == len(entries):
                return nonempty_sqs
            local_sqs = torch.zeros(len(entries), dtype=torch.float32, device=nonempty_sqs.device)
            nonempty_indices = torch.tensor(
                [entry_idx for entry_idx, _ in nonempty_entries],
                dtype=torch.long,
                device=nonempty_sqs.device,
            )
            local_sqs.index_copy_(0, nonempty_indices, nonempty_sqs)
            return local_sqs

        with torch.autograd.profiler.record_function(
            f"Muon-FSDP approx boundary global norm begin count={len(entries)}"
        ):
            max_stage_count = max(
                (len(plan["stages"]) for _, _, plan in entries if plan is not None), default=0
            )
            local_sqs: torch.Tensor | None = None
            global_sqs: torch.Tensor | None = None

            def launch_all_reduces() -> None:
                assert global_sqs is not None
                remaining_entry_indices = set(range(len(entries)))
                if self.fsdp_approx_local_boundary_flat_norm_all_reduce:
                    flat_groups: dict[int, tuple[torch.distributed.ProcessGroup, list[int]]] = {}
                    for entry_idx, (p, _, plan) in enumerate(entries):
                        if plan is None:
                            remaining_entry_indices.discard(entry_idx)
                            continue
                        flat_group = self._get_existing_flat_uneven_gather_group(p, plan)
                        if flat_group is None or get_pg_size(flat_group) <= 1:
                            continue
                        group_id = id(flat_group)
                        if group_id not in flat_groups:
                            flat_groups[group_id] = (flat_group, [])
                        flat_groups[group_id][1].append(entry_idx)
                        remaining_entry_indices.discard(entry_idx)

                    if flat_groups:
                        flat_count = sum(len(indices) for _, indices in flat_groups.values())
                        with torch.autograd.profiler.record_function(
                            "Muon-FSDP approx boundary global norm flat all-reduce "
                            f"groups={len(flat_groups)} count={flat_count}"
                        ):
                            for shard_group, entry_indices in flat_groups.values():
                                group_sqs = global_sqs[entry_indices].contiguous()
                                torch.distributed.all_reduce(
                                    group_sqs, op=torch.distributed.ReduceOp.SUM, group=shard_group
                                )
                                global_sqs[entry_indices] = group_sqs

                for stage_idx in range(max_stage_count):
                    stage_groups: dict[int, tuple[torch.distributed.ProcessGroup, list[int]]] = {}
                    for entry_idx, (_, _, plan) in enumerate(entries):
                        if entry_idx not in remaining_entry_indices:
                            continue
                        if plan is None:
                            continue
                        if stage_idx >= len(plan["stages"]):
                            continue
                        shard_group = plan["stages"][stage_idx]["shard_group"]
                        group_id = id(shard_group)
                        if group_id not in stage_groups:
                            stage_groups[group_id] = (shard_group, [])
                        stage_groups[group_id][1].append(entry_idx)

                    for shard_group, entry_indices in stage_groups.values():
                        if get_pg_size(shard_group) <= 1:
                            continue
                        group_sqs = global_sqs[entry_indices].contiguous()
                        torch.distributed.all_reduce(
                            group_sqs, op=torch.distributed.ReduceOp.SUM, group=shard_group
                        )
                        global_sqs[entry_indices] = group_sqs

            def compute_local_sqs_and_launch_reduces() -> None:
                nonlocal local_sqs, global_sqs
                with torch.autograd.profiler.record_function(
                    f"Muon-FSDP approx boundary global norm local-sq count={len(entries)}"
                ):
                    local_sqs = compute_local_sqs()
                global_sqs = local_sqs.clone()
                with torch.autograd.profiler.record_function(
                    "Muon-FSDP approx boundary global norm all-reduce "
                    f"stages={max_stage_count} count={len(entries)}"
                ):
                    launch_all_reduces()

            done_event = None
            first_device = entries[0][1].device
            if (
                self.fsdp_approx_local_boundary_async_norm_all_reduce
                and first_device.type == "cuda"
                and torch.cuda.is_available()
            ):
                with torch.cuda.device(first_device):
                    ready_event = torch.cuda.Event()
                    current_stream = torch.cuda.current_stream(first_device)
                    current_stream.record_event(ready_event)
                    comm_stream = self._get_fsdp_comm_stream(first_device)
                    for _, pre_ns_grad, _ in entries:
                        pre_ns_grad.record_stream(comm_stream)
                    with torch.cuda.stream(comm_stream):
                        comm_stream.wait_event(ready_event)
                        compute_local_sqs_and_launch_reduces()
                        assert local_sqs is not None and global_sqs is not None
                        local_sqs.record_stream(comm_stream)
                        global_sqs.record_stream(comm_stream)
                        done_event = torch.cuda.Event()
                        done_event.record(comm_stream)
            else:
                compute_local_sqs_and_launch_reduces()

            assert local_sqs is not None and global_sqs is not None

            return {
                "entries": entries,
                "local_sqs": local_sqs,
                "global_sqs": global_sqs,
                "done_event": done_event,
                "device": local_sqs.device,
                "finished": None,
            }

    def _finish_precompute_approx_local_boundary_global_norm_scales(
        self, work: dict[str, Any] | None
    ) -> dict[int, torch.Tensor]:
        """Finish batched boundary norm ratios after any async reductions complete."""
        if work is None:
            return {}

        finished = work.get("finished")
        if finished is not None:
            return finished

        entries = work["entries"]
        local_sqs = work["local_sqs"]
        global_sqs = work["global_sqs"]
        done_event = work.get("done_event")
        if done_event is not None:
            with torch.autograd.profiler.record_function(
                f"Muon-FSDP approx boundary global norm wait count={len(entries)}"
            ):
                with torch.cuda.device(work["device"]):
                    torch.cuda.current_stream(work["device"]).wait_event(done_event)

        with torch.autograd.profiler.record_function(
            f"Muon-FSDP approx boundary global norm ratio count={len(entries)}"
        ):
            local_norms = local_sqs.sqrt()
            global_norms = global_sqs.sqrt().clamp_min(1e-7)
            norm_scales = (local_norms / global_norms).to(dtype=torch.float32)
        result = {id(p): norm_scales[entry_idx] for entry_idx, (p, _, _) in enumerate(entries)}
        work["finished"] = result
        return result

    def _precompute_approx_local_boundary_global_norm_scales(
        self, updates: list
    ) -> dict[int, torch.Tensor]:
        """Batch local/global norm ratios for approximate local-boundary updates."""
        work = self._begin_precompute_approx_local_boundary_global_norm_scales(updates)
        return self._finish_precompute_approx_local_boundary_global_norm_scales(work)

    def _apply_orthogonal_muon_update(
        self, p: torch.Tensor, orth_update: torch.Tensor, update_mode: str, lr: float
    ) -> None:
        if update_mode == "gather":
            local_update = self._local_shard_from_full_update_like(p, orth_update)
        else:
            local_update = orth_update.to(dtype=p._local_tensor.dtype)
        self.pre_weight_update_fn_inplace(p._local_tensor, local_update)
        p._local_tensor.add_(local_update, alpha=-lr)
        self.post_weight_update_fn_inplace(p._local_tensor)

    def _try_apply_orthogonal_muon_update_foreach(
        self, chunk: list, orth_updates: torch.Tensor
    ) -> bool:
        if not self.fsdp_foreach_weight_update or not chunk:
            return False
        if not hasattr(torch, "_foreach_add_"):
            return False
        if orth_updates.ndim < 1 or int(orth_updates.shape[0]) != len(chunk):
            return False

        first_lr = chunk[0][3]
        if any(update[3] != first_lr for update in chunk):
            return False

        has_gather_update = any(update[2] == "gather" for update in chunk)
        if has_gather_update and not self.fsdp_foreach_gather_weight_update:
            return False

        param_tensors = [update[0]._local_tensor for update in chunk]
        target_dtype = param_tensors[0].dtype
        if any(param_tensor.dtype != target_dtype for param_tensor in param_tensors):
            return False

        update_tensors = []
        for batch_idx, (param_tensor, update) in enumerate(zip(param_tensors, chunk)):
            p, _, update_mode, _, _ = update
            if update_mode == "gather":
                update_tensor = self._local_shard_from_full_update_like(p, orth_updates[batch_idx])
            else:
                update_tensor = orth_updates[batch_idx]
                if update_tensor.dtype != target_dtype:
                    update_tensor = update_tensor.to(dtype=target_dtype)
            if tuple(update_tensor.shape) != tuple(param_tensor.shape):
                return False
            update_tensors.append(update_tensor)

        with torch.autograd.profiler.record_function(
            f"Muon-FSDP foreach weight update count={len(chunk)}"
        ):
            for param_tensor, update_tensor in zip(param_tensors, update_tensors):
                self.pre_weight_update_fn_inplace(param_tensor, update_tensor)
            torch._foreach_add_(param_tensors, update_tensors, alpha=-first_lr)
            for param_tensor in param_tensors:
                self.post_weight_update_fn_inplace(param_tensor)
        return True

    def _apply_batched_qkv_muon_updates(
        self, chunk: list, mode: str, shape: tuple[int, ...], split_dim: int
    ) -> None:
        assert self.qkv_split_shapes is not None
        grad_shape = chunk[0][1].shape
        qkv_total = sum(self.qkv_split_shapes)
        if split_dim == 0:
            num_query_groups = grad_shape[0] // qkv_total
        else:
            num_query_groups = grad_shape[1] // qkv_total

        split_components_by_update = []
        for _, pre_ns_grad, _, _, _ in chunk:
            if split_dim == 0:
                qkv_components = torch.split(
                    pre_ns_grad.view(num_query_groups, qkv_total, -1), self.qkv_split_shapes, dim=1
                )
                split_components_by_update.append(
                    [component.reshape(-1, grad_shape[-1]) for component in qkv_components]
                )
            else:
                qkv_components = torch.split(
                    pre_ns_grad.view(grad_shape[0], num_query_groups, qkv_total),
                    self.qkv_split_shapes,
                    dim=2,
                )
                split_components_by_update.append(
                    [component.reshape(grad_shape[0], -1) for component in qkv_components]
                )

        orth_components_by_update: list[list[torch.Tensor]] = [[] for _ in chunk]
        for component_idx, component_size in enumerate(self.qkv_split_shapes):
            if split_dim == 0:
                component_shape = (num_query_groups * component_size, grad_shape[-1])
            else:
                component_shape = (grad_shape[0], num_query_groups * component_size)
            with torch.autograd.profiler.record_function(
                "Muon-FSDP batched split-QKV NS/update "
                f"mode={mode} shape={shape} split_dim={split_dim} component={component_idx} "
                f"component_shape={component_shape} count={len(chunk)}"
            ):
                stacked_component = torch.stack(
                    [components[component_idx] for components in split_components_by_update], dim=0
                )
                orth_component_batch = self.scaled_orthogonalize_fn(stacked_component, None, None)
            for update_idx, orth_component in enumerate(orth_component_batch.unbind(0)):
                if split_dim == 0:
                    orth_components_by_update[update_idx].append(
                        orth_component.view(num_query_groups, component_size, grad_shape[-1])
                    )
                else:
                    orth_components_by_update[update_idx].append(
                        orth_component.view(grad_shape[0], num_query_groups, component_size)
                    )

        for components, (p, _, update_mode, lr, _) in zip(orth_components_by_update, chunk):
            cat_dim = 1 if split_dim == 0 else 2
            orth_update = torch.cat(components, dim=cat_dim).view(grad_shape)
            self._apply_orthogonal_muon_update(p, orth_update, update_mode, lr)

    def _apply_precomputed_muon_updates(
        self, updates: list, progress_callback: Callable[[], None] | None = None
    ) -> None:
        def maybe_progress() -> None:
            if progress_callback is not None:
                progress_callback()

        if not self.fsdp_batched_newton_schulz:
            for p, pre_ns_grad, update_mode, lr, group_kwargs in updates:
                self._apply_precomputed_muon_update(p, pre_ns_grad, update_mode, lr, group_kwargs)
                maybe_progress()
            return

        norm_scale_work = self._begin_precompute_approx_local_boundary_global_norm_scales(updates)
        precomputed_norm_scales: dict[int, torch.Tensor] | None = None

        def get_precomputed_norm_scales() -> dict[int, torch.Tensor]:
            nonlocal precomputed_norm_scales
            if precomputed_norm_scales is None:
                precomputed_norm_scales = (
                    self._finish_precompute_approx_local_boundary_global_norm_scales(
                        norm_scale_work
                    )
                )
            return precomputed_norm_scales

        batch_map: dict[
            tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype, torch.device], list
        ] = {}
        distributed_batch_map: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.device, int], list
        ] = {}
        partial_distributed_batch_map: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.device, int, int], list
        ] = {}
        qkv_batch_map: dict[tuple[str, tuple[int, ...], torch.dtype, torch.device, int], list] = {}
        fallback_updates = []
        shape_counts: dict[tuple[str, tuple[int, ...]], int] = {}
        fallback_shape_counts: dict[tuple[str, str, tuple[int, ...]], int] = {}

        def record_fallback(update, reason: str) -> None:
            _, pre_ns_grad, update_mode, _, _ = update
            shape = tuple(pre_ns_grad.shape) if pre_ns_grad is not None else ()
            key = (reason, update_mode, shape)
            fallback_shape_counts[key] = fallback_shape_counts.get(key, 0) + 1

        def fallback_reason(p, pre_ns_grad, update_mode: str) -> str:
            if update_mode == "distributed":
                return "distributed"
            if update_mode == "partial_distributed":
                return "partial_distributed"
            if pre_ns_grad is None:
                return "missing_pre_ns"
            if pre_ns_grad.ndim != 2:
                return f"ndim{pre_ns_grad.ndim}"
            if pre_ns_grad.numel() == 0:
                return "empty"
            if pre_ns_grad.numel() > self.fsdp_batched_newton_schulz_max_numel:
                return "too_large"
            if self._tp_partition_dim_for_param(p) is not None:
                return "tp_partition"
            if self._is_split_qkv_param(p):
                return "qkv_unbatched"
            return "other"

        for update in updates:
            p, pre_ns_grad, update_mode, _, _ = update
            distributed_key = self._fsdp_batched_distributed_ns_key(p, pre_ns_grad, update_mode)
            if distributed_key is not None:
                distributed_batch_map.setdefault(distributed_key, []).append(update)
                shape_counts[(distributed_key[0], distributed_key[1])] = (
                    shape_counts.get((distributed_key[0], distributed_key[1]), 0) + 1
                )
                continue
            partial_distributed_key = self._fsdp_batched_partial_distributed_ns_key(
                p, pre_ns_grad, update_mode
            )
            if partial_distributed_key is not None:
                partial_distributed_batch_map.setdefault(partial_distributed_key, []).append(update)
                shape_counts[(partial_distributed_key[0], partial_distributed_key[1])] = (
                    shape_counts.get((partial_distributed_key[0], partial_distributed_key[1]), 0)
                    + 1
                )
                continue
            key = self._fsdp_batched_ns_key(p, pre_ns_grad, update_mode)
            if key is not None:
                batch_map.setdefault(key, []).append(update)
                shape_counts[(key[0], key[1])] = shape_counts.get((key[0], key[1]), 0) + 1
                continue
            qkv_key = self._fsdp_batched_qkv_ns_key(p, pre_ns_grad, update_mode)
            if qkv_key is not None:
                qkv_batch_map.setdefault(qkv_key, []).append(update)
                shape_counts[(f"{qkv_key[0]}_qkv", qkv_key[1])] = (
                    shape_counts.get((f"{qkv_key[0]}_qkv", qkv_key[1]), 0) + 1
                )
                continue
            record_fallback(update, fallback_reason(p, pre_ns_grad, update_mode))
            fallback_updates.append(update)

        maybe_progress()

        batched_chunks = 0
        batched_updates = 0

        def apply_distributed_batches() -> None:
            nonlocal batched_chunks, batched_updates
            for key, candidate_updates in distributed_batch_map.items():
                for chunk in self._iter_batched_distributed_ns_chunks(candidate_updates):
                    if len(chunk) < 2:
                        for update in chunk:
                            record_fallback(update, "singleton_distributed_batch")
                        fallback_updates.extend(chunk)
                        continue
                    batched_chunks += 1
                    batched_updates += len(chunk)
                    mode, full_shape, _, _, _ = key
                    local_rows = [int(update[1].shape[0]) for update in chunk]
                    min_local_rows = min(local_rows)
                    max_local_rows = max(local_rows)
                    with torch.autograd.profiler.record_function(
                        "Muon-FSDP batched distributed NS/update "
                        f"mode={mode} full_shape={full_shape} count={len(chunk)} "
                        f"local_rows={min_local_rows}..{max_local_rows}"
                    ):
                        self._apply_batched_distributed_muon_updates(chunk)
                    maybe_progress()

        if self.fsdp_prioritize_distributed_ns:
            apply_distributed_batches()

        batch_items = list(batch_map.items())
        if norm_scale_work is not None and self.fsdp_approx_local_boundary_async_norm_all_reduce:
            batch_items.sort(key=lambda item: item[0][0] == "local_boundary")

        for key, candidate_updates in batch_items:
            for chunk in self._iter_batched_ns_chunks(candidate_updates):
                if len(chunk) < 2:
                    for update in chunk:
                        record_fallback(update, "singleton_batch")
                    fallback_updates.extend(chunk)
                    continue
                batched_chunks += 1
                batched_updates += len(chunk)
                mode, shape, scale_shape, _, _ = key
                with torch.autograd.profiler.record_function(
                    f"Muon-FSDP batched NS/update mode={mode} shape={shape} count={len(chunk)}"
                ):
                    stacked_pre_ns = torch.stack([update[1] for update in chunk], dim=0)
                    if mode == "local_boundary" and scale_shape:
                        orth_updates = self._approx_local_boundary_orthogonalize(
                            chunk[0][0], stacked_pre_ns
                        )
                    else:
                        orth_updates = self.scaled_orthogonalize_fn(stacked_pre_ns, None, None)
                        if (
                            mode == "local_boundary"
                            and self.fsdp_approx_local_boundary_global_norm_scale
                        ):
                            norm_scales = []
                            for update in chunk:
                                norm_scale = get_precomputed_norm_scales().get(id(update[0]))
                                if norm_scale is None:
                                    raise AssertionError(
                                        "Missing precomputed local-boundary norm scale "
                                        "for batched Muon update."
                                    )
                                norm_scales.append(
                                    norm_scale.to(
                                        device=orth_updates.device, dtype=orth_updates.dtype
                                    )
                                )
                            norm_scale_shape = (len(norm_scales),) + (1,) * (orth_updates.ndim - 1)
                            orth_updates.mul_(torch.stack(norm_scales).view(norm_scale_shape))
                    if not self._try_apply_orthogonal_muon_update_foreach(chunk, orth_updates):
                        for orth_update, (p, _, update_mode, lr, _) in zip(
                            orth_updates.unbind(0), chunk
                        ):
                            self._apply_orthogonal_muon_update(p, orth_update, update_mode, lr)
                maybe_progress()

        if not self.fsdp_prioritize_distributed_ns:
            apply_distributed_batches()

        for key, candidate_updates in partial_distributed_batch_map.items():
            for chunk in self._iter_batched_partial_distributed_ns_chunks(candidate_updates):
                if len(chunk) < 2:
                    for update in chunk:
                        record_fallback(update, "singleton_partial_distributed_batch")
                    fallback_updates.extend(chunk)
                    continue
                batched_chunks += 1
                batched_updates += len(chunk)
                mode, full_shape, _, _, _, partition_dim = key
                with torch.autograd.profiler.record_function(
                    "Muon-FSDP batched partial distributed NS/update "
                    f"mode={mode} full_shape={full_shape} count={len(chunk)} "
                    f"partition_dim={partition_dim}"
                ):
                    self._apply_batched_partial_distributed_muon_updates(chunk)
                maybe_progress()

        for key, candidate_updates in qkv_batch_map.items():
            for chunk in self._iter_batched_ns_chunks(candidate_updates):
                if len(chunk) < 2:
                    for update in chunk:
                        record_fallback(update, "singleton_qkv_batch")
                    fallback_updates.extend(chunk)
                    continue
                batched_chunks += 1
                batched_updates += len(chunk)
                mode, shape, _, _, split_dim = key
                with torch.autograd.profiler.record_function(
                    f"Muon-FSDP batched split-QKV update mode={mode} "
                    f"shape={shape} split_dim={split_dim} count={len(chunk)}"
                ):
                    self._apply_batched_qkv_muon_updates(chunk, mode, shape, split_dim)
                maybe_progress()

        self._maybe_log_fsdp_batched_ns_summary(
            candidate_updates=(
                sum(len(updates) for updates in batch_map.values())
                + sum(len(updates) for updates in distributed_batch_map.values())
                + sum(len(updates) for updates in partial_distributed_batch_map.values())
                + sum(len(updates) for updates in qkv_batch_map.values())
            ),
            batched_chunks=batched_chunks,
            batched_updates=batched_updates,
            fallback_updates=len(fallback_updates),
            shape_counts=shape_counts,
            fallback_shape_counts=fallback_shape_counts,
        )

        for p, pre_ns_grad, update_mode, lr, group_kwargs in fallback_updates:
            precomputed_norm_scale = None
            if update_mode == "local_boundary":
                precomputed_norm_scale = get_precomputed_norm_scales().get(id(p))
            self._apply_precomputed_muon_update(
                p,
                pre_ns_grad,
                update_mode,
                lr,
                group_kwargs,
                precomputed_norm_scale=precomputed_norm_scale,
            )
            maybe_progress()

        if precomputed_norm_scales is None:
            get_precomputed_norm_scales()

    def _apply_precomputed_muon_update(
        self,
        p: torch.Tensor,
        pre_ns_grad: torch.Tensor | None,
        update_mode: str,
        lr: float,
        group_kwargs: dict[str, Any],
        precomputed_norm_scale: torch.Tensor | None = None,
    ) -> None:
        if update_mode == "gather":
            if pre_ns_grad is None:
                return
            with torch.autograd.profiler.record_function(
                f"Muon-FSDP individual NS/update mode=gather shape={tuple(pre_ns_grad.shape)}"
            ):
                orth_update = super(FSDPTensorParallelMuon, self).orthogonalize(
                    p, pre_ns_grad, **group_kwargs
                )
            self._apply_orthogonal_muon_update(p, orth_update, update_mode, lr)
            return

        assert pre_ns_grad is not None
        if update_mode in ("distributed", "partial_distributed"):
            with torch.autograd.profiler.record_function(
                f"Muon-FSDP individual NS/update mode={update_mode} "
                f"shape={tuple(pre_ns_grad.shape)}"
            ):
                orth_update = self._distributed_fsdp_orthogonalize(p, pre_ns_grad, group_kwargs).to(
                    dtype=p._local_tensor.dtype
                )
            self._apply_orthogonal_muon_update(p, orth_update, update_mode, lr)
            return

        full_shape_boundary_scale = self._uses_approx_local_boundary_full_shape_scale(
            p, update_mode
        )
        norm_scale = None
        if (
            update_mode == "local_boundary"
            and self.fsdp_approx_local_boundary_global_norm_scale
            and not full_shape_boundary_scale
        ):
            if pre_ns_grad.numel() == 0:
                return
            norm_scale = precomputed_norm_scale
            if norm_scale is None:
                norm_scale = self._get_approx_local_boundary_global_norm_scale(p, pre_ns_grad)

        with torch.autograd.profiler.record_function(
            f"Muon-FSDP individual NS/update mode={update_mode} shape={tuple(pre_ns_grad.shape)}"
        ):
            if full_shape_boundary_scale:
                orth_update = self._approx_local_boundary_orthogonalize(p, pre_ns_grad).to(
                    dtype=p._local_tensor.dtype
                )
            else:
                orth_update = super(FSDPTensorParallelMuon, self).orthogonalize(
                    p, pre_ns_grad, **group_kwargs
                )
                if norm_scale is not None:
                    orth_update.mul_(norm_scale)
                orth_update = orth_update.to(dtype=p._local_tensor.dtype)
        self._apply_orthogonal_muon_update(p, orth_update, update_mode, lr)

    def _attach_boundary_ready_events(self, batches: list[dict[str, Any]]) -> None:
        if not self.fsdp_overlap_boundary_ready_event:
            return
        if not torch.cuda.is_available():
            return

        ready_events: dict[torch.device, torch.cuda.Event] = {}
        for batch in batches:
            device = batch["device"]
            if device.type != "cuda":
                continue
            event = ready_events.get(device)
            if event is None:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    torch.cuda.current_stream(device).record_event(event)
                ready_events[device] = event
            batch["_boundary_ready_event"] = event

    def _direct_boundary_gather_item_ref(self, p: torch.Tensor) -> torch.Tensor:
        """Return a metadata tensor for direct pre-NS gather-buffer planning.

        The returned tensor is not used as data.  It only supplies shape, dtype,
        and device to the existing batch planner before the real pre-NS values
        are written directly into the stage-0 gather buffer.
        """
        mom_local = self.state[p]["momentum_buffer"]._local_tensor
        wire_dtype = self._boundary_gather_wire_dtype(mom_local.dtype)
        if p._local_tensor.dtype == wire_dtype:
            return p._local_tensor
        return torch.empty(p._local_tensor.shape, dtype=wire_dtype, device=p._local_tensor.device)

    def _start_overlap_boundary_gathers(
        self, all_updates: list, boundary_update_indices: list[int], *, direct_pre_ns: bool = False
    ) -> dict[str, Any]:
        if direct_pre_ns:
            boundary_items = [
                (all_updates[i][0], self._direct_boundary_gather_item_ref(all_updates[i][0]))
                for i in boundary_update_indices
            ]
        else:
            boundary_items = [
                (all_updates[i][0], self._prepare_boundary_gather_tensor(all_updates[i][1]))
                for i in boundary_update_indices
            ]
        gathered_boundary_updates: list[torch.Tensor | None] = [None] * len(boundary_items)
        completed_item_indices: set[int] = set()
        batches = self._build_full_uneven_local_tensor_gather_batches(
            boundary_items, gathered_boundary_updates, completed_item_indices
        )
        self._attach_boundary_ready_events(batches)

        batch_iter = iter(batches)
        pending_queue: list[tuple[dict[str, Any], int]] = []
        prefetch_batches = (
            self.fsdp_overlap_boundary_prefetch_batches if self.fsdp_overlap_comm_compute else 1
        )
        use_prefetch_scratch_scopes = (
            max(prefetch_batches, self.fsdp_overlap_boundary_post_compute_prefetch_batches) > 1
        )
        for slot in range(prefetch_batches):
            next_batch = next(batch_iter, None)
            if next_batch is None:
                break
            if use_prefetch_scratch_scopes:
                next_batch["_scratch_scope"] = ("boundary_prefetch", slot)
            pending = self._start_gather_full_uneven_local_tensor_batch_async(
                boundary_items,
                next_batch,
                all_updates=all_updates if direct_pre_ns else None,
                boundary_update_indices=boundary_update_indices if direct_pre_ns else None,
            )
            pending_queue.append((pending, slot))

        return {
            "boundary_items": boundary_items,
            "gathered_boundary_updates": gathered_boundary_updates,
            "completed_item_indices": completed_item_indices,
            "batch_iter": batch_iter,
            "pending_queue": pending_queue,
            "use_prefetch_scratch_scopes": use_prefetch_scratch_scopes,
            "direct_pre_ns": direct_pre_ns,
        }

    def _overlap_boundary_gather_and_update(
        self,
        all_updates: list,
        boundary_update_indices: list[int],
        gather_state: dict[str, Any] | None = None,
        applied_update_indices: set[int] | None = None,
    ) -> None:
        if gather_state is None:
            gather_state = self._start_overlap_boundary_gathers(
                all_updates, boundary_update_indices
            )

        boundary_items = gather_state["boundary_items"]
        gathered_boundary_updates = gather_state["gathered_boundary_updates"]
        completed_item_indices = gather_state["completed_item_indices"]
        batch_iter = gather_state["batch_iter"]
        if "pending_queue" in gather_state:
            pending_queue = gather_state["pending_queue"]
        else:
            pending_queue = []
            pending = gather_state.get("pending")
            if pending is not None:
                pending_queue.append((pending, 0))
        use_prefetch_scratch_scopes = gather_state.get(
            "use_prefetch_scratch_scopes",
            max(
                self.fsdp_overlap_boundary_prefetch_batches,
                self.fsdp_overlap_boundary_post_compute_prefetch_batches,
            )
            > 1,
        )
        direct_pre_ns = bool(gather_state.get("direct_pre_ns", False))
        applied_update_indices = applied_update_indices or set()

        deferred_distributed_updates = []
        deferred_partial_updates = []
        local_only_updates = []
        distributed_updates = []
        for idx, update in enumerate(all_updates):
            if idx in applied_update_indices:
                continue
            update_mode = update[2]
            if update_mode == "gather":
                continue
            if update_mode in ("distributed", "partial_distributed"):
                defer_update = self.fsdp_defer_distributed_ns_under_gather or (
                    update_mode == "partial_distributed"
                    and self.fsdp_defer_partial_distributed_ns_under_gather
                )
                if defer_update:
                    if (
                        update_mode == "partial_distributed"
                        and self.fsdp_async_partial_distributed_gather
                    ):
                        deferred_partial_updates.append((idx, update))
                    else:
                        deferred_distributed_updates.append((idx, update))
                else:
                    distributed_updates.append((idx, update))
                continue
            local_only_updates.append((idx, update))

        def sort_by_work(updates: list) -> None:
            updates.sort(
                key=lambda item: self._local_update_work_estimate(item[1][0], item[1][1]),
                reverse=True,
            )

        progressed_item_indices: list[int] = []

        next_scratch_slot = self.fsdp_overlap_boundary_prefetch_batches

        async_partial_pending_chunks: list[dict[str, Any]] = []
        if deferred_partial_updates:
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 2p start async partial distributed gather"
            ):
                async_partial_pending_chunks, fallback_partial_updates = (
                    self._start_async_partial_distributed_gathers(deferred_partial_updates)
                )
            deferred_distributed_updates.extend(fallback_partial_updates)

        def start_next_pending(slot: int) -> bool:
            next_batch = next(batch_iter, None)
            if next_batch is None:
                return False
            if use_prefetch_scratch_scopes:
                next_batch["_scratch_scope"] = ("boundary_prefetch", slot)
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 2c start next boundary gather"
            ):
                next_pending = self._start_gather_full_uneven_local_tensor_batch_async(
                    boundary_items,
                    next_batch,
                    all_updates=all_updates if direct_pre_ns else None,
                    boundary_update_indices=boundary_update_indices if direct_pre_ns else None,
                )
            pending_queue.append((next_pending, slot))
            return True

        def top_up_pending_queue(target_depth: int) -> None:
            nonlocal next_scratch_slot
            while len(pending_queue) < target_depth:
                slot = next_scratch_slot
                if not start_next_pending(slot):
                    return
                next_scratch_slot += 1

        def progress_boundary_gathers_during_local() -> None:
            if not self.fsdp_overlap_boundary_progress_during_local:
                return
            while pending_queue:
                pending, slot = pending_queue[0]
                if not self._overlap_gather_pending_completed(pending):
                    return
                pending_queue.pop(0)
                with torch.autograd.profiler.record_function(
                    "Muon-FSDP phase 2d progress completed boundary gather"
                ):
                    finished_item_indices = (
                        self._finish_gather_full_uneven_local_tensor_batch_async(
                            pending, gathered_boundary_updates
                        )
                    )
                progressed_item_indices.extend(finished_item_indices)
                start_next_pending(slot)

        sort_by_work(local_only_updates)
        sort_by_work(distributed_updates)
        if self.fsdp_overlap_local_ns_first:
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 3a local-only NS/update under gather"
            ):
                self._apply_precomputed_muon_updates(
                    [update for _, update in local_only_updates],
                    progress_callback=progress_boundary_gathers_during_local,
                )
            if distributed_updates:
                with torch.autograd.profiler.record_function(
                    "Muon-FSDP phase 3a distributed NS/update after local under gather"
                ):
                    self._apply_precomputed_muon_updates(
                        [update for _, update in distributed_updates],
                        progress_callback=progress_boundary_gathers_during_local,
                    )
        else:
            local_updates = local_only_updates + distributed_updates
            sort_by_work(local_updates)
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 3a local NS/update under gather"
            ):
                self._apply_precomputed_muon_updates(
                    [update for _, update in local_updates],
                    progress_callback=progress_boundary_gathers_during_local,
                )

        if self.fsdp_overlap_boundary_post_compute_prefetch_batches > len(pending_queue):
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 2e post-compute boundary gather prefetch"
            ):
                top_up_pending_queue(self.fsdp_overlap_boundary_post_compute_prefetch_batches)

        processed_item_indices: set[int] = set()
        deferred_boundary_updates_by_key: dict[Any, list] = {}

        def defer_key(update) -> tuple[str, Any] | None:
            p, pre_ns_grad, update_mode, _, _ = update
            qkv_key = self._fsdp_batched_qkv_ns_key(p, pre_ns_grad, update_mode)
            if qkv_key is not None:
                return ("qkv", qkv_key)
            ns_key = self._fsdp_batched_ns_key(p, pre_ns_grad, update_mode)
            if ns_key is not None:
                return ("ns", ns_key)
            return None

        def flush_deferred_boundary_updates(*, force: bool = False) -> None:
            ready_updates = []
            for key, updates in list(deferred_boundary_updates_by_key.items()):
                if force or len(updates) >= self.fsdp_overlap_defer_boundary_batch_size:
                    ready_updates.extend(updates)
                    del deferred_boundary_updates_by_key[key]
            if ready_updates:
                self._apply_precomputed_muon_updates(ready_updates)

        def process_completed_items(
            item_indices: set[int] | list[int], *, force: bool = False
        ) -> None:
            with torch.autograd.profiler.record_function("Muon-FSDP phase 3b boundary NS/update"):
                completed_updates = []
                for item_idx in item_indices:
                    if item_idx in processed_item_indices:
                        continue
                    update_idx = boundary_update_indices[item_idx]
                    p, local_pre_ns_grad, _, lr, group_kwargs = all_updates[update_idx]
                    gathered_pre_ns_grad = self._restore_boundary_gather_tensor(
                        gathered_boundary_updates[item_idx], local_pre_ns_grad
                    )
                    completed_updates.append((p, gathered_pre_ns_grad, "gather", lr, group_kwargs))
                    processed_item_indices.add(item_idx)
                if self.fsdp_overlap_defer_boundary_batch_size <= 1 or force:
                    self._apply_precomputed_muon_updates(completed_updates)
                    if force:
                        flush_deferred_boundary_updates(force=True)
                    return

                immediate_updates = []
                for update in completed_updates:
                    key = defer_key(update)
                    if key is None:
                        immediate_updates.append(update)
                    else:
                        deferred_boundary_updates_by_key.setdefault(key, []).append(update)
                if immediate_updates:
                    self._apply_precomputed_muon_updates(immediate_updates)
                flush_deferred_boundary_updates()

        process_completed_items(completed_item_indices)
        if progressed_item_indices:
            process_completed_items(progressed_item_indices)

        while pending_queue:
            pending, slot = pending_queue.pop(0)
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 2b finish overlapped boundary gather"
            ):
                finished_item_indices = self._finish_gather_full_uneven_local_tensor_batch_async(
                    pending, gathered_boundary_updates
                )
            start_next_pending(slot)
            process_completed_items(finished_item_indices)

        process_completed_items(set(range(len(boundary_update_indices))), force=True)

        if len(processed_item_indices) != len(boundary_update_indices):
            missing = sorted(set(range(len(boundary_update_indices))) - processed_item_indices)
            raise AssertionError(
                "Muon+M-FSDP overlap step missed boundary updates: "
                f"missing_boundary_item_indices={missing}."
            )

        if deferred_distributed_updates:
            deferred_distributed_updates.sort(
                key=lambda item: self._local_update_work_estimate(item[1][0], item[1][1]),
                reverse=True,
            )
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 3c deferred distributed NS/update"
            ):
                self._apply_precomputed_muon_updates(
                    [update for _, update in deferred_distributed_updates]
                )

        if async_partial_pending_chunks:
            with torch.autograd.profiler.record_function(
                "Muon-FSDP phase 3d async partial distributed NS/update"
            ):
                self._finish_async_partial_distributed_gathers(async_partial_pending_chunks)

    def _local_update_work_estimate(self, p: torch.Tensor, pre_ns_grad: torch.Tensor | None) -> int:
        if pre_ns_grad is not None:
            return pre_ns_grad.numel()
        return p._local_tensor.numel()

    def _needs_boundary_gather(self, dtensor: torch.Tensor) -> bool:
        assert isinstance(
            dtensor, _DTensor
        ), f"Detected non-DTensor during {type(self).__name__}: {dtensor}"
        local_tensor = dtensor._local_tensor
        return local_tensor.numel() > 0 and tuple(dtensor.shape) != tuple(local_tensor.shape)

    def _collect_boundary_indices_across_shard_groups(
        self, params: list[torch.Tensor], local_indices: list[int]
    ) -> set[int]:
        """Propagate boundary-gather decisions over all DTensor FSDP shard dimensions.

        In plain FSDP the optimizer's ``dp_group`` is the only shard group. In
        HFSDP, however, the optimizer wrapper may be scoped to the inner DP
        group while a DTensor can also be sharded on the outer DP dimension.
        Boundary parameters must be selected consistently by every rank that
        participates in any of the uneven gather stages.
        """
        result = set(local_indices)

        shard_groups: list[torch.distributed.ProcessGroup] = []
        for param in params:
            if isinstance(param, _DTensor):
                for shard_group in get_dtensor_data_parallel_shard_groups(param):
                    append_unique_process_group(shard_groups, shard_group)

        if not shard_groups and self.dp_group is not None:
            append_unique_process_group(shard_groups, self.dp_group)

        for shard_group in shard_groups:
            if get_pg_size(shard_group) <= 1:
                continue
            gathered_indices: list[list[int] | None] = [None] * get_pg_size(shard_group)
            torch.distributed.all_gather_object(gathered_indices, sorted(result), group=shard_group)
            result.update(
                idx
                for rank_indices in gathered_indices
                if rank_indices is not None
                for idx in rank_indices
            )

        return result

    def _get_mfsdp_param_layout(self, param: torch.Tensor, param_idx: int):
        """Return M-FSDP flat-buffer layout metadata for `param`.

        FSDP optimizer params keep a pointer to the original module parameter.
        M-FSDP attaches the original parameter's backing buffer and item id
        there, which lets Muon test whether the parameter's flat item interval
        crosses a shard boundary exactly.
        """
        orig_param = getattr(param, "orig_param", None)
        if orig_param is None:
            return None

        gbuf = getattr(orig_param, "_gbuf", None)
        item_id = getattr(orig_param, "_item_id", None)
        if gbuf is None or item_id is None or not hasattr(gbuf, "item_index_map"):
            raise AssertionError(
                "M-FSDP optimizer parameter is missing bucket metadata required "
                f"for Muon boundary gather detection: param_idx={param_idx}."
            )

        item_index = gbuf.item_index_map.get(item_id)
        if (
            item_index is None
            or not hasattr(item_index, "global_data_index")
            or not hasattr(item_index, "size")
        ):
            raise AssertionError(
                "M-FSDP optimizer parameter has invalid bucket item metadata required "
                f"for Muon boundary gather detection: param_idx={param_idx}, "
                f"item_id={item_id}."
            )

        bucket_index = getattr(gbuf, "bucket_index", None)
        shard_bucket_index = getattr(gbuf, "shard_bucket_index", None)
        if (
            bucket_index is None
            or shard_bucket_index is None
            or not hasattr(bucket_index, "global_data_index")
            or not hasattr(bucket_index, "size")
            or not hasattr(shard_bucket_index, "size")
        ):
            raise AssertionError(
                "M-FSDP optimizer parameter is missing bucket/shard metadata required "
                f"for Muon boundary gather detection: param_idx={param_idx}."
            )

        return gbuf, item_index, bucket_index, shard_bucket_index

    def _mfsdp_param_crosses_shard_boundary(self, param: torch.Tensor, param_idx: int) -> bool:
        """Return whether `param`'s flat M-FSDP item interval spans DP shards."""
        layout = self._get_mfsdp_param_layout(param, param_idx)
        if layout is None:
            raise AssertionError(
                "Expected M-FSDP parameter metadata while checking Muon boundary "
                f"gather requirement: param_idx={param_idx}."
            )

        gbuf, item_index, bucket_index, shard_bucket_index = layout
        if not getattr(gbuf, "is_data_distributed", True):
            return False

        item_size = int(item_index.size)
        if item_size == 0:
            return False

        bucket_start = int(bucket_index.global_data_index)
        bucket_size = int(bucket_index.size)
        shard_size = int(shard_bucket_index.size)
        if shard_size <= 0 or bucket_size % shard_size != 0:
            raise AssertionError(
                "Invalid M-FSDP shard metadata for Muon boundary gather detection: "
                f"param_idx={param_idx}, bucket_size={bucket_size}, shard_size={shard_size}."
            )

        item_start = int(item_index.global_data_index)
        item_end = item_start + item_size
        if item_start < bucket_start or item_end > bucket_start + bucket_size:
            raise AssertionError(
                "M-FSDP item interval falls outside its bucket during Muon boundary "
                f"gather detection: param_idx={param_idx}, item=({item_start}, {item_end}), "
                f"bucket=({bucket_start}, {bucket_start + bucket_size})."
            )

        first_shard = (item_start - bucket_start) // shard_size
        last_shard = (item_end - 1 - bucket_start) // shard_size
        crosses_boundary = first_shard != last_shard

        local_tensor = param._local_tensor
        if local_tensor.numel() > 0:
            local_is_partial = tuple(param.shape) != tuple(local_tensor.shape)
            if local_is_partial and not crosses_boundary:
                # HFSDP can introduce a second sharding dimension after the
                # flat bucket metadata has identified an inner-DP shard. The
                # caller globally unions these local split observations so
                # empty ranks still participate in the required collective.
                return False

        return crosses_boundary

    def _get_boundary_gather_param_indices(self, group: dict[str, Any]) -> set[int]:
        """Return globally-agreed parameters whose flat items cross FSDP shard boundaries."""
        params = group["params"]
        cache_key = tuple(id(param) for param in params)
        cached_indices = self._boundary_gather_indices_cache.get(cache_key)
        if cached_indices is not None:
            return cached_indices

        has_mfsdp_params = any(getattr(param, "orig_param", None) is not None for param in params)
        if has_mfsdp_params:
            local_boundary_indices = []
            for idx, param in enumerate(params):
                if getattr(param, "orig_param", None) is None:
                    raise AssertionError(
                        "Muon optimizer group mixes M-FSDP params with params lacking "
                        f"M-FSDP bucket metadata: param_idx={idx}."
                    )
                if self._mfsdp_param_crosses_shard_boundary(param, idx):
                    local_boundary_indices.append(idx)
                elif self._needs_boundary_gather(param):
                    local_boundary_indices.append(idx)
            result = self._collect_boundary_indices_across_shard_groups(
                params, local_boundary_indices
            )
            self._maybe_log_mfsdp_boundary_layout_summary(params, result)
            self._boundary_gather_indices_cache[cache_key] = result
            return result

        local_boundary_indices = [
            idx for idx, param in enumerate(params) if self._needs_boundary_gather(param)
        ]
        result = self._collect_boundary_indices_across_shard_groups(params, local_boundary_indices)
        self._boundary_gather_indices_cache[cache_key] = result
        return result

    def _copy_dtensor_chunk_metadata(self, dst, src) -> None:
        if hasattr(src._local_tensor, "__create_chunk_list__"):
            dst._local_tensor.__create_chunk_list__ = src._local_tensor.__create_chunk_list__
        if hasattr(src._local_tensor, "__create_write_items__"):
            dst._local_tensor.__create_write_items__ = src._local_tensor.__create_write_items__

    def _dtensor_from_local_like(self, dtensor_ref, local_tensor: torch.Tensor):
        dtensor = _DTensor.from_local(
            local_tensor=local_tensor,
            device_mesh=dtensor_ref.device_mesh,
            placements=dtensor_ref.placements,
            shape=dtensor_ref.shape,
            stride=dtensor_ref.stride(),
        )
        self._copy_dtensor_chunk_metadata(dtensor, dtensor_ref)
        return dtensor

    def _get_uneven_gather_plan(self, dtensor_ref) -> dict[str, Any] | None:
        """Return static metadata needed to gather a Megatron-FSDP uneven DTensor.

        The parameter layout is fixed after M-FSDP construction, so the chunk
        offsets/shapes only need to be exchanged once per boundary parameter.
        """
        cache_key = id(dtensor_ref)
        cached_plan = self._uneven_gather_plan_cache.get(cache_key)
        if cached_plan is not None:
            return cached_plan

        shard_mesh_dims = []
        for mesh_dim, placement in enumerate(dtensor_ref.placements):
            if isinstance(placement, (Shard, _StridedShard)):
                shard_mesh_dims.append(mesh_dim)
            elif isinstance(placement, Replicate):
                continue
            else:
                raise ValueError(
                    f"Unexpected placement {placement} at mesh dimension {mesh_dim}. "
                    "Expected Shard, _StridedShard, or Replicate."
                )

        if not shard_mesh_dims:
            return None

        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            update_uneven_dtensor_chunk_metadata(dtensor_ref)

        chunk_metadata_list = dtensor_ref._local_tensor.__create_chunk_list__()
        if len(chunk_metadata_list) != 1:
            raise ValueError(
                f"Expected exactly one chunk metadata per rank, got {len(chunk_metadata_list)}."
            )

        local_tensor = dtensor_ref._local_tensor
        local_chunk_metadata = chunk_metadata_list[0]

        def _normalize_chunk_info(chunk_info: dict[str, Any]) -> dict[str, Any]:
            shape = torch.Size(chunk_info["shape"])
            return {"shape": shape, "offset": tuple(chunk_info["offset"]), "numel": shape.numel()}

        local_chunks_info = [
            _normalize_chunk_info(
                {
                    "shape": torch.Size(local_tensor.shape),
                    "offset": tuple(local_chunk_metadata.offsets),
                }
            )
        ]
        stages = []
        for shard_mesh_dim in shard_mesh_dims:
            shard_group = dtensor_ref.device_mesh.get_group(shard_mesh_dim)
            group_chunks_info: list[list[dict[str, Any]] | None] = [None] * shard_group.size()
            torch.distributed.all_gather_object(
                group_chunks_info, local_chunks_info, group=shard_group
            )
            if any(chunks_info is None for chunks_info in group_chunks_info):
                raise AssertionError(
                    "Uneven DTensor gather metadata exchange returned an incomplete result."
                )

            stage_chunks_info = [
                [_normalize_chunk_info(chunk_info) for chunk_info in chunks_info]
                for chunks_info in group_chunks_info
                if chunks_info is not None
            ]
            stages.append(
                {
                    "shard_group": shard_group,
                    "rank_numels": [
                        sum(chunk_info["numel"] for chunk_info in chunks_info)
                        for chunks_info in stage_chunks_info
                    ],
                    "rank_chunk_counts": [len(chunks_info) for chunks_info in stage_chunks_info],
                }
            )
            local_chunks_info = [
                chunk_info for chunks_info in stage_chunks_info for chunk_info in chunks_info
            ]

        chunk_infos = local_chunks_info
        full_shape = torch.Size(dtensor_ref.shape)
        plan = {
            "shard_mesh_dims": tuple(shard_mesh_dims),
            "stages": stages,
            "chunk_infos": chunk_infos,
            "full_numel": full_shape.numel(),
            "is_contiguous_full_order": _chunk_infos_are_contiguous_full_order(
                full_shape, chunk_infos
            ),
        }
        if len(stages) == 1:
            plan["shard_group"] = stages[0]["shard_group"]
        self._uneven_gather_plan_cache[cache_key] = plan
        return plan

    def _get_global_max_original_local_numel(self, dtensor_ref) -> int:
        """Return a rank-consistent max original local shard size for an uneven DTensor."""
        plan = self._get_uneven_gather_plan(dtensor_ref)
        if plan is None:
            return int(dtensor_ref._local_tensor.numel())
        return max((int(chunk_info["numel"]) for chunk_info in plan["chunk_infos"]), default=0)

    def _get_partial_uneven_gather_plan(
        self, dtensor_ref, gather_mesh_dims: tuple[int, ...]
    ) -> dict[str, Any] | None:
        """Return a gather plan over only selected DTensor shard mesh dimensions."""
        cache_key = (id(dtensor_ref), gather_mesh_dims)
        if cache_key in self._partial_uneven_gather_plan_cache:
            return self._partial_uneven_gather_plan_cache[cache_key]

        if not gather_mesh_dims:
            self._partial_uneven_gather_plan_cache[cache_key] = None
            return None

        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            update_uneven_dtensor_chunk_metadata(dtensor_ref)

        chunk_metadata_list = dtensor_ref._local_tensor.__create_chunk_list__()
        if len(chunk_metadata_list) != 1:
            self._partial_uneven_gather_plan_cache[cache_key] = None
            return None

        local_tensor = dtensor_ref._local_tensor
        local_chunk_metadata = chunk_metadata_list[0]

        def _normalize_chunk_info(chunk_info: dict[str, Any]) -> dict[str, Any]:
            shape = torch.Size(chunk_info["shape"])
            return {"shape": shape, "offset": tuple(chunk_info["offset"]), "numel": shape.numel()}

        local_chunks_info = [
            _normalize_chunk_info(
                {
                    "shape": torch.Size(local_tensor.shape),
                    "offset": tuple(local_chunk_metadata.offsets),
                }
            )
        ]
        stages = []
        for shard_mesh_dim in gather_mesh_dims:
            shard_group = dtensor_ref.device_mesh.get_group(shard_mesh_dim)
            group_chunks_info: list[list[dict[str, Any]] | None] = [None] * shard_group.size()
            torch.distributed.all_gather_object(
                group_chunks_info, local_chunks_info, group=shard_group
            )
            if any(chunks_info is None for chunks_info in group_chunks_info):
                self._partial_uneven_gather_plan_cache[cache_key] = None
                return None

            stage_chunks_info = [
                [_normalize_chunk_info(chunk_info) for chunk_info in chunks_info]
                for chunks_info in group_chunks_info
                if chunks_info is not None
            ]
            stages.append(
                {
                    "shard_group": shard_group,
                    "rank_numels": [
                        sum(chunk_info["numel"] for chunk_info in chunks_info)
                        for chunks_info in stage_chunks_info
                    ],
                    "rank_chunk_counts": [len(chunks_info) for chunks_info in stage_chunks_info],
                }
            )
            local_chunks_info = [
                chunk_info for chunks_info in stage_chunks_info for chunk_info in chunks_info
            ]

        full_shape = torch.Size(dtensor_ref.shape)
        partial_offsets = []
        partial_shape = []
        for dim, full_dim in enumerate(full_shape):
            starts = [chunk_info["offset"][dim] for chunk_info in local_chunks_info]
            ends = [
                chunk_info["offset"][dim] + chunk_info["shape"][dim]
                for chunk_info in local_chunks_info
            ]
            start = min(starts)
            end = max(ends)
            if start < 0 or end > full_dim:
                self._partial_uneven_gather_plan_cache[cache_key] = None
                return None
            partial_offsets.append(start)
            partial_shape.append(end - start)

        normalized_chunk_infos = []
        assigned_numel = 0
        for chunk_info in local_chunks_info:
            normalized_offset = tuple(
                chunk_info["offset"][dim] - partial_offsets[dim]
                for dim in range(len(partial_shape))
            )
            normalized = {
                "shape": chunk_info["shape"],
                "offset": normalized_offset,
                "numel": chunk_info["numel"],
            }
            normalized_chunk_infos.append(normalized)
            assigned_numel += chunk_info["numel"]

        try:
            _assert_chunks_cover_full_tensor(
                torch.Size(partial_shape), normalized_chunk_infos, assigned_numel
            )
        except AssertionError:
            self._partial_uneven_gather_plan_cache[cache_key] = None
            return None

        plan = {
            "stages": stages,
            "chunk_infos": local_chunks_info,
            "partial_shape": torch.Size(partial_shape),
            "partial_offsets": tuple(partial_offsets),
            "partial_numel": assigned_numel,
            "is_contiguous_full_order": False,
        }
        self._partial_uneven_gather_plan_cache[cache_key] = plan
        return plan

    def _get_flat_uneven_gather_plan(
        self, dtensor_ref, plan: dict[str, Any], *, require_flat_enabled: bool = True
    ) -> dict[str, Any] | None:
        if require_flat_enabled and not self.fsdp_flat_batched_all_gather:
            return None
        if len(plan["stages"]) <= 1:
            return None
        if not plan["is_contiguous_full_order"]:
            return None

        shard_placement_types = (Shard, _StridedShard)
        for mesh_dim in plan["shard_mesh_dims"]:
            placement = dtensor_ref.placements[mesh_dim]
            if not isinstance(placement, shard_placement_types):
                return None
            if getattr(placement, "dim", None) != 0:
                return None

        flat_group = self._get_existing_flat_uneven_gather_group(dtensor_ref, plan)
        if flat_group is None:
            return None

        cache_key = (id(dtensor_ref), id(flat_group))
        if cache_key in self._flat_uneven_gather_plan_cache:
            return self._flat_uneven_gather_plan_cache[cache_key]

        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            update_uneven_dtensor_chunk_metadata(dtensor_ref)

        chunk_metadata_list = dtensor_ref._local_tensor.__create_chunk_list__()
        if len(chunk_metadata_list) != 1:
            self._flat_uneven_gather_plan_cache[cache_key] = None
            return None

        local_chunk_metadata = chunk_metadata_list[0]

        def _normalize_chunk_info(chunk_info: dict[str, Any]) -> dict[str, Any]:
            shape = torch.Size(chunk_info["shape"])
            return {"shape": shape, "offset": tuple(chunk_info["offset"]), "numel": shape.numel()}

        local_chunk_info = _normalize_chunk_info(
            {
                "shape": torch.Size(dtensor_ref._local_tensor.shape),
                "offset": tuple(local_chunk_metadata.offsets),
            }
        )

        flat_group_size = get_pg_size(flat_group)
        group_chunks_info: list[dict[str, Any] | None] = [None] * flat_group_size
        torch.distributed.all_gather_object(group_chunks_info, local_chunk_info, group=flat_group)
        if any(chunk_info is None for chunk_info in group_chunks_info):
            self._flat_uneven_gather_plan_cache[cache_key] = None
            return None

        flat_chunk_infos = [
            _normalize_chunk_info(chunk_info)
            for chunk_info in group_chunks_info
            if chunk_info is not None
        ]
        flat_rank_numels = [chunk_info["numel"] for chunk_info in flat_chunk_infos]
        assigned_numel = sum(flat_rank_numels)
        try:
            _assert_chunks_cover_full_tensor(dtensor_ref.shape, flat_chunk_infos, assigned_numel)
        except AssertionError:
            self._flat_uneven_gather_plan_cache[cache_key] = None
            return None

        full_shape = torch.Size(dtensor_ref.shape)

        def _chunk_flat_start(chunk_info: dict[str, Any]) -> int:
            flat_start = 0
            stride = 1
            for dim in range(len(full_shape) - 1, -1, -1):
                flat_start += chunk_info["offset"][dim] * stride
                stride *= full_shape[dim]
            return flat_start

        flat_sorted_rank_indices = sorted(
            range(flat_group_size), key=lambda rank: _chunk_flat_start(flat_chunk_infos[rank])
        )
        flat_sorted_chunk_infos = [flat_chunk_infos[rank] for rank in flat_sorted_rank_indices]
        active_rank_indices = tuple(
            rank for rank, rank_numel in enumerate(flat_rank_numels) if rank_numel > 0
        )
        active_group = flat_group
        if self.fsdp_flat_batched_all_gather_nonempty_group:
            active_group = self._get_flat_nonempty_gather_group(flat_group, active_rank_indices)
            if active_group is None:
                self._flat_uneven_gather_plan_cache[cache_key] = None
                return None
        flat_plan = {
            "flat_group": flat_group,
            "flat_active_group": active_group,
            "flat_group_size": flat_group_size,
            "flat_group_rank": torch.distributed.get_rank(flat_group),
            "flat_rank_numels": flat_rank_numels,
            "flat_active_rank_indices": active_rank_indices,
            "flat_active_rank_numels": [flat_rank_numels[rank] for rank in active_rank_indices],
            "flat_chunk_infos": flat_chunk_infos,
            "flat_sorted_rank_indices": flat_sorted_rank_indices,
            "flat_sorted_is_contiguous_full_order": _chunk_infos_are_contiguous_full_order(
                full_shape, flat_sorted_chunk_infos
            ),
        }
        self._flat_uneven_gather_plan_cache[cache_key] = flat_plan
        return flat_plan

    def _is_split_qkv_param(self, param: torch.Tensor) -> bool:
        return bool(self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(param))

    def _get_fsdp_distributed_ns_group(self, dtensor_ref) -> torch.distributed.ProcessGroup | None:
        if not self.fsdp_distributed_ns:
            return None
        if self.fsdp_distributed_ns_exclude_qkv and self._is_split_qkv_param(dtensor_ref):
            return None
        if (
            self.fsdp_distributed_ns_min_numel > 0
            and dtensor_ref.numel() < self.fsdp_distributed_ns_min_numel
        ):
            small_col_dim = self.fsdp_distributed_ns_small_col_dim
            if (
                small_col_dim <= 0
                or len(dtensor_ref.shape) < 2
                or dtensor_ref.shape[-1] > small_col_dim
            ):
                return None
        if len(dtensor_ref.shape) < 2 or dtensor_ref.shape[-2] <= dtensor_ref.shape[-1]:
            return None

        if self._is_split_qkv_param(dtensor_ref):
            split_dim = self._qkv_split_dim_for_shape(tuple(dtensor_ref.shape))
            if split_dim != 0:
                return None

        plan = self._get_uneven_gather_plan(dtensor_ref)
        if plan is None or not plan["is_contiguous_full_order"]:
            return None

        shard_placement_types = (Shard, _StridedShard)
        for mesh_dim in plan["shard_mesh_dims"]:
            placement = dtensor_ref.placements[mesh_dim]
            if not isinstance(placement, shard_placement_types):
                return None
            if getattr(placement, "dim", None) != 0:
                return None

        if len(plan["stages"]) == 1:
            return plan["stages"][0]["shard_group"]

        flat_plan = self._get_flat_uneven_gather_plan(dtensor_ref, plan, require_flat_enabled=False)
        if flat_plan is None:
            return None
        return flat_plan["flat_group"]

    def _get_fsdp_distributed_ns_group_rank_numels(
        self, dtensor_ref, fsdp_group: torch.distributed.ProcessGroup
    ) -> list[int] | None:
        plan = self._get_uneven_gather_plan(dtensor_ref)
        if plan is None:
            return None
        if len(plan["stages"]) == 1 and plan["stages"][0]["shard_group"] is fsdp_group:
            return plan["stages"][0]["rank_numels"]

        flat_plan = self._get_flat_uneven_gather_plan(dtensor_ref, plan, require_flat_enabled=False)
        if flat_plan is not None and flat_plan["flat_group"] is fsdp_group:
            return flat_plan["flat_rank_numels"]
        return None

    def _get_fsdp_distributed_ns_runtime_group(
        self, dtensor_ref, pre_ns_grad: torch.Tensor
    ) -> tuple[torch.distributed.ProcessGroup | None, bool]:
        fsdp_group = self._get_fsdp_distributed_ns_group(dtensor_ref)
        if fsdp_group is None:
            return None, False
        if not self.fsdp_distributed_ns_nonempty_group:
            return fsdp_group, True

        rank_numels = self._get_fsdp_distributed_ns_group_rank_numels(dtensor_ref, fsdp_group)
        if rank_numels is None:
            return fsdp_group, True

        group_ranks = torch.distributed.get_process_group_ranks(fsdp_group)
        if len(group_ranks) != len(rank_numels):
            return fsdp_group, True

        nonempty_global_ranks = tuple(
            rank for rank, rank_numel in zip(group_ranks, rank_numels) if rank_numel > 0
        )
        if len(nonempty_global_ranks) == len(group_ranks):
            return fsdp_group, True

        if not nonempty_global_ranks:
            return None, False

        global_rank = torch.distributed.get_rank()
        participates = global_rank in nonempty_global_ranks
        if not participates and pre_ns_grad.numel() != 0:
            raise AssertionError(
                "Muon+M-FSDP distributed NS nonempty subgroup excluded a rank " "with local data."
            )

        if len(nonempty_global_ranks) == 1:
            return None, participates

        cached_group = self._fsdp_nonempty_group_cache.get(nonempty_global_ranks)
        if cached_group is None:
            cached_group = torch.distributed.new_group(ranks=list(nonempty_global_ranks))
            self._fsdp_nonempty_group_cache[nonempty_global_ranks] = cached_group
        return cached_group, participates

    def _get_fsdp_partial_distributed_ns_plan(self, dtensor_ref) -> dict[str, Any] | None:
        if not self.fsdp_partial_distributed_ns or not self.fsdp_distributed_ns:
            return None
        if self.fsdp_boundary_gather_dtype not in ("fp32", "bf16"):
            return None
        if self._is_split_qkv_param(dtensor_ref):
            return None
        if len(dtensor_ref.shape) != 2:
            return None
        if (
            self.fsdp_distributed_ns_min_numel > 0
            and dtensor_ref.numel() < self.fsdp_distributed_ns_min_numel
        ):
            small_col_dim = self.fsdp_distributed_ns_small_col_dim
            if (
                small_col_dim <= 0
                or len(dtensor_ref.shape) < 2
                or dtensor_ref.shape[-1] > small_col_dim
            ):
                return None

        rows, cols = int(dtensor_ref.shape[-2]), int(dtensor_ref.shape[-1])
        partition_dim = 0 if rows > cols else 1
        partition_mesh_dims = []
        gather_mesh_dims = []
        shard_placement_types = (Shard, _StridedShard)
        for mesh_dim, placement in enumerate(dtensor_ref.placements):
            if isinstance(placement, Replicate):
                continue
            if not isinstance(placement, shard_placement_types):
                return None
            shard_dim = getattr(placement, "dim", None)
            if shard_dim not in (0, 1):
                return None
            if shard_dim == partition_dim:
                partition_mesh_dims.append(mesh_dim)
            else:
                gather_mesh_dims.append(mesh_dim)

        if not partition_mesh_dims or not gather_mesh_dims:
            return None

        partition_group = self._get_dtensor_mesh_dims_group(dtensor_ref, tuple(partition_mesh_dims))
        if partition_group is None:
            return None

        gather_plan = self._get_partial_uneven_gather_plan(dtensor_ref, tuple(gather_mesh_dims))
        if gather_plan is None:
            return None

        return {
            "partition_dim": partition_dim,
            "partition_group": partition_group,
            "gather_mesh_dims": tuple(gather_mesh_dims),
            "partition_mesh_dims": tuple(partition_mesh_dims),
            "gather_plan": gather_plan,
        }

    def _local_dtensor_row_offset(self, dtensor_ref) -> int:
        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            update_uneven_dtensor_chunk_metadata(dtensor_ref)
        chunk_metadata_list = dtensor_ref._local_tensor.__create_chunk_list__()
        if len(chunk_metadata_list) != 1:
            raise ValueError(
                f"Expected exactly one local DTensor chunk, got {len(chunk_metadata_list)}."
            )
        offsets = tuple(chunk_metadata_list[0].offsets)
        return int(offsets[0]) if offsets else 0

    def _split_local_qkv_components(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[tuple[int, int, int, int]]]]:
        assert self.qkv_split_shapes is not None
        local_row_start = self._local_dtensor_row_offset(p)
        local_row_end = local_row_start + pre_ns_grad.shape[0]
        component_prefix = [0]
        for split_size in self.qkv_split_shapes:
            component_prefix.append(component_prefix[-1] + split_size)

        group_rows = component_prefix[-1]
        if p.shape[0] % group_rows != 0:
            raise ValueError(
                "Split-QKV distributed NS expected the full row count to be "
                f"divisible by {group_rows}, got shape={tuple(p.shape)}."
            )
        num_query_groups = p.shape[0] // group_rows

        component_pieces: list[list[torch.Tensor]] = [[] for _ in self.qkv_split_shapes]
        component_segments: list[list[tuple[int, int, int, int]]] = [
            [] for _ in self.qkv_split_shapes
        ]
        component_offsets = [0 for _ in self.qkv_split_shapes]

        for group_idx in range(num_query_groups):
            group_start = group_idx * group_rows
            for component_idx, _ in enumerate(self.qkv_split_shapes):
                global_start = group_start + component_prefix[component_idx]
                global_end = group_start + component_prefix[component_idx + 1]
                overlap_start = max(local_row_start, global_start)
                overlap_end = min(local_row_end, global_end)
                if overlap_start >= overlap_end:
                    continue

                local_start = overlap_start - local_row_start
                local_end = overlap_end - local_row_start
                component_start = component_offsets[component_idx]
                component_end = component_start + (overlap_end - overlap_start)
                component_offsets[component_idx] = component_end
                component_pieces[component_idx].append(pre_ns_grad[local_start:local_end])
                component_segments[component_idx].append(
                    (local_start, local_end, component_start, component_end)
                )

        components = []
        for pieces in component_pieces:
            if pieces:
                components.append(torch.cat(pieces, dim=0))
            else:
                components.append(pre_ns_grad.new_empty((0, pre_ns_grad.shape[1])))
        return components, component_segments

    def _fsdp_boundary_update_mode(self, param: torch.Tensor) -> str:
        if self._get_fsdp_distributed_ns_group(param) is not None and not (
            self.fsdp_approx_distributed_ns_update and self.fsdp_approx_local_boundary_update
        ):
            return "distributed"
        if self._get_fsdp_partial_distributed_ns_plan(param) is not None:
            return "partial_distributed"
        if self.fsdp_approx_local_boundary_update:
            if self.fsdp_approx_local_boundary_exclude_qkv and self._is_split_qkv_param(param):
                return "gather"
            if (
                self.fsdp_approx_local_boundary_max_local_numel > 0
                and self._get_global_max_original_local_numel(param)
                > self.fsdp_approx_local_boundary_max_local_numel
            ):
                return "gather"
            return "local_boundary"
        return "gather"

    def _distributed_fsdp_orthogonalize(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor, group_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        partial_plan = self._get_fsdp_partial_distributed_ns_plan(p)
        if partial_plan is not None:
            partial_pre_ns_grad = self._gather_partial_distributed_pre_ns(
                p, pre_ns_grad, partial_plan
            )
            orth_update = newton_schulz_tp(
                partial_pre_ns_grad,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,
                tp_group=partial_plan["partition_group"],
                partition_dim=int(partial_plan["partition_dim"]),
                tp_mode="distributed",
                use_syrk=self.use_syrk,
                distributed_gram_recurrence=self.fsdp_distributed_ns_single_all_reduce,
                distributed_gram_refresh_interval=self.fsdp_distributed_ns_gram_refresh_interval,
            )
            scale_factor = get_muon_scale_factor(p.shape[-2], p.shape[-1], mode=self.scale_mode)
            orth_update.mul_(scale_factor * self.extra_scale_factor)
            return self._local_shard_from_partial_update_like(p, orth_update)

        fsdp_group, participates = self._get_fsdp_distributed_ns_runtime_group(p, pre_ns_grad)
        if not participates:
            return torch.empty_like(pre_ns_grad)
        if fsdp_group is None and not self.fsdp_distributed_ns_nonempty_group:
            raise AssertionError(
                "Muon+M-FSDP distributed NS was requested for an ineligible param."
            )

        if self._is_split_qkv_param(p):
            components, component_segments = self._split_local_qkv_components(p, pre_ns_grad)
            orth_update = torch.empty_like(pre_ns_grad)
            assert self.qkv_split_shapes is not None
            num_query_groups = p.shape[0] // sum(self.qkv_split_shapes)
            for component_idx, (component, segments) in enumerate(
                zip(components, component_segments)
            ):
                with torch.autograd.profiler.record_function(
                    "Muon-FSDP distributed split-QKV NS/update "
                    f"component={component_idx} local_shape={tuple(component.shape)}"
                ):
                    if fsdp_group is None:
                        orth_component = self.scaled_orthogonalize_fn(component, None, None)
                    else:
                        orth_component = newton_schulz_tp(
                            component,
                            steps=self.num_ns_steps,
                            coefficient_type=self.coefficient_type,
                            tp_group=fsdp_group,
                            partition_dim=0,
                            tp_mode="distributed",
                            use_syrk=self.use_syrk,
                            distributed_gram_recurrence=self.fsdp_distributed_ns_single_all_reduce,
                            distributed_gram_refresh_interval=(
                                self.fsdp_distributed_ns_gram_refresh_interval
                            ),
                        )
                        scale_factor = get_muon_scale_factor(
                            num_query_groups * self.qkv_split_shapes[component_idx],
                            p.shape[-1],
                            mode=self.scale_mode,
                        )
                        orth_component.mul_(scale_factor * self.extra_scale_factor)
                for local_start, local_end, component_start, component_end in segments:
                    orth_update[local_start:local_end].copy_(
                        orth_component[component_start:component_end]
                    )
            return orth_update

        if fsdp_group is None:
            orth_update = self.scaled_orthogonalize_fn(pre_ns_grad, None, None)
        else:
            orth_update = newton_schulz_tp(
                pre_ns_grad,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,
                tp_group=fsdp_group,
                partition_dim=0,
                tp_mode="distributed",
                use_syrk=self.use_syrk,
                distributed_gram_recurrence=self.fsdp_distributed_ns_single_all_reduce,
                distributed_gram_refresh_interval=self.fsdp_distributed_ns_gram_refresh_interval,
            )
            scale_factor = get_muon_scale_factor(p.shape[-2], p.shape[-1], mode=self.scale_mode)
            orth_update.mul_(scale_factor * self.extra_scale_factor)
        return orth_update

    def _prepare_padded_all_gather_buffers(
        self,
        local_buffer: torch.Tensor,
        rank_total_numels: list[int],
        shard_group: torch.distributed.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        max_rank_numel = max(rank_total_numels)
        group_size = len(rank_total_numels)
        if max_rank_numel == 0:
            raise AssertionError("Cannot padded-all-gather an empty uneven DTensor batch.")

        if local_buffer.numel() == max_rank_numel:
            padded_local_buffer = local_buffer
        else:
            padded_local_buffer = self._get_fsdp_gather_scratch_tensor(
                ("padded_local_buffer", id(shard_group), local_buffer.dtype, local_buffer.device),
                max_rank_numel,
                dtype=local_buffer.dtype,
                device=local_buffer.device,
            )
            if local_buffer.numel() > 0:
                padded_local_buffer[: local_buffer.numel()].copy_(local_buffer)
            if self.fsdp_padded_all_gather_zero_pad:
                padded_local_buffer[local_buffer.numel() : max_rank_numel].zero_()

        gathered_padded_buffer = self._get_fsdp_gather_scratch_tensor(
            (
                "gathered_padded_buffer",
                id(shard_group),
                local_buffer.dtype,
                local_buffer.device,
                group_size,
            ),
            group_size * max_rank_numel,
            dtype=local_buffer.dtype,
            device=local_buffer.device,
        )
        return padded_local_buffer, gathered_padded_buffer, max_rank_numel

    def _all_gather_padded_equal_size(
        self,
        local_buffer: torch.Tensor,
        rank_total_numels: list[int],
        shard_group: torch.distributed.ProcessGroup,
    ) -> tuple[torch.Tensor, int]:
        padded_local_buffer, gathered_padded_buffer, max_rank_numel = (
            self._prepare_padded_all_gather_buffers(local_buffer, rank_total_numels, shard_group)
        )
        torch.distributed.all_gather_into_tensor(
            gathered_padded_buffer, padded_local_buffer, group=shard_group
        )
        return gathered_padded_buffer, max_rank_numel

    def _flatten_tensor_for_uneven_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_contiguous():
            return tensor.view(-1)
        return tensor.reshape(-1)

    def _rank_buffers_from_pending_uneven_gather_stage(
        self, pending_stage: dict[str, Any]
    ) -> list[torch.Tensor]:
        group_size = pending_stage["group_size"]
        if pending_stage.get("skipped_nonempty_collective", False):
            local_buffer = pending_stage["local_buffer"]
            return [
                torch.empty(0, dtype=local_buffer.dtype, device=local_buffer.device)
                for _ in range(group_size)
            ]

        if pending_stage["use_padded_all_gather"]:
            gathered_padded_buffer = pending_stage["gathered_padded_buffer"]
            max_rank_numel = pending_stage["max_rank_numel"]
            collective_rank_indices = pending_stage.get(
                "collective_rank_indices", range(group_size)
            )
            rank_buffers = [
                gathered_padded_buffer[rank * max_rank_numel : (rank + 1) * max_rank_numel]
                for rank in range(len(collective_rank_indices))
            ]
        else:
            rank_buffers = pending_stage["group_tensors"]

        collective_rank_indices = pending_stage.get("collective_rank_indices")
        if collective_rank_indices is None:
            return rank_buffers

        local_buffer = pending_stage["local_buffer"]
        full_rank_buffers = [
            torch.empty(0, dtype=local_buffer.dtype, device=local_buffer.device)
            for _ in range(group_size)
        ]
        for collective_rank, original_rank in enumerate(collective_rank_indices):
            full_rank_buffers[original_rank] = rank_buffers[collective_rank]
        return full_rank_buffers

    def _reconstruct_full_tensor_from_rank_buffers(
        self,
        dtensor_ref,
        plan: dict[str, Any],
        rank_buffers: list[torch.Tensor],
        rank_buffer_offsets: list[int] | None = None,
    ) -> torch.Tensor:
        if self.fsdp_fast_reconstruct and plan["is_contiguous_full_order"]:
            chunks = []
            assigned_numel = 0
            for rank, rank_buffer in enumerate(rank_buffers):
                chunk_info = plan["chunk_infos"][rank]
                chunk_numel = chunk_info["numel"]
                if chunk_numel == 0:
                    continue
                source_offset = 0 if rank_buffer_offsets is None else rank_buffer_offsets[rank]
                chunks.append(rank_buffer[source_offset : source_offset + chunk_numel])
                assigned_numel += chunk_numel
            if assigned_numel != plan["full_numel"]:
                raise AssertionError(
                    "Fast uneven DTensor reconstruction did not cover the full tensor: "
                    f"assigned={assigned_numel}, expected={plan['full_numel']}."
                )
            if assigned_numel == 0:
                return torch.empty(
                    dtensor_ref.shape, dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
                )
            if len(chunks) == 1:
                full_flat = chunks[0].clone()
            else:
                full_flat = torch.cat(chunks)
            return full_flat.view(dtensor_ref.shape)

        full_tensor = torch.empty(
            dtensor_ref.shape, dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
        )
        assigned_numel = 0
        for rank, rank_buffer in enumerate(rank_buffers):
            chunk_info = plan["chunk_infos"][rank]
            chunk_numel = chunk_info["numel"]
            if chunk_numel == 0:
                continue
            source_offset = 0 if rank_buffer_offsets is None else rank_buffer_offsets[rank]
            chunk_shape = chunk_info["shape"]
            chunk_tensor = rank_buffer[source_offset : source_offset + chunk_numel].view(
                chunk_shape
            )
            slices = tuple(slice(o, o + s) for o, s in zip(chunk_info["offset"], chunk_shape))
            full_tensor[slices] = chunk_tensor
            assigned_numel += chunk_numel

        _assert_chunks_cover_full_tensor(dtensor_ref.shape, plan["chunk_infos"], assigned_numel)
        return full_tensor

    def _reconstruct_full_tensor_from_flat_buffer(
        self, dtensor_ref, plan: dict[str, Any], full_flat_buffer: torch.Tensor
    ) -> torch.Tensor:
        if full_flat_buffer.numel() != plan["full_numel"]:
            raise AssertionError(
                "Uneven DTensor flat reconstruction buffer size mismatch: "
                f"got {full_flat_buffer.numel()}, expected {plan['full_numel']}."
            )

        if self.fsdp_fast_reconstruct and plan["is_contiguous_full_order"]:
            return full_flat_buffer.view(dtensor_ref.shape)

        full_tensor = torch.empty(
            dtensor_ref.shape, dtype=full_flat_buffer.dtype, device=full_flat_buffer.device
        )
        assigned_numel = 0
        buffer_offset = 0
        for chunk_info in plan["chunk_infos"]:
            chunk_shape = chunk_info["shape"]
            chunk_numel = chunk_info["numel"]
            gathered_tensor = full_flat_buffer[buffer_offset : buffer_offset + chunk_numel]
            buffer_offset += chunk_numel
            offset = chunk_info["offset"]
            slices = tuple(slice(o, o + s) for o, s in zip(offset, chunk_shape))
            full_tensor[slices] = gathered_tensor.view(chunk_shape)
            assigned_numel += chunk_numel

        _assert_chunks_cover_full_tensor(dtensor_ref.shape, plan["chunk_infos"], assigned_numel)
        return full_tensor

    def _reconstruct_full_tensor_from_final_stage_buffers(
        self,
        dtensor_ref,
        plan: dict[str, Any],
        stage_idx: int,
        rank_buffers: list[torch.Tensor],
        rank_buffer_offsets: list[int],
    ) -> torch.Tensor:
        if stage_idx != len(plan["stages"]) - 1:
            raise AssertionError(
                "Direct uneven DTensor reconstruction requires the final gather stage: "
                f"got stage={stage_idx}, final={len(plan['stages']) - 1}."
            )

        stage = plan["stages"][stage_idx]
        if len(rank_buffers) != len(stage["rank_numels"]):
            raise AssertionError(
                "Uneven DTensor reconstruction rank buffer count mismatch: "
                f"got {len(rank_buffers)}, expected {len(stage['rank_numels'])}."
            )

        rank_chunk_counts = stage["rank_chunk_counts"]
        if self.fsdp_fast_reconstruct and plan["is_contiguous_full_order"]:
            chunks = []
            assigned_numel = 0
            for rank, rank_buffer in enumerate(rank_buffers):
                rank_numel = stage["rank_numels"][rank]
                if rank_numel == 0:
                    continue
                source_offset = rank_buffer_offsets[rank]
                chunks.append(rank_buffer[source_offset : source_offset + rank_numel])
                assigned_numel += rank_numel
            if assigned_numel != plan["full_numel"]:
                raise AssertionError(
                    "Fast batched uneven DTensor reconstruction did not cover the full tensor: "
                    f"assigned={assigned_numel}, expected={plan['full_numel']}."
                )
            if assigned_numel == 0:
                return torch.empty(
                    dtensor_ref.shape, dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
                )
            if len(chunks) == 1:
                full_flat = chunks[0].clone()
            else:
                full_flat = torch.cat(chunks)
            return full_flat.view(dtensor_ref.shape)

        full_tensor = torch.empty(
            dtensor_ref.shape, dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
        )
        assigned_numel = 0
        chunk_info_idx = 0
        for rank, rank_buffer in enumerate(rank_buffers):
            source_offset = rank_buffer_offsets[rank]
            expected_source_end = source_offset + stage["rank_numels"][rank]
            for _ in range(rank_chunk_counts[rank]):
                chunk_info = plan["chunk_infos"][chunk_info_idx]
                chunk_info_idx += 1
                chunk_shape = chunk_info["shape"]
                chunk_numel = chunk_info["numel"]
                if chunk_numel == 0:
                    continue
                gathered_tensor = rank_buffer[source_offset : source_offset + chunk_numel]
                source_offset += chunk_numel
                offset = chunk_info["offset"]
                slices = tuple(slice(o, o + s) for o, s in zip(offset, chunk_shape))
                full_tensor[slices] = gathered_tensor.view(chunk_shape)
                assigned_numel += chunk_numel
            if source_offset != expected_source_end:
                raise AssertionError(
                    "Batched uneven DTensor reconstruction consumed an unexpected rank size: "
                    f"rank={rank}, consumed={source_offset - rank_buffer_offsets[rank]}, "
                    f"expected={stage['rank_numels'][rank]}."
                )

        if chunk_info_idx != len(plan["chunk_infos"]):
            raise AssertionError(
                "Batched uneven DTensor reconstruction consumed an unexpected chunk count: "
                f"consumed={chunk_info_idx}, expected={len(plan['chunk_infos'])}."
            )
        _assert_chunks_cover_full_tensor(dtensor_ref.shape, plan["chunk_infos"], assigned_numel)
        return full_tensor

    def _reconstruct_full_tensor_from_flat_rank_buffers(
        self,
        dtensor_ref,
        flat_plan: dict[str, Any],
        rank_buffers: list[torch.Tensor],
        rank_buffer_offsets: list[int],
    ) -> torch.Tensor:
        flat_chunk_infos = flat_plan["flat_chunk_infos"]
        flat_rank_numels = flat_plan["flat_rank_numels"]
        if len(rank_buffers) != len(flat_rank_numels):
            raise AssertionError(
                "Flat uneven DTensor reconstruction rank buffer count mismatch: "
                f"got {len(rank_buffers)}, expected {len(flat_rank_numels)}."
            )

        if self.fsdp_fast_reconstruct and flat_plan["flat_sorted_is_contiguous_full_order"]:
            chunks = []
            assigned_numel = 0
            for rank in flat_plan["flat_sorted_rank_indices"]:
                chunk_numel = flat_rank_numels[rank]
                if chunk_numel == 0:
                    continue
                source_offset = rank_buffer_offsets[rank]
                chunks.append(rank_buffers[rank][source_offset : source_offset + chunk_numel])
                assigned_numel += chunk_numel
            if assigned_numel != torch.Size(dtensor_ref.shape).numel():
                raise AssertionError(
                    "Fast flat uneven DTensor reconstruction did not cover the full tensor: "
                    f"assigned={assigned_numel}, expected={torch.Size(dtensor_ref.shape).numel()}."
                )
            if assigned_numel == 0:
                return torch.empty(
                    dtensor_ref.shape, dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
                )
            if len(chunks) == 1:
                full_flat = chunks[0].clone()
            else:
                full_flat = torch.cat(chunks)
            return full_flat.view(dtensor_ref.shape)

        full_tensor = torch.empty(
            dtensor_ref.shape, dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
        )
        assigned_numel = 0
        for rank, (rank_buffer, chunk_info) in enumerate(zip(rank_buffers, flat_chunk_infos)):
            chunk_shape = chunk_info["shape"]
            chunk_numel = chunk_info["numel"]
            if chunk_numel == 0:
                continue
            source_offset = rank_buffer_offsets[rank]
            gathered_tensor = rank_buffer[source_offset : source_offset + chunk_numel]
            offset = chunk_info["offset"]
            slices = tuple(slice(o, o + s) for o, s in zip(offset, chunk_shape))
            full_tensor[slices] = gathered_tensor.view(chunk_shape)
            assigned_numel += chunk_numel

        _assert_chunks_cover_full_tensor(dtensor_ref.shape, flat_chunk_infos, assigned_numel)
        return full_tensor

    def _gather_full_uneven_local_tensor_like(
        self, dtensor_ref, local_tensor: torch.Tensor
    ) -> torch.Tensor | None:
        """Gather a local tensor using the uneven sharding layout of `dtensor_ref`."""
        plan = self._get_uneven_gather_plan(dtensor_ref)
        if plan is None:
            if redistribute_uneven_dtensor_to_replicated is None:
                raise RuntimeError(
                    "Megatron-FSDP `redistribute_uneven_dtensor_to_replicated` is required "
                    "to gather un-evenly sharded parameters for Muon step()."
                )
            local_dtensor = self._dtensor_from_local_like(dtensor_ref, local_tensor.contiguous())
            if not hasattr(local_dtensor._local_tensor, "__create_chunk_list__"):
                update_uneven_dtensor_chunk_metadata(local_dtensor)
            full_tensor = redistribute_uneven_dtensor_to_replicated(local_dtensor)._local_tensor
            self._copy_dtensor_chunk_metadata(dtensor_ref, local_dtensor)
            return None if local_tensor.numel() == 0 else full_tensor

        local_buffer = self._flatten_tensor_for_uneven_gather(local_tensor)
        if len(plan["stages"]) == 1:
            stage = plan["stages"][0]
            shard_group = stage["shard_group"]
            group_rank = torch.distributed.get_rank(shard_group)
            expected_local_numel = stage["rank_numels"][group_rank]
            if local_buffer.numel() != expected_local_numel:
                raise AssertionError(
                    "Uneven DTensor gather local buffer size mismatch: "
                    f"got {local_buffer.numel()}, expected {expected_local_numel}."
                )
            group_tensors = self._get_uneven_group_tensors(
                stage["rank_numels"],
                dtype=local_tensor.dtype,
                device=local_tensor.device,
                shard_group=shard_group,
            )
            torch.distributed.all_gather(group_tensors, local_buffer, group=shard_group)

            if local_tensor.numel() == 0:
                return None

            return self._reconstruct_full_tensor_from_rank_buffers(dtensor_ref, plan, group_tensors)

        for stage in plan["stages"]:
            shard_group = stage["shard_group"]
            group_rank = torch.distributed.get_rank(shard_group)
            expected_local_numel = stage["rank_numels"][group_rank]
            if local_buffer.numel() != expected_local_numel:
                raise AssertionError(
                    "Uneven DTensor gather local buffer size mismatch: "
                    f"got {local_buffer.numel()}, expected {expected_local_numel}."
                )
            group_tensors = self._get_uneven_group_tensors(
                stage["rank_numels"],
                dtype=local_tensor.dtype,
                device=local_tensor.device,
                shard_group=shard_group,
            )
            torch.distributed.all_gather(group_tensors, local_buffer, group=shard_group)
            local_buffer = torch.cat(group_tensors)

        if local_tensor.numel() == 0:
            return None

        return self._reconstruct_full_tensor_from_flat_buffer(dtensor_ref, plan, local_buffer)

    def _build_full_uneven_local_tensor_batches(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        results: list[torch.Tensor | None],
        completed_item_indices: set[int] | None = None,
        skip_item_indices: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        batches: dict[tuple[int, torch.dtype, torch.device], list[dict[str, Any]]] = {}

        for item_idx, (dtensor_ref, local_tensor) in enumerate(items):
            if skip_item_indices is not None and item_idx in skip_item_indices:
                continue

            plan = self._get_uneven_gather_plan(dtensor_ref)
            if plan is None:
                results[item_idx] = self._gather_full_uneven_local_tensor_like(
                    dtensor_ref, local_tensor
                )
                if completed_item_indices is not None:
                    completed_item_indices.add(item_idx)
                continue

            stage_group_key = tuple(id(stage["shard_group"]) for stage in plan["stages"])
            key = (stage_group_key, local_tensor.dtype, local_tensor.device)
            key_batches = batches.setdefault(key, [])
            element_size = self._boundary_gather_wire_element_size(local_tensor.dtype)
            if not key_batches:
                key_batches.append(
                    {
                        "dtype": local_tensor.dtype,
                        "device": local_tensor.device,
                        "element_size": element_size,
                        "item_indices": [],
                        "plans": [],
                        "stage_rank_total_numels": [
                            [0] * len(stage["rank_numels"]) for stage in plan["stages"]
                        ],
                    }
                )
            batch = key_batches[-1]
            if batch["item_indices"]:
                candidate_bytes = self._candidate_batch_gather_bytes(
                    batch["stage_rank_total_numels"], plan, element_size=element_size
                )
                if candidate_bytes > self.fsdp_batch_max_gather_bytes:
                    batch = {
                        "dtype": local_tensor.dtype,
                        "device": local_tensor.device,
                        "element_size": element_size,
                        "item_indices": [],
                        "plans": [],
                        "stage_rank_total_numels": [
                            [0] * len(stage["rank_numels"]) for stage in plan["stages"]
                        ],
                    }
                    key_batches.append(batch)
            batch["item_indices"].append(item_idx)
            batch["plans"].append(plan)
            for stage_idx, stage in enumerate(plan["stages"]):
                for rank, rank_numel in enumerate(stage["rank_numels"]):
                    batch["stage_rank_total_numels"][stage_idx][rank] += rank_numel

        for key_batches in batches.values():
            self._sort_gather_batches_by_size(key_batches)

        return [batch for key_batches in batches.values() for batch in key_batches]

    def _build_flat_full_uneven_local_tensor_batches(
        self, items: list[tuple[torch.Tensor, torch.Tensor]], skip_item_indices: set[int]
    ) -> list[dict[str, Any]]:
        batches: dict[tuple[int, torch.dtype, torch.device], list[dict[str, Any]]] = {}

        for item_idx, (dtensor_ref, local_tensor) in enumerate(items):
            if item_idx in skip_item_indices:
                continue

            plan = self._get_uneven_gather_plan(dtensor_ref)
            if plan is None:
                continue
            flat_plan = self._get_flat_uneven_gather_plan(dtensor_ref, plan)
            if flat_plan is None:
                continue

            if self.fsdp_flat_batched_all_gather_nonempty_group:
                collective_group = flat_plan["flat_active_group"]
                collective_rank_numels = flat_plan["flat_active_rank_numels"]
                active_rank_indices = flat_plan["flat_active_rank_indices"]
            else:
                collective_group = flat_plan["flat_group"]
                collective_rank_numels = flat_plan["flat_rank_numels"]
                active_rank_indices = tuple(range(flat_plan["flat_group_size"]))
            key = (
                id(collective_group),
                active_rank_indices,
                local_tensor.dtype,
                local_tensor.device,
            )
            key_batches = batches.setdefault(key, [])
            element_size = self._boundary_gather_wire_element_size(local_tensor.dtype)
            if not key_batches:
                key_batches.append(
                    {
                        "is_flat": True,
                        "dtype": local_tensor.dtype,
                        "device": local_tensor.device,
                        "element_size": element_size,
                        "item_indices": [],
                        "plans": [],
                        "flat_plans": [],
                        "flat_rank_total_numels": [0] * flat_plan["flat_group_size"],
                        "flat_collective_rank_indices": active_rank_indices,
                        "flat_collective_rank_total_numels": [0] * len(active_rank_indices),
                    }
                )
            batch = key_batches[-1]
            if batch["item_indices"]:
                candidate_bytes = self._candidate_rank_batch_gather_bytes(
                    batch["flat_collective_rank_total_numels"],
                    collective_rank_numels,
                    element_size=element_size,
                )
                if candidate_bytes > self.fsdp_batch_max_gather_bytes:
                    batch = {
                        "is_flat": True,
                        "dtype": local_tensor.dtype,
                        "device": local_tensor.device,
                        "element_size": element_size,
                        "item_indices": [],
                        "plans": [],
                        "flat_plans": [],
                        "flat_rank_total_numels": [0] * flat_plan["flat_group_size"],
                        "flat_collective_rank_indices": active_rank_indices,
                        "flat_collective_rank_total_numels": [0] * len(active_rank_indices),
                    }
                    key_batches.append(batch)

            batch["item_indices"].append(item_idx)
            batch["plans"].append(plan)
            batch["flat_plans"].append(flat_plan)
            for rank, rank_numel in enumerate(flat_plan["flat_rank_numels"]):
                batch["flat_rank_total_numels"][rank] += rank_numel
            for rank, rank_numel in enumerate(collective_rank_numels):
                batch["flat_collective_rank_total_numels"][rank] += rank_numel
            skip_item_indices.add(item_idx)

        for key_batches in batches.values():
            self._sort_gather_batches_by_size(key_batches)

        return [batch for key_batches in batches.values() for batch in key_batches]

    def _build_full_uneven_local_tensor_gather_batches(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        results: list[torch.Tensor | None],
        completed_item_indices: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        skip_item_indices = set(completed_item_indices or ())
        completed_without_batch = len(skip_item_indices)
        batches = []
        if self.fsdp_flat_batched_all_gather:
            batches.extend(
                self._build_flat_full_uneven_local_tensor_batches(items, skip_item_indices)
            )
        batches.extend(
            self._build_full_uneven_local_tensor_batches(
                items,
                results,
                completed_item_indices=completed_item_indices,
                skip_item_indices=skip_item_indices,
            )
        )
        self._maybe_log_fsdp_gather_batch_summary(
            items, batches, completed_without_batch=completed_without_batch
        )
        return batches

    def _gather_full_uneven_local_tensors_like(
        self, items: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> list[torch.Tensor | None]:
        """Batch uneven gathers for boundary tensors.

        Each boundary parameter still needs the same logical data as
        `_gather_full_uneven_local_tensor_like`. Batching concatenates local
        shards with the same dtype/device/group so the optimizer issues one
        all-gather per batch instead of one all-gather per boundary parameter.
        Ranks with empty local shards participate in the collective but do not
        reconstruct or orthogonalize the full boundary tensor.
        """
        results: list[torch.Tensor | None] = [None] * len(items)
        batches = self._build_full_uneven_local_tensor_gather_batches(items, results)

        for batch in batches:
            if batch.get("is_flat", False):
                self._gather_flat_full_uneven_local_tensor_batch(items, results, batch)
            else:
                self._gather_full_uneven_local_tensor_batch(items, results, batch)

        return results

    def _gather_flat_full_uneven_local_tensor_batch(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        results: list[torch.Tensor | None],
        batch: dict[str, Any],
    ) -> None:
        with torch.autograd.profiler.record_function("Muon-FSDP flat gather pack"):
            stage_state = self._prepare_flat_uneven_gather_batch(items, batch)
        pending_stage = self._start_uneven_gather_stage(stage_state, async_op=False)
        self._wait_uneven_gather_stage(pending_stage)
        with torch.autograd.profiler.record_function("Muon-FSDP flat gather reconstruct"):
            self._store_flat_uneven_gather_batch_results(items, results, batch, pending_stage)

    def _prepare_flat_uneven_gather_batch(
        self, items: list[tuple[torch.Tensor, torch.Tensor]], batch: dict[str, Any]
    ) -> dict[str, Any]:
        item_indices = batch["item_indices"]
        flat_plans = batch["flat_plans"]
        flat_group = flat_plans[0]["flat_group"]
        group_size = flat_plans[0]["flat_group_size"]
        group_rank = flat_plans[0]["flat_group_rank"]
        collective_rank_indices = batch.get(
            "flat_collective_rank_indices", tuple(range(group_size))
        )
        collective_group = (
            flat_plans[0]["flat_active_group"]
            if self.fsdp_flat_batched_all_gather_nonempty_group
            else flat_group
        )
        participates = group_rank in collective_rank_indices

        expected_item_numels = [
            flat_plan["flat_rank_numels"][group_rank] for flat_plan in flat_plans
        ]
        expected_local_numel = sum(expected_item_numels)
        current_buffers = [
            self._flatten_tensor_for_uneven_gather(items[item_idx][1]) for item_idx in item_indices
        ]
        if len(current_buffers) == 1:
            local_buffer = current_buffers[0]
            if local_buffer.numel() != expected_local_numel:
                raise AssertionError(
                    "Flat batched uneven DTensor gather buffer size mismatch: "
                    f"got {local_buffer.numel()}, expected {expected_local_numel}."
                )
        else:
            local_buffer = self._get_fsdp_gather_scratch_tensor(
                ("flat_batch_local_buffer", id(flat_group), batch["dtype"], batch["device"]),
                expected_local_numel,
                dtype=batch["dtype"],
                device=batch["device"],
            )
            local_offset = 0
            for buffer, expected_numel in zip(current_buffers, expected_item_numels):
                if buffer.numel() != expected_numel:
                    raise AssertionError(
                        "Flat batched uneven DTensor gather buffer size mismatch: "
                        f"got {buffer.numel()}, expected {expected_numel}."
                    )
                if expected_numel == 0:
                    continue
                local_buffer[local_offset : local_offset + expected_numel].copy_(buffer)
                local_offset += expected_numel
            if local_offset != expected_local_numel:
                raise AssertionError(
                    "Flat batched uneven DTensor gather local buffer size mismatch: "
                    f"packed {local_offset}, expected {expected_local_numel}."
                )

        fp8_dtype = self._fp8_boundary_gather_dtype()
        if self.fsdp_boundary_gather_dtype == "int8":
            batch["_int8_gather_scales"] = self._compute_int8_boundary_gather_scales(
                current_buffers, [flat_group]
            )
        elif fp8_dtype is not None:
            batch["_fp8_gather_scales"] = self._compute_fp8_boundary_gather_scales(
                current_buffers, [flat_group], fp8_dtype
            )
        local_buffer = self._maybe_quantize_boundary_gather_buffer(
            local_buffer, batch, stage_idx=0, item_numels=expected_item_numels
        )

        rank_offsets: list[list[int]] = []
        rank_total_numels: list[int] = []
        for rank in range(group_size):
            offsets = []
            total_numel = 0
            for flat_plan in flat_plans:
                offsets.append(total_numel)
                total_numel += flat_plan["flat_rank_numels"][rank]
            rank_offsets.append(offsets)
            rank_total_numels.append(total_numel)

        if rank_total_numels != batch["flat_rank_total_numels"]:
            raise AssertionError(
                "Flat batched uneven DTensor gather rank totals changed after batching."
            )

        collective_rank_total_numels = batch.get(
            "flat_collective_rank_total_numels", rank_total_numels
        )
        use_padded_all_gather = self._batch_uses_padded_all_gather(collective_rank_total_numels)
        return {
            "batch": batch,
            "plans": batch["plans"],
            "flat_plans": flat_plans,
            "stage_idx": 0,
            "shard_group": collective_group,
            "group_size": group_size,
            "rank_offsets": rank_offsets,
            "rank_total_numels": rank_total_numels,
            "collective_rank_indices": collective_rank_indices,
            "collective_rank_total_numels": collective_rank_total_numels,
            "local_buffer": local_buffer,
            "participates": participates,
            "use_padded_all_gather": use_padded_all_gather,
        }

    def _store_flat_uneven_gather_batch_results(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        results: list[torch.Tensor | None],
        batch: dict[str, Any],
        pending_stage: dict[str, Any],
    ) -> list[int]:
        rank_offsets = pending_stage["rank_offsets"]
        rank_buffers = self._rank_buffers_from_pending_uneven_gather_stage(pending_stage)

        for batch_item_idx, item_idx in enumerate(batch["item_indices"]):
            dtensor_ref, local_tensor = items[item_idx]
            if local_tensor.numel() == 0:
                results[item_idx] = None
                continue

            rank_buffer_offsets = [
                rank_offsets[rank][batch_item_idx] for rank in range(pending_stage["group_size"])
            ]
            full_tensor = self._reconstruct_full_tensor_from_flat_rank_buffers(
                dtensor_ref, batch["flat_plans"][batch_item_idx], rank_buffers, rank_buffer_offsets
            )
            results[item_idx] = self._maybe_dequantize_boundary_gather_result(
                full_tensor, local_tensor, batch, batch_item_idx
            )
        return batch["item_indices"]

    def _gather_full_uneven_local_tensor_batch(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        results: list[torch.Tensor | None],
        batch: dict[str, Any],
    ) -> None:
        item_indices = batch["item_indices"]
        plans = batch["plans"]
        current_buffers = [
            self._flatten_tensor_for_uneven_gather(items[item_idx][1]) for item_idx in item_indices
        ]
        final_stage_idx = len(plans[0]["stages"]) - 1

        stage_idx = 0
        stage_state = self._prepare_uneven_gather_stage(
            current_buffers, plans, batch, stage_idx, plans[0]["stages"][stage_idx]
        )
        pending_stage = self._start_uneven_gather_stage(stage_state, async_op=False)
        while True:
            if stage_idx == final_stage_idx:
                self._wait_uneven_gather_stage(pending_stage)
                self._store_uneven_gather_batch_results_from_stage(
                    items, results, batch, pending_stage
                )
                return
            next_stage_idx = stage_idx + 1
            next_stage = plans[0]["stages"][next_stage_idx]
            stage_state = self._prepare_uneven_gather_stage_from_previous_pending(
                pending_stage, plans, batch, next_stage_idx, next_stage
            )
            pending_stage = self._start_uneven_gather_stage(stage_state, async_op=False)
            stage_idx = next_stage_idx

    def _prepare_uneven_gather_stage(
        self,
        current_buffers: list[torch.Tensor],
        plans: list[dict[str, Any]],
        batch: dict[str, Any],
        stage_idx: int,
        stage: dict[str, Any],
    ) -> dict[str, Any]:
        with torch.autograd.profiler.record_function("Muon-FSDP gather pack"):
            shard_group = stage["shard_group"]
            group_size = get_pg_size(shard_group)
            group_rank = torch.distributed.get_rank(shard_group)

            expected_item_numels = [
                plan["stages"][stage_idx]["rank_numels"][group_rank] for plan in plans
            ]
            expected_local_numel = sum(expected_item_numels)

            if len(current_buffers) == 1:
                local_buffer = current_buffers[0]
                if local_buffer.numel() != expected_local_numel:
                    raise AssertionError(
                        "Batched uneven DTensor gather stage buffer size mismatch: "
                        f"stage={stage_idx}, got {local_buffer.numel()}, expected "
                        f"{expected_local_numel}."
                    )
            else:
                buffer_dtype = current_buffers[0].dtype
                local_buffer = self._get_fsdp_gather_scratch_tensor(
                    (
                        "batch_local_buffer",
                        stage_idx,
                        id(shard_group),
                        buffer_dtype,
                        batch["device"],
                    ),
                    expected_local_numel,
                    dtype=buffer_dtype,
                    device=batch["device"],
                )
                local_offset = 0
                for buffer, expected_numel in zip(current_buffers, expected_item_numels):
                    if buffer.numel() != expected_numel:
                        raise AssertionError(
                            "Batched uneven DTensor gather stage buffer size mismatch: "
                            f"stage={stage_idx}, got {buffer.numel()}, expected {expected_numel}."
                        )
                    if expected_numel == 0:
                        continue
                    local_buffer[local_offset : local_offset + expected_numel].copy_(buffer)
                    local_offset += expected_numel

                if local_offset != expected_local_numel:
                    raise AssertionError(
                        "Batched uneven DTensor gather local buffer size mismatch: "
                        f"stage={stage_idx}, packed {local_offset}, expected "
                        f"{expected_local_numel}."
                    )

            fp8_dtype = self._fp8_boundary_gather_dtype()
            if self.fsdp_boundary_gather_dtype == "int8" and stage_idx == 0:
                batch["_int8_gather_scales"] = self._compute_int8_boundary_gather_scales(
                    current_buffers,
                    [plan_stage["shard_group"] for plan_stage in plans[0]["stages"]],
                )
            elif fp8_dtype is not None and stage_idx == 0:
                batch["_fp8_gather_scales"] = self._compute_fp8_boundary_gather_scales(
                    current_buffers,
                    [plan_stage["shard_group"] for plan_stage in plans[0]["stages"]],
                    fp8_dtype,
                )
            local_buffer = self._maybe_quantize_boundary_gather_buffer(
                local_buffer, batch, stage_idx=stage_idx, item_numels=expected_item_numels
            )

            rank_offsets: list[list[int]] = []
            rank_total_numels: list[int] = []
            for rank in range(group_size):
                offsets = []
                total_numel = 0
                for plan in plans:
                    offsets.append(total_numel)
                    total_numel += plan["stages"][stage_idx]["rank_numels"][rank]
                rank_offsets.append(offsets)
                rank_total_numels.append(total_numel)

            use_padded_all_gather = self._batch_uses_padded_all_gather(rank_total_numels)
        return {
            "batch": batch,
            "plans": plans,
            "stage_idx": stage_idx,
            "shard_group": shard_group,
            "group_size": group_size,
            "rank_offsets": rank_offsets,
            "rank_total_numels": rank_total_numels,
            "local_buffer": local_buffer,
            "use_padded_all_gather": use_padded_all_gather,
        }

    def _prepare_uneven_gather_stage_from_previous_pending(
        self,
        pending_stage: dict[str, Any],
        plans: list[dict[str, Any]],
        batch: dict[str, Any],
        stage_idx: int,
        stage: dict[str, Any],
        *,
        wait_for_previous: bool = True,
    ) -> dict[str, Any]:
        if wait_for_previous:
            self._wait_uneven_gather_stage(pending_stage)
        with torch.autograd.profiler.record_function("Muon-FSDP gather fused unpack/pack"):
            previous_stage_idx = pending_stage["stage_idx"]
            if stage_idx != previous_stage_idx + 1:
                raise AssertionError(
                    "Fused uneven DTensor gather repack expected consecutive stages: "
                    f"previous={previous_stage_idx}, next={stage_idx}."
                )

            shard_group = stage["shard_group"]
            group_size = get_pg_size(shard_group)
            group_rank = torch.distributed.get_rank(shard_group)

            expected_item_numels = [
                plan["stages"][stage_idx]["rank_numels"][group_rank] for plan in plans
            ]
            expected_local_numel = sum(expected_item_numels)
            buffer_dtype = pending_stage["local_buffer"].dtype
            local_buffer = self._get_fsdp_gather_scratch_tensor(
                ("batch_local_buffer", stage_idx, id(shard_group), buffer_dtype, batch["device"]),
                expected_local_numel,
                dtype=buffer_dtype,
                device=batch["device"],
            )

            previous_rank_offsets = pending_stage["rank_offsets"]
            previous_rank_buffers = self._rank_buffers_from_pending_uneven_gather_stage(
                pending_stage
            )
            local_offset = 0
            for batch_item_idx, (plan, expected_numel) in enumerate(
                zip(plans, expected_item_numels)
            ):
                previous_item_numel = sum(plan["stages"][previous_stage_idx]["rank_numels"])
                if expected_numel != previous_item_numel:
                    raise AssertionError(
                        "Fused uneven DTensor gather repack saw an unexpected stage size: "
                        f"stage={stage_idx}, item={batch_item_idx}, got={previous_item_numel}, "
                        f"expected={expected_numel}."
                    )

                assigned_numel = 0
                if wait_for_previous:
                    for rank, rank_buffer in enumerate(previous_rank_buffers):
                        chunk_numel = plan["stages"][previous_stage_idx]["rank_numels"][rank]
                        if chunk_numel == 0:
                            continue
                        source_offset = previous_rank_offsets[rank][batch_item_idx]
                        target_start = local_offset + assigned_numel
                        target_end = target_start + chunk_numel
                        local_buffer[target_start:target_end].copy_(
                            rank_buffer[source_offset : source_offset + chunk_numel]
                        )
                        assigned_numel += chunk_numel
                else:
                    chunks = []
                    for rank, rank_buffer in enumerate(previous_rank_buffers):
                        chunk_numel = plan["stages"][previous_stage_idx]["rank_numels"][rank]
                        if chunk_numel == 0:
                            continue
                        source_offset = previous_rank_offsets[rank][batch_item_idx]
                        chunks.append(rank_buffer[source_offset : source_offset + chunk_numel])
                        assigned_numel += chunk_numel
                    target = local_buffer[local_offset : local_offset + expected_numel]
                    if len(chunks) == 1:
                        target.copy_(chunks[0])
                    elif chunks:
                        torch.cat(chunks, out=target)
                if assigned_numel != expected_numel:
                    raise AssertionError(
                        "Fused uneven DTensor gather repack did not cover the item: "
                        f"stage={stage_idx}, item={batch_item_idx}, assigned={assigned_numel}, "
                        f"expected={expected_numel}."
                    )
                local_offset += expected_numel

            if local_offset != expected_local_numel:
                raise AssertionError(
                    "Fused uneven DTensor gather repack local buffer size mismatch: "
                    f"packed={local_offset}, expected={expected_local_numel}."
                )

            local_buffer = self._maybe_quantize_boundary_gather_buffer(
                local_buffer, batch, stage_idx=stage_idx, item_numels=expected_item_numels
            )

            rank_offsets: list[list[int]] = []
            rank_total_numels: list[int] = []
            for rank in range(group_size):
                offsets = []
                total_numel = 0
                for plan in plans:
                    offsets.append(total_numel)
                    total_numel += plan["stages"][stage_idx]["rank_numels"][rank]
                rank_offsets.append(offsets)
                rank_total_numels.append(total_numel)

            use_padded_all_gather = self._batch_uses_padded_all_gather(rank_total_numels)
        return {
            "batch": batch,
            "plans": plans,
            "stage_idx": stage_idx,
            "shard_group": shard_group,
            "group_size": group_size,
            "rank_offsets": rank_offsets,
            "rank_total_numels": rank_total_numels,
            "local_buffer": local_buffer,
            "use_padded_all_gather": use_padded_all_gather,
        }

    def _start_uneven_gather_stage(
        self, stage_state: dict[str, Any], *, async_op: bool
    ) -> dict[str, Any]:
        device = stage_state["batch"]["device"]
        comm_stream = None
        if async_op and device.type == "cuda":
            with torch.cuda.device(device):
                comm_stream = self._get_fsdp_comm_stream(device)
                ready_event = stage_state["batch"].get("_boundary_ready_event")
                if ready_event is not None:
                    comm_stream.wait_event(ready_event)
                else:
                    comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(comm_stream):
                    return self._issue_uneven_gather_stage(
                        stage_state, async_op=True, comm_stream=comm_stream
                    )
        return self._issue_uneven_gather_stage(
            stage_state, async_op=async_op, comm_stream=comm_stream
        )

    def _issue_uneven_gather_stage(
        self, stage_state: dict[str, Any], *, async_op: bool, comm_stream: torch.cuda.Stream | None
    ) -> dict[str, Any]:
        batch = stage_state["batch"]
        shard_group = stage_state["shard_group"]
        local_buffer = stage_state["local_buffer"]
        rank_total_numels = stage_state.get(
            "collective_rank_total_numels", stage_state["rank_total_numels"]
        )
        pending = dict(stage_state)
        pending["comm_stream"] = comm_stream
        if not stage_state.get("participates", True):
            pending["work"] = None
            pending["skipped_nonempty_collective"] = True
            return pending

        with torch.autograd.profiler.record_function(
            self._gather_collective_nvtx_label(stage_state)
        ):
            if stage_state["use_padded_all_gather"]:
                padded_local_buffer, gathered_padded_buffer, max_rank_numel = (
                    self._prepare_padded_all_gather_buffers(
                        local_buffer, rank_total_numels, shard_group
                    )
                )
                pending["padded_local_buffer"] = padded_local_buffer
                pending["gathered_padded_buffer"] = gathered_padded_buffer
                pending["max_rank_numel"] = max_rank_numel
                pending["work"] = torch.distributed.all_gather_into_tensor(
                    gathered_padded_buffer,
                    padded_local_buffer,
                    group=shard_group,
                    async_op=async_op,
                )
            else:
                group_tensors = self._get_uneven_group_tensors(
                    rank_total_numels,
                    dtype=local_buffer.dtype,
                    device=local_buffer.device,
                    shard_group=shard_group,
                )
                pending["group_tensors"] = group_tensors
                pending["work"] = torch.distributed.all_gather(
                    group_tensors, local_buffer, group=shard_group, async_op=async_op
                )
        return pending

    def _wait_uneven_gather_stage(self, pending_stage: dict[str, Any]) -> None:
        with torch.autograd.profiler.record_function("Muon-FSDP gather wait"):
            work = pending_stage.get("work")
            if work is not None:
                work.wait()
            comm_stream = pending_stage.get("comm_stream")
            if comm_stream is not None:
                device = pending_stage["batch"]["device"]
                with torch.cuda.device(device):
                    torch.cuda.current_stream().wait_stream(comm_stream)

    def _block_current_stream_on_uneven_gather_stage(self, pending_stage: dict[str, Any]) -> bool:
        work = pending_stage.get("work")
        if work is None:
            return True
        block_current_stream = getattr(work, "block_current_stream", None)
        if block_current_stream is None:
            return False
        block_current_stream()
        return True

    def _overlap_gather_pending_completed(self, pending: dict[str, Any]) -> bool:
        def stage_completed(pending_stage: dict[str, Any]) -> bool:
            work = pending_stage.get("work")
            if work is None:
                return True
            is_completed = getattr(work, "is_completed", None)
            if is_completed is None:
                return False
            return bool(is_completed())

        if "pending_flat_stage" in pending:
            return stage_completed(pending["pending_flat_stage"])
        if "final_pending_stage" in pending:
            return all(stage_completed(stage) for stage in pending["pending_stages"]) and (
                stage_completed(pending["final_pending_stage"])
            )
        if "pending_stage" in pending:
            return stage_completed(pending["pending_stage"])
        return True

    def _finish_uneven_gather_stage(self, pending_stage: dict[str, Any]) -> list[torch.Tensor]:
        self._wait_uneven_gather_stage(pending_stage)
        return self._unpack_uneven_gather_stage(pending_stage)

    def _unpack_uneven_gather_stage(self, pending_stage: dict[str, Any]) -> list[torch.Tensor]:
        with torch.autograd.profiler.record_function("Muon-FSDP gather unpack"):
            batch = pending_stage["batch"]
            plans = pending_stage["plans"]
            stage_idx = pending_stage["stage_idx"]
            rank_offsets = pending_stage["rank_offsets"]
            rank_buffers = self._rank_buffers_from_pending_uneven_gather_stage(pending_stage)

            next_buffers = []
            for batch_item_idx, plan in enumerate(plans):
                item_numel = sum(plan["stages"][stage_idx]["rank_numels"])
                if plan["is_contiguous_full_order"]:
                    chunks = []
                    assigned_numel = 0
                    for rank, rank_buffer in enumerate(rank_buffers):
                        chunk_numel = plan["stages"][stage_idx]["rank_numels"][rank]
                        if chunk_numel == 0:
                            continue
                        source_offset = rank_offsets[rank][batch_item_idx]
                        chunks.append(rank_buffer[source_offset : source_offset + chunk_numel])
                        assigned_numel += chunk_numel
                    if assigned_numel != item_numel:
                        raise AssertionError(
                            "Batched uneven DTensor stage unpack did not cover the item: "
                            f"stage={stage_idx}, assigned={assigned_numel}, expected={item_numel}."
                        )
                    if item_numel == 0:
                        item_buffer = torch.empty(
                            0, dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
                        )
                    elif len(chunks) == 1:
                        item_buffer = chunks[0].clone()
                    else:
                        item_buffer = torch.cat(chunks)
                    next_buffers.append(item_buffer)
                    continue

                item_buffer = torch.empty(
                    item_numel, dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
                )
                item_offset = 0
                for rank, rank_buffer in enumerate(rank_buffers):
                    chunk_numel = plan["stages"][stage_idx]["rank_numels"][rank]
                    if chunk_numel == 0:
                        continue
                    source_offset = rank_offsets[rank][batch_item_idx]
                    item_buffer[item_offset : item_offset + chunk_numel].copy_(
                        rank_buffer[source_offset : source_offset + chunk_numel]
                    )
                    item_offset += chunk_numel
                if item_offset != item_numel:
                    raise AssertionError(
                        "Batched uneven DTensor stage unpack did not cover the item: "
                        f"stage={stage_idx}, assigned={item_offset}, expected={item_numel}."
                    )
                next_buffers.append(item_buffer)

            return next_buffers

    def _store_uneven_gather_batch_results(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        results: list[torch.Tensor | None],
        batch: dict[str, Any],
        current_buffers: list[torch.Tensor],
    ) -> list[int]:
        with torch.autograd.profiler.record_function("Muon-FSDP gather reconstruct"):
            item_indices = batch["item_indices"]
            plans = batch["plans"]
            for batch_item_idx, item_idx in enumerate(item_indices):
                dtensor_ref, local_tensor = items[item_idx]
                if local_tensor.numel() == 0:
                    results[item_idx] = None
                    continue

                plan = plans[batch_item_idx]
                full_tensor = self._reconstruct_full_tensor_from_flat_buffer(
                    dtensor_ref, plan, current_buffers[batch_item_idx]
                )
                results[item_idx] = self._maybe_dequantize_boundary_gather_result(
                    full_tensor, local_tensor, batch, batch_item_idx
                )
            return item_indices

    def _store_uneven_gather_batch_results_from_stage(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        results: list[torch.Tensor | None],
        batch: dict[str, Any],
        pending_stage: dict[str, Any],
    ) -> list[int]:
        item_indices = batch["item_indices"]
        plans = batch["plans"]
        stage_idx = pending_stage["stage_idx"]
        rank_offsets = pending_stage["rank_offsets"]
        rank_buffers = self._rank_buffers_from_pending_uneven_gather_stage(pending_stage)

        for batch_item_idx, item_idx in enumerate(item_indices):
            dtensor_ref, local_tensor = items[item_idx]
            if local_tensor.numel() == 0:
                results[item_idx] = None
                continue

            rank_buffer_offsets = [
                rank_offsets[rank][batch_item_idx] for rank in range(pending_stage["group_size"])
            ]
            full_tensor = self._reconstruct_full_tensor_from_final_stage_buffers(
                dtensor_ref, plans[batch_item_idx], stage_idx, rank_buffers, rank_buffer_offsets
            )
            results[item_idx] = self._maybe_dequantize_boundary_gather_result(
                full_tensor, local_tensor, batch, batch_item_idx
            )
        return item_indices

    def _prepare_uneven_gather_stage_direct_pre_ns(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        batch: dict[str, Any],
        all_updates: list,
        boundary_update_indices: list[int],
    ) -> dict[str, Any]:
        with torch.autograd.profiler.record_function(
            "Muon-FSDP gather direct pre-NS into pack buffer"
        ):
            item_indices = batch["item_indices"]
            plans = batch["plans"]
            stage_idx = 0
            stage = plans[0]["stages"][stage_idx]
            shard_group = stage["shard_group"]
            group_size = get_pg_size(shard_group)
            group_rank = torch.distributed.get_rank(shard_group)

            expected_item_numels = [
                plan["stages"][stage_idx]["rank_numels"][group_rank] for plan in plans
            ]
            expected_local_numel = sum(expected_item_numels)
            local_buffer = self._get_fsdp_gather_scratch_tensor(
                (
                    "direct_pre_ns_batch_local_buffer",
                    stage_idx,
                    id(shard_group),
                    batch["dtype"],
                    batch["device"],
                ),
                expected_local_numel,
                dtype=batch["dtype"],
                device=batch["device"],
            )
            prewire_buffers: dict[torch.dtype, torch.Tensor] = {}

            local_offset = 0
            for batch_item_idx, (item_idx, expected_numel) in enumerate(
                zip(item_indices, expected_item_numels)
            ):
                dtensor_ref, local_tensor_ref = items[item_idx]
                if local_tensor_ref.numel() != expected_numel:
                    raise AssertionError(
                        "Direct Muon+M-FSDP boundary pre-NS gather item size mismatch: "
                        f"item={batch_item_idx}, got={local_tensor_ref.numel()}, "
                        f"expected={expected_numel}."
                    )
                update_idx = boundary_update_indices[item_idx]
                p, _, update_mode, lr, group_kwargs = all_updates[update_idx]
                if p is not dtensor_ref:
                    raise AssertionError(
                        "Direct Muon+M-FSDP boundary pre-NS gather item/update mismatch."
                    )
                target_flat = local_buffer[local_offset : local_offset + expected_numel]
                mom_local = self.state[p]["momentum_buffer"]._local_tensor
                if expected_numel == 0:
                    ref_dtype = mom_local.dtype
                    ref_device = mom_local.device
                elif target_flat.dtype == mom_local.dtype:
                    target = target_flat.view(mom_local.shape)
                    self._compute_local_pre_ns_grad(p, group_kwargs, lr, out=target)
                    ref_dtype = target.dtype
                    ref_device = target.device
                else:
                    prewire_buffer = prewire_buffers.get(mom_local.dtype)
                    if prewire_buffer is None:
                        prewire_buffer = self._get_fsdp_gather_scratch_tensor(
                            (
                                "direct_pre_ns_batch_prewire_buffer",
                                stage_idx,
                                id(shard_group),
                                mom_local.dtype,
                                batch["device"],
                            ),
                            expected_local_numel,
                            dtype=mom_local.dtype,
                            device=batch["device"],
                        )
                        prewire_buffers[mom_local.dtype] = prewire_buffer
                    prewire_flat = prewire_buffer[local_offset : local_offset + expected_numel]
                    prewire = prewire_flat.view(mom_local.shape)
                    self._compute_local_pre_ns_grad(p, group_kwargs, lr, out=prewire)
                    target_flat.copy_(prewire_flat)
                    ref_dtype = prewire.dtype
                    ref_device = prewire.device
                # The packed scratch view can be reused before the completed
                # boundary update is processed.  Keep only a tiny dtype/device
                # reference for restoring the gathered wire dtype before exact NS.
                pre_ns_ref = torch.empty(0, dtype=ref_dtype, device=ref_device)
                all_updates[update_idx] = (p, pre_ns_ref, update_mode, lr, group_kwargs)
                local_offset += expected_numel

            if local_offset != expected_local_numel:
                raise AssertionError(
                    "Direct Muon+M-FSDP boundary pre-NS gather local buffer size mismatch: "
                    f"packed={local_offset}, expected={expected_local_numel}."
                )

            rank_offsets: list[list[int]] = []
            rank_total_numels: list[int] = []
            for rank in range(group_size):
                offsets = []
                total_numel = 0
                for plan in plans:
                    offsets.append(total_numel)
                    total_numel += plan["stages"][stage_idx]["rank_numels"][rank]
                rank_offsets.append(offsets)
                rank_total_numels.append(total_numel)

            use_padded_all_gather = self._batch_uses_padded_all_gather(rank_total_numels)
        return {
            "batch": batch,
            "plans": plans,
            "stage_idx": stage_idx,
            "shard_group": shard_group,
            "group_size": group_size,
            "rank_offsets": rank_offsets,
            "rank_total_numels": rank_total_numels,
            "local_buffer": local_buffer,
            "use_padded_all_gather": use_padded_all_gather,
        }

    def _start_gather_full_uneven_local_tensor_batch_async(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        batch: dict[str, Any],
        *,
        all_updates: list | None = None,
        boundary_update_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        with self._with_fsdp_gather_scratch_scope(batch.get("_scratch_scope")):
            direct_pre_ns = all_updates is not None
            if direct_pre_ns and boundary_update_indices is None:
                raise AssertionError(
                    "Direct Muon+M-FSDP boundary pre-NS gather requires boundary update indices."
                )
            if batch.get("is_flat", False):
                if direct_pre_ns:
                    raise AssertionError(
                        "Direct Muon+M-FSDP boundary pre-NS gather-buffer path "
                        "does not yet support flat gather batches."
                    )
                with torch.autograd.profiler.record_function("Muon-FSDP flat gather pack"):
                    stage_state = self._prepare_flat_uneven_gather_batch(items, batch)
                pending_stage = self._start_uneven_gather_stage(stage_state, async_op=True)
                return {"items": items, "batch": batch, "pending_flat_stage": pending_stage}

            item_indices = batch["item_indices"]
            plans = batch["plans"]
            current_buffers = [
                self._flatten_tensor_for_uneven_gather(items[item_idx][1])
                for item_idx in item_indices
            ]

            if batch["device"].type == "cuda" and len(plans[0]["stages"]) > 1:
                pending_stages = []
                final_stage_idx = len(plans[0]["stages"]) - 1
                with torch.cuda.device(batch["device"]):
                    comm_stream = self._get_fsdp_comm_stream(batch["device"])
                    ready_event = batch.get("_boundary_ready_event")
                    if ready_event is not None:
                        comm_stream.wait_event(ready_event)
                    else:
                        comm_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(comm_stream):
                        for stage_idx, stage in enumerate(plans[0]["stages"]):
                            if (
                                self.fsdp_fused_async_gather_repack
                                and stage_idx > 0
                                and pending_stages
                            ):
                                # The previous Work has already inserted a stream wait below,
                                # so repacking can stay GPU-queued without a CPU-side wait.
                                stage_state = (
                                    self._prepare_uneven_gather_stage_from_previous_pending(
                                        pending_stages[-1],
                                        plans,
                                        batch,
                                        stage_idx,
                                        stage,
                                        wait_for_previous=False,
                                    )
                                )
                            else:
                                if direct_pre_ns and stage_idx == 0:
                                    stage_state = self._prepare_uneven_gather_stage_direct_pre_ns(
                                        items, batch, all_updates, boundary_update_indices
                                    )
                                else:
                                    stage_state = self._prepare_uneven_gather_stage(
                                        current_buffers, plans, batch, stage_idx, stage
                                    )
                            pending_stage = self._issue_uneven_gather_stage(
                                stage_state, async_op=True, comm_stream=comm_stream
                            )
                            if stage_idx == final_stage_idx:
                                pending_stages.append(pending_stage)
                                break
                            if not self._block_current_stream_on_uneven_gather_stage(pending_stage):
                                if direct_pre_ns:
                                    raise AssertionError(
                                        "Direct Muon+M-FSDP boundary pre-NS gather-buffer path "
                                        "requires torch.distributed.Work.block_current_stream() "
                                        "for async multi-stage gathers."
                                    )
                                if stage_idx == 0:
                                    return {
                                        "items": items,
                                        "batch": batch,
                                        "current_buffers": current_buffers,
                                        "pending_stage": pending_stage,
                                    }
                                raise AssertionError(
                                    "Muon+M-FSDP async multi-stage gather requires "
                                    "torch.distributed.Work.block_current_stream()."
                                )
                            pending_stages.append(pending_stage)
                            if not self.fsdp_fused_async_gather_repack:
                                current_buffers = self._unpack_uneven_gather_stage(pending_stage)

                return {
                    "items": items,
                    "batch": batch,
                    "pending_stages": pending_stages[:-1],
                    "final_pending_stage": pending_stages[-1],
                }

            if direct_pre_ns:
                stage_state = self._prepare_uneven_gather_stage_direct_pre_ns(
                    items, batch, all_updates, boundary_update_indices
                )
            else:
                stage_state = self._prepare_uneven_gather_stage(
                    current_buffers, plans, batch, 0, plans[0]["stages"][0]
                )
            pending_stage = self._start_uneven_gather_stage(stage_state, async_op=True)
            return {"items": items, "batch": batch, "pending_stage": pending_stage}

    def _finish_gather_full_uneven_local_tensor_batch_async(
        self, pending: dict[str, Any], results: list[torch.Tensor | None]
    ) -> list[int]:
        batch = pending["batch"]
        with self._with_fsdp_gather_scratch_scope(batch.get("_scratch_scope")):
            items = pending["items"]
            plans = batch["plans"]
            if "pending_flat_stage" in pending:
                pending_stage = pending["pending_flat_stage"]
                self._wait_uneven_gather_stage(pending_stage)
                with torch.autograd.profiler.record_function("Muon-FSDP flat gather reconstruct"):
                    return self._store_flat_uneven_gather_batch_results(
                        items, results, batch, pending_stage
                    )

            if "final_pending_stage" in pending:
                for pending_stage in pending["pending_stages"]:
                    self._wait_uneven_gather_stage(pending_stage)
                final_pending_stage = pending["final_pending_stage"]
                self._wait_uneven_gather_stage(final_pending_stage)
                return self._store_uneven_gather_batch_results_from_stage(
                    items, results, batch, final_pending_stage
                )

            final_stage_idx = len(plans[0]["stages"]) - 1
            if final_stage_idx == 0:
                self._wait_uneven_gather_stage(pending["pending_stage"])
                return self._store_uneven_gather_batch_results_from_stage(
                    items, results, batch, pending["pending_stage"]
                )

            current_buffers = self._finish_uneven_gather_stage(pending["pending_stage"])

            for stage_idx in range(1, len(plans[0]["stages"])):
                stage_state = self._prepare_uneven_gather_stage(
                    current_buffers, plans, batch, stage_idx, plans[0]["stages"][stage_idx]
                )
                pending_stage = self._start_uneven_gather_stage(stage_state, async_op=False)
                if stage_idx == final_stage_idx:
                    self._wait_uneven_gather_stage(pending_stage)
                    return self._store_uneven_gather_batch_results_from_stage(
                        items, results, batch, pending_stage
                    )
                current_buffers = self._finish_uneven_gather_stage(pending_stage)

        raise AssertionError("Batched uneven DTensor gather did not reach its final stage.")

    def _gather_partial_distributed_pre_ns(
        self, p: torch.Tensor, pre_ns_grad: torch.Tensor, partial_plan: dict[str, Any]
    ) -> torch.Tensor:
        gather_tensor = self._prepare_boundary_gather_tensor(pre_ns_grad)
        gathered = self._gather_partial_uneven_local_tensor_like(
            p, gather_tensor, partial_plan["gather_plan"]
        )
        restored = self._restore_boundary_gather_tensor(gathered, pre_ns_grad)
        assert restored is not None
        return restored

    def _gather_partial_uneven_local_tensor_like(
        self, dtensor_ref, local_tensor: torch.Tensor, plan: dict[str, Any]
    ) -> torch.Tensor:
        if not plan["stages"]:
            return local_tensor

        batch = {
            "dtype": local_tensor.dtype,
            "device": local_tensor.device,
            "element_size": local_tensor.element_size(),
            "item_indices": [0],
            "plans": [plan],
        }
        current_buffers = [self._flatten_tensor_for_uneven_gather(local_tensor)]
        final_stage_idx = len(plan["stages"]) - 1
        for stage_idx, stage in enumerate(plan["stages"]):
            with torch.autograd.profiler.record_function(
                f"Muon-FSDP partial gather stage {stage_idx}"
            ):
                stage_state = self._prepare_uneven_gather_stage(
                    current_buffers, [plan], batch, stage_idx, stage
                )
                pending_stage = self._start_uneven_gather_stage(stage_state, async_op=False)
                if stage_idx == final_stage_idx:
                    self._wait_uneven_gather_stage(pending_stage)
                    return self._reconstruct_partial_tensor_from_final_stage_buffers(
                        plan, pending_stage
                    )
                current_buffers = self._finish_uneven_gather_stage(pending_stage)

        raise AssertionError("Partial uneven DTensor gather did not reach its final stage.")

    def _validate_batched_partial_gather_plans(self, plans: list[dict[str, Any]]) -> int:
        if not plans:
            return 0
        stage_count = len(plans[0]["stages"])
        for plan in plans[1:]:
            if len(plan["stages"]) != stage_count:
                raise AssertionError("Batched partial gather expected equal stage counts.")
            for stage_idx in range(stage_count):
                if (
                    plan["stages"][stage_idx]["shard_group"]
                    is not plans[0]["stages"][stage_idx]["shard_group"]
                ):
                    raise AssertionError("Batched partial gather expected matching shard groups.")
        return stage_count

    def _start_gather_partial_uneven_local_tensors_like_async(
        self, items: list[tuple[torch.Tensor, torch.Tensor]], plans: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if len(items) != len(plans):
            raise AssertionError(
                "Async partial uneven DTensor batch gather item/plan length mismatch: "
                f"items={len(items)}, plans={len(plans)}."
            )
        if not items:
            return {"immediate": []}

        stage_count = self._validate_batched_partial_gather_plans(plans)
        if stage_count == 0:
            return {"immediate": [local_tensor for _, local_tensor in items]}

        local_tensor = items[0][1]
        batch = {
            "dtype": local_tensor.dtype,
            "device": local_tensor.device,
            "element_size": local_tensor.element_size(),
            "item_indices": list(range(len(items))),
            "plans": plans,
        }
        current_buffers = [
            self._flatten_tensor_for_uneven_gather(item_local_tensor)
            for _, item_local_tensor in items
        ]
        with torch.autograd.profiler.record_function(
            f"Muon-FSDP async batched partial gather start count={len(items)}"
        ):
            stage_state = self._prepare_uneven_gather_stage(
                current_buffers, plans, batch, 0, plans[0]["stages"][0]
            )
            pending_stage = self._start_uneven_gather_stage(stage_state, async_op=True)
        return {"plans": plans, "batch": batch, "pending_stage": pending_stage}

    def _finish_gather_partial_uneven_local_tensors_like_async(
        self, pending: dict[str, Any]
    ) -> list[torch.Tensor]:
        if "immediate" in pending:
            return pending["immediate"]

        plans = pending["plans"]
        batch = pending["batch"]
        stage_count = self._validate_batched_partial_gather_plans(plans)
        final_stage_idx = stage_count - 1

        if final_stage_idx == 0:
            self._wait_uneven_gather_stage(pending["pending_stage"])
            return [
                self._reconstruct_partial_tensor_from_final_stage_buffers(
                    plan, pending["pending_stage"], batch_item_idx=batch_item_idx
                )
                for batch_item_idx, plan in enumerate(plans)
            ]

        current_buffers = self._finish_uneven_gather_stage(pending["pending_stage"])
        for stage_idx in range(1, stage_count):
            stage_state = self._prepare_uneven_gather_stage(
                current_buffers, plans, batch, stage_idx, plans[0]["stages"][stage_idx]
            )
            pending_stage = self._start_uneven_gather_stage(stage_state, async_op=False)
            if stage_idx == final_stage_idx:
                self._wait_uneven_gather_stage(pending_stage)
                return [
                    self._reconstruct_partial_tensor_from_final_stage_buffers(
                        plan, pending_stage, batch_item_idx=batch_item_idx
                    )
                    for batch_item_idx, plan in enumerate(plans)
                ]
            current_buffers = self._finish_uneven_gather_stage(pending_stage)

        raise AssertionError("Async batched partial uneven DTensor gather missed final stage.")

    def _gather_partial_uneven_local_tensors_like(
        self, items: list[tuple[torch.Tensor, torch.Tensor]], plans: list[dict[str, Any]]
    ) -> list[torch.Tensor]:
        if len(items) != len(plans):
            raise AssertionError(
                "Partial uneven DTensor batch gather item/plan length mismatch: "
                f"items={len(items)}, plans={len(plans)}."
            )
        if not items:
            return []
        if not plans[0]["stages"]:
            return [local_tensor for _, local_tensor in items]

        stage_count = self._validate_batched_partial_gather_plans(plans)

        local_tensor = items[0][1]
        batch = {
            "dtype": local_tensor.dtype,
            "device": local_tensor.device,
            "element_size": local_tensor.element_size(),
            "item_indices": list(range(len(items))),
            "plans": plans,
        }
        current_buffers = [
            self._flatten_tensor_for_uneven_gather(item_local_tensor)
            for _, item_local_tensor in items
        ]
        final_stage_idx = stage_count - 1
        for stage_idx, stage in enumerate(plans[0]["stages"]):
            with torch.autograd.profiler.record_function(
                f"Muon-FSDP batched partial gather stage {stage_idx} count={len(items)}"
            ):
                stage_state = self._prepare_uneven_gather_stage(
                    current_buffers, plans, batch, stage_idx, stage
                )
                pending_stage = self._start_uneven_gather_stage(stage_state, async_op=False)
                if stage_idx == final_stage_idx:
                    self._wait_uneven_gather_stage(pending_stage)
                    return [
                        self._reconstruct_partial_tensor_from_final_stage_buffers(
                            plan, pending_stage, batch_item_idx=batch_item_idx
                        )
                        for batch_item_idx, plan in enumerate(plans)
                    ]
                current_buffers = self._finish_uneven_gather_stage(pending_stage)

        raise AssertionError("Batched partial uneven DTensor gather did not reach its final stage.")

    def _reconstruct_partial_tensor_from_final_stage_buffers(
        self, plan: dict[str, Any], pending_stage: dict[str, Any], batch_item_idx: int = 0
    ) -> torch.Tensor:
        stage_idx = pending_stage["stage_idx"]
        if stage_idx != len(plan["stages"]) - 1:
            raise AssertionError(
                "Partial uneven DTensor reconstruction requires the final gather stage: "
                f"got stage={stage_idx}, final={len(plan['stages']) - 1}."
            )

        stage = plan["stages"][stage_idx]
        rank_buffers = self._rank_buffers_from_pending_uneven_gather_stage(pending_stage)
        rank_offsets = pending_stage["rank_offsets"]
        if len(rank_buffers) != len(stage["rank_numels"]):
            raise AssertionError(
                "Partial uneven DTensor reconstruction rank buffer count mismatch: "
                f"got {len(rank_buffers)}, expected={len(stage['rank_numels'])}."
            )

        partial_tensor = torch.empty(
            plan["partial_shape"], dtype=rank_buffers[0].dtype, device=rank_buffers[0].device
        )
        partial_offsets = plan["partial_offsets"]
        rank_chunk_counts = stage["rank_chunk_counts"]
        assigned_numel = 0
        chunk_info_idx = 0
        for rank, rank_buffer in enumerate(rank_buffers):
            source_offset = rank_offsets[rank][batch_item_idx]
            expected_source_end = source_offset + stage["rank_numels"][rank]
            for _ in range(rank_chunk_counts[rank]):
                chunk_info = plan["chunk_infos"][chunk_info_idx]
                chunk_info_idx += 1
                chunk_shape = chunk_info["shape"]
                chunk_numel = chunk_info["numel"]
                if chunk_numel == 0:
                    continue
                gathered_tensor = rank_buffer[source_offset : source_offset + chunk_numel]
                source_offset += chunk_numel
                slices = tuple(
                    slice(
                        chunk_info["offset"][dim] - partial_offsets[dim],
                        chunk_info["offset"][dim] - partial_offsets[dim] + chunk_shape[dim],
                    )
                    for dim in range(len(chunk_shape))
                )
                partial_tensor[slices] = gathered_tensor.view(chunk_shape)
                assigned_numel += chunk_numel
            if source_offset != expected_source_end:
                raise AssertionError(
                    "Partial uneven DTensor reconstruction consumed an unexpected rank size: "
                    f"rank={rank}, consumed={source_offset - rank_offsets[rank][batch_item_idx]}, "
                    f"expected={stage['rank_numels'][rank]}."
                )

        if chunk_info_idx != len(plan["chunk_infos"]):
            raise AssertionError(
                "Partial uneven DTensor reconstruction consumed an unexpected chunk count: "
                f"consumed={chunk_info_idx}, expected={len(plan['chunk_infos'])}."
            )
        if assigned_numel != plan["partial_numel"]:
            raise AssertionError(
                "Partial uneven DTensor reconstruction did not cover the partial tensor: "
                f"assigned={assigned_numel}, expected={plan['partial_numel']}."
            )
        return partial_tensor

    def _local_shard_from_partial_update_like(
        self, dtensor_ref, partial_update: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            update_uneven_dtensor_chunk_metadata(dtensor_ref)
        chunk_metadata_list = dtensor_ref._local_tensor.__create_chunk_list__()
        if len(chunk_metadata_list) != 1:
            raise ValueError(
                f"Expected exactly one local DTensor chunk, got {len(chunk_metadata_list)}."
            )

        chunk_metadata = chunk_metadata_list[0]
        slices = []
        for dim, (offset, size) in enumerate(zip(chunk_metadata.offsets, chunk_metadata.sizes)):
            if partial_update.shape[dim] == dtensor_ref.shape[dim]:
                slices.append(slice(offset, offset + size))
            elif partial_update.shape[dim] == size:
                slices.append(slice(None))
            else:
                raise AssertionError(
                    "Partial distributed Muon update has incompatible local shape: "
                    f"dim={dim}, partial_dim={partial_update.shape[dim]}, "
                    f"full_dim={dtensor_ref.shape[dim]}, local_size={size}."
                )

        local_update = partial_update[tuple(slices)]
        if local_update.dtype != dtensor_ref._local_tensor.dtype:
            local_update = local_update.to(dtype=dtensor_ref._local_tensor.dtype)
        return local_update

    def _local_shard_from_full_update_like(self, dtensor_ref, full_update: torch.Tensor):
        if not hasattr(dtensor_ref._local_tensor, "__create_chunk_list__"):
            raise ValueError(
                f"{dtensor_ref} is not a Megatron-FSDP DTensor parameter "
                "with DTensor._local_tensor.__create_chunk_list__. "
                "Verify that `update_uneven_dtensor_chunk_metadata` "
                "has been called on this uneven DTensor."
            )
        shard_metadata = dtensor_ref._local_tensor.__create_chunk_list__()[0]
        slices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(shard_metadata.offsets, shard_metadata.sizes)
        )
        local_update = full_update[slices]
        if local_update.dtype != dtensor_ref._local_tensor.dtype:
            local_update = local_update.to(dtype=dtensor_ref._local_tensor.dtype)
        return local_update

    @torch.no_grad()  # type: ignore[misc]
    def _local_muon_update(
        self, p: torch.Tensor, grad: torch.Tensor, group: dict[str, Any]
    ) -> None:
        """Local (non-DP) Muon update – identical to OrthogonalizedOptimizer.step body."""
        from emerging_optimizers import utils

        state = self.state[p]
        if grad.dtype != state["momentum_buffer"].dtype:
            grad = grad.to(dtype=state["momentum_buffer"].dtype)
        self._apply_weight_decay_inplace(p, grad, group["lr"], group["weight_decay"])
        state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
        if self.nesterov:
            grad = grad.lerp(state["momentum_buffer"], group["momentum"])
        else:
            grad = state["momentum_buffer"]
        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            orth_grad = self.orthogonalize(p, grad, **group_kwargs)
        self.pre_weight_update_fn_inplace(p, orth_grad)
        p.add_(orth_grad, alpha=-group["lr"])
        self.post_weight_update_fn_inplace(p)


class TensorParallelAdaptiveMuon(TensorParallelMuon, AdaptiveMuon):
    """Tensor Parallel Adaptive Muon optimizer.

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: The exponential decay rate for momentum.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to use decoupled weight decay.
        split_qkv: Whether to split QKV weights for orthogonalization.
        is_qkv_fn: Function to determine if a tensor is a QKV weight.
        qkv_split_shapes: Shapes for splitting QKV weights.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        use_syrk: Whether to use Triton SYRK Newton-Schulz kernels when supported.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", or "distributed").
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        use_syrk: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            use_syrk=use_syrk,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step function"""
        return AdaptiveMuon.step(self, closure)


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = (
        lambda p: getattr(p, "is_qkv", False)
        or getattr(getattr(p, "orig_param", None), "is_qkv", False)
        or _is_named_qkv_param(p)
    )
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _default_adam_based_eopt_config_to_kwargs(
    eopt_name, config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Convert OptimizerConfig to default emerging optimizer constructor kwargs."""
    kwargs = _kwargs_from_config(registry.get_optimizer_cls(eopt_name), eopt_name, config)
    kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
    }
)

# Register soap with default config
# TODO(skyw): register all emerging optimizers.
if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
        if eopt_name in _EMERGING_OPTIMIZERS:
            # skip already registered local versions, e.g. TensorParallel versions.
            continue
        _EMERGING_OPTIMIZERS[eopt_name] = EmergingOptimizerEntry(
            optimizer_cls=registry.get_optimizer_cls(eopt_name)
        )
