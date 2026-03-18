# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron muon optimizer wrapper to handle tensor-parallel."""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey

try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object

# TODO: Remove this separate try/except once the next version of emerging_optimizers
# (which includes Lion) is released. Then Lion can be imported in the block above.
try:
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import

    HAVE_LION = True
except ImportError:
    HAVE_LION = False


logger = logging.getLogger(__name__)


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
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
        pg_collection: Optional[ProcessGroupCollection] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, '
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
                mode="duplicated" if mode == "blockwise" else mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
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
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            # emerging-optimizers use None instead of -1 to indicate no tensor parallel
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            # split grouped attention parameters (e.g., QKV, GQA, etc.)
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            # Apply Newton-Schulz and scales to each component, concat back
            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class FSDPZeROTensorParallelMuon(TensorParallelMuon):
    """TensorParallelMuon extended for FSDP ZeRO-1/2/3.

    Supports all three ZeRO sharding strategies:

    * ``optim`` (ZeRO-1): optimizer state sharded; grads reduce-scattered.
    * ``optim_grads`` (ZeRO-2): optimizer state + grads sharded.
    * ``optim_grads_params`` (ZeRO-3): optimizer state + grads + model params sharded.

    For all three, ``finish_grad_sync()`` reduce-scatters gradients so each DP rank holds
    only a contiguous ``Shard(0)`` row-shard of the full (TP-local) 2D gradient.
    Newton-Schulz requires the full matrix.  This class restores that invariant by:

      1. Extracting the local shard tensor via ``.to_local()`` on any Shard(0) DTensor.
      2. Allgathering the DP row-shards across the DP group to reconstruct the
         TP-local, DP-full gradient matrix.
      3. Trimming FSDP bucket-padding rows using the declared global shape from the DTensor.
      4. Delegating to ``TensorParallelMuon.orthogonalize()`` which handles the remaining
         TP dimension (via ``newton_schulz_tp``).
      5. Extracting the local DP row-shard of the orthogonalized result, zero-padding back
         to the original shard size for the last rank when FSDP padding is present.

    Memory note for ZeRO-3: during each Muon step, a temporary ``full_grad`` buffer of
    shape ``(tp_local_rows, C)`` is materialized per Muon parameter for the allgather.
    Peak extra memory ≈ ``3 × (m × n) × sizeof(float32)`` per linear layer
    (allgather buffer + full NS output + shard padding).  Momentum itself stays sharded
    (``O(mn / dp_size)`` per rank), the same as the optimizer-state savings from ZeRO-3.

    For ``no_shard`` or single-rank DP, this class falls back transparently to the parent.
    """

    def __init__(self, params, dp_group=None, **kwargs):
        self.dp_group = dp_group
        super().__init__(params, **kwargs)

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize with DP-shard allgather for FSDP ZeRO >= 1."""
        if self.dp_group is None or get_pg_size(self.dp_group) == 1:
            return super().orthogonalize(p, grad, **kwargs)

        dp_size = get_pg_size(self.dp_group)
        dp_rank = get_pg_rank(self.dp_group)

        # The momentum buffer (grad) and param (p) may be Shard(0) DTensors whose
        # .shape[0] is the *global* row count.  Always extract the local shard tensor
        # so that shard_rows reflects the actual per-rank row count.
        try:
            from torch.distributed.tensor import DTensor as _DTensor

            _have_dtensor = True
        except ImportError:
            _DTensor = None  # type: ignore[assignment,misc]
            _have_dtensor = False

        grad_local = grad.to_local() if (_have_dtensor and isinstance(grad, _DTensor)) else grad
        shard_rows = grad_local.shape[0]

        # Allgather DP row-shards to reconstruct the TP-local, DP-full gradient matrix.
        # FSDP guarantees equal shard sizes across all DP ranks (buckets are padded to
        # dp_size * inner_dim), so a simple all_gather_into_tensor is sufficient.
        full_grad = torch.empty(
            shard_rows * dp_size, *grad_local.shape[1:],
            device=grad_local.device, dtype=grad_local.dtype,
        )
        torch.distributed.all_gather_into_tensor(
            full_grad, grad_local.contiguous(), group=self.dp_group
        )

        # Trim FSDP bucket-padding rows.
        # p.shape[0] on a DTensor is the declared full *global* row count (set via
        # shape=param.shape in from_local).  For TP column-parallel (partition_dim=0):
        # tp_local_rows = global / tp_size.  For all other params tp_size_dim0 == 1.
        p_global_rows = (
            p.shape[0] if (_have_dtensor and isinstance(p, _DTensor)) else shard_rows * dp_size
        )

        tp_group = (
            self.pg_collection.expt_tp
            if getattr(p, 'expert_tp', False)
            else self.pg_collection.tp
        ) if self.pg_collection else None

        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        tp_size_dim0 = get_pg_size(tp_group) if (partition_dim == 0 and tp_group is not None) else 1
        tp_local_rows = p_global_rows // max(tp_size_dim0, 1)

        full_grad = full_grad[:tp_local_rows]

        # Apply NS to the full TP-local, DP-full gradient.
        # super().orthogonalize() re-computes tp_group/partition_dim internally, which is
        # correct: the TP dimension of the reconstructed full_grad is unchanged.
        orth_full_grad = super().orthogonalize(p, full_grad, **kwargs)

        # Extract the local DP row-shard.  If the last rank had fewer real rows than
        # shard_rows (FSDP padding), zero-fill the extra rows so padded elements get a
        # no-op update.
        start_row = dp_rank * shard_rows
        end_row = min(start_row + shard_rows, tp_local_rows)
        orth_shard = orth_full_grad[start_row:end_row]

        if orth_shard.shape[0] < shard_rows:
            pad_rows = shard_rows - orth_shard.shape[0]
            pad = torch.zeros(
                pad_rows, *grad_local.shape[1:],
                device=grad_local.device, dtype=grad_local.dtype,
            )
            orth_shard = torch.cat([orth_shard, pad], dim=0)

        return orth_shard


def _get_mfsdp_models(model_chunks):
    """Extract list of MegatronFSDP instances from FSDP-wrapped model chunks."""
    mfsdp_models = []
    for chunk in model_chunks:
        # FullyShardedDataParallel delegates finish_grad_sync / start_param_sync
        # from its .module (MegatronFSDP).  install_optimized_model_weights lives
        # directly on MegatronFSDP, so we need the inner module reference.
        if hasattr(chunk, 'finish_grad_sync') and hasattr(chunk, 'module'):
            mfsdp_models.append(chunk.module)
    if not mfsdp_models:
        raise RuntimeError(
            "Could not find any MegatronFSDP instances in model_chunks. "
            "Ensure the model is wrapped with FullyShardedDataParallel."
        )
    return mfsdp_models


class FSDPMuonChainedOptimizer:
    """Thin FSDP-protocol adapter wrapping a Muon-based MegatronOptimizer.

    Injects the MegatronFSDP step contract around the inner optimizer:
      1. finish_grad_sync()               — waits for async grad sync, attaches grads
                                            (reduce-scatters for ZeRO-1/2/3,
                                            allreduces for no_shard)
      2. inner_optimizer.step()           — Muon NS + weight update + Adam
                                            (FSDPZeROTensorParallelMuon allgathers
                                            sharded gradients before NS for ZeRO-1/2/3)
      3. install_optimized_model_weights() — copies fp32 main weights → bf16 model weights
                                            (writes to sharded bf16 buffer for ZeRO-3)

    All other attribute accesses are delegated to the inner optimizer via __getattr__,
    making this class transparent to the training loop.
    """

    def __init__(self, inner: MegatronOptimizer, mfsdp_models: list):
        # Use object.__setattr__ to avoid triggering our own __getattr__ during init.
        object.__setattr__(self, 'inner', inner)
        object.__setattr__(self, '_mfsdp_models', mfsdp_models)

    @torch.no_grad()
    def step(self):
        """FSDP-aware optimizer step: sync grads → inner step → install weights."""
        for mfsdp in self._mfsdp_models:
            if not mfsdp.model_auto_sync:
                mfsdp.finish_grad_sync()
        result = self.inner.step()
        for mfsdp in self._mfsdp_models:
            mfsdp.install_optimized_model_weights()
        return result

    def zero_grad(self, set_to_none: bool = True):
        """Zero optimizer gradients. FSDP grad buffer is zeroed by the training loop."""
        self.inner.zero_grad(set_to_none)

    def __getattr__(self, name: str):
        """Delegate all other attribute accesses to the inner optimizer."""
        return getattr(object.__getattribute__(self, 'inner'), name)


def get_megatron_fsdp_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> "FSDPMuonChainedOptimizer":
    """Muon optimizer factory for Megatron-FSDP, supporting all ZeRO strategies.

    Supports all four ``data_parallel_sharding_strategy`` values:

    * ``no_shard`` (ZeRO-0): params/grads are Replicate DTensors; plain
      ``TensorParallelMuon`` is used — no extra communication.
    * ``optim`` / ``optim_grads`` / ``optim_grads_params`` (ZeRO-1/2/3):
      ``finish_grad_sync()`` reduce-scatters gradients into per-rank Shard(0) DTensors.
      ``FSDPZeROTensorParallelMuon`` allgathers across the DP group before
      Newton-Schulz, then re-shards the orthogonalized update.

    HSDP (``outer_dp_sharding_strategy != no_shard``) is not supported and is blocked
    by argument validation in ``arguments.py``.

    ``FP32Optimizer`` is used for all strategies (instead of
    ``Float16OptimizerWithFloat16Params``) because FSDP attaches gradients directly via
    ``finish_grad_sync()``, bypassing the ``main_grad`` mechanism.

    Args:
        config: Optimizer configuration.
        model_chunks: FSDP-wrapped model chunks.
        config_overrides: Per-parameter group overrides.
        use_gloo_process_groups: If false, disable Gloo process groups.
        layer_wise_distributed_optimizer: If true, use LayerWiseDistributedOptimizer.
        pg_collection: Process group collection; defaults to MPU process groups.
    """
    assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers is not installed."
    assert not config.fp16, 'Muon with fp16 is not supported.'

    # Muon currently reuses Adam config infrastructure.
    config.optimizer = 'adam'

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(
        logger, logging.INFO,
        f'Setting up Megatron-FSDP Muon optimizer with config {config}'
    )

    def muon_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    linear_params = []
    nonlinear_params = []
    qkv_split_shapes = None
    for model_chunk in model_chunks:
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    muon_kwargs = {
        "lr": config.lr,
        "momentum_beta": config.muon_momentum,
        "use_nesterov": config.muon_use_nesterov,
        "weight_decay": config.weight_decay,
        "fp32_matmul_prec": config.muon_fp32_matmul_prec,
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "split_qkv": config.muon_split_qkv,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "qkv_split_shapes": qkv_split_shapes,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "mode": config.muon_tp_mode,
    }

    # Freeze nonlinear params so _get_param_groups only sees linear params for Muon.
    for param in nonlinear_params:
        param.requires_grad = False
    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)

    # Split expert params out when not using layer-wise distributed optimizer.
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        for group in list(linear_param_groups):
            if group['is_expert_parallel']:
                expert_param_groups.append(group)
                linear_param_groups.remove(group)

    # Choose Muon variant based on the ZeRO sharding strategy.
    # - no_shard (ZeRO-0): params/grads are full Replicate DTensors; plain TensorParallelMuon.
    # - optim / optim_grads / optim_grads_params (ZeRO-1/2/3): finish_grad_sync() performs
    #   reduce_scatter so each rank holds a Shard(0) row-shard of every gradient.
    #   FSDPZeROTensorParallelMuon allgathers across the DP group before Newton-Schulz
    #   and re-shards the orthogonalized result back to the local row-shard.
    fsdp_sharding_strategy = model_chunks[0].ddp_config.data_parallel_sharding_strategy
    if fsdp_sharding_strategy != 'no_shard':
        muon_cls = FSDPZeROTensorParallelMuon
        dp_group = pg_collection.dp_cp
        muon_kwargs['dp_group'] = dp_group
    else:
        muon_cls = TensorParallelMuon

    # FP32Optimizer.prepare_grads() guards with hasattr(param, 'main_grad'),
    # which is safe for DTensors that receive grad directly from finish_grad_sync().
    muon_base = muon_cls(linear_param_groups, **muon_kwargs)
    muon_opt = FP32Optimizer(muon_base, config, muon_init_state_fn)
    optimizers = [muon_opt]

    if expert_param_groups:
        expert_muon_base = muon_cls(expert_param_groups, **muon_kwargs)
        expert_muon_opt = FP32Optimizer(expert_muon_base, config, muon_init_state_fn)
        setattr(expert_muon_opt, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_muon_opt)

    # Restore nonlinear; freeze linear so Adam only gets non-linear params.
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # Adam for non-linear params via the standard FSDP path in get_megatron_optimizer().
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # Restore all params.
    for param in linear_params:
        param.requires_grad = True

    # The FSDP branch of get_megatron_optimizer() may return a single optimizer
    # (not always a ChainedOptimizer) for single-chunk models.
    adam_optimizers = getattr(chained_adam, 'chained_optimizers', [chained_adam])
    n_muon = 1 + (1 if expert_param_groups else 0)
    init_fns = n_muon * [muon_init_state_fn] + len(adam_optimizers) * [adam_init_state_fn]
    optimizers += adam_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(
            logger, logging.INFO,
            'Using LayerWiseDistributedOptimizer for Muon + Megatron-FSDP'
        )
        # Must temporarily unset config.bf16 to prevent LayerWiseDistributedOptimizer
        # from re-wrapping each optimizer with Float16OptimizerWithFloat16Params.
        # That wrapper is incompatible with FSDP DTensor params (no .main_grad).
        # Also force async_allgather=False: DDP bucket infrastructure is absent in FSDP.
        reset_config_bf16 = config.bf16
        config.bf16 = False
        inner = LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=init_fns,
            async_allgather=False,
        )
        config.bf16 = reset_config_bf16
    else:
        inner = ChainedOptimizer(optimizers)

    mfsdp_models = _get_mfsdp_models(model_chunks)
    return FSDPMuonChainedOptimizer(inner, mfsdp_models)


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    # TODO: Mutating config.optimizer is a side effect; clean up after
    # https://github.com/NVIDIA/Megatron-LM/pull/3638 lands.
    # Set the nonlinear optimizer for muon (used for embeddings, biases, norms).
    config.optimizer = config.muon_scalar_optimizer

    assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers is not installed."
    if config.muon_scalar_optimizer == 'lion':
        assert HAVE_LION, (
            "Lion optimizer requires a version of 'emerging_optimizers' that includes Lion. "
            "Please upgrade to use --muon-scalar-optimizer lion."
        )

    # Dist-opt is not supported due to strong coupling with how DDP init grad buffer
    # In theory we can change DDP to enable use muon and dist-opt-adam together
    if config.use_distributed_optimizer:
        raise Exception('muon with dist optimizer is not supported.')
    # only support bf16 w/o loss scale now
    if config.fp16:
        raise Exception('muon with fp16 is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    # TODO(boxiangw): Improve usability after optimizer refactor
    # TODO(boxiangw): support precision aware optimizer
    def muon_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    def lion_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)

    nonlinear_init_state_fn = (
        lion_init_state_fn if config.muon_scalar_optimizer == 'lion' else adam_init_state_fn
    )

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # add flag for qkv parameter
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            # TODO(deyuf): currently only allow 2D non-embedding weight to avoid breaking
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    muon_kwargs = {
        "lr": config.lr,
        "momentum_beta": config.muon_momentum,
        "use_nesterov": config.muon_use_nesterov,
        "weight_decay": config.weight_decay,
        "fp32_matmul_prec": config.muon_fp32_matmul_prec,
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "split_qkv": config.muon_split_qkv,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "qkv_split_shapes": qkv_split_shapes,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "mode": config.muon_tp_mode,
    }

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    # if layerwise distributed optimizer is not used, need to handle ep params separately
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        for group in linear_param_groups:
            if group['is_expert_parallel']:
                expert_param_groups.append(group)
                linear_param_groups.remove(group)

    optimizer = TensorParallelMuon(linear_param_groups, **muon_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, muon_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, muon_init_state_fn)

    optimizers.append(optimizer)

    # expert optimizer exists meaning layerwise distributed optimizer is not used
    if len(expert_param_groups) > 0:
        expert_optimizer = TensorParallelMuon(expert_param_groups, **muon_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, muon_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, muon_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    init_fns = [muon_init_state_fn] + len(chained_adam.chained_optimizers) * [
        nonlinear_init_state_fn
    ]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Muon')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=init_fns,
            model_chunks=model_chunks,
            async_allgather=config.overlap_param_gather,
        )
    return ChainedOptimizer(optimizers)
