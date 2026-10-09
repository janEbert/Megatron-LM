# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The owner-compute orthogonalized optimizer for MFSDP v2.

Implements the Muon-style orthogonalizing optimizer step on top of MFSDP v2's
all-`RowAtomic` placements: the Newton-Schulz kernels come from `emerging_optimizers`
through a composed `OrthogonalizedOptimizer` (see `orthogonalization`), and the
communication runs through the `gather_scatter` module over the group's `DBuffer`s.

Every >=2D parameter is contracted to a 2D matrix `(shape[0], rest)` for orthogonalization.
Like the upstream `OrthogonalizedOptimizer`, `<2D` parameters are not contracted and error
when `orthogonalize` is called; subclasses may reject them earlier.
"""

import contextlib
import dataclasses
from collections.abc import Callable, Generator, Sequence
from typing import Any, overload, override

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT

from ..dbuffer import DBuffer
from ..gather_scatter import gather, scatter
from ..owner_planning import ns_cost_fn
from ..parameter_group import FsdpParameterGroup, get_containing_parameter_group
from ..placement import RowAtomic
from ..range import Range, intersect_ranges
from .composed_optimizer import ComposedOptimizer
from .orthogonalization import (
    OrthogonalizedOptimizer,
    _require_emerging_optimizers,
    apply_update,
    compute_orthogonalization_inputs,
    orthogonalize_owned,
)


def _fsdp_group_of(param: torch.nn.Parameter) -> FsdpParameterGroup:
    """Return the owning `FsdpParameterGroup`, raising for unsharded parameters."""
    fsdp_group = get_containing_parameter_group(param)
    if fsdp_group is None:
        raise RuntimeError(
            "FsdpOrthogonalizedOptimizer parameters must be FSDP-sharded; "
            f"parameter {param!r} is not owned by an FsdpParameterGroup."
        )
    return fsdp_group


@dataclasses.dataclass
class _ActiveGroup:
    """One FSDP group's grad-bearing parameters for this step, with their torch group.

    Step-local plan record, built in rank-identical discovery order: the two-phase
    posting completes discovery of every active group before any communication is
    posted, and every phase then iterates these records in the same order.

    Attributes:
        fsdp_group: The FSDP parameter group whose active parameters this record is.
        scratch: The group's scratch `DBuffer` (see `FsdpOrthogonalizedOptimizer
            ._scratch_for`): holds the local pre-orthogonalization chunks as the
            gather source, then the received result shards as the scatter destination.
        tensor_indices: The group's participating tensor indices, in buffer order.
        tensor_to_owner: Participating tensor index to owner global rank; assigned by
            the joint balancing.
        torch_group: The `torch.optim` param group holding the hyperparameters.
        lr: The learning rate, snapshotted from `torch_group` when this record is
            built, so the whole step applies one consistent value even if the param
            group is adjusted while the step is executing (relevant once the phases
            run deferred across calls).
        destination: The owner's full pre-orthogonalization buffers per owned tensor
            index; filled in by `_step_gather`.
    """

    fsdp_group: FsdpParameterGroup
    scratch: DBuffer
    tensor_indices: tuple[int, ...]
    tensor_to_owner: dict[int, int]
    torch_group: dict[str, Any]
    destination: dict[int, torch.Tensor] | None = None
    lr: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.lr = self.torch_group["lr"]


class FsdpOrthogonalizedOptimizer(ComposedOptimizer):
    """Owner-compute orthogonalized optimizer for all-`RowAtomic` MFSDP v2 parameters.

    A drop-in `torch.optim.Optimizer` for the training loop and checkpointer: through
    `ComposedOptimizer`, it delegates `param_groups`/`defaults`/`state` to the composed
    `OrthogonalizedOptimizer` (held as `self._inner`), which provides only the
    Newton-Schulz kernel, weight decay, and the pre/post weight-update hooks. The `step`
    override replaces the inner's per-parameter all-gather path with the pipelined
    owner-compute algorithm.

    Each step runs in two posting phases: first every active group's gather is
    posted (local pre-orthogonalization chunks computed on the caller's stream,
    P2P stream-ordered on each group's own CUDA stream), then the groups'
    Newton-Schulz work, scatter, and apply run per group on its stream. Because
    no scatter is enqueued before every gather, no scatter's Newton-Schulz
    dependency can head-of-line-block another group's gather on the shared NCCL
    communicator, so the groups' Newton-Schulz work genuinely overlaps. Owner
    compute is balanced jointly across all active groups. At the end, the caller's
    stream is ordered behind every group stream and the touched groups' model
    weights are synced from their main weights.

    Args:
        params: Parameters or param-group dicts to optimize.
        inner_optimizer: Pre-built `OrthogonalizedOptimizer` over the same parameters;
            its arguments (`lr`, `momentum`, `weight_decay`, `nesterov`, ...) are the
            optimization hyperparameters.
        dp_mesh: Device mesh of the FSDP data-parallel group. Every parameter group's
            mesh must be this mesh.
        num_ns_steps: Newton-Schulz iteration count, used by the owner load-balancing
            cost heuristic. If `None`, defaults to 1.
    """

    # The composed inner provides the EO-specific hooks the step delegates to.
    _inner: OrthogonalizedOptimizer

    def __init__(
        self,
        params: ParamsT,
        inner_optimizer: OrthogonalizedOptimizer,
        dp_mesh: DeviceMesh,
        num_ns_steps: int | None = None,
    ) -> None:
        _require_emerging_optimizers()

        if num_ns_steps is None:
            num_ns_steps = 1
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}.")

        self.dp_mesh = dp_mesh
        self._num_ns_steps: int = num_ns_steps
        self._scratches: dict[FsdpParameterGroup, DBuffer] = {}
        self._streams: dict[FsdpParameterGroup, torch.cuda.Stream] = {}

        # Disable properties while initializing this instance. We'd either have a missing attribute
        # or would reset the inner optimizer's attributes.
        with self._without_property_methods():
            super().__init__(params, {})
        self._inner = inner_optimizer

        # Validate: FSDP-sharded, dimensionalities, one mesh, all-`RowAtomic` placements.
        params_list = self._all_params()
        self._validate_param_ndims(params_list)
        seen: set[FsdpParameterGroup] = set()
        for param in params_list:
            fsdp_group = _fsdp_group_of(param)
            if fsdp_group in seen:
                continue
            seen.add(fsdp_group)
            if fsdp_group.mesh is not self.dp_mesh:
                raise ValueError(
                    "FsdpOrthogonalizedOptimizer requires every parameter group's mesh "
                    f"to be the optimizer's dp_mesh; {fsdp_group!r} has a different mesh."
                )
            for buf in (fsdp_group.main_weight, fsdp_group.model_weight, fsdp_group.main_grad):
                if buf is None:
                    continue
                if not isinstance(buf, DBuffer):
                    raise ValueError(
                        "FsdpOrthogonalizedOptimizer requires non-quantized buffers, but "
                        f"{fsdp_group!r} has a {type(buf).__name__} buffer."
                    )
                if not all(isinstance(p, RowAtomic) for p in buf.placements):
                    raise ValueError(
                        "FsdpOrthogonalizedOptimizer requires all-RowAtomic placements "
                        "(parameter=RowAtomic, gradient=RowAtomic, optimizer=RowAtomic), but "
                        f"{fsdp_group!r} has non-RowAtomic placements "
                        f"{[type(p).__name__ for p in buf.placements]}."
                    )

    def _validate_param_ndims(self, params: Sequence[torch.Tensor]) -> None:
        """Validate parameter dimensionalities; subclasses may restrict them.

        The base class performs no dimensionality filtering: every >=2D parameter is
        contracted to a 2D matrix `(shape[0], rest)` for orthogonalization, and `<2D`
        parameters — like in the upstream `OrthogonalizedOptimizer` — error when
        `orthogonalize` is called. Subclasses may override this to reject parameters
        earlier; Muon's convention, for instance, rejects `<2D` parameters at
        construction.
        """
        pass

    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        """Performs lazy momentum-state initialization, delegated to the inner optimizer."""
        self._inner._init_group(group, skip_non_grad_params=skip_non_grad_params)

    def _this_rank(self) -> int:
        """This rank's global rank."""
        return self.dp_mesh.get_rank()

    def _device(self) -> torch.device:
        """The DP mesh's device."""
        return torch.device(self.dp_mesh.device_type)

    def _sharded_dtensor(self, fsdp_group: FsdpParameterGroup, tensor_index: int) -> DTensor:
        """Return a group's sharded parameter as a `DTensor`, asserting its type.

        The sharded parameters are constructed as `nn.Parameter`s over `DBuffer` DTensors,
        which preserves the `DTensor` class at runtime; the static types only see
        `nn.Parameter`. Each use asserts the `DTensor` view, so a misconstructed group
        fails loudly instead of dispatching wrongly.
        """
        param = fsdp_group.fsdp_parameters[tensor_index].sharded
        assert isinstance(
            param, DTensor
        ), f"Sharded parameter {tensor_index} of {fsdp_group!r} is not a `DTensor`."
        return param

    def _scratch_for(self, fsdp_group: FsdpParameterGroup) -> DBuffer:
        """Return the group's scratch buffer, pooled and reused across steps.

        Same layout, placements, and mesh as the group's buffers. The chunks and
        updates are fp32 end to end — the Newton-Schulz kernel computes in fp32
        (`orthogonalize_owned` casts its input; `scatter` requires matching source and
        destination dtypes) — so one fp32 scratch serves as the gather source and the
        scatter destination.
        """
        scratch = self._scratches.get(fsdp_group)
        if scratch is None:
            main = fsdp_group.main_weight
            scratch = self._scratches[fsdp_group] = DBuffer.empty(
                main.mesh, main.placements, main.layout.tensor_shapes, torch.float32, main.device
            )
        return scratch

    def _stream_for(self, fsdp_group: FsdpParameterGroup) -> torch.cuda.Stream | None:
        """Return the pipelining stream for one group, pooled and reused across steps.

        Returns `None` on CPU, where everything runs on the current stream.
        """
        if not torch.cuda.is_available():
            return None
        if fsdp_group not in self._streams:
            self._streams[fsdp_group] = torch.cuda.Stream(device=self._device())
        return self._streams[fsdp_group]

    @contextlib.contextmanager
    def _p2p_scope(self, stream: torch.cuda.Stream | None) -> Generator[None, None, None]:
        """Run P2P on `stream`, which waits on the caller's stream at entry.

        No-op without CUDA, where everything runs on the caller's current stream. The caller's
        stream itself is never blocked; the step waits on the group streams before reading the
        results.
        """
        if stream is None:
            yield
            return
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            yield

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single pipelined owner-compute optimization step."""
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        # Insertion-ordered: the loops below post P2P ops and collectives whose
        # order must match on every rank — NCCL matches P2P FIFO per peer pair
        # (tags unsupported), and `sync_model_weight_from_main_weight` is a
        # collective on the shared mesh. Set iteration order is id-hash dependent
        # and can differ across ranks; the param list is rank-identical.
        touched: dict[FsdpParameterGroup, None] = {}
        temporaries: list[Any] = []

        # One `_ActiveGroup` per active FSDP group of this step, in rank-identical
        # order: the two-phase posting below completes discovery of every active
        # group before any communication is posted.
        actives: list[_ActiveGroup] = []
        for torch_group in self.param_groups:
            self._init_group(torch_group)
            params = [p for p in torch_group["params"] if p.grad is not None]
            if not params:
                continue
            # Membership by identity: tensor `==` is elementwise, so `in` on a
            # tensor list could raise for multi-element params.
            grad_bearing = set(params)
            fsdp_groups: dict[FsdpParameterGroup, None] = {}
            for param in params:
                fsdp_group = _fsdp_group_of(param)
                fsdp_groups[fsdp_group] = None
                touched[fsdp_group] = None
            for fsdp_group in fsdp_groups:
                active_indices = [
                    i
                    for i, fp in enumerate(fsdp_group.fsdp_parameters)
                    if fp.sharded in grad_bearing
                ]
                if not active_indices:
                    continue
                # The `orthogonalize` kernel rejects <2D inputs, but only the owner
                # calls it: a per-rank raise there would leave the other ranks
                # deadlocked in the gather/scatter P2P. Validate rank-uniformly from
                # the rank-identical layouts before posting any communication.
                tensor_shapes = fsdp_group.main_weight.layout.tensor_shapes
                for i in active_indices:
                    if len(tensor_shapes[i]) < 2:
                        raise ValueError(
                            f"Only 2D or higher-dimensional parameters can be "
                            f"orthogonalized; parameter {i} of {fsdp_group!r} has "
                            f"shape {tuple(tensor_shapes[i])}."
                        )
                actives.append(
                    _ActiveGroup(
                        fsdp_group=fsdp_group,
                        scratch=self._scratch_for(fsdp_group),
                        tensor_indices=tuple(active_indices),
                        tensor_to_owner={},
                        torch_group=torch_group,
                    )
                )

        # Balance the owner compute jointly across every active group: per-group
        # balancing repeats the same one-sided choice in every group of a group set
        # with a single boundary parameter, compounding the imbalance across the
        # step. Deterministic and rank-identical for the rank-identical `actives`.
        self._balance_owners_jointly(actives)

        # Post every group's gather before any group's scatter: a scatter's NCCL
        # kernel is stream-ordered behind its group's Newton-Schulz work, so if
        # scatter(k) is enqueued first, it head-of-line-blocks gather(k+1) on the
        # shared communicator FIFO and serializes the groups' Newton-Schulz work.
        # With every gather ahead of every scatter, group k+1's gather completes
        # while group k orthogonalizes, so the Newton-Schulz work truly overlaps.
        for group in actives:
            group.destination = self._step_gather(group)
            temporaries.append(group.destination)
        for group in actives:
            temporaries.extend(self._step_apply(group))

        # Order the caller's stream behind every group stream. All step temporaries stay
        # referenced until after these waits, so their blocks cannot be reused before
        # the pending group work completes (the caching allocator assumes the
        # allocation stream; the waits make any later reuse stream-ordered after it).
        for stream in self._streams.values():
            torch.cuda.current_stream().wait_stream(stream)
        temporaries.clear()
        for fsdp_group in touched:
            fsdp_group.sync_model_weight_from_main_weight()
        return loss

    def _balance_owners_jointly(self, actives: list[_ActiveGroup]) -> None:
        """Balance the owner compute jointly across the step's active groups.

        Mirrors `assign_owner_work`'s greedy — single-holder tensors first, then the rest
        in descending cost order, each to its least-loaded holder — but with one shared
        running-cost total across every active group, so boundary tensors of different
        groups compensate one another's imbalance instead of repeating the same
        one-sided per-group choice. Owners are global ranks; candidate ties follow buffer
        order, cost ties keep the rank-identical (group order, buffer order) iteration.
        Deterministic and rank-identical for the rank-identical `actives`.
        """
        cost_fn = ns_cost_fn(self._num_ns_steps)
        ranks = self.dp_mesh.mesh.flatten().tolist()
        running_cost: dict[int, float] = {rank: 0.0 for rank in ranks}
        boundary: list[tuple[float, _ActiveGroup, int, list[int]]] = []
        for group in actives:
            buffer = group.fsdp_group.main_weight
            layout = buffer.layout
            rank_ranges: dict[int, Range] = {
                rank: layout.get_rank_range(self.dp_mesh, buffer.placements, rank) for rank in ranks
            }
            # Buffer order, rather than global rank order, breaks candidate ties.
            ranks_in_buffer_order = sorted(ranks, key=lambda rank: (rank_ranges[rank].start, rank))
            for i in group.tensor_indices:
                tensor_range = layout.get_tensor_range(i)
                candidates = [
                    rank
                    for rank in ranks_in_buffer_order
                    if intersect_ranges(tensor_range, rank_ranges[rank]).numel > 0
                ]
                if not candidates:
                    raise RuntimeError(
                        f"No eligible owner for tensor {i} with shape "
                        f"{layout.tensor_shapes[i]}; no rank owns a shard."
                    )
                cost = cost_fn(layout.tensor_shapes[i])
                if len(candidates) > 1:
                    boundary.append((cost, group, i, candidates))
                    continue
                group.tensor_to_owner[i] = candidates[0]
                running_cost[candidates[0]] += cost
        # Descending cost (longest processing time first); ties keep the iteration order.
        boundary.sort(key=lambda item: -item[0])
        for _cost, group, i, candidates in boundary:
            owner = min(candidates, key=lambda rank: running_cost[rank])
            group.tensor_to_owner[i] = owner
            running_cost[owner] += _cost

    def _step_gather(self, group: _ActiveGroup) -> dict[int, torch.Tensor]:
        """Stage 1 for one group's active parameters: compute the local
        pre-orthogonalization chunks and post the group's gather.

        The chunks are written into the group's scratch `DBuffer`, which the gather reads;
        the owner's `destination` buffers are allocated here and receive the gathered full
        tensors. The gather's P2P runs on the group's own stream, which waits on the
        caller's stream at entry; the caller's stream itself is never blocked. Returns the
        `destination` buffers, which the caller must keep referenced until it has ordered
        its stream behind every group stream; they are also recorded on `group` for
        `_step_apply`.
        """
        buffer = group.scratch
        layout = buffer.layout
        this_rank = self._this_rank()
        my_range = layout.get_rank_range(self.dp_mesh, buffer.placements, this_rank)

        destination: dict[int, torch.Tensor] = {}
        for i in group.tensor_indices:
            tensor_range = layout.get_tensor_range(i)
            if intersect_ranges(tensor_range, my_range).numel == 0:
                continue
            param = self._sharded_dtensor(group.fsdp_group, i)
            grad = param.grad
            assert isinstance(
                grad, DTensor
            ), f"Gradient of sharded parameter {i} is not a `DTensor`."
            momentum = self.state[param]["momentum_buffer"]
            # In-place through the persistent `main_grad` view (`l2` mutates the grad
            # buffer): safe under the zero_grad contract — main_grad is overwritten or
            # zeroed before each step's accumulation (reduce_partial_gradients), so the
            # decay term cannot leak into the next step.
            chunk = compute_orthogonalization_inputs(
                self._inner,
                param.to_local(),
                momentum.to_local(),
                grad.to_local(),
                group.torch_group,
                group.lr,
            )
            buffer.get_tensor_view(i).copy_(chunk)
            if group.tensor_to_owner[i] == this_rank:
                destination[i] = torch.empty(
                    tensor_range.numel, dtype=buffer.dtype, device=buffer.device
                )
        with self._p2p_scope(self._stream_for(group.fsdp_group)):
            gather(buffer, destination, tensor_to_owner=group.tensor_to_owner)
        return destination

    def _step_apply(self, group: _ActiveGroup) -> list[Any]:
        """Stages 2 and 3 for one group's active parameters, on the group's stream:
        orthogonalize the gathered inputs, scatter the updates, and apply them.

        With every group's gather posted before any group's scatter, this group's Newton-Schulz
        work overlaps the other groups' gathers and Newton-Schulz work. The scatter receives
        the remote-owned result shards directly into the scratch `DBuffer`; the owner slices
        its own shard from its full update.

        Returns the step temporaries, which the caller must keep referenced until it has
        ordered its stream behind every group stream.
        """
        buffer = group.scratch
        layout = buffer.layout
        destination = group.destination
        assert destination is not None, "`_step_gather` must run before `_step_apply`."
        this_rank = self._this_rank()
        group_kwargs = {k: v for k, v in group.torch_group.items() if k != "params"}
        tensor_shapes = group.fsdp_group.main_weight.layout.tensor_shapes
        my_range = layout.get_rank_range(self.dp_mesh, buffer.placements, this_rank)

        # Stages 2 and 3, on the group's stream.
        stream = self._stream_for(group.fsdp_group)
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            updates = orthogonalize_owned(
                self._inner,
                this_rank,
                group.tensor_to_owner,
                destination,
                tensor_shapes,
                group_kwargs,
            )
            scatter(updates, buffer, tensor_to_owner=group.tensor_to_owner)
            for i in group.tensor_indices:
                tensor_range = layout.get_tensor_range(i)
                shard = intersect_ranges(tensor_range, my_range)
                if shard.numel == 0:
                    continue
                if group.tensor_to_owner[i] == this_rank:
                    offset = shard.start - tensor_range.start
                    update_shard = updates[i].reshape(-1)[offset : offset + shard.numel]
                else:
                    update_shard = buffer.get_tensor_view(i)
                param_local = self._sharded_dtensor(group.fsdp_group, i).to_local()
                apply_update(self._inner, param_local, update_shard, group.lr)
            return [updates]
