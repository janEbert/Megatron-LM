# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for the delegating plumbing of composed optimizers."""

import threading

import torch
import torch.nn as nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer.composed_optimizer import (
    ComposedOptimizer,
)


class _Composed(ComposedOptimizer):
    """Minimal concrete composed optimizer over a pre-built inner."""

    def __init__(self, params, inner: torch.optim.Optimizer) -> None:
        with self._without_property_methods():
            super().__init__(params, {})
        self._inner = inner

    def step(self, closure=None):
        """No-op step: the plumbing is under test, not the optimization."""
        return None


def _inner(params) -> torch.optim.Optimizer:
    """A plain `torch.optim` optimizer as the composed inner."""
    return torch.optim.SGD(params, lr=0.1)


def test_base_init_lands_despite_delegating_properties():
    """`torch.optim.Optimizer.__init__` completes with the descriptors present.

    The base assigns `param_groups`/`defaults`/`state` as plain instance attributes;
    without `_without_property_methods`, those assignments would hit the delegating
    setters and a missing `_inner`. The base-init entries are dropped afterwards, since
    every read delegates to the inner.
    """
    param = nn.Parameter(torch.zeros(2))
    outer = _Composed([param], _inner([param]))
    assert "param_groups" not in outer.__dict__
    assert "defaults" not in outer.__dict__
    assert "state" not in outer.__dict__


def test_reads_delegate_to_the_inner():
    """The three delegated names are the inner optimizer's objects."""
    param = nn.Parameter(torch.zeros(2))
    inner = _inner([param])
    outer = _Composed([param], inner)
    assert outer.param_groups is inner.param_groups
    assert outer.defaults is inner.defaults
    assert outer.state is inner.state


def test_writes_delegate_to_the_inner():
    """Mutations and reassignments through the outer reach the inner's objects."""
    param = nn.Parameter(torch.zeros(2))
    inner = _inner([param])
    outer = _Composed([param], inner)

    outer.param_groups[0]["lr"] = 0.2
    assert inner.param_groups[0]["lr"] == 0.2
    outer.defaults["momentum"] = 0.9
    assert inner.defaults["momentum"] == 0.9
    outer.state[param]["step"] = 3
    assert inner.state[param]["step"] == 3

    replacement = {"lr": 0.3}
    outer.defaults = replacement
    assert inner.defaults is replacement


def test_all_params_flattens_all_groups():
    """`_all_params` flattens the params across every param group."""
    p0 = nn.Parameter(torch.zeros(2))
    p1 = nn.Parameter(torch.zeros(3))
    outer = _Composed([{"params": [p0]}, {"params": [p1]}], _inner([p0, p1]))
    assert outer._all_params() == [p0, p1]


def test_concurrent_constructions_are_serialized():
    """Constructions of the same type from several threads complete without interference.

    The delegating descriptors live on the type, so without the lock, concurrent
    constructions could interleave their delete/restore phases and leave one
    constructing with live descriptors. This smoke test constructs concurrently and
    verifies the descriptors are intact afterwards.
    """
    errors: list[Exception] = []

    def construct() -> None:
        try:
            for _ in range(20):
                param = nn.Parameter(torch.zeros(2))
                _Composed([param], _inner([param]))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=construct) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert isinstance(ComposedOptimizer.param_groups, property)
    assert isinstance(ComposedOptimizer.defaults, property)
    assert isinstance(ComposedOptimizer.state, property)
