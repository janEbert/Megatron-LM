# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Plumbing for optimizers composing a pre-built inner optimizer.

The concrete optimizer exposes the inner's `param_groups`/`defaults`/`state` through
delegating data descriptors, so training loops and checkpointers see one coherent
optimizer. `torch.optim.Optimizer.__init__`, however, assigns those names as plain
instance attributes — assignments a data descriptor would intercept and forward to the
not-yet-set `_inner`. `_without_property_methods` therefore temporarily removes the
descriptors for the duration of the base `__init__`.

The descriptors are defined on the type and shared by every instance, so the removal
window is process-global: `_delegation_lock`, also defined on the type, serializes
concurrent constructions so their delete/restore phases cannot interleave.
"""

import contextlib
import threading
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, ClassVar

import torch


class ComposedOptimizer(torch.optim.Optimizer):
    """`torch.optim.Optimizer` delegating `param_groups`/`defaults`/`state` to an inner.

    Concrete classes hold a pre-built inner optimizer in `self._inner` — set right after
    the base `__init__`, run inside `_without_property_methods` — and define the actual
    `step`. Reads and writes of the three delegated names go to the inner, so both
    optimizers share one set of bookkeeping objects.
    """

    _inner: torch.optim.Optimizer

    # Serializes `_without_property_methods` across concurrent constructions: the
    # delegating descriptors live on the type (shared by every instance), so without the
    # lock, two constructions could interleave their delete/restore phases and leave one
    # constructing with live descriptors. Re-entrant because a subclass's `__init__` may
    # wrap its own base-`__init__` (and thereby this context) again on the same thread.
    _delegation_lock: ClassVar[threading.RLock] = threading.RLock()

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Delegate `param_groups` to the inner optimizer."""
        return self._inner.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        """Set `param_groups` on the inner optimizer."""
        self._inner.param_groups = value

    @property
    def defaults(self) -> dict[str, Any]:
        """Delegate `defaults` to the inner optimizer."""
        return self._inner.defaults

    @defaults.setter
    def defaults(self, value: dict[str, Any]) -> None:
        """Set `defaults` on the inner optimizer."""
        self._inner.defaults = value

    @property
    def state(self) -> defaultdict[torch.Tensor, Any]:
        """Delegate `state` to the inner optimizer."""
        return self._inner.state

    @state.setter
    def state(self, value: defaultdict[torch.Tensor, Any]) -> None:
        """Set `state` on the inner optimizer."""
        self._inner.state = value

    def _all_params(self) -> list[torch.nn.Parameter]:
        """Flatten this optimizer's params from its (now-materialized) groups."""
        return [p for group in self.param_groups for p in group["params"]]

    @contextlib.contextmanager
    def _without_property_methods(self) -> Iterator[None]:
        """Temporarily remove the delegating property descriptors.

        The properties are data descriptors on this class (inherited by subclasses), so
        `delattr` must target the defining class (found via the MRO), not `type(self)`
        (which is the subclass and does not own the descriptors).
        """
        names = ["param_groups", "defaults", "state"]
        saved: dict[str, tuple[type, property]] = {}
        with self._delegation_lock:
            for name in names:
                for cls_ in type(self).__mro__:
                    descriptor = cls_.__dict__.get(name)
                    if descriptor is not None and isinstance(descriptor, property):
                        saved[name] = (cls_, descriptor)
                        try:
                            delattr(cls_, name)
                        except AttributeError:
                            pass
                        break
            try:
                yield
            finally:
                for name in names:
                    self.__dict__.pop(name, None)
                for name, (cls_, descriptor) in saved.items():
                    setattr(cls_, name, descriptor)
