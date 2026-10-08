# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Function-lifetime event ownership shared by phase and loop emitters.

No channel is released by lexical position. Reuse requires a separate proof
that every asynchronous participant consumed the previous notification.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EventChannel:
    source_core: str
    source_pipe: str
    destination_core: str
    destination_pipe: str
    event_id: int

    @property
    def cross(self):
        return self.source_core != self.destination_core


class EventRegistry:

    def __init__(self, *, local_ids, cross_ids):
        from .local_sync import LocalSyncError
        for values in (local_ids, cross_ids):
            if (type(values) is not tuple or len(set(values)) != len(values) or
                    any(type(value) is not int or value < 0 for value in values)):
                raise LocalSyncError("event registry requires distinct nonnegative explicit IDs")
        self.local_ids, self.cross_ids = local_ids, cross_ids
        self._local, self._cross, self._records = {}, set(), []

    @property
    def records(self):
        return tuple(self._records)

    def require_ids(self, *, local_ids=None, cross_ids=None):
        from .local_sync import LocalSyncError
        if ((local_ids is not None and local_ids != self.local_ids) or
            (cross_ids is not None and cross_ids != self.cross_ids)):
            raise LocalSyncError("emitter IDs must match its shared event registry")

    def allocate(self, source_core, source_pipe, destination_core, destination_pipe):
        from .local_sync import EventCapacityError, LocalSyncError, _CORE_PIPES
        if (source_pipe not in _CORE_PIPES.get(source_core, ()) or
                destination_pipe not in _CORE_PIPES.get(destination_core, ())):
            raise LocalSyncError("event channel is outside its core/pipe domain")
        cross = source_core != destination_core
        # CANN TPipe::AllocEventID selects eventPool_ + EventToIndex(evt).
        # HardEvent::<src>_<dst> therefore has its own local ID bitmap; a
        # destination-only pool falsely exhausts different HardEvents together.
        pool = self._cross if cross else self._local.setdefault(
            (destination_core, source_pipe, destination_pipe), set())
        candidates = self.cross_ids if cross else self.local_ids
        event = next((value for value in candidates if value not in pool), None)
        if event is None:
            raise EventCapacityError("function-level event channel capacity exhausted")
        pool.add(event)
        channel = EventChannel(source_core, source_pipe, destination_core, destination_pipe, event)
        self._records.append(channel)
        return channel
