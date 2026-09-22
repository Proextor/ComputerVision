"""Thread-safe latest-value publication with bounded per-client delivery."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
import time
from typing import Callable

from .schemas import SceneSnapshot


@dataclass(frozen=True)
class SnapshotPublication:
    sequence: int
    published_at: datetime
    monotonic_at: float
    snapshot_json: str

    @property
    def snapshot(self) -> SceneSnapshot:
        # The broker holds immutable JSON, so callers cannot mutate published state.
        return SceneSnapshot.model_validate_json(self.snapshot_json)


@dataclass(eq=False)
class Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[SnapshotPublication] = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    pending: SnapshotPublication | None = None
    scheduled: bool = False
    active: bool = True


class SnapshotBroker:
    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._clock = clock
        self._utc_now = utc_now
        self._lock = Lock()
        self._sequence = 0
        self._latest: SnapshotPublication | None = None
        self._subscribers: set[Subscriber] = set()

    def publish(self, snapshot: SceneSnapshot) -> SnapshotPublication:
        payload = snapshot.model_dump_json(exclude_none=False)
        with self._lock:
            self._sequence += 1
            publication = SnapshotPublication(self._sequence, self._utc_now(),
                                              self._clock(), payload)
            self._latest = publication
            for subscriber in list(self._subscribers):
                subscriber.pending = publication
                if not subscriber.scheduled:
                    subscriber.scheduled = True
                    try:
                        subscriber.loop.call_soon_threadsafe(self._deliver, subscriber)
                    except RuntimeError:
                        subscriber.active = False
                        self._subscribers.discard(subscriber)
            return publication

    def _deliver(self, subscriber: Subscriber) -> None:
        with self._lock:
            if not subscriber.active:
                return
            publication = subscriber.pending
            subscriber.pending = None
            subscriber.scheduled = False
        if publication is None:
            return
        if subscriber.queue.full():
            subscriber.queue.get_nowait()
        subscriber.queue.put_nowait(publication)

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> tuple[Subscriber, SnapshotPublication | None]:
        subscriber = Subscriber(loop=loop)
        with self._lock:
            self._subscribers.add(subscriber)
            latest = self._latest
        return subscriber, latest

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            subscriber.active = False
            subscriber.pending = None
            self._subscribers.discard(subscriber)

    def latest(self) -> SnapshotPublication | None:
        with self._lock:
            return self._latest

    def age_seconds(self) -> float | None:
        with self._lock:
            latest = self._latest
        return max(0.0, self._clock() - latest.monotonic_at) if latest else None

    @property
    def publication_sequence(self) -> int:
        with self._lock:
            return self._sequence

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
