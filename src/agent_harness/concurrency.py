"""并发上限与排队。

面经 A9 的结论：并发天花板通常在上游配额，不在机器。
所以本地要有**有界并发 + 排队可见**：在途数受限、排队时长进指标，
而不是"请求都塞进去然后一起变慢"。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConcurrencyStats:
    limit: int
    max_in_flight: int = 0
    wait_times: list[float] = field(default_factory=list)
    completed: int = 0

    @property
    def total_wait(self) -> float:
        return sum(self.wait_times)

    @property
    def queue_waits(self) -> int:
        return len(self.wait_times)

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "max_in_flight": self.max_in_flight,
            "completed": self.completed,
            "queue_waits": len(self.wait_times),
            "total_wait_s": round(self.total_wait, 3),
            "max_wait_s": round(max(self.wait_times) if self.wait_times else 0.0, 3),
        }


class BoundedRunner:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit 至少为 1")
        self.semaphore = threading.BoundedSemaphore(limit)
        self.stats = ConcurrencyStats(limit=limit)
        self._lock = threading.Lock()
        self._in_flight = 0

    def map(self, jobs: Iterable[Callable[[], Any]]) -> list[Any]:
        results: list[Any] = []
        lock = threading.Lock()

        def worker(job: Callable[[], Any]) -> None:
            started_waiting = time.monotonic()
            with self.semaphore:
                waited = time.monotonic() - started_waiting
                with self._lock:
                    self._in_flight += 1
                    self.stats.max_in_flight = max(self.stats.max_in_flight, self._in_flight)
                    self.stats.wait_times.append(waited)
                try:
                    value = job()
                finally:
                    with self._lock:
                        self._in_flight -= 1
                        self.stats.completed += 1
                with lock:
                    results.append(value)

        threads = [threading.Thread(target=worker, args=(job,)) for job in jobs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results
