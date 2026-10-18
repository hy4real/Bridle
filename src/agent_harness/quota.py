"""配额：预留 + 结算。

为什么不能「检查 + 扣减」（面经 A6-④ 的原话）：
余额 100，两个各消耗 80 的请求**同时**通过检查，最后消耗 160。
唯一正确的做法是先原子预留，再按实际用量结算：

    可用额度 = 已授权额度 − 已结算消耗 − 在途预留

这条不变量由一个锁保护；并发测试会证明它真的成立。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Reservation:
    id: str
    amount: float
    settled: bool = False


class QuotaLedger:
    def __init__(self, limit: float) -> None:
        self.limit = limit
        self._settled = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()
        self._reservations: dict[str, Reservation] = {}

    @property
    def available(self) -> float:
        with self._lock:
            return max(0.0, self.limit - self._settled - self._reserved)

    @property
    def settled(self) -> float:
        return self._settled

    @property
    def reserved(self) -> float:
        return self._reserved

    def reserve(self, amount: float, *, key: str) -> Reservation | None:
        with self._lock:
            if self.limit - self._settled - self._reserved < amount:
                return None
            self._reserved += amount
            reservation = Reservation(key, amount)
            self._reservations[key] = reservation
            return reservation

    def settle(self, reservation: Reservation, actual: float) -> None:
        with self._lock:
            if reservation.settled:
                raise RuntimeError("重复结算")
            self._reserved -= reservation.amount
            self._settled += min(actual, reservation.amount)  # 结算不超过预留
            self._reservations[reservation.id] = Reservation(
                reservation.id, reservation.amount, settled=True)

    def release(self, reservation: Reservation) -> None:
        with self._lock:
            self._reserved -= reservation.amount
            self._reservations.pop(reservation.id, None)
