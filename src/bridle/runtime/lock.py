"""run 锁：同一个 run 同一时刻只许一个进程动它。

要挡的是这两种情况：
- 同一个 run 被两个进程同时恢复；
- 第一个进程其实还活着，有人就把它留下的 `unknown` 解除成「没执行」，于是又执行了一遍。

用 `flock`：进程被杀（包括 SIGKILL）时由内核释放，不会留下需要人工清理的死锁文件。
锁文件本身不删，删了反而会让两个进程各锁各的 inode。
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..effects import RunLocked, validate_run_id


@contextmanager
def run_lock(state_dir: Path, run_id: str) -> Iterator[None]:
    """拿不到就立刻抛 `RunLocked`，不排队：排队等一个可能跑很久的 run 没有意义。"""
    validate_run_id(run_id)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(state_dir / f"run-{run_id}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunLocked(run_id) from error
        yield
    finally:
        os.close(descriptor)  # 关闭即释放
