import os
import warnings
from datetime import timedelta
from typing import Any, List, Optional

from torch import distributed as dist
from torch.distributed.constants import default_pg_timeout

__all__ = [
    "init",
    "is_initialized",
    "size",
    "rank",
    "local_size",
    "local_rank",
    "is_main",
    "barrier",
    "gather",
    "all_gather",
]


def init(timeout: timedelta = default_pg_timeout) -> None:
    if "RANK" not in os.environ:
        warnings.warn("Environment variable 'RANK' is not set. Skipping distributed initialization.")
        return
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timeout,
    )


def is_initialized() -> bool:
    return dist.is_initialized()


def size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    dist.barrier()


def gather(obj: Any, dst: int = 0) -> Optional[List[Any]]:
    if is_main():
        objs = [None for _ in range(size())]
        dist.gather_object(obj, objs, dst=dst)
        return objs
    else:
        dist.gather_object(obj, dst=dst)
        return None


def all_gather(obj: Any) -> List[Any]:
    objs = [None for _ in range(size())]
    dist.all_gather_object(objs, obj)
    return objs
