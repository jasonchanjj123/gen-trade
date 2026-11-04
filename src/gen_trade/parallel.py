"""Portable process workers, with one numerical-library thread per worker."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os

_thread_limiter = None


def initialize_worker(initializer=None, initargs=()) -> None:
    global _thread_limiter
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "RAYON_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    _thread_limiter = threadpool_limits(limits=1)
    if initializer is not None:
        initializer(*initargs)


def process_pool(workers: int, *, initializer=None, initargs=()) -> ProcessPoolExecutor:
    # Spawn avoids inheriting a live BLAS/Rayon runtime and works on macOS/Linux.
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                               initializer=initialize_worker, initargs=(initializer, initargs))
