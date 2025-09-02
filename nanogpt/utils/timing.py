"""timing utils"""
import time
from loguru import logger
import functools
from typing import Callable, Any, TypeVar, cast

T = TypeVar("T")

class TimeIt:
    """Timing utility usable as both decorator and context manager.

    Args:
        name (str): Optional name for logging.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.start_time: float = 0.0

    def __enter__(self) -> "TimeIt":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        if exc_type is None:
            logger.info(f"'{self.name}' executed in {elapsed:.4f} seconds.")
        else:
            logger.error(f"'{self.name}' failed after {elapsed:.4f} seconds with error: {exc_val}")

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with self.__class__(self.name or func.__name__):
                return func(*args, **kwargs)
        return cast(Callable[..., T], wrapper)
