"""工具调用重试中间件。

接口/工具异常时指数退避自动重试，支持配置最大重试次数、基础间隔、
退避倍数、最大间隔与随机抖动（避免重试风暴）。
既可作装饰器使用，也可包裹任意可调用对象。
"""
from __future__ import annotations

import functools
import random
import time
from typing import Callable, Tuple, Type, TypeVar

from utils.common import get_logger

logger = get_logger("middleware.retry")

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    jitter: float = 0.3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """指数退避重试装饰器。

    第 n 次重试延迟 = min(base_delay * backoff_factor**n, max_delay) * (1 ± jitter)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(
                            "调用 %s 失败，已达最大重试次数 %d：%s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
                        raise
                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    # 对称抖动
                    delay *= 1 + random.uniform(-jitter, jitter)
                    delay = max(delay, 0.0)
                    logger.warning(
                        "调用 %s 第 %d 次失败：%s，%.2fs 后重试",
                        func.__name__,
                        attempt,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator
