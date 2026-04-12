from __future__ import annotations

import logging
from typing import Any


class _LoggerShim:
    def __init__(self) -> None:
        self._logger = logging.getLogger("loguru-shim")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(*args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> None:
        self._logger.info(*args, **kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(*args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        self._logger.error(*args, **kwargs)

    def exception(self, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(*args, **kwargs)

    def add(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def remove(self, *args: Any, **kwargs: Any) -> None:
        return None

    def bind(self, **kwargs: Any) -> "_LoggerShim":
        return self


logger = _LoggerShim()
