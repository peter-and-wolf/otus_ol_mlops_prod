import logging

from os import PathLike
from typing import Any


class StdoutTracker:
  def __init__(self, path: str | PathLike[Any] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, filename=path)
  
  def log_params(self, params: dict[str, Any]) -> None:
    for k, v in params.items():
      logging.info(f"hyperparam: {k}={v}")

  def log_metric(self, name: str, value: int | float, step: int | None = None) -> None:
    logging.info(f"{str(step)+': ' if step is not None else ''}{name}={value}")