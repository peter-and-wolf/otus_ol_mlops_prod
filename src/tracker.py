
from typing import Any, Protocol


class ExperimentTracker(Protocol):
  def log_params(self, params: dict[str, Any]) -> None:
    """ Logs a bunch of hyperparams """

  def log_metric(self, name: str, value: int | float, step: int | None = None) -> None:
    """ Logs a single metric with name and value """


