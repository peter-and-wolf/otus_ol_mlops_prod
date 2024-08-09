from typing import Any, Optional
from contextlib import contextmanager

import mlflow # type: ignore


class MLFlowTracker:
  
  def start_run(self):
    mlflow.start_run()

  def stop_run(self):
    mlflow.end_run()

  def log_params(self, params: dict[str, Any]) -> None:
    mlflow.log_params(params)

  def log_metric(self, name: str, value: int | float, step: int | None = None) -> None:
    mlflow.log_metric(name, value, step=step)


@contextmanager
def start_tracker():
  tracker = MLFlowTracker()
  tracker.start_run()
  try:
    yield tracker
  finally:
    tracker.stop_run()



