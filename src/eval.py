
from pathlib import Path
from typing import Annotated, Any, Callable

import torch
from torch import nn
import torchmetrics
import numpy as np

import typer

from model import MNISTClassifier
from tracker import ExperimentTracker
from stdout_tracker import StdoutTracker


def eval(
    model: MNISTClassifier, 
    X: Any, 
    y: Any, 
    loss_fn: Callable,
    metric_fn: Callable,
    tracker: ExperimentTracker) -> None:
  pred = model(X)
  loss = loss_fn(pred, y)
  accuracy = metric_fn(pred, y)
  tracker.log_metric('loss', loss)
  tracker.log_metric('accuracy', accuracy)


def main(
  test_path: Annotated[Path, typer.Option()] = Path('data/test_data.npy'),
  model_path: Annotated[Path, typer.Option()] = Path('data/model.pt')
  ) -> None:
  
  test_data = np.load(test_path)
  X = torch.tensor(test_data[:, 1:]).float().reshape(test_data.shape[0], 1, 28, 28)
  y = torch.tensor(test_data[:, 0]).long()

  model = MNISTClassifier()
  model.load_state_dict(torch.load(model_path, weights_only=True))
  model.eval()

  tracker = StdoutTracker()

  loss_fn = nn.CrossEntropyLoss()
  metric_fn = torchmetrics.Accuracy(task='multiclass', num_classes=10)

  eval(
    model=model, 
    X=X, 
    y=y, 
    loss_fn=loss_fn, 
    metric_fn=metric_fn, 
    tracker=tracker
  )


if __name__ == '__main__':
  typer.run(main)