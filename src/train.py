from pathlib import Path
from typing import Annotated

import typer # type: ignore

import torch
from torch import nn
import torchmetrics

from loop import Loop, TrainingLoop, ValidationLoop
from model import MNISTClassifier
from dataset import get_dataloaders

from tracker import ExperimentTracker
from stdout_tracker import StdoutTracker
from mlflow_tracker import MLFlowTracker, start_tracker


def train(epochs: int,
          train_loop: Loop, 
          test_loop: Loop,
          tracker: ExperimentTracker):
  for _ in range(epochs):
    train_loop.run(tracker)
    test_loop.run(tracker)


def main(
    train_path: Annotated[Path, typer.Option()] = Path('data/train_data.npy'),
    model_path: Annotated[Path, typer.Option()] = Path('data/model.pt'),
    epochs: Annotated[int, typer.Option()] = 3, 
    batch_size: Annotated[int, typer.Option()] = 32, 
    lr: Annotated[float, typer.Option()] = .01
    ):
  
  loss_fn = nn.CrossEntropyLoss()
  metric_fn = torchmetrics.Accuracy(task='multiclass', num_classes=10)
  model = MNISTClassifier()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  train_dataloader, test_dataloader = get_dataloaders(train_path, batch_size=batch_size)
  # tracker = StdoutTracker()

  train_loop = TrainingLoop(
    train_dataloader,
    model=model,
    loss_fn=loss_fn,
    metric_fn=metric_fn,
    optimizer=optimizer
  )

  test_loop = ValidationLoop(
    test_dataloader,
    model=model,
    loss_fn=loss_fn,
    metric_fn=metric_fn,
  )

  with start_tracker(MLFlowTracker()) as tracker:

    tracker.log_params({
      'loss_function': 'CrossEntropyLoss',
      'metric_function': 'Accuracy',
      'optimizer_class': 'SGD',
      'batch_size': batch_size,
      'epochs': epochs,
      'lr': lr,
    })

    train(epochs, train_loop, test_loop, tracker)
  
  torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
  typer.run(main)
