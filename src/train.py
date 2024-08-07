from pathlib import Path
from typing import Annotated

import typer # type: ignore

import torch
from torch import nn
import torchmetrics

from loop import Loop
from steps import TrainingStep, ValidationLoop
from model import MNISTClassifier
from dataset import get_dataloaders

from tracker import ExperimentTracker
from mlflow_tracker import start_tracker


def train(
    epochs: int,
    train_loop: Loop, 
    test_loop: Loop
  ) -> None:
  
  for _ in range(epochs):
    train_loop.run()
    test_loop.run()


def main(
    train_path: Annotated[Path, typer.Option()] = Path('data/train_data.npy'),
    model_path: Annotated[Path, typer.Option()] = Path('data/model.pt'),
    epochs: Annotated[int, typer.Option()] = 3, 
    batch_size: Annotated[int, typer.Option()] = 32, 
    lr: Annotated[float, typer.Option()] = .01
  ) -> None:
  
  loss_fn = nn.CrossEntropyLoss()
  metric_fn = torchmetrics.Accuracy(task='multiclass', num_classes=10)
  model = MNISTClassifier()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  train_dataloader, test_dataloader = get_dataloaders(train_path, batch_size=batch_size)

  with start_tracker() as tracker:

    tracker.log_params({
      'loss_function': 'CrossEntropyLoss',
      'metric_function': 'Accuracy',
      'optimizer_class': 'SGD',
      'batch_size': batch_size,
      'epochs': epochs,
      'lr': lr,
    })

    train_step = TrainingStep(
      model=model,
      loss_fn=loss_fn,
      metric_fn=metric_fn,
      tracker=tracker,
      optimizer=optimizer
    )

    test_loop = ValidationLoop(
      model=model,
      loss_fn=loss_fn,
      metric_fn=metric_fn,
      tracker=tracker
    )

    train(
      epochs, 
      Loop(train_dataloader, train_step),
      Loop(test_dataloader, test_loop),
    )
  
  torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
  typer.run(main)
