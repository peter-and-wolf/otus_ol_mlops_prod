from typing import Any, Callable

import torch
from torch import nn
from torch import optim

from tracker import ExperimentTracker


class LoopStep:
  def __init__(
    self,
    model: nn.Module,
    loss_fn: Callable,
    metric_fn: Callable,
    tracker: ExperimentTracker,
  ):
    self.model = model
    self.loss_fn = loss_fn
    self.metric_fn = metric_fn
    self.tracker = tracker


class TrainingStep(LoopStep):
  def __init__(
    self,
    model: nn.Module,
    loss_fn: Callable,
    metric_fn: Callable,
    tracker: ExperimentTracker,
    optimizer: optim.Optimizer
  ):
    super().__init__(
      model,
      loss_fn,
      metric_fn,
      tracker,
    )
    self.optimizer = optimizer

  def open(self):
    self.model.train()

  def close(self):
    pass

  def make(self, X: Any, y: Any, num: int):
    pred = self.model(X)
    loss = self.loss_fn(pred, y)
    accuracy = self.metric_fn(pred, y)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    self.tracker.log_metric(f'training_loss', loss, num)
    self.tracker.log_metric(f'training_accuracy', accuracy, num)


class ValidationStep(LoopStep):
  def open(self):
    torch.set_grad_enabled(False)

  def close(self):
    torch.set_grad_enabled(True)

  def make(self, X: Any, y: Any, num: int):
    pred = self.model(X)
    loss = self.loss_fn(pred, y)
    accuracy = self.metric_fn(pred, y)
    self.tracker.log_metric(f'validation_loss', loss, num)
    self.tracker.log_metric(f'validation_accuracy', accuracy, num)