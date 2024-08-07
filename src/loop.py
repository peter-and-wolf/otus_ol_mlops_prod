from typing import Any, Callable

from tqdm import tqdm as progress

import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader

from tracker import ExperimentTracker


class Loop():
  def __init__(
      self,
      name: str, 
      dataloader: DataLoader[Any], 
      model: nn.Module,
      loss_fn: Callable,
      metric_fn: Callable) -> None:
    self.name = name
    self.step = 0
    self.dataloader = dataloader
    self.model = model
    self.loss_fn = loss_fn
    self.metric_fn = metric_fn


  def run(self, tracker: ExperimentTracker) -> None:
    self._setup()
    for X, y in progress(self.dataloader, total=len(self.dataloader), desc=self.name):
      loss, accuracy = self._step(X, y)
      
      tracker.log_metric(f'{self.name.lower()}_loss', loss, self.step)
      tracker.log_metric(f'{self.name.lower()}_accuracy', accuracy, self.step)

      
  def _setup(self) -> None:
    pass

  def _post_step(self, loss: Any) -> None:
    pass

  def _step(self, X: Any, y: Any) -> tuple[int, int]:
    pred = self.model(X)
    loss = self.loss_fn(pred, y)
    accuracy = self.metric_fn(pred, y)
    self.step += 1

    self._post_step(loss)

    return loss.item(), accuracy


class TrainingLoop(Loop):
  def __init__(
      self,
      dataloader: DataLoader[Any], 
      model: nn.Module,
      loss_fn: Callable,
      metric_fn: Callable,
      optimizer: optim.Optimizer,
    ) -> None:
    super().__init__('Train', dataloader, model, loss_fn, metric_fn)
    self.optimizer = optimizer

  def _setup(self) -> None:
    self.model.train()

  def _post_step(self, loss: Any) -> None:
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()


class ValidationLoop(Loop):
  def __init__(
      self,
      dataloader: DataLoader[Any], 
      model: nn.Module,
      loss_fn: Callable,
      metric_fn: Callable,
    ) -> None:
    super().__init__('Val', dataloader, model, loss_fn, metric_fn)

  @torch.no_grad()
  def run(self, tracker: ExperimentTracker) -> None:
    return super().run(tracker)
  
  def _setup(self) -> None:
    self.model.eval()


  

