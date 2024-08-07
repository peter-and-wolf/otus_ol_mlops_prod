from typing import Any, Protocol
from contextlib import contextmanager

from tqdm import tqdm as progress
from torch.utils.data.dataloader import DataLoader


class Step(Protocol):
  def open(self) -> None:
    ...

  def close(self) -> None:
    ...

  def make(self, X: Any, y: Any, num: int) -> tuple[Any, Any]:
    ...
  

class Loop():
  def __init__(
      self,
      dataloader: DataLoader[Any],
      step: Step) -> None:
    self.dataloader = dataloader
    self.step = step
    self.step_num = 1

  def run(self) -> None:
    with self._loop_init(): 
      for X, y in progress(self.dataloader, total=len(self.dataloader)):
        self.step.make(X, y, self.step_num)
        self.step_num += 1
      
  @contextmanager
  def _loop_init(self):
    self.step.open()
    try:
      yield
    finally:
      self.step.close()



  

