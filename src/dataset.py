from os import PathLike
from typing import Any

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class MNISTDatasetCsv(Dataset):
  def __init__(self, path: str | PathLike[Any]):
    data = np.load(path)
    self.data = torch.tensor(data[:, 1:]).float().reshape(data.shape[0], 1, 28, 28)
    self.labels = torch.tensor(data[:, 0]).long()

  def __len__(self) -> int:
    return self.labels.shape[0]

  def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
    return self.data[index], self.labels[index]
  

def get_dataloaders(path: str | PathLike[Any], batch_size: int, ratio: float =.8) -> tuple[DataLoader[Any], DataLoader[Any]]:
  assert(ratio > 0.1 and ratio < 0.9)
  
  dataset = MNISTDatasetCsv(path)

  train_size = int(ratio * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

  return train_dataloader, val_dataloader

