from pathlib import Path
from typing import Annotated, Callable

import pandas as pd
import numpy as np

import typer


def main(
  train_in_path: Annotated[Path, typer.Option()] = Path('data/train_data.csv'),
  test_in_path: Annotated[Path, typer.Option()] = Path('data/test_data.csv'),
  train_out_path: Annotated[Path, typer.Option()] = Path('data/test_data.npy'),
  test_out_path: Annotated[Path, typer.Option()] = Path('data/test_data.npy')
  ) -> None:

  train_df = pd.read_csv(train_in_path, header=None, dtype=float)
  test_df = pd.read_csv(test_in_path, header=None, dtype=float)

  train_mean = train_df.values[:, 1:].mean()
  train_std = train_df.values[:, 1:].std()

  train_df.values[:, 1:] -= train_mean
  train_df.values[:, 1:] /= train_std
  test_df.values[:, 1:] -= train_mean
  test_df.values[:, 1:] /= train_std
  
  np.save(train_out_path, train_df)
  np.save(test_out_path, test_df)


if __name__ == '__main__':
  typer.run(main)