import abc
from pathlib import Path
from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule, seed_everything
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DataCollatorWithPadding

from src import DATA_PATH
from src.data import LABEL_COLUMN
from src.training import LABEL_MAP
from src.training.datasets import BertTweetsDataset
from src.training.samplers import get_sampler


class BaseTweetsDataModule(LightningDataModule, metaclass=abc.ABCMeta):
    def __init__(
        self,
        train_dataset_filename: Path,
        test_dataset_filename: Path,
        val_dataset_filename: Path,
        sampler_name: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 5,
        max_token_len: int = 512,
        seed: int = 2022,
    ):
        super().__init__()

        self._train_dataset_path = DATA_PATH.joinpath('datasets', train_dataset_filename)
        self._test_dataset_path = DATA_PATH.joinpath('datasets', test_dataset_filename)
        self._val_dataset_path = DATA_PATH.joinpath('datasets', val_dataset_filename)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._sampler_name = sampler_name
        self._seed = seed
        self._max_token_len = max_token_len

    def setup(self, stage: Optional[str] = None) -> None:
        seed_everything(self._seed)

        if stage in (None, 'fit'):
            self._train_df = pd.read_csv(self._train_dataset_path)
            self._val_df = pd.read_csv(self._val_dataset_path)

            self._val_df[LABEL_COLUMN] = self._map_labels_int(self._val_df[LABEL_COLUMN].values)
            self._train_df[LABEL_COLUMN] = self._map_labels_int(self._train_df[LABEL_COLUMN].values)

        if stage in (None, 'test'):
            self._test_df = pd.read_csv(self._test_dataset_path)
            labels = self._map_labels_int(self._test_df[LABEL_COLUMN].values)
            self._test_df[LABEL_COLUMN] = labels

    @staticmethod
    def _map_labels_int(labels: list[str]) -> list[int]:
        return [int(LABEL_MAP[label]) for label in labels]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.create_dataloader(self._train_df, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.create_dataloader(self._val_df, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.create_dataloader(self._test_df, shuffle=False)

    def create_dataloader(self, data: pd.DataFrame, shuffle: bool):
        kwargs: dict = {}
        if shuffle and self._sampler_name:
            kwargs['sampler'] = get_sampler(
                sampler_name=self._sampler_name,
                labels=data[LABEL_COLUMN].values,
                batch_size=self._batch_size,
            )
        else:
            kwargs['shuffle'] = shuffle

        return self._get_dataloader(data, **kwargs)

    @abc.abstractmethod
    def _get_dataloader(self, data: pd.DataFrame, **kwargs) -> DataLoader:
        pass


class BertTweetsDataModule(BaseTweetsDataModule):
    def __init__(
        self,
        train_dataset_filename: Path,
        test_dataset_filename: Path,
        val_dataset_filename: Path,
        tokenizer: BertTokenizer,
        sampler_name: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 2718,
        max_token_len: int = 512,
    ):
        super().__init__(
            train_dataset_filename=train_dataset_filename,
            test_dataset_filename=test_dataset_filename,
            val_dataset_filename=val_dataset_filename,
            sampler_name=sampler_name,
            batch_size=batch_size,
            num_workers=num_workers,
            max_token_len=max_token_len,
            seed=seed,
        )
        self._tokenizer = tokenizer

    def _get_dataloader(self, data: pd.DataFrame, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=BertTweetsDataset(
                data=data,
                tokenizer=self._tokenizer,
                max_token_len=self._max_token_len,
            ),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            pin_memory=True,
            collate_fn=DataCollatorWithPadding(self._tokenizer),
            **kwargs,
        )
