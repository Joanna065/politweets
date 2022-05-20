import abc
from abc import ABCMeta

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.data import LABEL_COLUMN, TEXT_COLUMN


class BaseTweetsDataset(Dataset, metaclass=ABCMeta):
    def __init__(
            self,
            data: pd.DataFrame,
            max_token_len: int = 512,
    ):
        self._data = data
        self._max_token_len = max_token_len

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        data_row = self._data.iloc[index]
        text = data_row[TEXT_COLUMN]
        label = int(data_row[LABEL_COLUMN])

        return {
            **self._process_sample(text),
            'label': label,
        }

    @abc.abstractmethod
    def _process_sample(self, text: str) -> dict[str, Tensor]:
        pass


class BertTweetsDataset(BaseTweetsDataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            max_token_len: int = 512,
    ):
        super().__init__(
            data=data,
            max_token_len=max_token_len,
        )
        self._tokenizer = tokenizer

    def _process_sample(self, text: str) -> dict[str, Tensor]:
        tokenized_text = self._tokenizer(
            text=text,
            truncation=True,
            add_special_tokens=True,
            max_length=self._max_token_len,
            return_token_type_ids=False,
            padding=False,  # no padding, doing later in dataloader step
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': tokenized_text['input_ids'].squeeze(0),
            'attention_mask': tokenized_text['attention_mask'].squeeze(0),
        }
