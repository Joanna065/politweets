import abc
from time import time
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchmetrics import Accuracy, F1, MetricCollection, Precision, Recall
from transformers import (AdamW, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

from src.training import LABEL_MAP


class BertTweetsClassifier(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(
            self,
            model_name: str,
            learning_rate: float = 1e-5,
            adam_epsilon: float = 1e-8,
            weight_decay: float = 0.01,
            warmup_steps: int = 0,
            num_labels: int = 5,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        metrics = MetricCollection({
            'accuracy': Accuracy(average='micro'),
            'precision': Precision(average='micro'),
            'recall': Recall(average='micro'),
            'f1_score': F1(average='micro'),
            'f1_score_per_class': F1(average='none', num_classes=num_labels),
        })
        self._train_metrics = metrics.clone(prefix='train/')
        self._val_metrics = metrics.clone(prefix='val/')
        self._test_metrics = metrics.clone(prefix='test/')

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if
                    not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        # calculate total learning steps
        # call len(self.train_dataloader()) should be fixed in pytorch-lightning v1.6
        self._total_train_steps = (
                self.trainer.max_epochs
                * len(self.trainer._data_connector._train_dataloader_source.dataloader())
                * self.trainer.accumulate_grad_batches
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self._total_train_steps,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def forward(self, input_ids: Tensor, labels: Tensor, attention_mask: Optional[Tensor] = None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._common_step(batch, step_type='train')

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        return self._common_step(batch, step_type='val')

    def test_step(self, batch: dict[str, Tensor], batch_idx: int):
        return self._common_step(batch, step_type='test')

    def _common_step(self, batch: dict[str, Tensor], step_type: str) -> dict[str, Tensor]:
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs['loss']
        if step_type != 'test':
            self.log(name=f'{step_type}/loss', value=loss, on_epoch=False, on_step=True)

        output = {
            'loss': loss if step_type == 'train' else loss.detach(),
            'logits': outputs['logits'].detach(),
            'labels': batch['labels'].detach(),
        }
        return output

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time()

    def training_epoch_end(self, outputs) -> None:
        epoch_time = time() - self._epoch_start_time
        self.log('train/epoch_time', epoch_time, on_epoch=True, on_step=False)

        logits = torch.cat([out['logits'] for out in outputs]).float()
        labels = torch.cat([out['labels'] for out in outputs]).int()

        metrics = self._train_metrics(logits, labels)
        self._epoch_log_metrics(metrics, step_type=self._train_metrics.prefix)

    def validation_epoch_end(self, outputs) -> None:
        logits = torch.cat([out['logits'] for out in outputs]).float()
        labels = torch.cat([out['labels'] for out in outputs]).int()

        metrics = self._val_metrics(logits, labels)
        self._epoch_log_metrics(metrics, step_type=self._val_metrics.prefix)

    def test_epoch_end(self, outputs) -> None:
        logits = torch.cat([out['logits'] for out in outputs]).float()
        labels = torch.cat([out['labels'] for out in outputs]).int()

        metrics = self._test_metrics(logits, labels)
        self._epoch_log_metrics(metrics, step_type=self._test_metrics.prefix)

    def _epoch_log_metrics(self, metric_dict: dict[str, Tensor], step_type: str) -> None:
        f1_class_key = f'{step_type}f1_score_per_class'
        labels_str = [f'{f1_class_key}/{label_name}' for label_name in LABEL_MAP.keys()]
        f1_class = metric_dict.pop(f1_class_key)
        metrics_per_class = dict(zip(labels_str, f1_class))

        metrics = metric_dict | metrics_per_class
        self.log_dict(metrics, on_epoch=True, on_step=False)
