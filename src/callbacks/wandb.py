from typing import Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torchmetrics.functional import confusion_matrix

from src import PARAMS_FILE_PATH
from src.training import LABEL_MAP

"""
Inspired by template: https://github.com/ashleve/lightning-hydra-template
"""


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason."
    )


class WatchModel(Callback):
    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=True)


class LogParamsFile(Callback):
    """
    Log params.yaml file as an artifact to save configuration values from experiment run.
    """

    def __init__(self):
        self.param_filepath = PARAMS_FILE_PATH

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        artifact = wandb.Artifact(
            name='params_file',
            type='config',
            metadata={'wandb_run_name': experiment.name, 'wandb_run_id': experiment.id},
        )
        artifact.add_file(local_path=self.param_filepath)

        run = logger.experiment
        run.log_artifact(artifact)


class LogConfusionMatrix(Callback):
    """
    Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return logits and labels.
    """

    def __init__(self, validation_only: bool = True):
        self.val_logits = []
        self.val_labels = []
        self.validation_only = validation_only
        self.train_logits = []
        self.train_labels = []
        self.ready = True
        self.uniq_labels = LABEL_MAP.keys()

    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.ready = True

    def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: dict[str, Tensor],
            batch: Any,
            batch_idx: int,
            unused: Optional[int] = 0,
    ) -> None:
        if self.validation_only:
            return
        self.train_logits.append(outputs['logits'])
        self.train_labels.append(outputs['labels'])

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: dict[str, Tensor],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        if self.ready:
            self.val_logits.append(outputs['logits'])
            self.val_labels.append(outputs['labels'])

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.validation_only:
            return

        logits = torch.cat(self.train_logits).float()
        labels = torch.cat(self.train_labels).int()

        self._log_confusion_matrix(trainer, logits, labels, step_type='train')

        self.train_logits.clear()
        self.train_labels.clear()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.ready:
            logits = torch.cat(self.val_logits).float()
            labels = torch.cat(self.val_labels).int()

            self._log_confusion_matrix(trainer, logits, labels, step_type='val')

            self.val_logits.clear()
            self.val_labels.clear()

    def _log_confusion_matrix(
            self,
            trainer: Trainer,
            logits: Tensor,
            labels: Tensor,
            step_type: str,
    ) -> None:
        logger = get_wandb_logger(trainer)
        experiment = logger.experiment

        conf_matrix = confusion_matrix(
            preds=logits,
            target=labels,
            num_classes=len(self.uniq_labels),
            normalize='true',
        )

        plt.figure(figsize=(10, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(
            conf_matrix.cpu().numpy(),
            xticklabels=self.uniq_labels,
            yticklabels=self.uniq_labels,
            annot=True,
            annot_kws={'size': 8},
            fmt='g',
            cmap='Blues',
        )
        experiment.log(
            {f"{step_type}/confusion_matrix/{trainer.current_epoch}": wandb.Image(plt)},
            commit=False,
        )

        plt.clf()
