import datetime
import json
import logging

import wandb
import yaml
from src import CHECKPOINTS_PATH, DATA_PATH, LOGS_PATH, PARAMS_FILE_PATH, RESULTS_PATH
from src.callbacks.wandb import LogConfusionMatrix, LogParamsFile, WatchModel
from src.training import LABEL_MAP
from src.training.bert_model import BertTweetsClassifier
from src.training.datamodules import BertTweetsDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DATASET_PATH = DATA_PATH.joinpath('datasets')

with PARAMS_FILE_PATH.open(mode='r') as f:
    config = yaml.full_load(f)

if config.get("seed"):
    seed_everything(config['seed'], workers=True)

model_name = config['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name)
datamodule = BertTweetsDataModule(tokenizer=tokenizer, **config['datamodule'])

model = BertTweetsClassifier(
    num_labels=len(LABEL_MAP.keys()),
    **config['model'],
)

logger.info("Initialize wandb logger...")
config_wandb = config["wandb_logger"]
EXP_NAME = f'{config_wandb["name"]}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
wandb_logger = WandbLogger(
    entity=config_wandb['entity'],
    project=config_wandb['project'],
    name=EXP_NAME,
    save_dir=LOGS_PATH,
)
checkpoint_dir = CHECKPOINTS_PATH.joinpath(EXP_NAME)
checkpoint_dir.mkdir(exist_ok=True, parents=True)
callbacks = [
    WatchModel(),
    LogParamsFile(),
    LogConfusionMatrix(validation_only=True),
    ModelCheckpoint(
        dirpath=checkpoint_dir,
        **config['callbacks']['checkpoint'],
    ),
    LearningRateMonitor(logging_interval='step'),
    EarlyStopping(**config['callbacks']['early_stopping']),
]
trainer = Trainer(
    **config['trainer'],
    logger=wandb_logger,
    callbacks=callbacks,
)

logger.info("Starting training...")
wandb.require(experiment="service")  # opt-in for lightning and DDP / multiprocessing
trainer.fit(model=model, datamodule=datamodule)

logger.info("Evaluating test split for best checkpoint model...")
trainer.test(datamodule=datamodule)

metrics, *_ = trainer.test(datamodule=datamodule)

result_dir = RESULTS_PATH.joinpath(EXP_NAME)
result_dir.mkdir(exist_ok=True, parents=True)
with result_dir.joinpath('test_metrics.json').open('w') as file:
    json.dump(metrics, file, indent=2)
