from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent
PARAMS_FILE_PATH = PROJECT_PATH / 'params.yaml'

STORAGE_PATH = PROJECT_PATH / 'storage'
DATA_PATH = STORAGE_PATH / 'dataset'
CHECKPOINTS_PATH = STORAGE_PATH / 'checkpoints'
LOGS_PATH = STORAGE_PATH / 'logs'
RESULTS_PATH = STORAGE_PATH / 'results'
