import pandas as pd
import yaml

from src import DATA_PATH, PARAMS_FILE_PATH
from src.data import TEXT_COLUMN
from src.data.preprocessing import PreprocessTextPipeline, get_drop_indices_by_min_words

with PARAMS_FILE_PATH.open(mode='r') as f:
    config = yaml.full_load(f)

train_filename = config['dataset']['train_filename']
val_filename = config['dataset']['val_filename']
test_filename = config['dataset']['test_filename']

df_train = pd.read_csv(DATA_PATH.joinpath('datasets', f'{train_filename}'))
df_val = pd.read_csv(DATA_PATH.joinpath('datasets', f'{val_filename}'))
df_test = pd.read_csv(DATA_PATH.joinpath('datasets', f'{test_filename}'))

print(f"Loaded train data records num: {len(df_train)}")
print(f"Loaded val data records num: {len(df_val)}")
print(f"Loaded test data records num: {len(df_test)}")

MIN_WORDS = config['dataset']['minimum_words']
STAGES = config['dataset']['preprocessing']

for df, split_name in [(df_train, 'train'), (df_val, 'val'), (df_test, 'test')]:
    print(f"Processing {split_name} data...")

    # get tweets with minimum words limit constraint
    if MIN_WORDS:
        drop_indices = get_drop_indices_by_min_words(df, min_words=int(MIN_WORDS))

        print(f"Dropping rows with less than {MIN_WORDS} words in tweet main content.")
        df = df.drop(drop_indices, axis=0)
        print(f"Dataframe records update num: {len(df)}")

    # process textual data
    if STAGES:
        process_pipe = PreprocessTextPipeline(stages=STAGES)
        texts = df[TEXT_COLUMN]
        df[TEXT_COLUMN] = process_pipe(texts)

    # save updated dataset file
    df.to_csv(DATA_PATH.joinpath('datasets', f'tweets_{split_name}.csv'), index=False)
