import numpy as np 
import pandas as pd 
import json 
import shutil
import os
import subprocess 

from src.data_engineering.parquet_loader import load_relavant_data
from src.data_engineering.encoder import encode_row
from src.data_engineering.chunking import process_chunk
from src.data_engineering.folds import label_folds,convert_to_tfrecords,version_dataset
from src.data_engineering.create_kaggle_dataset import create_kaggle_dataset

from src.config import CFG


# loading competition data
train_df = pd.read_csv(f'{CFG.data_root}/train.csv')


#loading competiton label encoded file

with open(f'{CFG.data_root}/sign_to_prediction_index_map.json') as json_file:
    label_dict = json.load(json_file)


# createsa a new column with respective fold labelled
df = label_folds(train_df,n_splits=CFG.n_splits,seed = CFG.seed)

dataset_path, Dataset = create_kaggle_dataset(n_splits=CFG.n_splits)

convert_to_tfrecords(df=df ,n_splits=CFG.n_splits,chunk_size=CFG.chunk_size,n_part=CFG.n_part,dataset_name=Dataset, part=CFG.part ,label_dict=label_dict)

# Check if the dataset already exists
result = subprocess.run(
    ["kaggle", "datasets", "status", f"jbsbhanc/{Dataset}"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

if result.returncode == 0:
    print("Dataset already exists. Uploading new version...")
    version_dataset(Dataset)
else:
    print("Creating dataset for the first time...")
    subprocess.run(
        ["kaggle", "datasets", "create", "-p", dataset_path, "--public"],
        check=True
    )