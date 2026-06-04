import json
import os
from src.config import CFG

def create_kaggle_dataset(n_splits):
    # naming dataset    
    Dataset = f"ISLR-{n_splits}fold-JBSB"
    #remove if dataset already exist
    shutil.rmtree(f"/tmp/{Dataset}",ignore_errors=True)

    os.makedirs(f"/tmp/{Dataset}",exist_ok = True)

    with open ('/kaggle/input/datasets/jbsbhanc/kaggleapi/kaggle.json') as f:
        kaggle_creds = json.load(f)

    os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
    os.environ['KAGGLE_KEY'] = kaggle_creds['key']

    dataset_path = f"/tmp/{Dataset}"
    #initialize kaggle dataset (metadata json automatically created)

    subprocess.run(['kaggle',
                    'datasets',
                    'init',
                    '-p',
                    dataset_path],

                    check=True
                )

    with open(f"/tmp/{Dataset}/dataset-metadata.json") as f:
        dataset_meta = json.load(f)
        dataset_meta['id'] = f"jbsbhanc/{Dataset}"
        dataset_meta['title'] = Dataset

    with open (f"/tmp/{Dataset}/dataset-metadata.json",'w') as outfile:
        json.dump(dataset_meta,outfile)

    shutil.copy(
        f"/tmp/{Dataset}/dataset-metadata.json",
        f"/tmp/{Dataset}/meta.json"
    )

    return dataset_path,Dataset
    
