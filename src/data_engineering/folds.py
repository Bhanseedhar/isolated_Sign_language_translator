from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from chunking import split_dataframe,process_chunk
from multiprocessing import cpu_count
from datetime import datetime

def label_folds(df, n_splits=5, seed=42):

    df = df.copy()

    df['fold'] = -1

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for fold_idx, (_, valid_idx) in enumerate(kfold.split(df)):
        df.loc[valid_idx, 'fold'] = fold_idx

    return df

def convert_to_tfrecords(df,n_splits,chunk_size,n_part,dataset_name, part,label_dict ):
    """takes rows of aparticular fold and converts them to tfrecord"""
    for fold in range(n_splits):   
        
        rows = df [df['fold']== fold] 
        chunks = split_dataframe(rows,chunk_size)
        part_size = len(chunks)//n_part
        last = (part+1)*part_size if part != n_part -1 else len(chunks) # +1 ?
        chunks = chunks[part*part_size:last]

        N= [len(x) for x in chunks]
        Parallel(n_jobs = cpu_count())(delayed(process_chunk)(x, f'/tmp/{dataset_name}/fold{fold}-{i}-{n}.tfrecords',label_dict) for i ,(x,n) in enumerate(zip(chunks,N)))


def version_dataset(dataset_name):

    version_name = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    print(version_name)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "version",
            "-m",
            version_name,
            "-p",
            f"/tmp/{dataset_name}",
            "-r",
            "zip"
        ],
        check=True
    )