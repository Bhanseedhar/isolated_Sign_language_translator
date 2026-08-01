from src.training_utils.train_folds import train_fold
from src.config import CFG,get_strategy
from src.data_engineering.tfdecoder import get_tfrec_dataset
import glob








strategy, REPLICAS, IS_TPU = get_strategy()

TRAIN_FILENAMES = glob.glob('/kaggle/input/islr-5fold/*.tfrecords') ##makesure the dataset u create will be named here exatly

def train_folds(CFG,folds,strategy, summary=True):
    for fold in folds:
        if fold != 'all':
            all_files = TRAIN_FILENAMES
            train_files = [x for x in all_files if f'fold{fold}' not in x]
            valid_files = [x for x in all_files if f'fold{fold}' in x ]
        else:
            train_files = TRAIN_FILENAMES
            valid_files = None
        model , cv, history = train_fold(CFG,fold,train_files,valid_files,strategy,summary,)
    return

train_folds(CFG, [0],strategy=strategy)