from src.training_utils.train_folds import train_folds
from src.config import CFG,get_strategy
from src.data_engineering.tfdecoder import get_tfrec_dataset
import glob


strategy, REPLICAS, IS_TPU = get_strategy()

TRAIN_FILENAMES = glob.glob('/kaggle/input/islr-5fold/*.tfrecords') ##makesure the dataset u create will be named here exatly


train_folds(CFG, [0],strategy=strategy)