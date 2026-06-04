import pandas as pd 
import numpy as np 
from src.config import rows_per_frame

def load_relavant_data(pq_path):
    """ from the landmarks of 543 we only load relavant data like the coordinates of landmarks """
    data_columns=['x','y','z']
    data = pd.read_parquet(pq_path,columns=data_columns)
    n_frames = int(len(data)/rows_per_frame)

    data = data.values.reshape(n_frames,rows_per_frame,len(data_columns))
    return data.astype(np.float32)


