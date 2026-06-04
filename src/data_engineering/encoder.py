import tensorflow as tf
from src.data_engineering.parquet_loader import load_relavant_data

def encode_row(row,label_dict):
    coordinates = load_relavant_data(f'/kaggle/input/competitions/asl-signs/{row.path}')
    coordinates_encoded = coordinates.tobytes()
    participant_id = int(row.participant_id)
    sequence_id = int(row.sequence_id)
    sign = int(label_dict[row.sign])

    record_bytes = tf.train.Example(features = tf.train.Features(feature = {
        'coordinates' :  tf.train.Feature(bytes_list = tf.train.BytesList(value= [coordinates_encoded])),
        'participant_id' : tf.train.Feature(int64_list = tf.train.Int64List(value=[participant_id])),
        'sequence_id':tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_id])),
        'sign':tf.train.Feature(int64_list=tf.train.Int64List(value=[sign])),    
    })).SerializeToString()
    return record_bytes

