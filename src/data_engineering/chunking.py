import tensorflow as tf
from tqdm import tqdm

from src.data_engineering.encoder import encode_row

def split_dataframe(df,chunk_size=1000):
    chunks = list()
    num_chunks=len(df)//chunk_size +1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def process_chunk(chunk,tfrecord_name,label_dict):
    options = tf.io.TFRecordOptions(compression_type ='GZIP',compression_level=9)
    with tf.io.TFRecordWriter(tfrecord_name,options=options) as file_writer:
        for _,row in tqdm(chunk.iterrows()):
            record_bytes = encode_row(row,label_dict)
            file_writer.write(record_bytes)
            del record_bytes
        file_writer.close()
