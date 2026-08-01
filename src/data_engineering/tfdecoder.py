import tensorflow as tf
import numpy as np
from tensorflow import Tensor
from src.preprocess.preprocess_utils import Preprocess,len_interpolation

from src.config import (
    POINT_LANDMARKS,
    MAX_LEN,
    ROWS_PER_FRAME,
    LHAND,
    RHAND,
    LLIP,
    RLIP,
    LPOSE,
    RPOSE,
    LEYE,
    REYE,
    LNOSE,
    RNOSE,
    PAD,
    CHANNELS,
    NUM_CLASSES,
)



def decode_tfrec(record_bytes):
    features = tf.io.parse_single_example(record_bytes,{
        'coordinates' : tf.io.FixedLenFeature([],tf.string),
        'sign' : tf.io.FixedLenFeature([],tf.int64),
    })
    out ={}
    out['coordinates'] = tf.reshape(
        tf.io.decode_raw(features['coordinates'],tf.float32),
        (-1,ROWS_PER_FRAME,3)
        )
    
    out['sign']= features['sign']
    return out

def filter_nans_tf(x, ref_point=POINT_LANDMARKS):
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(tf.gather(x,ref_point,axis=1)), axis=[-2,-1]))
    x = tf.boolean_mask(x, mask, axis=0)
    return x

def flip_lr(input_tensor):
    x,y,z = tf.unstack(input_tensor,axis = -1)
    x = 1-x
    new_input_tensor = tf.stack([x,y,z],-1)
    new_input_tensor = tf.transpose(new_input_tensor,[1,0,2])
    lhand = tf.gather(new_input_tensor,LHAND,axis =0)
    rhand = tf.gather(new_input_tensor,RHAND,axis =0)
    new_input_tensor = tf.tensor_scatter_nd_update(
        new_input_tensor,
        tf.constant(LHAND)[...,None],
        rhand
    )
    new_input_tensor = tf.tensor_scatter_nd_update(
        new_input_tensor,
        tf.constant(RHAND)[...,None],
        lhand
    )
    llip = tf.gather(new_input_tensor, LLIP, axis=0)
    rlip = tf.gather(new_input_tensor, RLIP, axis=0)
    new_input_tensor = tf.tensor_scatter_nd_update(new_input_tensor, tf.constant(LLIP)[...,None], rlip)
    new_input_tensor = tf.tensor_scatter_nd_update(new_input_tensor, tf.constant(RLIP)[...,None], llip)
    lpose = tf.gather(new_input_tensor, LPOSE, axis=0)
    rpose = tf.gather(new_input_tensor, RPOSE, axis=0)
    new_input_tensor = tf.tensor_scatter_nd_update(new_input_tensor, tf.constant(LPOSE)[...,None], rpose)
    new_input_tensor = tf.tensor_scatter_nd_update(new_input_tensor, tf.constant(RPOSE)[...,None], lpose)
    leye = tf.gather(new_input_tensor, LEYE, axis=0)
    reye = tf.gather(new_input_tensor, REYE, axis=0)
    new_input_tensor = tf.tensor_scatter_nd_update(new_input_tensor, tf.constant(LEYE)[...,None], reye)
    new_input_tensor = tf.tensor_scatter_nd_update(new_input_tensor, tf.constant(REYE)[...,None], leye)
    lnose = tf.gather(new_input_tensor, LNOSE, axis=0)
    rnose = tf.gather(new_input_tensor, RNOSE, axis=0)
    new_input_tensor = tf.tensor_scatter_nd_update(new_input_tensor, tf.constant(LNOSE)[...,None], rnose)
    new_input_tensor = tf.tensor_scatter_nd_update(new_input_tensor, tf.constant(RNOSE)[...,None], lnose)
    new_input_tensor = tf.transpose(new_input_tensor, [1,0,2])
    return new_input_tensor



def resample(x,rate = (0.8,1.2)):
    rate =tf.random.uniform((),rate[0],rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate * tf.cast(length,tf.float32),tf.int32)
    new_x = len_interpolation(x,new_size)
    return new_x



def spatial_random_affine(tensor,
    scale = (0.8,1.2),
    shear = (-0.15,0.15),
    shift = (-0.1,0.1),
    degree= (-30,30),
    ):

    center = tf.constant([0.5,0.5])
    if scale is not None:
        scale = tf.random.uniform((),*scale)
        tensor = scale * tensor
    if shear is not None:
        tensor_xy = tensor[...,:2]
        tensor_z = tensor[...,2:]
        shear_x = shear_y = tf.random.uniform((),*shear)
        if tf.random.uniform(()) < 0.5 :
            shear_x = 0.
        else:
            shear_y =0.
        shear_matrix = tf.identity([
            [1.,shear_x],
            [shear_y,1.]
        ])
        tensor_xy = tensor_xy @ shear_matrix
        center = center + [shear_y,shear_x]
        tensor = tf.concat([tensor_xy,tensor_z], axis = -1)
    if degree is not None:
        tensor_xy = tensor[...,:2]
        tensor_z = tensor[...,2:]
        tensor_xy -= center
        degree = tf.random.uniform((),*degree)
        radian = degree/180*np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity([
            [c,s],
            [-s, c],
        ])
        tensor_xy = tensor_xy @ rotate_mat
        tensor_xy += center
        tensor = tf.concat([tensor_xy,tensor_z], axis=-1)

    if shift is not None:
        shift = tf.random.uniform((),*shift)
        tensor = tensor + shift

    return tensor

def temporal_crop(x, length=MAX_LEN):
    l = tf.shape(x)[0]
    offset = tf.random.uniform((), 0, tf.clip_by_value(l-length,1,length), dtype=tf.int32)
    x = x[offset:offset+length]
    return x

def temporal_mask(x, size=(0.2,0.4), mask_value=float('nan')):
    l = tf.shape(x)[0]
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l-mask_size,1,l), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(x,tf.range(mask_offset, mask_offset+mask_size)[...,None],tf.fill([mask_size,543,3],mask_value))
    return x
def spatial_mask(x, size=(0.2,0.4), mask_value=float('nan')):
    mask_offset_y = tf.random.uniform(())
    mask_offset_x = tf.random.uniform(())
    mask_size = tf.random.uniform((), *size)
    mask_x = (mask_offset_x<x[...,0]) & (x[...,0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y<x[...,1]) & (x[...,1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x = tf.where(mask[...,None], mask_value, x)
    return x
def augment_fn(x, always=False, max_len=None):
    if tf.random.uniform(())<0.8 or always:
        x = resample(x, (0.5,1.5))
    if tf.random.uniform(())<0.5 or always:
        x = flip_lr(x)
    if max_len is not None:
        x = temporal_crop(x, max_len)
    if tf.random.uniform(())<0.75 or always:
        x = spatial_random_affine(x)
    if tf.random.uniform(())<0.5 or always:
        x = temporal_mask(x)
    if tf.random.uniform(())<0.5 or always:
        x = spatial_mask(x)
    return x


def preprocess(x,augment = False, max_len= MAX_LEN):


    # x = {
    #     "coordinates" : Tensor(...),
    #     "sign": integer
    # }
    coord = x['coordinates']
    coord = filter_nans_tf(coord)
    if augment :
        coord = augment_fn(coord,max_len=max_len)
    coord = tf.ensure_shape(coord,(None,ROWS_PER_FRAME,3))
    preprocess_ = Preprocess(max_len = max_len)
    # remove fake batch dimension by using [0]
    processed_features = preprocess_(coord)[0]
    processed_features = tf.cast(processed_features,tf.float32)
    label =tf.one_hot(x["sign"],NUM_CLASSES)

    return processed_features,label


def get_tfrec_dataset(tfrecords, batch_size=64, max_len=64, drop_remainder=False, augment=False, shuffle=False, repeat=False):
    # Initialize dataset with TFRecords
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE, compression_type='GZIP')
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    ds = ds.map(lambda x: preprocess(x, augment=augment, max_len=max_len), tf.data.AUTOTUNE)

    if repeat: 
        ds = ds.repeat()
        
    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = (False)
        ds = ds.with_options(options)
    
    if batch_size:
        ds = ds.padded_batch(batch_size, padding_values=PAD, padded_shapes=([max_len,CHANNELS],[NUM_CLASSES]), drop_remainder=drop_remainder)

    ds = ds.prefetch(tf.data.AUTOTUNE)
        
    return ds