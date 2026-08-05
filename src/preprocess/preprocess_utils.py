import numpy as np
import tensorflow as tf


ROWS_PER_FRAME = 543
MAX_LEN = 384
CROP_LEN = MAX_LEN
NUM_CLASSES  = 250
PAD = -100.
NOSE=[
    1,2,98,327
]
LNOSE = [98]
RNOSE = [327]
LIP = [ 0, 
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE #+POSE

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6*NUM_NODES

def len_interpolation(x,target_len,method='random'):
    length = tf.shape(x)[1]
    target_len = tf.maximum(1,target_len)
    if method =='random':
        if tf.random.uniform(()) < 0.33 :
            x = tf.image.resize(x,(target_len,length),'bilinear')
        elif tf.random.uniform(()) < 0.5:
            x = tf.image.resize(x, (target_len,tf.shape(x)[1]),'bicubic')
        else:
            x = tf.image.resize(x, (target_len,tf.shape(x)[1]),'nearest')
    else:
        x = tf.image.resize(x, (target_len,tf.shape(x)[1]),method)
    return x

def tf_nan_mean(x, axis=0, keepdims=False):

    nan_mask = tf.math.is_nan(x)
    values = tf.where(
        nan_mask,
        tf.zeros_like(x),
        x
    )
    total = tf.reduce_sum(
        values,
        axis=axis,
        keepdims=keepdims
    )
    valid_mask = tf.where(
        nan_mask,
        tf.zeros_like(x),
        tf.ones_like(x)
    )
    count = tf.reduce_sum(
        valid_mask,
        axis=axis,
        keepdims=keepdims
    )
    mean = total / count

    return mean

def tf_nan_std(x,center = None,axis =0,keepdims=False):
    if center is None:
        center = tf_nan_mean(x,axis=axis, keepdims=True)
    
    d = x - center
    std = tf.math.sqrt(tf_nan_mean(d*d,axis = axis, keepdims=keepdims))

    return std



#--------------------------------------------------------------------------------------------------------
class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        if inputs.shape.rank == 3:
            x = inputs[None,...]
        else:
            x = inputs
        
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        
        x = (x - mean)/std

        if self.max_len is not None:
            x = x[:,:self.max_len]
        length = tf.shape(x)[1]
        x = x[...,:2]

        
        dx = tf.pad(
            x[:, 1:] - x[:, :-1],
            [[0, 0], [0, 1], [0, 0], [0, 0]]
                )

        dx2 = tf.pad(
            x[:, 2:] - x[:, :-2],
            [[0, 0], [0, 2], [0, 0], [0, 0]]
                )


        x = tf.concat([
            tf.reshape(x, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx2, (-1,length,2*len(self.point_landmarks))),
        ], axis = -1)
        
        x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)
        
        return x