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
    x = {
        "coordinates" : Tensor(...),
        "sign": integer
    }
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
