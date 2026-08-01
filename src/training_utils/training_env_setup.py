import gc
from src.config import seed_everything, CFG 
from tensorflow.keras import mixed_precision
import os
import tensorflow as tf


def setup_training_env(CFG):
    """
    Prepare the TF training env
    """ 
    seed_everything(CFG.seed)
    #clear previous TF session
    tf.keras.backend.clear_session()

    gc.collect()  # Run Python garbage collection
    tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation

    if CFG.fp16:
        try:
            policy = mixed_precision.Policy("mixed_bfloat16")
        except:
            policy = mixed_precision.Policy("mixed_float16")
    
    else:
        policy = mixed_precision.Policy("float32")
    mixed_precision.set_global_policy(policy)
    