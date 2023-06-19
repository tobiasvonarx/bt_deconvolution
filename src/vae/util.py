from tensorflow.keras.metrics import RootMeanSquaredError
import tensorflow.keras.backend as KB
import numpy as np
import pandas as pd
# from seaborn import heatmap
import io
from tensorflow.image import decode_png
from tensorflow import expand_dims, summary
import tensorflow as tf
from constants import LOSS_WEIGHT
# from constants import log_dir

def RMSEloss(true,pred):
    return KB.sqrt(KB.mean(KB.square(true-pred)))

float32ify = lambda df: df.astype({c: np.float32 for c in df.select_dtypes(include='float64').columns})

"""KLloss = lambda m,lv: tf.reduce_mean(
    tf.reduce_sum(
        -0.5 * (1 + lv - tf.square(m) - tf.exp(lv)),
        axis=1
    )
)"""

def MSEloss(true, pred):
    return LOSS_WEIGHT*KB.mean(KB.square(true-pred))

 
def KLloss(mean, log_var):
    kl_loss =  - LOSS_WEIGHT * KB.sum(1 + log_var - KB.square(mean) - KB.exp(log_var), axis = 1)
    return kl_loss

# numerical thing
def RMSEnum(true, pred):
    prop_error = RootMeanSquaredError()
    prop_error.update_state(pred, true)
    return prop_error.result().numpy()
