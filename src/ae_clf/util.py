from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.backend import sqrt, mean, square
import numpy as np
import pandas as pd
import seaborn as sns
import io
from tensorflow.image import decode_png
from tensorflow import expand_dims, summary
# from constants import log_dir

RMSEloss = lambda true, pred: sqrt(mean(square(true-pred)))
float32ify = lambda df: df.astype({c: np.float32 for c in df.select_dtypes(include='float64').columns})

# numerical thing
def RMSEnum(true, pred):
    prop_error = RootMeanSquaredError()
    prop_error.update_state(pred, true)
    return prop_error.result().numpy()