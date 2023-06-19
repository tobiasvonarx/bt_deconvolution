from constants import BATCH_SIZE, EPOCHS, LR, STUDY
from util import RMSEloss, RMSEnum
from callbacks import init_callbacks
from preprocess import get_true_proportions
from tensorflow.keras.metrics import AUC, Accuracy
from tensorflow.keras.optimizers.experimental import Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from constants import cell_types

def compile(model,t=None):
    # assert t != None
    model.compile(
        optimizer=Nadam(LR),
        # loss=lambda a,b: sqrt(mean(square(a-b))),
        loss=RMSEloss,
        metrics=["accuracy"]
        # run_eagerly=True
    )

def train(model, x_train, x_test, prop_train, prop_test,t=None):
    callbacks = init_callbacks(model, x_test, prop_test)
    model.fit(
        x=x_train,
        y=x_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        # validation_data=(x_test,x_test),
        shuffle=True,
        verbose=0,
        workers=4,
        callbacks=callbacks
    )