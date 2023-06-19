from constants import BATCH_SIZE, EPOCHS, LR
from util import RMSEloss, RMSEnum
from callbacks import init_callbacks
from inference import *
from preprocess import get_true_proportions
from tensorflow.keras.metrics import AUC, Accuracy
from tensorflow.keras.optimizers import Adam
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
    callbacks = init_callbacks(model, x_train, x_test, prop_train, prop_test)
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

    

def evaluate(model, X, true_proportions):
    reord_proportions = infer_cell_proportions(model, X, true_proportions).set_axis(cell_types, axis=1)
    # print(reord_proportions)
    diff = reord_proportions-true_proportions
    m = diff.mean(axis=0)
    stddev = diff.std(axis=0)
    losses = []
    for cell_type in cell_types:
        losses.append(RMSEloss(true_proportions[cell_type], reord_proportions[cell_type]).numpy())
    loss = RMSEloss(reord_proportions, true_proportions).numpy()
    # print(reord_proportions)
    # print(pd.DataFrame(true_proportions))
    return loss, losses, m, stddev

