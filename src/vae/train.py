from constants import BATCH_SIZE, EPOCHS, LR
from util import RMSEloss
from callbacks import init_callbacks
from tensorflow.keras.optimizers import Adam, Nadam
import pandas as pd

def compile(model):
    model.compile(
        optimizer=Nadam(LR),
        loss=RMSEloss,
        metrics=["accuracy"]
        # run_eagerly=True
    )

def train(model, x_train, x_test):
    callbacks = init_callbacks(model, x_test)
    y_train = pd.concat(x_train, axis=1)

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        # validation_data=(x_test,x_test),
        verbose=0,
        shuffle=True,
        callbacks=callbacks
    )

