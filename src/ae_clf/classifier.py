from tensorflow.keras.experimental import LinearModel
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from util import RMSEloss
from constants import log_dir_clf, K, TENSORBOARD
from sklearn.linear_model import ElasticNet
import sys
import pandas as pd

def train_classifier(x, y, x_test, y_test):
    def tune_clf(t):
        clf = ElasticNet(alpha=t.suggest_float("alpha", 0.1, 1.0), l1_ratio=t.suggest_float("l1_ratio", 0.1, 1.0))
        clf.fit(x, y)
        return evaluate(clf, x_test, y_test)

    if len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
        clf = ElasticNet(alpha=0.5)
        clf.fit(x, y)
        return clf
        # import optuna
        # study = optuna.create_study(direction="minimize")
        # study.optimize(tune_clf, n_trials=100)
        # print(study.best_params)
        # assert False
    if TENSORBOARD:
        tensorboard = TensorBoard(log_dir=log_dir_clf)
        callbacks = [tensorboard]
    else:
        callbacks = []
    model = LinearModel(units=K)
    model.compile(Adam(), loss=RMSEloss)
    model.fit(x, y, epochs=150, callbacks=callbacks, validation_data=(x_test, y_test), verbose=0)
    return model

    
def evaluate(model, Z, true_proportions):
    y = pd.DataFrame(model.predict(Z), columns=true_proportions.columns, index=true_proportions.index)
    # y = y.sub(y.min(axis=1), axis=0)
    # y = y.div(y.sum(axis=1), axis=0)
    y = y.clip(lower=0, upper=1)
    # y = y.div(y.sum(axis=1), axis=0)
    # diff = y - true_proportions
    # print(diff.describe())
    loss = RMSEloss(y, true_proportions).numpy()
    return loss
