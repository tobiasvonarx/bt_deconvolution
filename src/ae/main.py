__import__("matplotlib").use('tkagg')
from constants import *
from model import *
from preprocess import *
from train import *
from inference import *
from process_results import *
from util import *
import pandas as pd
import sys

tf.config.list_physical_devices('GPU')
pd.set_option("display.precision", 6)

def run(t=None):
    assert t != None
    model = AE(N_COL*K,t)
    compile(model,t)

    train(model, x_train, x_test, prop_train, prop_test,t)

    train_loss = evaluate(model, x_test, prop_test)
    test_loss = evaluate(model, x_train, prop_train)

    print(f"cell proportion RMSE loss on train set {train_loss[0]}")
    print(f"cell proportion RMSE loss on test set {test_loss[0]}")
    return (train_loss[0]+test_loss[0])/2

def study():
    import optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(run, n_trials=30)
    print(study.best_params)

def main(N, predict=False):
    if len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
        from gbm_neftel import preprocess
        x_train, x_test, prop_train, prop_test, _, _ = preprocess.get_data("mm")
    else:
        markers, raw_mixes = get_markers_mixes(N)
        true_proportions = get_true_proportions()
        x_train, x_test, prop_train, prop_test = preprocess_mixes(markers, raw_mixes, true_proportions)

    model = AE(N*K)
    compile(model)

    train(model, x_train, x_test, prop_train, prop_test)

    train_loss = evaluate(model, x_train, prop_train)
    test_loss = evaluate(model, x_test, prop_test)

    print(f"cell proportion RMSE loss on train set {train_loss[0]}")
    print(f"cell proportion RMSE loss on test set {test_loss[0]}")
    
    test_y_pred = infer_cell_proportions(model, x_test, prop_test).set_axis(cell_types, axis=1)
    losses = []
    for cell_type in cell_types:
        losses.append(RMSEnum(prop_test[cell_type], test_y_pred[cell_type]))
    print(losses)
    if not predict:
        return train_loss, test_loss
    # to get predicted test proportions:
    return test_y_pred, prop_test, test_loss[0]

if __name__ == "__main__":
    if STUDY and len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
        from gbm_neftel import preprocess
        x_train, x_test, prop_train, prop_test, _, _ = preprocess.get_data("mm")
        study()
    elif STUDY:
        markers, raw_mixes = get_markers_mixes(N_COL)
        true_proportions = get_true_proportions()
        x_train, x_test, prop_train, prop_test = preprocess_mixes(markers, raw_mixes, true_proportions)
        study()
    else:
        main(N_COL)

        
