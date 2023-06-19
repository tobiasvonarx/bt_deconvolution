from constants import *
from model import *
from preprocess import *
from train import *
from classifier import *
from util import *
import sys

tf.config.list_physical_devices('GPU')
pd.set_option("display.precision", 6)

def run(t=None):
    assert t != None
    model = VAE(N_COL,t)
    compile(model)

    train(model, x_train_list, x_test_list)

    z_train = model.concat([encoder.predict(x_train_list[i]) for i,encoder in enumerate(model.encoders)])
    z_test = model.concat([encoder.predict(x_test_list[i]) for i,encoder in enumerate(model.encoders)])

    clf = train_classifier(z_train, prop_train, z_test, prop_test)

    test_loss = evaluate(clf, z_test, prop_test)

    return test_loss

def study():
    import optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(run, n_trials=40)
    print(study.best_params)

def main(N, predict=False):
    if len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
        from gbm_neftel import preprocess
        x_train, x_test, prop_train, prop_test, markers, _ = preprocess.get_data("ss")
    else:
        markers, markers_all, raw_mixes = get_markers_mixes()
        true_proportions = get_true_proportions()
        x_train, x_test, prop_train, prop_test = preprocess_mixes(markers_all, raw_mixes, true_proportions, True)

    x_train_list = [x_train[markers[i]] for i in range(K)]
    x_test_list = [x_test[markers[i]] for i in range(K)]

    model = VAE(N)
    compile(model)

    train(model, x_train_list, x_test_list)

    z_train = model.concat([encoder.predict(x_train_list[i]) for i,encoder in enumerate(model.encoders)])
    z_test = model.concat([encoder.predict(x_test_list[i]) for i,encoder in enumerate(model.encoders)])

    clf = train_classifier(z_train, prop_train, z_test, prop_test)

    train_loss = evaluate(clf, z_train, prop_train)

    test_loss = evaluate(clf, z_test, prop_test)

    print(f"cell proportion RMSE loss on train set {train_loss}")
    print(f"cell proportion RMSE loss on test set {test_loss}")

    test_y_pred = pd.DataFrame(clf.predict(z_test), columns=prop_test.columns, index=prop_test.index)
    test_y_pred = test_y_pred.clip(lower=0, upper=1)

    if not predict:
        return train_loss, test_loss
    return clf.predict(z_test), prop_test, test_loss


if __name__ == "__main__":
    if STUDY and len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
        from gbm_neftel import preprocess
        x_train, x_test, prop_train, prop_test, markers, _ = preprocess.get_data("ss")
        x_train_list = [x_train[markers[i]] for i in range(K)]
        x_test_list = [x_test[markers[i]] for i in range(K)]
        study()
    elif STUDY:
        markers, markers_all, raw_mixes = get_markers_mixes()
        true_proportions = get_true_proportions()
        x_train, x_test, prop_train, prop_test = preprocess_mixes(markers_all, raw_mixes, true_proportions, True)
        x_train_list = [x_train[markers[i]] for i in range(K)]
        x_test_list = [x_test[markers[i]] for i in range(K)]
        study()
    else:
        main(N_COL)
