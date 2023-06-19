import pandas as pd
from process_results import get_correlation, reorder_pred
from util import log
import sys

def predict_proportions(model, mixes):
    df = pd.DataFrame(model.predict(mixes), index=mixes.index) 
    # if len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
    #     df = df.div(df.sum(axis=1), axis=0)
    return df


def infer_cell_proportions(model, mixes, true_proportions, epoch=None, name=None):
    pred_proportions = predict_proportions(model.encoder, mixes)
    corr = get_correlation(true_proportions, pred_proportions)
    if epoch != None:
        log(corr, epoch, name)
    # print("predicted proportions")
    # print(pred_proportions)
    reord_proportions = reorder_pred(corr, pred_proportions)
    # print("reordered proportions")
    # print(reord_proportions)
    # if epoch == None:
    #     print(get_correlation(true_proportions,reord_proportions))
    return reord_proportions