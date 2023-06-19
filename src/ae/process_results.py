import pandas as pd
import numpy as np
from constants import cell_types, K
import sys

def get_correlation(true_prop, pred_prop):
    both_proportions = pd.concat([true_prop, pred_prop], axis=1)
    # print(true_prop)
    # print(pred_prop)
    # print(both_proportions)
    # assert False
    corr = both_proportions.corr()[cell_types].loc[range(K)]
    return corr

def reorder_pred(corr, pred_prop):
    # reorder
    reordered_proportions = pred_prop.copy()

    highest_corrs = corr.unstack().sort_values(ascending=False)

    assigned_cts = [] # original cell types
    assigned_js = [] # predicted order

    for (ct, j), _ in highest_corrs.items():
        if ct not in assigned_cts and j not in assigned_js:
            i = cell_types.index(ct)
            reordered_proportions[i] = pred_prop[j]
            # print(f"assigning column {j=} of predictions to {ct=}")
            assigned_cts.append(ct)
            assigned_js.append(j)
    assert len(assigned_cts) == K
    assert len(assigned_js) == K

    return reordered_proportions