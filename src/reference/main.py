from constants import *
import pandas as pd
import numpy as np
from tensorflow.keras.backend import sqrt, mean, square
from sklearn.model_selection import train_test_split
import sys

float32ify = lambda df: df.astype({c: np.float32 for c in df.select_dtypes(include='float64').columns})
RMSEloss = lambda true, pred: sqrt(mean(square(true-pred)))

def get_markers_mixes(N):
    # subset N most expressed marker genes per cell type
    # markers = pd.read_csv(true_marker_path, usecols=range(N_COL+1), index_col=0, header=None)
    markers = pd.read_csv(true_marker_path, index_col=0, header=None)
    markers.index.rename("cell_type", inplace=True)
    # assert markers.shape==(K,N_COL)

    # per mix, get the expressions of the marker gene
    # each mix is a column of the mix file, rows are the gene expressions
    mixes = [pd.read_csv(path) for path in mix_paths]
    mixes = [float32ify(mix) for mix in mixes]

    usable_markers = mixes[0].index
    for mix in mixes:
        usable_markers = usable_markers.intersection(mix.index)

    # markers that are nowhere NaN expressed
    # list of top N_COL markers per cell type
    markers_cell_types = [markers.loc[cell_type] for cell_type in cell_types]
    for i in range(K):
        markers_cell_types[i] = markers_cell_types[i][markers_cell_types[i].isin(usable_markers)].head(N)

    markers_all = pd.Index(pd.concat(markers_cell_types, ignore_index=True))
    markers_cell_types = [pd.Index(markers_cell_type) for markers_cell_type in markers_cell_types]

    assert len(markers_all)==K*N

    return markers_all, markers_cell_types, mixes

def preprocess_mixes(markers, mixes, true_proportions):
    # only care about expression of subsetted marker genes
    # filter out marker genes not in the intersection and merge all
    all_mixes = pd.concat([mix[mix.index.isin(markers)].T for mix in mixes],axis=0)
    all_mixes.reset_index(inplace=True,drop=True)
    scenarios_x = np.vsplit(all_mixes, 4)
    scenarios_y = np.vsplit(true_proportions, 4)

    # train/test scenarios
    x_train = pd.concat([scenarios_x[0],scenarios_x[3],scenarios_x[2]], axis=0)
    x_test = scenarios_x[1]
    prop_train = pd.concat([scenarios_y[0],scenarios_y[3],scenarios_y[2]], axis=0)
    prop_test = scenarios_y[1]
    x_train, x_test, prop_train, prop_test = train_test_split(all_mixes, true_proportions, test_size=TEST_SIZE, random_state=2)
    return x_train, x_test, prop_train, prop_test

def get_true_proportions():
    true_proportions = pd.concat([pd.read_csv(path).reset_index(drop=True) for path in true_prop_paths],axis=0)
    true_proportions.reset_index(inplace=True,drop=True)
    return true_proportions

true_prop = get_true_proportions()

def main(N_COL, predict=False):
    if len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
        from gbm_neftel import preprocess
        _, x_test, _, prop_test, markers_celltype, markers_all = preprocess.get_data("mm")
    else: 
        markers_all, markers_celltype, mixes = get_markers_mixes(N_COL)
        _, x_test, _, prop_test = preprocess_mixes(markers_all, mixes, true_prop)

    df = pd.DataFrame()

    for i,markers in enumerate(markers_celltype):
        # expression of markers of a cell type
        m = x_test[markers].agg("mean", axis=1)
        df[cell_types[i]] = m
    df = df.div(df.sum(axis=1), axis=0)

    diff = df-prop_test
    print(diff.describe())

    m = diff.mean(axis=0)
    stddev = diff.std(axis=0)
    losses = []
    for cell_type in cell_types:
        losses.append(RMSEloss(prop_test[cell_type], df[cell_type]).numpy())
        # print(f"loss for {cell_type} is {loss}")
    loss = RMSEloss(prop_test, df).numpy()
    print(f"cell proportion RMSE loss is {loss}")
    print(losses)
    # both_proportions = pd.concat([true_prop, means.set_axis(range(K), axis=1)], axis=1)
    # corr = both_proportions.corr()[cell_types].loc[range(K)]
    # print(corr)

    if not predict:
        return loss, losses, m, stddev
    return df, prop_test, loss

if __name__ == "__main__":
    loss = main(N_COL)
    print(f"cell proportion RMSE loss is {loss}")