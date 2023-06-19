import pandas as pd
import numpy as np
from gbm_neftel.constants import marker_path, true_prop_path, gexp_path, test_split, cell_types, SEED
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import sys

def get_data(normalize=False):
    float32ify = lambda df: df.astype({c: np.float32 for c in df.select_dtypes(include='float64').columns})

    markers = pd.read_csv(marker_path, index_col=0, header=None)
    markers.index.rename("cell_type", inplace=True)
    markers_celltypes = [pd.Index(markers.loc[cell_type]) for cell_type in cell_types]
    markers_all = pd.Index(markers.T.to_numpy().flatten())

    gexp = float32ify(pd.read_csv(gexp_path, index_col=["patient", "pseudopatient"]))
    # gexp = gexp.applymap(lambda x: np.log2(x+1))

    if normalize == "ss":
        gexp = pd.DataFrame(StandardScaler().fit_transform(gexp), index=gexp.index, columns=gexp.columns)
    elif normalize == "log":
        gexp = gexp.applymap(lambda x: np.log2(x+1))
    elif normalize == "mm":
        gexp = pd.DataFrame(MinMaxScaler().fit_transform(gexp), index=gexp.index, columns=gexp.columns)
    
    true_prop = pd.read_csv(true_prop_path, index_col=["patient", "pseudopatient"])

    patients = gexp.index.get_level_values(0).unique().tolist()

    train_patients, test_patients = train_test_split(patients, random_state=10, test_size=test_split)

    train_x = gexp.loc[train_patients]
    test_x = gexp.loc[test_patients]
    train_y = true_prop.loc[train_patients]
    test_y = true_prop.loc[test_patients]
    return train_x, test_x, train_y, test_y, markers_celltypes, markers_all

if __name__ == "__main__":
    train_x, test_x, train_y, test_y, markers_celltypes, markers_all = get_data(True)
    print(train_x)
