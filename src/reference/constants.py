import sys

N_COL = 50 # how many columns of marker genes to subset per cell type
K = 3 # number of cell types
TEST_SIZE = 0.3

mix_paths = [
    "../../data/mixa_filtered.txt",
    "../../data/mixb_filtered.txt",
    "../../data/mixc_filtered.txt",
    "../../data/mixd_filtered.txt"
]

true_prop_paths = [
    "../../data/seta.txt",
    "../../data/setb.txt",
    "../../data/setc.txt",
    "../../data/setd.txt"
]

cell_types = ["bcell", "epi", "fib"]
if len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
    from gbm_neftel import constants
    cell_types = constants.cell_types

true_marker_path = "../../data/true_markers.csv"