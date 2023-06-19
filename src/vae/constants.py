from datetime import datetime
import sys

N_COL = 50 # how many columns of marker genes to subset per cell type
K = 3 # number of cell types
EPOCHS = 160
BATCH_SIZE = 8
DROPOUT_RATE_ENCODER = 0.2
DROPOUT_RATE_DECODER = 0.2
PATIENCE = 7
TEST_SIZE = 0.3

STUDY = False
TENSORBOARD = False

LR = 1e-3
LOSS_WEIGHT = 5e-5

# hidden layer in encoder
M1 = 20
# hidden layer in decoder
M2 = 64

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
    K = len(cell_types) 
    M2 = 135
    DROPOUT_RATE_ENCODER = 0.5
    DROPOUT_RATE_DECODER = 0.4

true_marker_path = "../../data/true_markers.csv"

log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_clf = "./logs_clf/" + datetime.now().strftime("%Y%m%d-%H%M%S")
