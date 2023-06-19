marker_path = "../../data/GBM_Neftel/markers.csv"
true_prop_path = "../../data/GBM_Neftel/gbm_neftel_true_prop.csv"
gexp_path = "../../data/GBM_Neftel/gbm_neftel_gexp.csv"
test_split = 0.3
malignant_states = [f"metasig{i}" for i in range(1,6)]
tme_states = ["Macrophage", "Oligodendrocyte", "T-cell"]
cell_types = malignant_states + tme_states
SEED = 1
