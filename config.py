"""Shared configuration for the experiments in
"Counterfactual Explanations for Recommendation: Are We Hitting the Target?"

Paths, dataset sizes, and checkpoint names for the two backbones used in the paper:
MF (called "MLP" in the code and checkpoint names) and VAE.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                # data/<dataset>/... (see data/README.md)
CHECKPOINTS_DIR = ROOT / "checkpoints"  # trained recommenders and LXR explainers
RESULTS_DIR = ROOT / "results"          # metric pickles, tables and figures

DATASETS = ["ML1M", "Yahoo", "Pinterest"]
RECOMMENDERS = ["MLP", "VAE"]           # "MLP" is the MF backbone in the paper
EXPLAINERS = ["jaccard", "cosine", "lime", "shap", "accent", "spinrec", "lxr"]

# ---------------------------------------------------------------------------
# Evaluation hyperparameters (Section 4 of the paper)
# ---------------------------------------------------------------------------
K = 20          # recommendation list length
PI = 10         # maximum explanation size; K_e = 1..PI
TAU = 0.7       # list-similarity threshold for Counterfactual Validity
C = 5           # rank offset for Rank Degradation
RBO_P = 0.9     # RBO persistence

# ---------------------------------------------------------------------------
# Dataset and model settings
# ---------------------------------------------------------------------------
OUTPUT_TYPE = {"MLP": "single", "VAE": "multiple"}

NUM_USERS = {"ML1M": 6037, "Yahoo": 13797, "Pinterest": 19155}
NUM_ITEMS = {"ML1M": 3381, "Yahoo": 4604, "Pinterest": 9362}

VAE_CONFIG = {"enc_dims": [512, 128], "dropout": 0.5, "anneal_cap": 0.2, "total_anneal_steps": 200000}
PINTEREST_VAE_CONFIG = {"enc_dims": [256, 64], "dropout": 0.5, "anneal_cap": 0.2, "total_anneal_steps": 200000}

RECOMMENDER_CHECKPOINTS = {
    ("ML1M", "MLP"): "MLP1_ML1M_0.0076_256_7.pt",
    ("ML1M", "VAE"): "VAE_ML1M_0.0007_128_10.pt",
    ("Yahoo", "MLP"): "MLP2_Yahoo_0.0083_128_1.pt",
    ("Yahoo", "VAE"): "VAE_Yahoo_0.0001_128_13.pt",
    ("Pinterest", "MLP"): "MLP_Pinterest_0.0062_512_21_0.pt",
    ("Pinterest", "VAE"): "VAE_Pinterest_12_18_0.0001_256.pt",
}

HIDDEN_DIM = {
    ("ML1M", "MLP"): 32,
    ("Yahoo", "MLP"): 32,
    ("Pinterest", "MLP"): 512,
    ("ML1M", "VAE"): None,
    ("Yahoo", "VAE"): None,
    ("Pinterest", "VAE"): None,
}

# (checkpoint file, hidden size) of the pretrained LXR explainer for each backbone
LXR_CHECKPOINTS = {
    ("ML1M", "MLP"): ("LXR_ML1M_MLP_12_39_64_11.59908096547193_0.1414854294885049.pt", 64),
    ("ML1M", "VAE"): ("LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt", 128),
    ("Yahoo", "MLP"): ("LXR_Yahoo_MLP_neg-pos_combined_last_29_37_128_12.40692505393434_0.19367009952856118.pt", 128),
    ("Yahoo", "VAE"): ("LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_18.958765029913238_4.92235962483309.pt", 128),
    ("Pinterest", "MLP"): ("LXR_Pinterest_MLP_0_5_16_10.059416809308486_0.705778173474644.pt", 16),
    ("Pinterest", "VAE"): ("LXR_Pinterest_VAE_comb_4_27_32_6.3443735346179855_1.472868807603448.pt", 32),
}


def dataset_dir(data_name):
    return DATA_DIR / data_name


def expl_dict_path(data_name, recommender_name, expl_name):
    """Explanation dictionary {user_id: [(history_item, score), ...]} for one explainer."""
    return dataset_dir(data_name) / f"{recommender_name}_{expl_name}_expl_dict.pkl"


def metrics_path(data_name, recommender_name, expl_name):
    """Per-K_e metric curves produced by evaluate_metrics.ipynb."""
    return RESULTS_DIR / data_name / f"{recommender_name}_{expl_name}_metrics.pkl"
