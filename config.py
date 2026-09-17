# config.py — STEP 1 (Baseline: No Compression, Stable Learning)

# ─────────────────────────────────────────────
# FEDERATED LEARNING SETTINGS
# ─────────────────────────────────────────────
NUM_CLIENTS = 5
NUM_ROUNDS = 20

FRACTION_FIT = 1.0
MIN_FIT_CLIENTS = 3
MIN_AVAILABLE_CLIENTS = 3


# ─────────────────────────────────────────────
# LOCAL TRAINING SETTINGS
# ─────────────────────────────────────────────
LOCAL_EPOCHS = 3          # 🔥 Stable
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = "adam"


# ─────────────────────────────────────────────
# FEDPROX
# ─────────────────────────────────────────────
USE_FEDPROX = False
MU = 0.01


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
DATASET = "CIFAR10"
DATA_DIR = "./data"

NUM_CLASSES = 10
IMG_SIZE = 32


# 🔥 STEP 2: START WITH IID (VERY IMPORTANT)
NON_IID = True
NUM_SHARDS_PER_CLIENT = 2


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
MODEL = "CNN"
DROPOUT = 0.25


# ─────────────────────────────────────────────
# 🚨 STEP 1: DISABLE COMPRESSION
# ─────────────────────────────────────────────
COMPRESSION_ENABLED = True
COMPRESSION_METHOD = "combined"

QUANTIZATION_BITS = 8
SPARSIFICATION_RATIO = 0.2


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
RESULTS_DIR = "./results"
LOG_FILE = "./results/logs.txt"

EVAL_EVERY = 1
SAVE_PLOTS = True


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42


# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"