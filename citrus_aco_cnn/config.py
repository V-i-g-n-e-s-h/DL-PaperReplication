"""Centralised hyper-parameters and random seeds."""

from pathlib import Path

IMG_SIZE    = 128  # 128×128 keeps memory footprint low
BATCH_SIZE  = 32
EPOCHS_CNN  = 40   # backbone
EPOCHS_HEAD = 40   # softmax head

ACO_ANTS = 40
ACO_ITERS = 80
ACO_EVAP = 0.20

NUM_CLASSES = 4
SEED = 42
DATA_DIR = Path.home() / ".tensorflow_datasets"
