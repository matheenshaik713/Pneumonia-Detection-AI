from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ==========================
# Training Configuration
# ==========================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 2

LEARNING_RATE = 0.0001

RANDOM_STATE = 42

# Binary Classification
NUM_CLASSES = 1

DROPOUT_RATE = 0.3

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]
REPORTS_DIR = PROJECT_ROOT / "reports"