from pathlib import Path

# Sampling and segmentation
FS = 360  # Hz
SEGMENT_SAMPLES = 1000  # fixed window length centered on R
PRE_SAMPLES = SEGMENT_SAMPLES // 2
POST_SAMPLES = SEGMENT_SAMPLES - PRE_SAMPLES

# Class mapping: MIT-BIH annotation symbols -> label index
# Only these classes are used; others are ignored.
CLASS_MAP = {
    "N": 0,  # Normal beat
    "L": 1,  # Left bundle branch block beat (LBBB)
    "R": 2,  # Right bundle branch block beat (RBBB)
    "A": 3,  # Atrial premature beat (APB)
    "V": 4,  # Premature ventricular contraction (PVC)
}
CLASSES = ["Normal", "LBBB", "RBBB", "APB", "PVC"]
NUM_CLASSES = len(CLASSES)

# Directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mitdb"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"

# Training defaults
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
PATIENCE = 5
KFOLDS = 10
SEED = 42
