"""
Central configuration for the T-RAG pipeline.

All paths, hyperparameters, and constants live here.
When running on Kaggle/Colab, update the DATASET_ROOT and PROJECT_ROOT
to match the remote environment paths.
"""

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    """Resolve a path from an environment variable, or use a local default."""
    value = os.getenv(name)
    if not value:
        return default

    if os.name != "nt" and len(value) >= 2 and value[1] == ":":
        drive = value[0].lower()
        rest = value[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{rest}").expanduser()

    return Path(value).expanduser()


# ─────────────────────────────────────────────
# PATHS — Update these when running on a different machine
# ─────────────────────────────────────────────

# Root of this project
PROJECT_ROOT = _path_from_env(
    "MIMIC_PROJECT_ROOT",
    Path(__file__).resolve().parents[1],
)

# Root of the MIMIC-CXR dataset
DATASET_ROOT = _path_from_env(
    "MIMIC_DATASET_ROOT",
    PROJECT_ROOT / "mimics" / "test_dataset",
)

# Derived paths
IMAGES_DIR = DATASET_ROOT / "images"
REPORTS_DIR = DATASET_ROOT / "reports"

RECORD_LIST_CSV = DATASET_ROOT / "cxr-record-list.csv"
STUDY_LIST_CSV = DATASET_ROOT / "cxr-study-list.csv"
PROVIDER_LIST_CSV = DATASET_ROOT / "cxr-provider-list.csv"

# Output directory for generated data artifacts
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEMPORAL_PATIENTS_PATH = (
    DATA_DIR / "temporal_patients.json"
)

# Checkpoints
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# SUBSET CONFIGURATION
# ─────────────────────────────────────────────

MAX_STUDIES = None
MIN_STUDIES_PER_PATIENT = 10

# ─────────────────────────────────────────────
# IMAGE EMBEDDING (DenseNet121)
# ─────────────────────────────────────────────

# Feature vector dimension from DenseNet121 and ResNet50
IMAGE_EMBEDDING_DIM = 1024  # Legacy (single-view)
AP_EMBEDDING_DIM = 1024
LV_EMBEDDING_DIM = 2048
DUAL_IMAGE_EMBEDDING_DIM = AP_EMBEDDING_DIM + LV_EMBEDDING_DIM  # 3072

# CLAF fusion — shared projection space
# Fusion dim logic will map the ensemble output back to this if needed
FUSION_DIM = 512

# View weights (AP > LV)
AP_VIEW_WEIGHT = 0.7
LV_VIEW_WEIGHT = 0.3


# ─────────────────────────────────────────────
# 14 CheXpert PATHOLOGIES
# ─────────────────────────────────────────────

PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


# ─────────────────────────────────────────────
# EMBEDDING CONFIGURATION
# ─────────────────────────────────────────────

# BiomedCLIP text encoder for shared multimodal text embeddings
BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

EMBEDDING_MODEL_NAME = BIOMEDCLIP_MODEL_NAME
EMBEDDING_DIM = 512


# ─────────────────────────────────────────────
# TEMPORAL MODELING
# ─────────────────────────────────────────────

# Window embeddings
WINDOW_TEMPORAL_FEATURE_DIM = 2  # trend + intensity
WINDOW_EMBEDDING_DIM = EMBEDDING_DIM + WINDOW_TEMPORAL_FEATURE_DIM  # 514

# Trajectory embeddings
TEMPORAL_FEATURE_DIM = 8
TRAJECTORY_EMBEDDING_DIM = EMBEDDING_DIM + TEMPORAL_FEATURE_DIM  # 520

# Window configuration
WINDOW_SIZE = 3
WINDOW_STEP = 1
RECENCY_MAX_WINDOWS = None


# ─────────────────────────────────────────────
# RETRIEVAL CONFIGURATION
# ─────────────────────────────────────────────

TOP_K_REPORTS = 5
TOP_K_WINDOWS = 5
TOP_K_TRAJECTORIES = 3


# ─────────────────────────────────────────────
# GENERATION CONFIGURATION
# ─────────────────────────────────────────────

LLM_BACKEND = "qwen2.5-vl-3b"


# ─────────────────────────────────────────────
# VLM MODELS (Comparative Benchmark)
# ─────────────────────────────────────────────

VLM_MODELS = {
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "nv-reason-cxr-3b": "nvidia/NV-Reason-CXR-3B",
    "minicpm-v-4.6": "openbmb/MiniCPM-V-4.6",
}

VLM_MAX_NEW_TOKENS = 1024
VLM_TEMPERATURE = 0.3
VLM_QUANTIZE = "4bit"   # "fp16" | "8bit" | "4bit"


# ─────────────────────────────────────────────
# FUSION (CLAF - BiomedCLIP)
# ─────────────────────────────────────────────

CLAF_DIM = 512
CLAF_IMAGE_WEIGHT = 0.7
CLAF_TEXT_WEIGHT = 0.3


# ─────────────────────────────────────────────
# VISION MODULE (Auxiliary)
# ─────────────────────────────────────────────

VISION_MODEL_NAME = "densenet121-res224-mimic_ch"
IMAGE_SIZE = 224
DETECTION_THRESHOLD = 0.5


# ─────────────────────────────────────────────
# DATASET SPLIT
# ─────────────────────────────────────────────

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.20
TEST_SPLIT  = 0.10
SPLIT_SEED  = 42


# ─────────────────────────────────────────────
# CLASSIFIER TRAINING
# ─────────────────────────────────────────────

CLASSIFIER_LR = 3e-4
CLASSIFIER_EPOCHS = 50
CLASSIFIER_BATCH_SIZE = 256
CLASSIFIER_PATIENCE = 10


# ─────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────
"""
Key Design Choices:

1. BiomedCLIP embeddings (512-dim) are used everywhere
   -> ensures consistency across retrieval, training, and fusion

2. Temporal features are intentionally minimal:
   - window: 2 feature (trend)
   - trajectory: 8 features

3. No DenseNet embeddings used for retrieval or training
   -> vision module is only for explainability (Grad-CAM)

4. Retrieval is multi-level:
   - reports
   - windows
   - trajectories

5. Generation defaults to retrieval-based (no LLM dependency)
"""

#cloudflared tunnel --protocol http2 --edge-ip-version 4 --url http://localhost:8501