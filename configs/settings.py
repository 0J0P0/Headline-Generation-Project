from pathlib import Path


PROJ_DIR = Path.cwd()

DATA_DIR = PROJ_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJ_DIR / "models"

SEED = 959


MAX_INPUT_LEN = 2048
MAX_TARGET_LEN = 8