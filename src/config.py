"""Project-wide constants and paths."""

from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
MERGED_DIR = ROOT / "models" / "merged"

# Model
BASE_MODEL = "unsloth/Qwen3.5-4B-Base"
MAX_SEQ_LEN = 2048
SEED = 42

# Local data files (used by evaluate.py)
KOREA_BANK_DATA = DATA_RAW / "korea-bank-700-cleaned.jsonl"
