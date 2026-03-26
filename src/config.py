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

# HF Hub datasets
HF_KOREAN_MONO = "alwaysgood/ko-news-split-512"

# Local data files (fallback)
HK_DATA = DATA_RAW / "hk.jsonl"
MK_DATA = DATA_RAW / "mk.jsonl"
KOREA_BANK_DATA = DATA_RAW / "korea-bank-700-cleaned.jsonl"
NAVER_DATA = DATA_RAW / "naver_terms_clean.jsonl"

# Naver filter
NAVER_EXCLUDE_CATEGORIES = {"학문명백과"}
