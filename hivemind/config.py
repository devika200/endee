"""
HiveMind — AI Research Paper Assistant on Endee
Configuration file: single source of truth for all settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === ENDEE CONFIGURATION ===
ENDEE_HOST = os.getenv("ENDEE_HOST", "https://didactic-space-adventure-rq7vqw449pphpw9p-8080.app.github.dev")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")

# INDEX NAMES
KNOWLEDGE_BASE_INDEX = "knowledge_base"
KNOWLEDGE_BASE_FP32_INDEX = "knowledge_base_fp32"
KNOWLEDGE_BASE_FP16_INDEX = "knowledge_base_fp16"
KNOWLEDGE_BASE_INT8_INDEX = "knowledge_base_int8"
SESSION_MEMORY_INDEX = "session_memory"

# === EMBEDDINGS ===
EMBEDDING_PROVIDER = "voyage"
EMBEDDING_MODEL = "voyage-3-lite"
EMBEDDING_DIM = 512
DISTANCE_METRIC = "cosine"
QUANTIZATION = "fp16"
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# === RERANKER ===
RERANKER_PROVIDER = "voyage"
RERANKER_MODEL = "rerank-2"

# === LLM ===
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === RETRIEVAL PARAMETERS ===
TOP_K_RETRIEVAL = 20
TOP_K_RERANKED = 5
TOP_K_MEMORY = 3

# === SPARSE SEARCH ===
SPARSE_MODEL = "endee_bm25"

# === DATA PATHS ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# DATASET FILES
PAPERS_JSON = PROCESSED_DIR / "papers.json"
CHUNKS_JSON = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_NPY = PROCESSED_DIR / "embeddings.npy"
SPARSE_VECTORS_JSON = PROCESSED_DIR / "sparse_vectors.json"
VOCABULARY_PATH = PROCESSED_DIR / "vocabulary.json"

# === ARXIV FETCH CONFIG ===
ARXIV_CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "cs.AI"]
PAPERS_PER_CATEGORY = 2500
YEAR_MIN = 2019
YEAR_MAX = 2026
ARXIV_YEAR_START = 2019
ARXIV_YEAR_END = 2024
ARXIV_DELAY_SECONDS = 10

# === EVALUATION ===
EVAL_DATASET = "scifact"
EVAL_NUM_QUERIES = 100
EVAL_RESULTS_FILE = BASE_DIR / "evaluation" / "results.csv"

# === SESSION MEMORY ===
MEMORY_EXPIRY_DAYS = 7

# === FILTER CONFIG ===
DEFAULT_PREFILTER_THRESHOLD = 15000
DEFAULT_BOOST_PERCENTAGE = 10

# === BATCH SIZES ===
EMBED_BATCH_SIZE = 128
INSERT_BATCH_SIZE = 100

# === VALIDATION ===
def validate_config():
    """Validate required API keys and paths"""
    errors = []
    
    if not VOYAGE_API_KEY:
        errors.append("VOYAGE_API_KEY not set in .env")
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY not set in .env")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    # Ensure directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "evaluation").mkdir(exist_ok=True)
    
    return True

if __name__ == "__main__":
    validate_config()
    print("Configuration validated successfully")
