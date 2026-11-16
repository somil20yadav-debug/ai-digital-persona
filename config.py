# config.py
"""
Global configuration for YOU - AI Writing Style Clone
"""

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"  # full endpoint for generator
OLLAMA_BASE_URL = "http://localhost:11434"          # base endpoint for health checks
OLLAMA_MODEL = "llama3"                             # or your persona model like "sam-persona"

# Embedding model settings
EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = "data/samples"
EMBED_DIR = "models/embeddings"

# Database settings
DATABASE_URL = "sqlite:///data/you.db"              # SQLite DB file
CHROMA_DIR = "models/chroma"                        # for vector db if used later

# Sample/storage requirements
MIN_SAMPLE_LENGTH = 100                             # minimum chars per sample
MIN_SAMPLES = 3                                     # required samples before enabling chat
