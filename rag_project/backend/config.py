import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
VECTOR_DIR = os.getenv("VECTOR_DIR", "vectorstore/faiss_index")

# RAG Settings
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))

# Provider Selection
PROVIDER = os.getenv("PROVIDER", "ollama")

# Ollama Settings
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
