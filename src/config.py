import os
from dotenv import load_dotenv
from src.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","text-embedding-3-small")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

logger.info(f"Configuration loaded - Index: {PINECONE_INDEX_NAME}, Model: {EMBEDDING_MODEL}")
