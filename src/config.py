import os
from dotenv import load_dotenv

load_dotenv()

PINECODE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","text-embedding-3-small")
CHUNK_SIZE =1000
CHUNK_OVERLAP = 200
