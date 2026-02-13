from langchain_openai import OpenAIEmbeddings
from src.config import EMBEDDING_MODEL
from src.logger import get_logger

logger = get_logger(__name__)

class EmbeddingsManager:

    def __init__(self):
        logger.info(f"Initializing EmbeddingsManager with model: {EMBEDDING_MODEL}")
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        logger.info("EmbeddingsManager initialized successfully")

    def get_embeddings(self):
        logger.debug("Getting embeddings instance")
        return self.embeddings
