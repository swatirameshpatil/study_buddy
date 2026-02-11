from langchain_openai import OpenAIEmbeddings
from src.config import EMBEDDING_MODEL

class EmbeddingsManager:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    def get_embeddings(self):
        return self.embeddings