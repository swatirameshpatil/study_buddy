import os
from src.document_processor import DocumentProcessor
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.config import PINECONE_INDEX_NAME
from src.EmbeddingsManager import EmbeddingsManager

class StudyBuddy:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embeddings_manager = EmbeddingsManager()
        self.document_processor = DocumentProcessor(self.embeddings_manager)

    @property
    def vectorestore(self):
        """Lazy-loaded Pinecone vectorstore."""
        embeddings = self.embeddings_manager.get_embeddings()
        return PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
        )

    def process_upload(self, file_path:str, user_id:str)-> dict:
        """Process uploaded file → Pinecone."""
        stats = self.document_processor.process_and_index(file_path, user_id)

        # Upsert chunks
        chunks = self.document_processor.text_splitter.split_documents(
            self.document_processor.load_document(file_path, user_id)
        )

        self.vectorstore.add_documents(chunks)

        stats["indexed"] = True
        return stats
    
    def retireve(self, query:str, user_id:str, k:int=3):
        """Retrieve relevant chunks for user."""
        return self.vectorestore.similarirty_search(
            query,
            k=k,
            filter={"user_id":user_id}
        )