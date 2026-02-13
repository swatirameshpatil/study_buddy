import os
from src.document_processor import DocumentProcessor
from src.logger import get_logger
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.config import PINECONE_INDEX_NAME
from src.EmbeddingsManager import EmbeddingsManager

logger = get_logger(__name__)

class StudyBuddy:
    def __init__(self):
        logger.info("Initializing StudyBuddy")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logger.info("Pinecone client initialized")
        self.embeddings_manager = EmbeddingsManager()
        self.document_processor = DocumentProcessor(self.embeddings_manager)
        #self._vectorstore = None
        logger.info("StudyBuddy initialized successfully")

    @property
    def vectorestore(self):
        """Lazy-loaded Pinecone vectorstore."""
        logger.info("Loading Pinecone vectorstore")
        #if self._vectorstore is None:
        embeddings = self.embeddings_manager.get_embeddings()
        index_name = os.getenv("PINECONE_INDEX_NAME")
        logger.info(f"Using index: {index_name} for vectorstore And embedding model: {embeddings.model}")
        self._vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
            )
        logger.debug(f"Pinecone vectorstore loaded successfully {self._vectorstore}")
        return self._vectorstore

    def process_upload(self, file_path:str, user_id:str = "default_user")-> dict:
        """Process uploaded file → Pinecone."""
        logger.info(f"Processing upload: {file_path} for user: {user_id}")
        stats = self.document_processor.process_and_index(file_path, user_id)

        # Upsert chunks
        logger.info(f"Upserting {stats['chunks_count']} chunks to vectorstore")
        chunks = self.document_processor.text_splitter.split_documents(
            self.document_processor.load_document(file_path, user_id)
        )

        self.vectorestore.add_documents(chunks)
        logger.info(f"Successfully upserted {len(chunks)} chunks")

        stats["indexed"] = True
        return stats
    
    def retrieve(self, query:str, user_id:str, k:int=3):
        """Retrieve relevant chunks for user."""
        logger.info(f"Retrieving top {k} chunks for query: {query[:50]} (user: {user_id})")
        results = self.vectorestore.similarity_search(
            query,
            k=k,
            filter={"user_id":user_id}
        )
        logger.info(f"Retrieved {len(results)} results")
        return results
