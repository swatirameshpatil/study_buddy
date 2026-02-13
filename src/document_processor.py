from src.EmbeddingsManager import EmbeddingsManager
from src.logger import get_logger
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = get_logger(__name__)

class DocumentProcessor:
    def __init__(self, embeddings_manager: EmbeddingsManager):
        logger.info("Initializing DocumentProcessor")
        self.embeddings_manager = embeddings_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        logger.info(f"DocumentProcessor initialized - chunk_size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    
    def load_document(self, file_path:str, user_id:str) ->List[Document]:
            """Load single file with metadata."""
            logger.info(f"Loading document: {file_path} for user: {user_id}")
            path = Path(file_path)
            filename = path.name
            if path.suffix.lower() =='.pdf':
                 logger.debug(f"Detected PDF file: {filename}")
                 try:
                     # Try standard extraction first
                     loader = PyMuPDFLoader(str(path))
                     docs = loader.load()
                 except KeyError as e:
                     # Handle PDFs with malformed font descriptors by using dict mode
                     logger.warning(f"Standard PDF extraction failed ({str(e)}), trying with dict mode")
                     loader = PyMuPDFLoader(str(path), mode="page")
                     docs = loader.load()
            else:
                 logger.debug(f"Detected non-PDF file: {filename}")
                 loader = UnstructuredFileLoader(str(path))
                 docs = loader.load()
            
            logger.info(f"Loaded {len(docs)} pages from {filename}")
            for doc in docs:
                 doc.metadata.update({
                      "user_id":user_id,
                      "filename": filename,
                      "source":file_path
                 })
                 return docs
     
    def process_and_index(self, file_path: str, user_id: str)->dict:
             """Full pipeline: load → split → index → return stats."""
             logger.info(f"Starting document processing pipeline for {file_path}")
             #load
             raw_docs = self.load_document(file_path, user_id)
     
             #split
             chunks = self.text_splitter.split_documents(raw_docs)
             logger.info(f"Split document into {len(chunks)} chunks")
     
             # Index (we'll wire vectorstore in orchestrator)
             result = {
                 "status":"success",
                 "chunks_count":len(chunks),
                 "filename":Path(file_path).name,
                 "raw_pages":len(raw_docs),
             }
             logger.info(f"Processing complete: {len(chunks)} chunks from {len(raw_docs)} pages")
             return result
