from EmbeddingsManager import EmbeddingsManager
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_OVERLAP, CHUNK_SIZE

class DocumentProcessor:
    def __init__(self, embeddings_manager: EmbeddingsManager):
        self.embeddings_manager = embeddings_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    
    def load_document(self, file_path:str, user_id:str) ->List[Document]:
            """Load single file with metadata."""
            path = Path(file_path)
            filename = path.name
            if path.suffix.lower() =='.pdf':
                 loader = PyPDFLoader(str(path))
            else:
                 loader = UnstructuredFileLoader(str(path))

            docs = loader.load()
            for doc in docs:
                 doc.metadata.update({
                      "user_id":user_id,
                      "filename": filename,
                      "source":file_path
                 })
                 return docs
     
    def process_and_index(self, file_path: str, user_id: str)->dict:
             """Full pipeline: load → split → index → return stats."""
             #load
             raw_docs = self.load_document(file_path, user_id)
     
             #split
             chunks = self.text_splitter.split_documents(raw_docs)
     
             # Index (we'll wire vectorstore in orchestrator)
             return {
                 "status":"success",
                 "chunks_count":len(chunks),
                 "filename":Path(file_path).name,
                 "raw_pages":len(raw_docs),
             }