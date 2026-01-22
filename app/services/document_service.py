"""
Document service for managing document ingestion and storage.
"""
import uuid
from typing import Dict, Any, List
from app.pipelines.ingestion import DocumentIngester
from app.pipelines.chunking import DocumentChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentService:
    """Service for document management."""
    
    def __init__(self):
        self.ingester = DocumentIngester()
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.vector_db_service = VectorDBService()
        self._documents = {}  # In-memory storage (use DB in production)
    
    def ingest_document(
        self,
        file_path: str,
        filename: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document: extract text, chunk, embed, and store.
        
        Args:
            file_path: Path to the document file
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            Dictionary with document_id and status
        """
        try:
            # Step 1: Extract text
            doc_data = self.ingester.ingest_file(
                file_path=file_path,
                filename=filename,
                metadata=metadata
            )
            document_id = doc_data["document_id"]
            
            # Step 2: Chunk text
            chunks = self.chunker.chunk_text(
                text=doc_data["text"],
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    **(doc_data.get("metadata", {}))
                }
            )
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Step 3: Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings(chunk_texts)
            
            # Step 4: Prepare vectors for storage
            vector_ids = [chunk["chunk_id"] for chunk in chunks]
            vector_metadata = [
                {
                    "text": chunk["text"],
                    "document_id": document_id,
                    "chunk_id": chunk["chunk_id"],
                    "filename": filename,
                    **chunk.get("metadata", {})
                }
                for chunk in chunks
            ]
            
            # Step 5: Store in vector database
            self.vector_db_service.upsert_vectors(
                vectors=embeddings,
                ids=vector_ids,
                metadata=vector_metadata
            )
            
            # Step 6: Store document metadata
            self._documents[document_id] = {
                "document_id": document_id,
                "filename": filename,
                "chunks_count": len(chunks),
                "status": "processed",
                "metadata": doc_data.get("metadata", {})
            }
            
            logger.info(f"Successfully ingested document {document_id} with {len(chunks)} chunks")
            
            return {
                "document_id": document_id,
                "status": "success",
                "chunks_created": len(chunks),
                "message": f"Document ingested successfully with {len(chunks)} chunks"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Get status of a document."""
        if document_id not in self._documents:
            raise ValueError(f"Document {document_id} not found")
        
        return self._documents[document_id]
