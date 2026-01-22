"""
Document chunking pipeline for optimal retrieval.
"""
from typing import List, Dict, Any
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentChunker:
    """Intelligent document chunking with overlap."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            metadata: Additional metadata for chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        # Split by sentences first for better semantic boundaries
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{metadata.get('document_id', 'doc')}_{chunk_id}",
                    "text": chunk_text,
                    "start_index": start,
                    "end_index": start + len(chunk_text),
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": chunk_id
                    }
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
                start += len(chunk_text) - len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": f"{metadata.get('document_id', 'doc')}_{chunk_id}",
                "text": chunk_text,
                "start_index": start,
                "end_index": start + len(chunk_text),
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": chunk_id
                }
            })
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting (can be enhanced with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Get last chunk_overlap characters, but try to break at word boundary
        overlap = text[-self.chunk_overlap:]
        first_space = overlap.find(' ')
        if first_space > 0:
            return overlap[first_space + 1:]
        return overlap
