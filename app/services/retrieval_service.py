"""
Retrieval service with confidence scoring and self-correction.
"""
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService
from app.services.llm_service import LLMService
from app.utils.logger import setup_logger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = setup_logger(__name__)


class RetrievalService:
    """Service for retrieval with confidence scoring and retries."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_db_service = VectorDBService()
        self.llm_service = LLMService()
        self.confidence_threshold = settings.confidence_threshold
        self.max_retries = settings.max_retries
    
    def retrieve_and_answer(
        self,
        query: str,
        top_k: int = None,
        confidence_threshold: Optional[float] = None,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents and generate answer with confidence scoring.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            confidence_threshold: Custom confidence threshold
            include_citations: Whether to include citations
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        threshold = confidence_threshold or self.confidence_threshold
        top_k = top_k or settings.retrieval_top_k
        
        # Try retrieval with retries
        for attempt in range(self.max_retries):
            try:
                # Generate query embedding
                query_embedding = self.embedding_service.generate_embedding(query)
                
                # Retrieve relevant chunks
                results = self.vector_db_service.query_vectors(
                    query_vector=query_embedding,
                    top_k=top_k
                )
                
                if not results:
                    logger.warning("No results retrieved from vector database")
                    return {
                        "answer": "I couldn't find any relevant information to answer your question.",
                        "citations": [],
                        "confidence_score": 0.0,
                        "retrieval_metadata": {
                            "retrieval_count": 0,
                            "attempt": attempt + 1
                        }
                    }
                
                # Calculate confidence scores
                confidence_scores = [r["score"] for r in results]
                max_confidence = max(confidence_scores) if confidence_scores else 0.0
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                
                # Filter by confidence threshold
                filtered_results = [
                    r for r in results if r["score"] >= threshold
                ]
                
                # If no results meet threshold, use top result anyway but with lower confidence
                if not filtered_results:
                    logger.warning(f"No results met confidence threshold {threshold}, using top result")
                    filtered_results = results[:1]
                    max_confidence = results[0]["score"]
                
                # Prepare context chunks
                context_chunks = [
                    {
                        "text": r["metadata"].get("text", ""),
                        "chunk_id": r["id"],
                        "document_id": r["metadata"].get("document_id", ""),
                        "score": r["score"],
                        "metadata": r["metadata"]
                    }
                    for r in filtered_results
                ]
                
                # Generate answer
                llm_response = self.llm_service.generate_response(
                    query=query,
                    context_chunks=context_chunks,
                    include_citations=include_citations
                )
                
                # Build citations
                citations = self._build_citations(context_chunks) if include_citations else []
                
                return {
                    "answer": llm_response["answer"],
                    "citations": citations,
                    "confidence_score": max_confidence,
                    "retrieval_metadata": {
                        "retrieval_count": len(filtered_results),
                        "total_retrieved": len(results),
                        "avg_confidence": avg_confidence,
                        "max_confidence": max_confidence,
                        "threshold_used": threshold,
                        "attempt": attempt + 1
                    }
                }
                
            except Exception as e:
                logger.error(f"Retrieval attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                # Exponential backoff for retries
                import time
                time.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded")
    
    def _build_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build citation objects from retrieved chunks."""
        citations = []
        for chunk in chunks:
            citations.append({
                "document_id": chunk.get("document_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                "confidence_score": chunk.get("score", 0.0),
                "metadata": chunk.get("metadata", {})
            })
        return citations
