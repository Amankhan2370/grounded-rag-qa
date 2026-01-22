"""
Pydantic models for API request and response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for document queries."""
    query: str = Field(..., description="The question to answer", min_length=1)
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    include_citations: bool = Field(True, description="Include source citations in response")
    confidence_threshold: Optional[float] = Field(None, description="Custom confidence threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the main topic of the document?",
                "top_k": 5,
                "include_citations": True
            }
        }


class Citation(BaseModel):
    """Citation model for source references."""
    document_id: str
    chunk_id: str
    text: str
    page_number: Optional[int] = None
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response model for document queries."""
    answer: str
    citations: List[Citation]
    confidence_score: float
    retrieval_metadata: Dict[str, Any]
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The document discusses...",
                "citations": [],
                "confidence_score": 0.85,
                "retrieval_metadata": {},
                "query": "What is the main topic?"
            }
        }


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""
    document_id: str
    status: str
    chunks_created: int
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentStatus(BaseModel):
    """Model for document status."""
    document_id: str
    status: str
    chunks_count: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    services: Dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
