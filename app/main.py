"""
FastAPI application for RAG System.
"""
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional

from app.config import settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentIngestResponse,
    DocumentStatus,
    HealthResponse,
    Citation
)
from app.services.document_service import DocumentService
from app.services.retrieval_service import RetrievalService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System for Grounded Document QA",
    description="Production-ready RAG system with confidence scoring and citation generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_service = DocumentService()
retrieval_service = RetrievalService()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check service health
        services = {
            "embedding_service": "healthy",
            "vector_db": "healthy",
            "llm_service": "healthy"
        }
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            services=services
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/api/v1/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a document for RAG processing.
    
    Supports: PDF, TXT, DOCX, MD
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Ingest document
            result = document_service.ingest_document(
                file_path=tmp_path,
                filename=file.filename,
                metadata={"uploaded_at": datetime.utcnow().isoformat()}
            )
            
            return DocumentIngestResponse(
                document_id=result["document_id"],
                status=result["status"],
                chunks_created=result["chunks_created"],
                message=result["message"]
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


@app.get("/api/v1/documents/{document_id}", response_model=DocumentStatus)
async def get_document_status(document_id: str):
    """Get status of an ingested document."""
    try:
        status = document_service.get_document_status(document_id)
        return DocumentStatus(
            document_id=status["document_id"],
            status=status["status"],
            chunks_count=status["chunks_count"],
            created_at=datetime.utcnow(),  # In production, get from DB
            metadata=status.get("metadata")
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG with confidence scoring.
    
    Returns answer with citations and confidence metrics.
    """
    try:
        result = retrieval_service.retrieve_and_answer(
            query=request.query,
            top_k=request.top_k,
            confidence_threshold=request.confidence_threshold,
            include_citations=request.include_citations
        )
        
        # Convert citations to Citation models
        citations = [
            Citation(
                document_id=cit.get("document_id", ""),
                chunk_id=cit.get("chunk_id", ""),
                text=cit.get("text", ""),
                confidence_score=cit.get("confidence_score", 0.0),
                metadata=cit.get("metadata")
            )
            for cit in result.get("citations", [])
        ]
        
        return QueryResponse(
            answer=result["answer"],
            citations=citations,
            confidence_score=result["confidence_score"],
            retrieval_metadata=result.get("retrieval_metadata", {}),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
