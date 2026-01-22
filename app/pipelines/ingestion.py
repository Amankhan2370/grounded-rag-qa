"""
Document ingestion pipeline for processing various file formats.
"""
import os
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentIngester:
    """Handle document ingestion from various formats."""
    
    SUPPORTED_FORMATS = {'.pdf', '.txt', '.docx', '.md'}
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
    
    def ingest_file(
        self,
        file_path: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document file and extract text.
        
        Args:
            file_path: Path to the file
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            Dictionary with document_id and extracted text
        """
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        document_id = str(uuid.uuid4())
        
        try:
            if file_ext == '.pdf':
                text = self._extract_from_pdf(file_path)
            elif file_ext == '.txt':
                text = self._extract_from_txt(file_path)
            elif file_ext == '.docx':
                text = self._extract_from_docx(file_path)
            elif file_ext == '.md':
                text = self._extract_from_md(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_ext}")
            
            return {
                "document_id": document_id,
                "text": text,
                "filename": filename,
                "file_type": file_ext,
                "metadata": {
                    **(metadata or {}),
                    "file_size": os.path.getsize(file_path)
                }
            }
        except Exception as e:
            logger.error(f"Error ingesting file {filename}: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"TXT extraction error: {str(e)}")
            raise
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX extraction error: {str(e)}")
            raise
    
    def _extract_from_md(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Markdown extraction error: {str(e)}")
            raise
