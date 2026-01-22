"""
Embedding service for generating vector embeddings.
"""
from typing import List, Dict, Any
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.provider = settings.embedding_provider
        self.dimension = settings.embedding_dimension
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the embedding client."""
        if self.provider == "openai":
            try:
                import openai
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API key not configured")
                self._client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("Initialized OpenAI embedding client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise
        else:
            # Fallback to sentence transformers
            try:
                from sentence_transformers import SentenceTransformer
                self._client = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = 384
                logger.info("Initialized SentenceTransformer embedding client")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer: {str(e)}")
                raise
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        
        try:
            if self.provider == "openai" and self._client:
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self._client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
            else:
                # Use sentence transformers
                embeddings = self._client.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False
                ).tolist()
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]
