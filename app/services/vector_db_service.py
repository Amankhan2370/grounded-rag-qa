"""
Vector database service for storing and retrieving embeddings.
"""
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorDBService:
    """Service for vector database operations."""
    
    def __init__(self):
        self.db_type = settings.vector_db_type
        self.index_name = settings.pinecone_index_name
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the vector database client."""
        if self.db_type == "pinecone":
            try:
                import pinecone
                if not settings.pinecone_api_key:
                    raise ValueError("Pinecone API key not configured")
                
                pinecone.init(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment
                )
                self._client = pinecone.Index(self.index_name)
                logger.info(f"Initialized Pinecone index: {self.index_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {str(e)}")
                raise
        elif self.db_type == "chromadb":
            try:
                import chromadb
                self._client = chromadb.Client()
                self._collection = self._client.get_or_create_collection(
                    name=self.index_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Initialized ChromaDB collection: {self.index_name}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported vector DB type: {self.db_type}")
    
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Upsert vectors into the database.
        
        Args:
            vectors: List of embedding vectors
            ids: List of vector IDs
            metadata: List of metadata dictionaries
            
        Returns:
            True if successful
        """
        try:
            if self.db_type == "pinecone":
                # Format for Pinecone
                vectors_to_upsert = [
                    (id, vector, meta) for id, vector, meta in zip(ids, vectors, metadata)
                ]
                self._client.upsert(vectors=vectors_to_upsert)
            elif self.db_type == "chromadb":
                self._collection.upsert(
                    embeddings=vectors,
                    ids=ids,
                    metadatas=metadata
                )
            
            logger.info(f"Upserted {len(vectors)} vectors")
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise
    
    def query_vectors(
        self,
        query_vector: List[float],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of results with ids, scores, and metadata
        """
        try:
            if self.db_type == "pinecone":
                results = self._client.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
                return [
                    {
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata
                    }
                    for match in results.matches
                ]
            elif self.db_type == "chromadb":
                results = self._collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    where=filter_dict
                )
                return [
                    {
                        "id": results["ids"][0][i],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                        "metadata": results["metadatas"][0][i]
                    }
                    for i in range(len(results["ids"][0]))
                ]
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            raise
