"""
LLM service for generating responses with RAG.
"""
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMService:
    """Service for LLM interactions."""
    
    def __init__(self):
        self.provider = settings.llm_provider
        self.temperature = settings.temperature
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client."""
        if self.provider == "openai":
            try:
                import openai
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API key not configured")
                self._client = openai.OpenAI(api_key=settings.openai_api_key)
                self.model = "gpt-4-turbo-preview"
                logger.info("Initialized OpenAI LLM client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise
        elif self.provider == "anthropic":
            try:
                import anthropic
                if not settings.anthropic_api_key:
                    raise ValueError("Anthropic API key not configured")
                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                self.model = "claude-3-opus-20240229"
                logger.info("Initialized Anthropic LLM client")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            include_citations: Whether to include citations
            
        Returns:
            Dictionary with answer and metadata
        """
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Create prompt
        prompt = self._create_prompt(query, context, include_citations)
        
        try:
            if self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                answer = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=self.temperature,
                    system=self._get_system_prompt(),
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.content[0].text
            
            return {
                "answer": answer,
                "model": self.model,
                "provider": self.provider
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", chunk.get("metadata", {}).get("text", ""))
            context_parts.append(f"[Document {i}]\n{text}\n")
        return "\n".join(context_parts)
    
    def _create_prompt(
        self,
        query: str,
        context: str,
        include_citations: bool
    ) -> str:
        """Create the RAG prompt."""
        citation_instruction = ""
        if include_citations:
            citation_instruction = (
                "\nWhen referencing information from the context, "
                "cite the document number (e.g., [Document 1]). "
                "Only use information that is explicitly stated in the provided context."
            )
        
        prompt = f"""Based on the following context, answer the question. 
If the answer cannot be found in the context, say so clearly.{citation_instruction}

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return (
            "You are a helpful assistant that answers questions based on provided context. "
            "Always ground your answers in the provided context and cite your sources. "
            "If the context doesn't contain enough information to answer the question, "
            "say so clearly rather than making up information."
        )
