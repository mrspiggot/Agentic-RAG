# Contains OpenAIEmbeddingModel class
# File: rag_system/corpus/embedding/openai_embedding.py
# Instruction: Replace the entire content of this file.

import logging
from typing import List, Any, Optional

# Use relative import for base class
from .base_embedding import EmbeddingModel, EmbeddingError

# Attempt to import LangChain OpenAIEmbeddings safely
try:
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    OpenAIEmbeddings = None # type: ignore
    LANGCHAIN_OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class OpenAIEmbeddingModel(EmbeddingModel):
    """
    EmbeddingModel implementation using OpenAI embeddings via Langchain.
    """
    # Default model if not specified otherwise - check OpenAI docs for current best
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initializes the OpenAIEmbeddingModel.

        Args:
            api_key: The OpenAI API key. If None, relies on environment variable OPENAI_API_KEY.
            model_name: The specific OpenAI embedding model to use (e.g., 'text-embedding-3-small').
                        Defaults to DEFAULT_MODEL if None.

        Raises:
            ImportError: If langchain-openai is not installed.
            EmbeddingError: If initialization of the underlying Langchain embedding fails.
        """
        if not LANGCHAIN_OPENAI_AVAILABLE or not OpenAIEmbeddings:
            raise ImportError("langchain-openai package is required for OpenAIEmbeddingModel.")

        self.model_name = model_name or self.DEFAULT_MODEL
        self.api_key = api_key
        logger.info(f"Initializing OpenAIEmbeddingModel with model: {self.model_name}")

        try:
            self._client: OpenAIEmbeddings = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key # Pass key if provided, None otherwise
            )
            logger.debug("Langchain OpenAIEmbeddings client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Langchain OpenAIEmbeddings: {e}", exc_info=True)
            raise EmbeddingError(f"Failed to initialize OpenAI embeddings client: {e}") from e


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        if not texts:
            return []
        logger.debug(f"Embedding {len(texts)} documents using model {self.model_name}...")
        try:
            embeddings = self._client.embed_documents(texts)
            logger.debug(f"Successfully generated {len(embeddings)} document embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed documents with OpenAI: {e}", exc_info=True)
            raise EmbeddingError(f"Failed to embed documents: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query text."""
        logger.debug(f"Embedding query using model {self.model_name}...")
        try:
            embedding = self._client.embed_query(text)
            logger.debug("Successfully generated query embedding.")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query with OpenAI: {e}", exc_info=True)
            raise EmbeddingError(f"Failed to embed query: {e}") from e

    def get_langchain_embedding_function(self) -> Any:
        """Returns the underlying Langchain OpenAIEmbeddings instance."""
        return self._client