# Defines EmbeddingModel (Interface)
# File: rag_system/corpus/embedding/base_embedding.py
# Instruction: Replace the entire content of this file.

import abc
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Custom exception for EmbeddingModel errors."""
    pass

class EmbeddingModel(abc.ABC):
    """
    Abstract Base Class (Interface) for embedding models.

    Defines the contract for generating vector embeddings for text.
    """

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: A list of document texts to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            The order corresponds to the input texts.

        Raises:
            EmbeddingError: If embedding fails.
        """
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding as a list of floats.

        Raises:
            EmbeddingError: If embedding fails.
        """
        pass

    @abc.abstractmethod
    def get_langchain_embedding_function(self) -> Any:
        """
        Returns the underlying embedding function object compatible with Langchain
        (e.g., an instance of Langchain's Embeddings).

        This is needed to initialize Langchain-based vector stores like Chroma.

        Returns:
            An object compatible with Langchain's embedding function requirements.
            Type hint is Any to avoid strict Langchain dependency here.
        """
        pass

    def get_name(self) -> str:
        """Return the name of the embedding model implementation."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Default representation."""
        return f"<{self.get_name()}>"