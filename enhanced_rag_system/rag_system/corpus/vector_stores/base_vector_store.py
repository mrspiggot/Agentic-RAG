# Defines VectorStore (Interface)
# File: rag_system/corpus/vector_stores/base_vector_store.py
# Instruction: Replace the entire content of this file.

import abc
import logging
from typing import List, Any, Tuple, Optional

# Use relative import for Document type hint
from ...data_models.document import Document

logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception for Vector Store errors."""
    pass

class VectorStore(abc.ABC):
    """
    Abstract Base Class (Interface) for vector stores.

    Defines the contract for storing, managing, and searching document vectors.
    """

    @abc.abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store. This typically involves generating
        embeddings using the configured embedding function during initialization.

        Args:
            documents: A list of Document objects to add.

        Returns:
            A list of IDs for the added documents.

        Raises:
            VectorStoreError: If adding documents fails.
        """
        pass

    @abc.abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """
        Perform a similarity search based on a query string.
        The implementation will typically embed the query using the configured
        embedding function and search the vector store.

        Args:
            query: The query text.
            k: The number of results to return.
            **kwargs: Additional search parameters specific to the implementation.

        Returns:
            A list of Document objects most similar to the query.

        Raises:
            VectorStoreError: If the search fails.
        """
        pass

    @abc.abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search and return results with similarity scores.

        Args:
            query: The query text.
            k: The number of results to return.
            **kwargs: Additional search parameters specific to the implementation.

        Returns:
            A list of tuples, where each tuple contains a Document object and its
            similarity score (higher score typically means more similar, but interpretation
            depends on the underlying distance metric).

        Raises:
            VectorStoreError: If the search fails.
        """
        pass

    # Optional methods that might be useful, depending on implementation needs
    # @abc.abstractmethod
    # def persist(self):
    #     """Persist the vector store to disk (if applicable)."""
    #     pass

    # @abc.abstractmethod
    # def load(self):
    #     """Load the vector store from disk (if applicable)."""
    #     pass

    @abc.abstractmethod
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None) -> Any:
        """
        Get a Langchain-compatible retriever instance configured for this vector store.

        Args:
            search_type: Type of search (e.g., "similarity", "mmr").
            search_kwargs: Dictionary of arguments for the search (e.g., {"k": 5}).

        Returns:
            A Langchain retriever object (type Any to avoid strict Langchain dependency here).
        """
        pass

    def get_name(self) -> str:
        """Return the name of the vector store implementation."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Default representation."""
        return f"<{self.get_name()}>"