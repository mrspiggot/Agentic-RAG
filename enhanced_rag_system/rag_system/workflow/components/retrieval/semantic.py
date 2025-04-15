# Contains SemanticRetriever class
# File: rag_system/workflow/components/retrieval/semantic.py
# Instruction: Replace the entire content of this file.

import logging
from typing import List, Optional

# Use relative imports
from .base_retriever import Retriever, RetrieverError
from ....data_models.document import Document
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration
from ....corpus.vector_stores.base_vector_store import VectorStore, VectorStoreError # Import base VectorStore

logger = logging.getLogger(__name__)

class SemanticRetriever(Retriever):
    """
    Retrieves documents based on semantic similarity using a VectorStore.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initializes the SemanticRetriever.

        Args:
            vector_store: An initialized instance conforming to the VectorStore interface.

        Raises:
            ValueError: If vector_store is not provided.
        """
        if not vector_store:
            raise ValueError("SemanticRetriever requires a valid VectorStore instance.")
        self.vector_store = vector_store
        logger.debug(f"SemanticRetriever initialized with VectorStore: {vector_store.get_name()}")

    def retrieve(self, query: str, state: WorkflowState, config: Configuration) -> List[Document]:
        """
        Performs semantic similarity search using the configured VectorStore.

        Args:
            query: The query string to search for.
            state: The current WorkflowState (used for context, potentially logging).
            config: The application Configuration object to get settings like 'k'.

        Returns:
            A list of relevant Document objects found by the vector store.

        Raises:
            RetrieverError: If the underlying vector store search fails.
        """
        k = config.get_retrieval_k()
        logger.info(f"Performing semantic retrieval for query '{query[:50]}...' with k={k}")

        try:
            # Delegate the search to the vector store instance
            retrieved_docs = self.vector_store.similarity_search(query=query, k=k)
            logger.info(f"Semantic retrieval found {len(retrieved_docs)} documents.")

            # Note: Updating WorkflowState (history, url_usage) is typically handled
            # by the calling node function in RAGWorkflowManager after receiving these results.
            return retrieved_docs

        except VectorStoreError as e:
            logger.error(f"VectorStore search failed during semantic retrieval: {e}", exc_info=True)
            raise RetrieverError(f"Semantic retrieval failed due to VectorStore error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during semantic retrieval: {e}", exc_info=True)
            raise RetrieverError(f"Unexpected error during semantic retrieval: {e}") from e