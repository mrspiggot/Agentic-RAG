# Defines Retriever (Interface)
# File: rag_system/workflow/components/retrieval/base_retriever.py
# Instruction: Replace the entire content of this file.

import abc
import logging
from typing import List, Optional

# Use relative imports for data models and config
from ....data_models.document import Document
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration

logger = logging.getLogger(__name__)

class RetrieverError(Exception):
    """Custom exception for Retriever errors."""
    pass

class Retriever(abc.ABC):
    """
    Abstract Base Class (Interface) for document retrieval strategies.

    Concrete implementations will provide different methods for finding documents
    relevant to a query (e.g., semantic search, keyword search, web search).
    """

    @abc.abstractmethod
    def retrieve(self, query: str, state: WorkflowState, config: Configuration) -> List[Document]:
        """
        Retrieve documents relevant to the query based on the current state and config.

        Args:
            query: The query string to retrieve documents for.
            state: The current WorkflowState object, providing context (like history)
                   and allowing updates (like url_usage_tracking).
            config: The application Configuration object for settings (like 'k').

        Returns:
            A list of retrieved Document objects. Implementations should handle
            updating the WorkflowState (e.g., url_usage_tracking, retrieval_history)
            appropriately within this method or expect the calling node function to do so
            based on the returned documents.

        Raises:
            RetrieverError: If the retrieval process encounters a critical error.
        """
        pass

    def get_name(self) -> str:
        """
        Returns the identifiable name of the retriever strategy.
        Defaults to the class name. Concrete classes can override if needed.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Default representation showing the retriever name."""
        return f"<{self.get_name()}>"