# Defines DocumentLoader (Interface)
# File: rag_system/corpus/loaders/base_loader.py
# Instruction: Create this file or replace its entire content.

import abc
import logging
from typing import List, Any

# Use relative import for Document type hint
from ...data_models.document import Document

logger = logging.getLogger(__name__)

class LoaderError(Exception):
    """Custom exception for DocumentLoader errors."""
    pass

class DocumentLoader(abc.ABC):
    """
    Abstract Base Class (Interface) for document loaders.

    Defines the contract for loading content from a source identifier (e.g., a URL, a file path).
    """

    @abc.abstractmethod
    def load(self, identifier: str) -> List[Document]:
        """
        Loads content corresponding to the given identifier.

        An identifier could be a URL, file path, database ID, etc.
        Implementations should handle fetching and initial parsing into Document objects.

        Args:
            identifier: The source identifier (e.g., "https://example.com/page").

        Returns:
            A list of Document objects loaded from the source. A single source might
            yield multiple Document objects if the loader performs initial chunking
            (though often splitting is done later).

        Raises:
            LoaderError: If loading fails for any reason (e.g., network error, file not found, parsing error).
            FileNotFoundError: Specifically if the identifier is a path that doesn't exist.
        """
        pass

    def get_name(self) -> str:
        """Return the name of the loader implementation."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Default representation."""
        return f"<{self.get_name()}>"