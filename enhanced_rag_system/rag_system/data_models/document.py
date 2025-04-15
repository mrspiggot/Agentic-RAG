# Defines Document class
# File: rag_system/data_models/document.py
# Instruction: Replace the entire content of this file.

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime, timezone

# Attempt to import LangChain Document for type hinting and conversion,
# but make it optional so the data model itself doesn't strictly require LangChain installed.
try:
    from langchain_core.documents import Document as LCDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LCDocument = None # Define as None if LangChain is not available
    LANGCHAIN_AVAILABLE = False


@dataclass
class Document:
    """
    Represents a processed piece of text content within the RAG system.
    Includes content and associated metadata.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())) # Unique ID for each Document instance

    def __post_init__(self):
        """Ensure standard metadata fields exist after initialization."""
        # Ensure ID is in metadata as well for potential downstream use
        self.metadata.setdefault('id', self.id)
        self.metadata.setdefault('timestamp_utc', datetime.now(timezone.utc).isoformat())
        # Ensure 'source' exists, defaulting to unknown if not provided
        self.metadata.setdefault('source', 'unknown')
        # Ensure 'source_type' exists, defaulting to unknown
        self.metadata.setdefault('source_type', 'unknown')


    def __str__(self) -> str:
        """String representation showing ID, source, and content snippet."""
        source = self.metadata.get('source', 'unknown')
        snippet_len = 50
        snippet = self.content[:snippet_len].replace('\n', ' ')
        if len(self.content) > snippet_len:
            snippet += "..."
        return f"Document(id={self.id}, source='{source}', content='{snippet}')"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"Document(id='{self.id}', content='{self.content[:100]}...', metadata={self.metadata})"

    @classmethod
    def from_langchain_document(cls, lc_document: Any) -> 'Document':
        """
        Factory method to create a Document instance from a LangChain Document.

        Args:
            lc_document: An instance of langchain_core.documents.Document.

        Returns:
            A new Document instance.

        Raises:
            TypeError: If lc_document is not of the expected type or LangChain is unavailable.
            AttributeError: If lc_document is missing expected attributes.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed, cannot create from LangChain document.")
        if LCDocument and isinstance(lc_document, LCDocument):
            # Create a copy of metadata to avoid modifying the original lc_document metadata
            metadata_copy = lc_document.metadata.copy() if lc_document.metadata else {}
            # Use existing ID if present in metadata, otherwise generate new one
            doc_id = metadata_copy.get('id', str(uuid.uuid4()))
            metadata_copy['id'] = doc_id # Ensure ID is consistently in metadata

            return cls(
                id=doc_id,
                content=lc_document.page_content,
                metadata=metadata_copy
            )
        else:
            raise TypeError(f"Expected a LangChain Document, but got {type(lc_document)}")

    def to_langchain_document(self) -> Any:
        """
        Converts this Document instance into a LangChain Document.

        Returns:
            An instance of langchain_core.documents.Document.

        Raises:
            ImportError: If LangChain is not installed.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed, cannot convert to LangChain document.")
        if LCDocument:
            # Pass a copy of metadata to avoid potential upstream modifications
            return LCDocument(
                page_content=self.content,
                metadata=self.metadata.copy()
            )
        else:
            # This case should theoretically not be reached if LANGCHAIN_AVAILABLE is True
            raise ImportError("LangChain Document class could not be imported.")