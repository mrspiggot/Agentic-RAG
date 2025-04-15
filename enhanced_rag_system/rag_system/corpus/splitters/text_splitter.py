# Contains TextSplitter class/interface
# File: rag_system/corpus/splitters/text_splitter.py
# Instruction: Create this file or replace its entire content.

import logging
from typing import List

# Use relative import for Document type hint
from ...data_models.document import Document

# Attempt safe import of Langchain splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # Check if tiktoken is available, as from_tiktoken_encoder relies on it
    import tiktoken
    LANGCHAIN_SPLITTER_AVAILABLE = True
except ImportError:
    RecursiveCharacterTextSplitter = None # type: ignore
    tiktoken = None # type: ignore
    LANGCHAIN_SPLITTER_AVAILABLE = False


logger = logging.getLogger(__name__)

class SplitterError(Exception):
    """Custom exception for TextSplitter errors."""
    pass


class TextSplitter:
    """
    Handles splitting of Document objects into smaller chunks using Langchain.
    Currently wraps RecursiveCharacterTextSplitter based on tiktoken encoding.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the TextSplitter.

        Args:
            chunk_size: The target size of each chunk (measured by tiktoken).
            chunk_overlap: The overlap between consecutive chunks (measured by tiktoken).

        Raises:
            ImportError: If langchain_text_splitters or tiktoken is not installed.
            ValueError: If chunk_size or chunk_overlap are invalid.
            SplitterError: If the underlying splitter fails to initialize.
        """
        if not LANGCHAIN_SPLITTER_AVAILABLE:
            raise ImportError("langchain-text-splitters and tiktoken packages are required for TextSplitter.")
        if not RecursiveCharacterTextSplitter or not tiktoken:
             # Should be caught by above, but defensive check
             raise ImportError("Failed to import necessary components from langchain/tiktoken.")

        if chunk_size <= 0:
             raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
             raise ValueError("chunk_overlap cannot be negative.")
        # Langchain splitter handles overlap >= size, but we can add a warning
        if chunk_overlap >= chunk_size:
            logger.warning(f"chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). This might lead to unexpected behavior.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            # Initialize the Langchain splitter
            self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"TextSplitter initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        except Exception as e:
            logger.error(f"Failed to initialize RecursiveCharacterTextSplitter: {e}", exc_info=True)
            raise SplitterError(f"Failed to initialize Langchain splitter: {e}") from e


    def process(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into smaller chunks.

        Args:
            documents: The list of Document objects to split.

        Returns:
            A list of new Document objects representing the smaller chunks.
            Metadata from the original document is typically preserved.

        Raises:
            SplitterError: If the splitting process fails.
        """
        if not documents:
            logger.warning("TextSplitter received no documents to process.")
            return []
        if not self._splitter:
             raise SplitterError("Splitter is not initialized.")

        logger.info(f"Splitting {len(documents)} documents into chunks...")
        try:
            # 1. Convert our Documents to Langchain Documents for the splitter
            langchain_docs_in = [doc.to_langchain_document() for doc in documents]

            # 2. Perform the split using the Langchain splitter
            langchain_docs_out = self._splitter.split_documents(langchain_docs_in)

            # 3. Convert the resulting Langchain Documents back to our Document format
            split_docs_out = [Document.from_langchain_document(lc_doc) for lc_doc in langchain_docs_out]

            logger.info(f"Split {len(documents)} documents into {len(split_docs_out)} chunks.")
            return split_docs_out

        except Exception as e:
            logger.error(f"Failed to split documents: {e}", exc_info=True)
            raise SplitterError(f"Document splitting failed: {e}") from e