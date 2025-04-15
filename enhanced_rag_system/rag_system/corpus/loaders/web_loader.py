# File: rag_system/corpus/loaders/web_loader.py
# Instruction: Replace the entire content of this file.
#              (Corrected WebBaseLoader instantiation FOR REAL this time)

import logging
from typing import List, Any, Optional, Dict

# Use relative imports
from .base_loader import DocumentLoader, LoaderError
from ...data_models.document import Document

# Attempt safe import of Langchain loader
try:
    from langchain_community.document_loaders import WebBaseLoader
    LANGCHAIN_LOADER_AVAILABLE = True
except ImportError:
    WebBaseLoader = None # type: ignore
    LANGCHAIN_LOADER_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebDocumentLoader(DocumentLoader):
    """
    Loads document content from a web URL using Langchain's WebBaseLoader.
    Handles loading a single URL per call to `load`.
    """

    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    def __init__(self, request_headers: Optional[Dict[str, str]] = None, **loader_kwargs: Any):
        """
        Initializes the WebDocumentLoader.

        Args:
            request_headers: Optional dictionary of headers for HTTP requests. Defaults to DEFAULT_HEADERS.
            **loader_kwargs: Additional keyword arguments passed directly to Langchain's WebBaseLoader
                              (e.g., bs_get_text_kwargs, continue_on_failure).

        Raises:
            ImportError: If langchain_community is not installed.
        """
        if not LANGCHAIN_LOADER_AVAILABLE or not WebBaseLoader:
            raise ImportError("langchain_community package is required for WebDocumentLoader.")

        self.headers = request_headers or self.DEFAULT_HEADERS
        self.loader_kwargs = loader_kwargs
        logger.debug(f"WebDocumentLoader initialized. Headers: {self.headers}, Kwargs: {self.loader_kwargs}")

    def load(self, identifier: str) -> List[Document]:
        """
        Loads content from the given URL (identifier).

        Args:
            identifier: The URL to load content from.

        Returns:
            A list containing one or more Document objects (WebBaseLoader might
            sometimes split pages, though usually returns one per URL).
            Returns an empty list if loading fails for the specific URL.

        Raises:
            LoaderError: Only if the identifier is fundamentally invalid (not raised on load failure).
        """
        url = identifier
        logger.info(f"Attempting to load content from URL: {url}")

        if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
             logger.error(f"Invalid identifier for WebDocumentLoader: Must be a valid URL starting with http/https. Got: {url}")
             raise LoaderError(f"Invalid identifier for WebDocumentLoader: {url}")

        try:
            # *** CORRECTED INSTANTIATION ***
            # Use the web_paths keyword argument, passing the single URL in a list
            loader = WebBaseLoader(
                web_paths=[url], # Use web_paths=[url] instead of web_path=url
                header_template=self.headers,
                **self.loader_kwargs
            )

            langchain_docs = loader.load()

            loaded_documents: List[Document] = []
            if langchain_docs:
                for lc_doc in langchain_docs:
                    metadata = lc_doc.metadata or {}
                    # Ensure 'source' is always the URL we tried to load
                    metadata['source'] = url
                    metadata['source_type'] = 'web_load'
                    loaded_documents.append(Document.from_langchain_document(lc_doc))
                logger.info(f"Successfully loaded and converted {len(loaded_documents)} document section(s) from {url}")
            else:
                 logger.warning(f"WebBaseLoader returned no documents for URL: {url}")

            return loaded_documents

        except Exception as e:
            logger.error(f"Failed to load content from URL '{url}': {e}", exc_info=False)
            return [] # Return empty list on failure