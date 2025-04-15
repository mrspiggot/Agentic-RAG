# File: rag_system/corpus/vector_stores/chroma_vector_store.py
# Instruction: Replace the entire content of this file.
#              (Removed self.client.persist() call in add_documents)

import logging
from typing import List, Any, Tuple, Optional

# Use relative import for base class and Document
from .base_vector_store import VectorStore, VectorStoreError
from ...data_models.document import Document
# Use relative import for EmbeddingModel interface for type hinting
from ..embedding.base_embedding import EmbeddingModel

# Attempt to import LangChain Chroma elements safely
try:
    from langchain_chroma import Chroma
    # Import base classes for type checking if possible
    from langchain_core.vectorstores import VectorStore as LangchainVectorStoreBase
    from langchain_core.embeddings import Embeddings as LangchainEmbeddingsBase
    LANGCHAIN_CHROMA_AVAILABLE = True
except ImportError:
    Chroma = None # type: ignore
    LangchainVectorStoreBase = None # type: ignore
    LangchainEmbeddingsBase = None # type: ignore
    LANGCHAIN_CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class ChromaVectorStore(VectorStore):
    """
    VectorStore implementation using ChromaDB via Langchain.
    Handles persistence automatically when initialized with persist_directory.
    """

    def __init__(self,
                 persist_directory: str,
                 collection_name: str,
                 embedding_model: EmbeddingModel):
        """
        Initializes the ChromaVectorStore. Attempts to load an existing store
        from the persist_directory, or sets up for creation if it doesn't exist.

        Args:
            persist_directory: Path to the directory where ChromaDB data is stored.
            collection_name: Name of the collection within ChromaDB.
            embedding_model: An instance conforming to the EmbeddingModel interface,
                             used to get the Langchain-compatible embedding function.

        Raises:
            ImportError: If langchain-chroma or underlying dependencies are not installed.
            VectorStoreError: If initialization fails for other reasons.
        """
        if not LANGCHAIN_CHROMA_AVAILABLE or not Chroma:
            raise ImportError("langchain-chroma package is required for ChromaVectorStore.")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client: Optional[Chroma] = None # Initialize client as Optional

        try:
            langchain_embedding_function = self.embedding_model.get_langchain_embedding_function()
            # Check if the returned object is usable by Langchain Chroma
            if not LANGCHAIN_CHROMA_AVAILABLE or not LangchainEmbeddingsBase or \
               not isinstance(langchain_embedding_function, LangchainEmbeddingsBase):
                 # Log detailed types if check fails
                 actual_type = type(langchain_embedding_function).__name__ if langchain_embedding_function else 'None'
                 expected_type = LangchainEmbeddingsBase.__name__ if LangchainEmbeddingsBase else 'LangchainEmbeddingsBase'
                 logger.error(f"Type mismatch for embedding function. Expected subclass of '{expected_type}', got '{actual_type}'.")
                 raise TypeError(f"EmbeddingModel provided an incompatible embedding function type '{actual_type}'.")

            logger.info(f"Initializing ChromaDB vector store: "
                        f"directory='{self.persist_directory}', "
                        f"collection='{self.collection_name}'")

            # Initialize Chroma with persist directory.
            self.client = Chroma(
                collection_name=self.collection_name,
                embedding_function=langchain_embedding_function,
                persist_directory=self.persist_directory
            )
            # Check if client seems valid (basic check)
            if not hasattr(self.client, 'similarity_search'):
                 raise VectorStoreError("Initialized Chroma client appears invalid (missing core methods).")

            logger.info("ChromaDB client initialized successfully.")

        except TypeError as te:
             logger.error(f"Type error during Chroma initialization: {te}", exc_info=True)
             raise VectorStoreError(f"Type error during Chroma setup: {te}") from te
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            raise VectorStoreError(f"ChromaDB client initialization failed: {e}") from e

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Adds documents to the Chroma collection. Persistence is handled by Chroma
        when initialized with a persist_directory.
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return []
        if not self.client:
             raise VectorStoreError("Chroma client not initialized.")

        logger.info(f"Adding {len(documents)} documents to Chroma collection '{self.collection_name}'...")
        try:
            lc_documents = [doc.to_langchain_document() for doc in documents]
            added_ids = self.client.add_documents(documents=lc_documents)

            # *** REMOVED self.client.persist() call ***
            # Persistence is managed by Chroma when persist_directory is set.

            if added_ids is None:
                # Chroma sometimes returns None on success if IDs aren't generated/returned explicitly
                logger.info(f"Successfully processed add_documents for {len(documents)} documents (no IDs returned). Chroma store persisted implicitly.")
                # Return empty list or maybe document IDs if available
                return [doc.id for doc in documents] # Return our generated IDs
            else:
                 logger.info(f"Successfully added {len(added_ids)} documents. Chroma store persisted implicitly.")
                 return added_ids
        except Exception as e:
            logger.error(f"Failed to add documents to Chroma: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to add documents: {e}") from e

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Performs similarity search using Chroma."""
        if not self.client:
             raise VectorStoreError("Chroma client not initialized.")
        logger.debug(f"Performing similarity search for query: '{query[:50]}...' with k={k}")
        try:
            results_lc = self.client.similarity_search(query=query, k=k, **kwargs)
            results_docs = [Document.from_langchain_document(doc) for doc in results_lc]
            logger.debug(f"Similarity search returned {len(results_docs)} documents.")
            return results_docs
        except Exception as e:
            logger.error(f"Similarity search failed in Chroma: {e}", exc_info=True)
            raise VectorStoreError(f"Similarity search failed: {e}") from e

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[Tuple[Document, float]]:
        """Performs similarity search with scores using Chroma."""
        if not self.client:
             raise VectorStoreError("Chroma client not initialized.")
        logger.debug(f"Performing similarity search with score for query: '{query[:50]}...' with k={k}")
        try:
            results_with_scores = self.client.similarity_search_with_score(query=query, k=k, **kwargs)
            converted_results = [
                (Document.from_langchain_document(doc), score)
                for doc, score in results_with_scores
            ]
            logger.debug(f"Similarity search with score returned {len(converted_results)} documents.")
            return converted_results
        except Exception as e:
            logger.error(f"Similarity search with score failed in Chroma: {e}", exc_info=True)
            raise VectorStoreError(f"Similarity search with score failed: {e}") from e

    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None) -> Any:
        """Gets a Langchain retriever instance."""
        if not self.client:
            raise VectorStoreError("Chroma client not initialized.")
        if not LANGCHAIN_CHROMA_AVAILABLE or not LangchainVectorStoreBase:
             raise ImportError("Langchain core components not available for creating retriever.")

        logger.debug(f"Getting Langchain retriever: search_type='{search_type}', search_kwargs={search_kwargs}")
        try:
            effective_search_kwargs = search_kwargs if search_kwargs is not None else {}
            if search_type not in ["similarity", "mmr"]:
                 logger.warning(f"Unsupported search_type '{search_type}'. Defaulting to 'similarity'.")
                 search_type = "similarity"
            # Ensure 'k' is present
            if "k" not in effective_search_kwargs:
                 # Use k from config if available, else default
                 # Cannot access self.config here easily, use a reasonable default
                 effective_search_kwargs["k"] = 4
                 logger.debug(f"Defaulting retriever 'k' to {effective_search_kwargs['k']}")

            retriever = self.client.as_retriever(
                search_type=search_type,
                search_kwargs=effective_search_kwargs
            )
            return retriever
        except Exception as e:
            logger.error(f"Failed to get retriever from Chroma client: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to get retriever: {e}") from e