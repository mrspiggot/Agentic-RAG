# Contains DocumentCorpus class
# File: rag_system/corpus/corpus_manager.py
# Instruction: Replace the entire content of this file.

import logging
from typing import Optional, List

# Use relative imports
from ..config.settings import Configuration
from ..data_models.document import Document
from .datasources.base_datasource import DataSource
from .loaders.base_loader import DocumentLoader
from .splitters.text_splitter import TextSplitter # Assuming TextSplitter is concrete or has default impl
from .embedding.base_embedding import EmbeddingModel, EmbeddingError
from .vector_stores.base_vector_store import VectorStore, VectorStoreError

# Import concrete types for default initialization (consider factories later if needed)
from .embedding.openai_embedding import OpenAIEmbeddingModel
from .vector_stores.chroma_vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class CorpusManagerError(Exception):
    """Custom exception for CorpusManager errors."""
    pass

class DocumentCorpus:
    """
    Manages the lifecycle of the document corpus: loading, processing,
    embedding, indexing, and providing access to the vector store.
    """

    def __init__(self,
                 config: Configuration,
                 embedding_model: Optional[EmbeddingModel] = None,
                 vector_store: Optional[VectorStore] = None):
        """
        Initializes the DocumentCorpus manager.

        Sets up the embedding model and vector store based on configuration
        or provided instances.

        Args:
            config: The application Configuration object.
            embedding_model: (Optional) Pre-initialized embedding model instance.
                             If None, creates OpenAIEmbeddingModel based on config.
            vector_store: (Optional) Pre-initialized vector store instance.
                          If None, creates ChromaVectorStore based on config.

        Raises:
            CorpusManagerError: If essential components cannot be initialized.
        """
        self.config = config
        self._is_ready = False # Tracks if the vector store is initialized/loaded

        try:
            # Initialize Embedding Model
            if embedding_model:
                self.embedding_model = embedding_model
                logger.info(f"Using provided embedding model: {embedding_model.get_name()}")
            else:
                # Create default (OpenAI) based on config
                logger.info("Initializing default OpenAI embedding model...")
                openai_api_key = self.config.get_api_key("openai")
                self.embedding_model = OpenAIEmbeddingModel(api_key=openai_api_key)

            # Initialize Vector Store
            if vector_store:
                self.vector_store = vector_store
                logger.info(f"Using provided vector store: {vector_store.get_name()}")
                self._is_ready = True # Assume provided store is ready
            else:
                # Create default (Chroma) based on config
                logger.info("Initializing default Chroma vector store...")
                db_path = self.config.get_vector_db_path()
                collection_name = self.config.get_collection_name()
                self.vector_store = ChromaVectorStore(
                    persist_directory=db_path,
                    collection_name=collection_name,
                    embedding_model=self.embedding_model # Pass our embedding model instance
                )
                # ChromaVectorStore init attempts to load/connect, so consider it ready
                self._is_ready = True

            logger.info("DocumentCorpus initialized successfully.")

        except (EmbeddingError, VectorStoreError, ImportError) as e:
            logger.error(f"Failed to initialize DocumentCorpus components: {e}", exc_info=True)
            self._is_ready = False
            # Depending on severity, could re-raise or allow creation but keep _is_ready=False
            raise CorpusManagerError(f"Failed to initialize corpus components: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error initializing DocumentCorpus: {e}", exc_info=True)
             self._is_ready = False
             raise CorpusManagerError(f"Unexpected error initializing corpus: {e}") from e


    def is_ready(self) -> bool:
        """Checks if the vector store is initialized and ready for use."""
        return self._is_ready

    def get_vector_store(self) -> VectorStore:
        """
        Returns the initialized vector store instance.

        Raises:
            CorpusManagerError: If the vector store is not ready.
        """
        if not self.is_ready() or not self.vector_store:
            raise CorpusManagerError("Vector store is not initialized or ready.")
        return self.vector_store

    def build_index(self,
                    data_source: DataSource,
                    loader: DocumentLoader,
                    splitter: Optional[TextSplitter] = None):
        """
        Builds or updates the vector store index from a data source.

        Orchestrates loading, splitting, embedding, and adding documents.

        Args:
            data_source: The DataSource providing the source identifiers (e.g., URLs).
            loader: The DocumentLoader to fetch content for each identifier.
            splitter: (Optional) The TextSplitter to use. If None, uses a default
                      TextSplitter based on configuration.

        Raises:
            CorpusManagerError: If any step in the indexing pipeline fails.
        """
        if not self.is_ready() or not self.vector_store:
             raise CorpusManagerError("Cannot build index: Vector store not ready.")

        logger.info(f"Starting index build process using source: {type(data_source).__name__}")

        # Initialize default splitter if none provided
        if splitter is None:
             try:
                 chunk_size = self.config.get_chunk_size()
                 chunk_overlap = self.config.get_chunk_overlap()
                 logger.info(f"Using default TextSplitter (size={chunk_size}, overlap={chunk_overlap})")
                 # Assumes TextSplitter is concrete here, adjust if it's an interface/factory needed
                 from .splitters.text_splitter import TextSplitter as DefaultSplitter # Local import
                 splitter = DefaultSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
             except Exception as e:
                  logger.error(f"Failed to initialize default TextSplitter: {e}", exc_info=True)
                  raise CorpusManagerError(f"Failed to initialize TextSplitter: {e}") from e

        try:
            # 1. Get Source Identifiers (URLs)
            source_identifiers = data_source.get_urls()
            if not source_identifiers:
                 logger.warning("Data source provided no identifiers (e.g., URLs). Index build skipped.")
                 return
            logger.info(f"Found {len(source_identifiers)} source identifiers to process.")

            # 2. Load Documents
            # Assuming the loader handles fetching content for the list of identifiers
            # Modify if loader expects one identifier at a time
            logger.info("Loading documents...")
            # WebLoader currently takes list in init, let's adapt to OOD interface
            # For now, assuming loader might need adaptation or we pass identifiers differently
            # Placeholder: Load one by one (less efficient but fits interface) - NEEDS REVISIT
            all_loaded_docs: List[Document] = []
            processed_count = 0
            for identifier in source_identifiers:
                 try:
                      docs = loader.load(identifier) # Load expects one identifier
                      all_loaded_docs.extend(docs)
                      processed_count += 1
                      if processed_count % 10 == 0: # Log progress periodically
                           logger.info(f"Loaded content for {processed_count}/{len(source_identifiers)} sources...")
                 except Exception as load_err:
                      logger.warning(f"Failed to load document from '{identifier}': {load_err}", exc_info=False) # Less verbose for single failures
            logger.info(f"Successfully loaded content, resulting in {len(all_loaded_docs)} initial document sections.")
            if not all_loaded_docs:
                 logger.warning("No documents were successfully loaded. Index build cannot proceed.")
                 return

            # 3. Split Documents
            logger.info("Splitting documents...")
            split_docs = splitter.process(all_loaded_docs)
            logger.info(f"Split documents into {len(split_docs)} chunks.")
            if not split_docs:
                 logger.warning("Splitting resulted in zero chunks. Index build cannot proceed.")
                 return

            # 4. Add to Vector Store (Embeddings handled internally by store/langchain)
            logger.info("Adding document chunks to vector store...")
            added_ids = self.vector_store.add_documents(split_docs)
            logger.info(f"Successfully added {len(added_ids)} chunks to the vector store.")

            logger.info("Index build process completed successfully.")

        except Exception as e:
            logger.error(f"Index build failed during processing: {e}", exc_info=True)
            raise CorpusManagerError(f"Index build failed: {e}") from e

    def __repr__(self) -> str:
        return (f"<DocumentCorpus(ready={self._is_ready}, "
                f"vector_store={self.vector_store.get_name()}, "
                f"embedding_model={self.embedding_model.get_name()})>")