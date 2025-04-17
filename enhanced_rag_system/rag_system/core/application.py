# File: rag_system/core/application.py
# (Complete file content adding provider/model overrides to process_question)

import logging
from typing import Optional, List

# Relative imports
from ..config.settings import Configuration
from ..corpus.corpus_manager import DocumentCorpus, CorpusManagerError
from ..corpus.datasources.base_datasource import DataSource # Interface
from ..corpus.datasources.file_datasource import URLFileSource # Concrete
from ..corpus.loaders.base_loader import DocumentLoader # Interface
from ..corpus.loaders.web_loader import WebDocumentLoader # Concrete
from ..corpus.splitters.text_splitter import TextSplitter # Concrete for now
from ..workflow.engine import RAGWorkflowManager, WorkflowError
from ..llm.factories import LLMProviderFactory
from ..data_models.result import Result

logger = logging.getLogger(__name__)

class RAGApplication:
    """
    Main application class orchestrating the RAG system components.
    Initializes configuration, corpus manager, workflow manager, and handles
    high-level operations like index building and question processing.
    """
    def __init__(self, config: Configuration):
        if not isinstance(config, Configuration):
            raise TypeError("RAGApplication requires a valid Configuration object.")
        self.config = config
        self.doc_corpus: Optional[DocumentCorpus] = None
        self.workflow_manager: Optional[RAGWorkflowManager] = None # Default workflow manager
        self.llm_provider_factory = LLMProviderFactory() # Keep factory instance

        try:
            logger.info("Initializing RAGApplication components...")
            self.doc_corpus = DocumentCorpus(config=self.config)
            # Initialize the *default* workflow manager based on config at startup
            self.workflow_manager = RAGWorkflowManager(
                config=self.config,
                doc_corpus=self.doc_corpus,
                llm_provider_factory=self.llm_provider_factory
                # Rely on default factories within RAGWorkflowManager if not passed
            )
            logger.info("RAGApplication initialized successfully with default workflow manager.")
        except (CorpusManagerError, WorkflowError) as e:
            logger.error(f"Failed to initialize RAGApplication: {e}", exc_info=True)
            raise
        except Exception as e:
             logger.error(f"Unexpected error initializing RAGApplication: {e}", exc_info=True)
             raise


    def initialize_corpus(self, url_file_path: Optional[str] = None):
        """
        Initializes or builds the document corpus index.
        Uses the document corpus instance created during application init.
        """
        if not self.doc_corpus:
             raise CorpusManagerError("DocumentCorpus is not initialized.")

        logger.info(f"Starting corpus initialization/build...")
        data_source: Optional[DataSource] = None
        if url_file_path:
            logger.info(f"Using URL file source: {url_file_path}")
            try: data_source = URLFileSource(file_path=url_file_path)
            except FileNotFoundError: logger.error(f"URL file not found: {url_file_path}"); raise
        else:
            logger.warning("No URL file path provided to initialize_corpus. Using URLs from configuration.")
            urls_from_config = self.config.get_document_urls()
            if not urls_from_config:
                 logger.error("No URL file provided and no URLs found in configuration. Cannot build index.")
                 raise ValueError("No document sources specified.")
            # Use a simple inline class or function for ConfigDataSource if needed
            from ..corpus.datasources.base_datasource import DataSource as BaseDataSource
            class ConfigDataSource(BaseDataSource):
                 def __init__(self, urls): self._urls = urls
                 def get_urls(self) -> List[str]: return self._urls
            data_source = ConfigDataSource(urls_from_config)

        # Use standard Web Loader and Text Splitter based on config
        loader = WebDocumentLoader() # Can add headers/kwargs from config if needed
        splitter = TextSplitter(
            chunk_size=self.config.get_chunk_size(),
            chunk_overlap=self.config.get_chunk_overlap()
        )

        try:
            # Build index using the application's doc_corpus instance
            self.doc_corpus.build_index(
                data_source=data_source,
                loader=loader,
                splitter=splitter
            )
            logger.info("Corpus initialization/build completed.")
        except (CorpusManagerError, FileNotFoundError, ValueError) as e:
             logger.error(f"Corpus initialization failed: {e}", exc_info=True); raise
        except Exception as e:
             logger.error(f"Unexpected error during corpus initialization: {e}", exc_info=True)
             raise CorpusManagerError(f"Unexpected corpus initialization error: {e}") from e

    # *** MODIFIED FUNCTION: process_question ***
    def process_question(self,
                         question: str,
                         selected_provider: Optional[str] = None,
                         selected_model: Optional[str] = None
                         ) -> Result:
        """
        Processes a user question through the RAG workflow.

        Allows dynamic selection of LLM provider and model, overriding defaults if provided.

        Args:
            question: The user's question string.
            selected_provider: The LLM provider name selected by the user (e.g., 'openai').
            selected_model: The specific LLM model name selected by the user (e.g., 'gpt-4o-mini').

        Returns:
            A Result object containing the answer and execution details.

        Raises:
            ValueError: If the question is empty.
            WorkflowError: If the workflow manager cannot be initialized or execution fails.
        """
        if not question:
             logger.error("Cannot process empty question.")
             raise ValueError("Question cannot be empty.")

        if not self.doc_corpus or not self.doc_corpus.is_ready():
            # Check corpus readiness before proceeding further
            logger.error("Document corpus is not initialized or ready. Please build index first.")
            raise WorkflowError("Corpus not ready. Cannot process question.")

        # Determine which workflow manager instance to use
        target_workflow_manager: Optional[RAGWorkflowManager] = None

        if selected_provider and selected_model:
            # If user made a selection, create a temporary, specifically configured manager
            logger.info(f"User selected LLM: Provider='{selected_provider}', Model='{selected_model}'. Creating temporary workflow manager.")
            try:
                target_workflow_manager = RAGWorkflowManager(
                    config=self.config, # Use main config for keys, etc.
                    doc_corpus=self.doc_corpus, # Share the indexed corpus
                    llm_provider_factory=self.llm_provider_factory,
                    # Pass the overrides to the constructor
                    provider_override=selected_provider,
                    model_override=selected_model
                )
                logger.info("Temporary workflow manager created successfully.")
            except Exception as e:
                 logger.error(f"Failed to create temporary workflow manager for selection: {e}", exc_info=True)
                 # Fall back to default manager or raise error? Let's raise for clarity.
                 raise WorkflowError(f"Failed to configure workflow for selected LLM ({selected_provider}/{selected_model}): {e}") from e
        else:
            # Use the default manager initialized at startup
            logger.info("Using default workflow manager initialized at startup.")
            target_workflow_manager = self.workflow_manager

        if not target_workflow_manager:
             logger.error("Workflow Manager is not available (default or temporary).")
             raise WorkflowError("WorkflowManager could not be initialized or selected.")


        logger.info(f"Processing question: '{question[:100]}...' using workflow manager configured for "
                    f"{selected_provider or self.config.get_llm_provider_name()}/"
                    f"{selected_model or self.config.get_llm_model_name()}")
        try:
            # Get the initial list of URLs used for indexing (or defaults) for tracking
            # This might need refinement if indexing changes dynamically
            initial_urls = self.config.get_document_urls()

            # Run the workflow using the chosen manager instance
            result = target_workflow_manager.run_workflow(
                 question=question,
                 initial_source_urls=initial_urls
            )
            logger.info("Question processing completed.")
            return result
        except WorkflowError as e:
             logger.error(f"Workflow execution failed: {e}", exc_info=True); raise
        except Exception as e:
             logger.error(f"Unexpected error during question processing: {e}", exc_info=True)
             raise WorkflowError(f"Unexpected question processing error: {e}") from e