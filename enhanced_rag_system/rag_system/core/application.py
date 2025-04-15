# File: rag_system/core/application.py
# Instruction: Modify the indicated line within the initialize_corpus method.

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
    # (__init__ method remains the same)
    def __init__(self, config: Configuration):
        if not isinstance(config, Configuration):
            raise TypeError("RAGApplication requires a valid Configuration object.")
        self.config = config
        self.doc_corpus: Optional[DocumentCorpus] = None
        self.workflow_manager: Optional[RAGWorkflowManager] = None
        try:
            logger.info("Initializing RAGApplication components...")
            self.doc_corpus = DocumentCorpus(config=self.config)
            llm_provider_factory = LLMProviderFactory()
            self.workflow_manager = RAGWorkflowManager(
                config=self.config,
                doc_corpus=self.doc_corpus,
                llm_provider_factory=llm_provider_factory
            )
            logger.info("RAGApplication initialized successfully.")
        except (CorpusManagerError, WorkflowError) as e:
            logger.error(f"Failed to initialize RAGApplication: {e}", exc_info=True)
            raise
        except Exception as e:
             logger.error(f"Unexpected error initializing RAGApplication: {e}", exc_info=True)
             raise


    def initialize_corpus(self, url_file_path: Optional[str] = None):
        """
        Initializes or builds the document corpus index.
        (Docstring remains the same)
        """
        if not self.doc_corpus:
             raise CorpusManagerError("DocumentCorpus is not initialized.")

        logger.info(f"Starting corpus initialization/build...")
        data_source: Optional[DataSource] = None # Initialize
        if url_file_path:
            logger.info(f"Using URL file source: {url_file_path}")
            try:
                data_source = URLFileSource(file_path=url_file_path)
            except FileNotFoundError:
                 logger.error(f"URL file not found: {url_file_path}")
                 raise # Re-raise FileNotFoundError
        else:
            logger.warning("No URL file path provided to initialize_corpus. Using URLs from configuration.")
            urls_from_config = self.config.get_document_urls()
            if not urls_from_config:
                 logger.error("No URL file provided and no URLs found in configuration. Cannot build index.")
                 raise ValueError("No document sources specified.")
            from ..corpus.datasources.base_datasource import DataSource as BaseDataSource
            class ConfigDataSource(BaseDataSource):
                 def __init__(self, urls): self._urls = urls
                 def get_urls(self) -> List[str]: return self._urls
            data_source = ConfigDataSource(urls_from_config)

        # *** CORRECTED INSTANTIATION ***
        # Remove the incorrect 'urls=[]' argument
        loader = WebDocumentLoader()
        # Pass other relevant WebLoader init args from config if needed, e.g.:
        # loader = WebDocumentLoader(request_headers=self.config.get_custom_headers()) # Example

        splitter = TextSplitter(
            chunk_size=self.config.get_chunk_size(),
            chunk_overlap=self.config.get_chunk_overlap()
        )

        try:
            self.doc_corpus.build_index(
                data_source=data_source,
                loader=loader,
                splitter=splitter
            )
            logger.info("Corpus initialization/build completed.")
        except (CorpusManagerError, FileNotFoundError, ValueError) as e:
             logger.error(f"Corpus initialization failed: {e}", exc_info=True)
             raise
        except Exception as e:
             logger.error(f"Unexpected error during corpus initialization: {e}", exc_info=True)
             raise CorpusManagerError(f"Unexpected corpus initialization error: {e}") from e

    # (process_question method remains the same)
    def process_question(self, question: str) -> Result:
        if not question:
             logger.error("Cannot process empty question.")
             raise ValueError("Question cannot be empty.")
        if not self.workflow_manager:
            raise WorkflowError("WorkflowManager is not initialized.")
        if not self.doc_corpus or not self.doc_corpus.is_ready():
            logger.warning("Document corpus is not marked as ready. Workflow might lack context.")

        logger.info(f"Processing question: '{question[:100]}...'")
        try:
            initial_urls = self.config.get_document_urls()
            result = self.workflow_manager.run_workflow(
                 question=question,
                 initial_source_urls=initial_urls
            )
            logger.info("Question processing completed.")
            return result
        except WorkflowError as e:
             logger.error(f"Workflow execution failed: {e}", exc_info=True)
             raise
        except Exception as e:
             logger.error(f"Unexpected error during question processing: {e}", exc_info=True)
             raise WorkflowError(f"Unexpected question processing error: {e}") from e