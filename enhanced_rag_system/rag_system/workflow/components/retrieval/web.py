# File: rag_system/workflow/components/retrieval/web.py
# Instruction: Create this file or replace its entire content.

import os
import logging
from typing import List, Optional, Dict, Any

# Use relative imports
from .base_retriever import Retriever, RetrieverError
from ....config.settings import Configuration
from ....data_models.document import Document
from ....data_models.workflow_state import WorkflowState

# Attempt safe import of Tavily tool
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TavilySearchResults = None # type: ignore
    TAVILY_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebSearchRetriever(Retriever):
    """
    Retrieves information using web search via the Tavily API.
    """

    def __init__(self, config: Configuration):
        """
        Initializes the WebSearchRetriever.

        Args:
            config: The application Configuration object.

        Raises:
            ImportError: If 'tavily-python' or required langchain_community components are not installed.
            RetrieverError: If the Tavily API key is missing.
        """
        if not TAVILY_AVAILABLE or not TavilySearchResults:
            raise ImportError("Packages 'tavily-python' and 'langchain-community' are required for WebSearchRetriever.")

        self.config = config
        self.api_key = self.config.get_tavily_api_key()
        self.k = self.config.get_web_search_results_k()

        # Check if API key is present (either from config reading .env or directly in env)
        if not self.api_key and not os.getenv("TAVILY_API_KEY"):
            logger.error("Tavily API key not found in config or environment variable (TAVILY_API_KEY).")
            raise RetrieverError("Tavily API key is required for WebSearchRetriever but was not found.")
        elif not self.api_key:
            logger.warning("Tavily API key not found directly in config. Relying on TAVILY_API_KEY environment variable.")

        try:
            # Initialize the Tavily search tool from Langchain
            # Note: Langchain tool handles finding key from env if api_key=None
            self.search_tool = TavilySearchResults(api_key=self.api_key, k=self.k)
            logger.info(f"WebSearchRetriever initialized with k={self.k}")
        except Exception as e:
            logger.error(f"Failed to initialize TavilySearchResults tool: {e}", exc_info=True)
            raise RetrieverError(f"Failed to initialize Tavily search tool: {e}") from e


    def retrieve(self, query: str, state: WorkflowState, config: Configuration) -> List[Document]:
        """
        Performs a web search using the Tavily tool.

        Args:
            query: The query string to search the web for.
            state: The current WorkflowState (used for context, logging).
            config: The application Configuration object (k value is read during init).

        Returns:
            A list of Document objects created from the search results.

        Raises:
            RetrieverError: If the web search API call fails.
        """
        node_name = "retrieve_web_search" # Approximate node name
        logger.info(f"Performing web search for query: '{query[:100]}...'")
        state.add_log("INFO", f"Starting web search retrieval.", node=node_name)

        try:
            # Invoke the Tavily tool
            # The tool returns a list of dictionaries
            search_results: List[Dict[str, Any]] = self.search_tool.invoke({"query": query})

            documents: List[Document] = []
            if not isinstance(search_results, list):
                 logger.warning(f"Tavily search returned unexpected type: {type(search_results)}. Expected list.")
                 return []

            # Convert results to our Document format
            for i, result in enumerate(search_results):
                 if not isinstance(result, dict):
                      logger.warning(f"Skipping invalid search result item (not a dict): {result!r}")
                      continue

                 content = result.get("content", "")
                 url = result.get("url", f"tavily_result_{i+1}") # Use URL as source, provide fallback
                 title = result.get("title", f"Web Result {i+1}")

                 # Create metadata dictionary
                 metadata = {
                     "source": url,
                     "title": title,
                     "source_type": "web_search",
                     "search_rank": i + 1,
                     "query": query # Store the query that produced this result
                 }
                 # Create Document object
                 doc = Document(content=content, metadata=metadata)
                 documents.append(doc)

            logger.info(f"Web search retrieval found {len(documents)} results.")
            # Note: Updating WorkflowState (history, url_usage) is handled by the
            # calling node function in RAGWorkflowManager.
            return documents

        except Exception as e:
            logger.error(f"Web search retrieval failed: {e}", exc_info=True)
            state.add_log("ERROR", f"Web search retrieval failed: {e}", node=node_name)
            # Raise error to potentially trigger alternative paths in workflow
            raise RetrieverError(f"Web search failed: {e}") from e

    def get_name(self) -> str:
        return "WebSearchRetriever"