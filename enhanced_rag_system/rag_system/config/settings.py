# File: rag_system/config/settings.py
# Instruction: Replace the entire content of this file.
#              (Added factual_threshold and quality_threshold settings)

import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional

# Initialize logger for this module
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class Configuration:
    """
    Handles loading and access to all application configuration settings.
    Loads settings primarily from environment variables (.env file).
    Provides default values for core settings if environment variables are not set.
    """

    def __init__(self, dotenv_path: Optional[str] = None):
        """ Initializes the Configuration object. """
        try:
            load_dotenv(dotenv_path=dotenv_path)
            logger.info(f"Loaded .env file from: {dotenv_path or 'default locations'}")
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")

        # --- API Keys ---
        self._openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self._tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
        self._anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        self._google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")

        # --- LLM Settings ---
        self._llm_provider_name: str = os.getenv("LLM_PROVIDER", "openai").lower()
        self._llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._llm_temperature: float = self._get_env_variable_as_float("LLM_TEMPERATURE", 0.0)

        # --- Retrieval Settings ---
        self._retrieval_k: int = self._get_env_variable_as_int("RETRIEVAL_K", 7)
        self._relevance_threshold: int = self._get_env_variable_as_int("RELEVANCE_THRESHOLD", 4)
        self._min_relevance_threshold: int = self._get_env_variable_as_int("MIN_RELEVANCE_THRESHOLD", 2)
        self._web_search_results_k: int = self._get_env_variable_as_int("WEB_SEARCH_RESULTS", 5)

        # --- Document Processing Settings ---
        self._chunk_size: int = self._get_env_variable_as_int("CHUNK_SIZE", 500)
        self._chunk_overlap: int = self._get_env_variable_as_int("CHUNK_OVERLAP", 50)

        # --- Workflow Settings ---
        self._max_transform_attempts: int = self._get_env_variable_as_int("MAX_TRANSFORM_ATTEMPTS", 5)
        # Add thresholds for new evaluators
        self._factual_threshold: int = self._get_env_variable_as_int("FACTUAL_THRESHOLD", 3) # Default 3
        self._quality_threshold: int = self._get_env_variable_as_int("QUALITY_THRESHOLD", 3) # Default 3

        # --- Storage Settings ---
        default_vector_db_path = str(Path(__file__).parent.parent / "chroma_db")
        self._vector_db_path: str = os.getenv("VECTOR_DB_PATH", default_vector_db_path)
        self._collection_name: str = os.getenv("COLLECTION_NAME", "rag-chroma-enhanced")

        # --- Document Source URLs ---
        self._default_urls: List[str] = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            "https://huggingface.co/blog/llm-agents",
        ]
        self._document_urls_str: Optional[str] = os.getenv("DOCUMENT_URLS")

        self._validate_config()
        logger.info("Configuration loaded.")
        logger.debug(f"Configuration details: {self!r}")


    def _get_env_variable_as_int(self, name: str, default: int) -> int:
        value_str = os.getenv(name)
        if value_str is None: return default
        try: return int(value_str)
        except ValueError:
            logger.warning(f"Invalid int value for env var '{name}': '{value_str}'. Using default: {default}.")
            return default

    def _get_env_variable_as_float(self, name: str, default: float) -> float:
        value_str = os.getenv(name)
        if value_str is None: return default
        try: return float(value_str)
        except ValueError:
            logger.warning(f"Invalid float value for env var '{name}': '{value_str}'. Using default: {default}.")
            return default

    def _validate_config(self):
        """Perform basic validation (e.g., check ranges)."""
        # (Validation logic for retrieval k, thresholds, chunking remains the same)
        if self._retrieval_k <= 0: logger.warning(f"RETRIEVAL_K invalid ({self._retrieval_k}). Using 1."); self._retrieval_k = 1
        if not (1 <= self._relevance_threshold <= 5): logger.warning(f"RELEVANCE_THRESHOLD ({self._relevance_threshold}) invalid. Clamping."); self._relevance_threshold = max(1, min(5, self._relevance_threshold))
        if not (1 <= self._min_relevance_threshold <= 5): logger.warning(f"MIN_RELEVANCE_THRESHOLD ({self._min_relevance_threshold}) invalid. Clamping."); self._min_relevance_threshold = max(1, min(5, self._min_relevance_threshold))
        if self._min_relevance_threshold > self._relevance_threshold: logger.warning(f"MIN_RELEVANCE > RELEVANCE threshold. Setting min=relevance."); self._min_relevance_threshold = self._relevance_threshold
        if self._chunk_size <= 0: logger.warning(f"CHUNK_SIZE invalid ({self._chunk_size}). Using 100."); self._chunk_size = 100
        if self._chunk_overlap < 0: logger.warning(f"CHUNK_OVERLAP invalid ({self._chunk_overlap}). Using 0."); self._chunk_overlap = 0
        if self._chunk_overlap >= self._chunk_size: logger.warning(f"CHUNK_OVERLAP >= CHUNK_SIZE. Using 0."); self._chunk_overlap = 0
        # Add validation for new thresholds
        if not (1 <= self._factual_threshold <= 5): logger.warning(f"FACTUAL_THRESHOLD ({self._factual_threshold}) invalid. Clamping."); self._factual_threshold = max(1, min(5, self._factual_threshold))
        if not (1 <= self._quality_threshold <= 5): logger.warning(f"QUALITY_THRESHOLD ({self._quality_threshold}) invalid. Clamping."); self._quality_threshold = max(1, min(5, self._quality_threshold))


    # --- Getter Methods ---
    def get_api_key(self, provider: str) -> Optional[str]:
        provider_lower = provider.lower()
        if provider_lower == "openai": return self._openai_api_key
        elif provider_lower == "tavily": return self._tavily_api_key
        elif provider_lower == "anthropic": return self._anthropic_api_key
        elif provider_lower == "google" or provider_lower == "gemini": return self._google_api_key
        else: logger.warning(f"API key requested for unknown provider: {provider}"); return None
    def get_openai_api_key(self) -> Optional[str]: return self._openai_api_key
    def get_tavily_api_key(self) -> Optional[str]: return self._tavily_api_key
    def get_llm_provider_name(self) -> str: return self._llm_provider_name
    def get_llm_model_name(self) -> str: return self._llm_model
    def get_llm_temperature(self) -> float: return self._llm_temperature
    def get_retrieval_k(self) -> int: return self._retrieval_k
    def get_relevance_threshold(self) -> int: return self._relevance_threshold
    def get_min_relevance_threshold(self) -> int: return self._min_relevance_threshold
    def get_web_search_results_k(self) -> int: return self._web_search_results_k
    def get_chunk_size(self) -> int: return self._chunk_size
    def get_chunk_overlap(self) -> int: return self._chunk_overlap
    def get_max_transform_attempts(self) -> int: return self._max_transform_attempts
    # Add getters for new thresholds
    def get_factual_threshold(self) -> int: return self._factual_threshold
    def get_quality_threshold(self) -> int: return self._quality_threshold
    def get_vector_db_path(self) -> str:
        db_path = Path(self._vector_db_path)
        try: db_path.mkdir(parents=True, exist_ok=True)
        except OSError as e: logger.error(f"Failed to create vector DB directory '{db_path}': {e}")
        return str(db_path)
    def get_collection_name(self) -> str: return self._collection_name
    def get_document_urls(self) -> List[str]:
        if self._document_urls_str:
            urls = [url.strip() for url in self._document_urls_str.split(",") if url.strip()]
            if urls: logger.info(f"Using document URLs from environment variable."); return urls
            else: logger.warning("DOCUMENT_URLS env var set but empty. Using defaults.")
        logger.info("Using default document URLs.")
        return self._default_urls

    def __repr__(self) -> str:
        masked_openai = f"{self._openai_api_key[:5]}..." if self._openai_api_key else "Not Set"
        masked_tavily = f"{self._tavily_api_key[:5]}..." if self._tavily_api_key else "Not Set"
        return (f"{self.__class__.__name__}(provider='{self._llm_provider_name}', model='{self._llm_model}', "
                f"retrieval_k={self._retrieval_k}, relevance_th={self._relevance_threshold}, "
                f"factual_th={self._factual_threshold}, quality_th={self._quality_threshold}, "
                f"db_path='{self._vector_db_path}', collection='{self._collection_name}', "
                f"openai_key='{masked_openai}', tavily_key='{masked_tavily}')")