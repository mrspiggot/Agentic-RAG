# Contains OpenAIProvider class
# File: rag_system/llm/providers/openai_provider.py
# Instruction: Replace the entire content of this file.

import logging
from typing import List, Optional, Dict, Any

# Attempt to import LangChain elements safely
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    ChatOpenAI = None # type: ignore
    LANGCHAIN_OPENAI_AVAILABLE = False

from .base_provider import ILLMProvider, LLMProviderError
# Use relative imports for interaction module
from ..interaction import ILLMInteraction, LangchainLLMInteraction, LLMInteractionError

logger = logging.getLogger(__name__)

class OpenAIProvider(ILLMProvider):
    """
    LLM Provider implementation for OpenAI models using Langchain.
    """
    PROVIDER_NAME = "openai"

    # Predefined list of common models. Could be dynamic later.
    _COMMON_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview", # Alias might exist
        "gpt-4",
        "gpt-3.5-turbo",
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAIProvider.

        Args:
            api_key: The OpenAI API key. If None, it's expected to be set in the environment
                     for Langchain's ChatOpenAI to pick up automatically.

        Raises:
            ImportError: If langchain-openai is not installed.
        """
        if not LANGCHAIN_OPENAI_AVAILABLE or not ChatOpenAI:
            raise ImportError("langchain-openai package is required for OpenAIProvider.")

        self.api_key = api_key
        if not self.api_key:
            logger.warning("OpenAI API key not provided during initialization. Relying on environment variables (OPENAI_API_KEY).")
        logger.debug("OpenAIProvider initialized.")


    def get_provider_name(self) -> str:
        """Returns the provider name."""
        return self.PROVIDER_NAME

    def get_available_models(self) -> List[str]:
        """Returns a predefined list of common OpenAI models."""
        # For now, return a static list.
        # Future enhancement: Could query OpenAI API if a method becomes available,
        # or use a more extensive static list.
        logger.debug(f"Returning static list of available OpenAI models: {self._COMMON_MODELS}")
        return self._COMMON_MODELS

    def create_interaction(self, model_name: str, **kwargs) -> ILLMInteraction:
        """
        Creates a LangchainLLMInteraction for the specified OpenAI model.

        Args:
            model_name: The specific OpenAI model name (e.g., "gpt-4o-mini").
            **kwargs: Additional arguments passed directly to ChatOpenAI constructor
                      (e.g., temperature, max_tokens from Configuration).

        Returns:
            An instance of LangchainLLMInteraction wrapping ChatOpenAI.

        Raises:
            ValueError: If the model_name is not recognized (currently based on static list).
            LLMProviderError: If ChatOpenAI instantiation fails (e.g., bad API key if provided).
        """
        # Optional: Validate model_name against get_available_models() list
        # if model_name not in self.get_available_models():
        #     logger.error(f"Model '{model_name}' is not in the known list for OpenAIProvider.")
        #     raise ValueError(f"Unknown or unsupported OpenAI model: {model_name}")

        logger.debug(f"Creating LLM interaction for OpenAI model: {model_name} with kwargs: {kwargs}")
        try:
            # Instantiate the Langchain ChatOpenAI model
            # Pass the API key if provided during init, otherwise let ChatOpenAI find it
            chat_model_instance = ChatOpenAI(
                model_name=model_name,
                api_key=self.api_key, # Will be None if not provided, ChatOpenAI handles env var
                **kwargs # Pass through other settings like temperature
            )

            # Wrap it in our interaction layer
            interaction = LangchainLLMInteraction(chat_model=chat_model_instance)
            logger.info(f"Successfully created ILLMInteraction for OpenAI model: {model_name}")
            return interaction

        except Exception as e:
            logger.error(f"Failed to create ChatOpenAI instance for model '{model_name}': {e}", exc_info=True)
            # Catch potential authentication errors if API key is explicitly passed and invalid
            if "authentication" in str(e).lower():
                 raise LLMProviderError(f"OpenAI authentication failed. Check API key. Error: {e}") from e
            raise LLMProviderError(f"Failed to create OpenAI interaction for model '{model_name}': {e}") from e