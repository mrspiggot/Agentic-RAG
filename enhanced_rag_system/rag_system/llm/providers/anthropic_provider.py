# Placeholder for AnthropicProvider
# File: rag_system/llm/providers/anthropic_provider.py
# Instruction: Create this new file with the following content.

import logging
from typing import List, Optional, Dict, Any

# Attempt to import LangChain elements safely
try:
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_ANTHROPIC_AVAILABLE = True
except ImportError:
    ChatAnthropic = None # type: ignore
    LANGCHAIN_ANTHROPIC_AVAILABLE = False

# Use relative imports for base classes and interaction module
from .base_provider import ILLMProvider, LLMProviderError
from ..interaction import ILLMInteraction, LangchainLLMInteraction, LLMInteractionError

logger = logging.getLogger(__name__)

class AnthropicProvider(ILLMProvider):
    """
    LLM Provider implementation for Anthropic (Claude) models using Langchain.
    """
    PROVIDER_NAME = "anthropic"

    # Common Anthropic models (as of early 2025, check Anthropic docs for current)
    _COMMON_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AnthropicProvider.

        Args:
            api_key: The Anthropic API key. If None, it's expected to be set in the
                     environment (ANTHROPIC_API_KEY).

        Raises:
            ImportError: If langchain-anthropic is not installed.
            LLMProviderError: If the API key is explicitly required but missing.
        """
        if not LANGCHAIN_ANTHROPIC_AVAILABLE or not ChatAnthropic:
            raise ImportError("langchain-anthropic package is required for AnthropicProvider.")

        self.api_key = api_key
        # Anthropic client within Langchain might strictly require the key.
        # Let's check explicitly, unlike OpenAI which has stronger env var fallback handling.
        # Check if key is None *and* environment variable is not set.
        if not self.api_key and not os.getenv("ANTHROPIC_API_KEY"):
             logger.error("Anthropic API key not provided during initialization and ANTHROPIC_API_KEY environment variable is not set.")
             # Decide whether to raise error now or let ChatAnthropic potentially fail later.
             # Let's raise early for clarity.
             raise LLMProviderError("Anthropic API key is required but was not provided and ANTHROPIC_API_KEY env var is not set.")
        elif not self.api_key:
             logger.warning("Anthropic API key not provided during initialization. Relying on environment variable (ANTHROPIC_API_KEY).")

        logger.debug("AnthropicProvider initialized.")

    def get_provider_name(self) -> str:
        """Returns the provider name."""
        return self.PROVIDER_NAME

    def get_available_models(self) -> List[str]:
        """Returns a predefined list of common Anthropic models."""
        logger.debug(f"Returning static list of available Anthropic models: {self._COMMON_MODELS}")
        return self._COMMON_MODELS

    def create_interaction(self, model_name: str, **kwargs) -> ILLMInteraction:
        """
        Creates a LangchainLLMInteraction for the specified Anthropic model.

        Args:
            model_name: The specific Anthropic model name (e.g., "claude-3-sonnet-20240229").
            **kwargs: Additional arguments passed directly to ChatAnthropic constructor
                      (e.g., temperature, max_tokens_to_sample).

        Returns:
            An instance of LangchainLLMInteraction wrapping ChatAnthropic.

        Raises:
            ValueError: If the model_name is not recognized (currently based on static list).
            LLMProviderError: If ChatAnthropic instantiation fails (e.g., bad API key).
        """
        # Optional: Validate model_name against get_available_models() list
        # if model_name not in self.get_available_models():
        #     logger.error(f"Model '{model_name}' is not in the known list for AnthropicProvider.")
        #     raise ValueError(f"Unknown or unsupported Anthropic model: {model_name}")

        logger.debug(f"Creating LLM interaction for Anthropic model: {model_name} with kwargs: {kwargs}")
        try:
            # Instantiate the Langchain ChatAnthropic model
            chat_model_instance = ChatAnthropic(
                model=model_name, # Note: parameter name is 'model' for ChatAnthropic
                anthropic_api_key=self.api_key, # Pass key if provided, otherwise relies on env var
                **kwargs # Pass through other settings like temperature
            )

            # Wrap it in our interaction layer
            interaction = LangchainLLMInteraction(chat_model=chat_model_instance)
            logger.info(f"Successfully created ILLMInteraction for Anthropic model: {model_name}")
            return interaction

        except Exception as e:
            logger.error(f"Failed to create ChatAnthropic instance for model '{model_name}': {e}", exc_info=True)
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                 raise LLMProviderError(f"Anthropic authentication failed. Check API key. Error: {e}") from e
            raise LLMProviderError(f"Failed to create Anthropic interaction for model '{model_name}': {e}") from e

# Required import for the API key check within __init__
import os