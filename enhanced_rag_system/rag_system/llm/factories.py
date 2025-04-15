# File: rag_system/llm/factories.py
# Instruction: Replace the entire content of this file.

import logging
from typing import Optional

# Use relative imports to access Configuration and provider elements
from ..config.settings import Configuration, ConfigurationError
from .providers.base_provider import ILLMProvider, LLMProviderError
from .providers.openai_provider import OpenAIProvider
# Import the new providers
from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)

class LLMProviderFactory:
    """
    Factory for creating instances of LLM providers based on configuration.
    Follows the Factory pattern.
    """

    @staticmethod
    def create_provider(config: Configuration) -> ILLMProvider:
        """
        Creates an LLM provider instance based on the name specified in the config.

        Args:
            config: The Configuration object holding settings, including the provider name and API keys.

        Returns:
            An instance conforming to the ILLMProvider interface.

        Raises:
            ConfigurationError: If the specified provider name is missing or unsupported.
            LLMProviderError: If the provider cannot be instantiated (e.g., missing API key required by the provider class).
        """
        provider_name = config.get_llm_provider_name()
        if not provider_name:
            logger.error("LLM Provider name not specified in configuration.")
            raise ConfigurationError("LLM_PROVIDER name is missing in configuration.")

        logger.info(f"Attempting to create LLM provider for: '{provider_name}'")
        provider_name_lower = provider_name.lower()

        # Retrieve the specific API key for the selected provider from Configuration
        # The concrete Provider class will handle logic if key is None vs empty string vs value
        api_key = config.get_api_key(provider_name_lower)

        try:
            if provider_name_lower == OpenAIProvider.PROVIDER_NAME:
                return OpenAIProvider(api_key=api_key)
            elif provider_name_lower == AnthropicProvider.PROVIDER_NAME:
                 # AnthropicProvider checks for key presence (itself or env var) in __init__
                 return AnthropicProvider(api_key=api_key)
            elif provider_name_lower == GeminiProvider.PROVIDER_NAME:
                 # GeminiProvider checks for key presence (itself or env var) in __init__
                 return GeminiProvider(api_key=api_key)
            # Add elif blocks here for future providers
            else:
                logger.error(f"Unsupported LLM provider specified: '{provider_name}'")
                # Raise ConfigurationError as it's an unsupported config value
                raise ConfigurationError(f"Unsupported LLM provider configured: {provider_name}")

        except ImportError as e:
             logger.error(f"Missing dependency for provider '{provider_name}': {e}", exc_info=True)
             # Raise error indicating setup issue
             raise LLMProviderError(f"Cannot create provider '{provider_name}' due to missing dependency. Please install the required package. Error: {e}") from e
        except LLMProviderError as e: # Catch errors raised during provider init (like missing keys)
             logger.error(f"Error initializing provider '{provider_name}': {e}", exc_info=False) # Log less verbosely
             raise # Re-raise specific provider errors
        except Exception as e:
             logger.error(f"Unexpected error creating provider '{provider_name}': {e}", exc_info=True)
             raise LLMProviderError(f"Unexpected failure creating provider '{provider_name}': {e}") from e