# File: rag_system/llm/factories.py
# (Complete file content with modified create_provider method)

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

    # *** MODIFIED FUNCTION: create_provider ***
    @staticmethod
    def create_provider(config: Configuration, provider_name_override: Optional[str] = None) -> ILLMProvider:
        """
        Creates an LLM provider instance based on the provided override name or the name specified in the config.

        Args:
            config: The Configuration object holding settings (used for API keys and default provider).
            provider_name_override: If provided, use this name. Otherwise, use the default from config.

        Returns:
            An instance conforming to the ILLMProvider interface.

        Raises:
            ConfigurationError: If the specified provider name is missing or unsupported.
            LLMProviderError: If the provider cannot be instantiated (e.g., missing API key required by the provider class).
        """
        # Determine the effective provider name to use
        target_provider_name = provider_name_override if provider_name_override else config.get_llm_provider_name()

        if not target_provider_name:
            logger.error("LLM Provider name could not be determined (neither override nor config default found).")
            raise ConfigurationError("LLM_PROVIDER name is missing or could not be determined.")

        logger.info(f"Attempting to create LLM provider for effective name: '{target_provider_name}'")
        provider_name_lower = target_provider_name.lower()

        # Retrieve the specific API key for the TARGET provider from Configuration
        # The concrete Provider class will handle logic if key is None vs empty string vs value
        api_key = config.get_api_key(provider_name_lower) # Get key for the target provider

        try:
            # Instantiate the correct provider based on the target name
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
                logger.error(f"Unsupported LLM provider specified: '{target_provider_name}'")
                raise ConfigurationError(f"Unsupported LLM provider configured: {target_provider_name}")

        except ImportError as e:
             logger.error(f"Missing dependency for provider '{target_provider_name}': {e}", exc_info=True)
             raise LLMProviderError(f"Cannot create provider '{target_provider_name}' due to missing dependency. Please install the required package. Error: {e}") from e
        except LLMProviderError as e: # Catch errors raised during provider init (like missing keys)
             logger.error(f"Error initializing provider '{target_provider_name}': {e}", exc_info=False)
             raise # Re-raise specific provider errors
        except Exception as e:
             logger.error(f"Unexpected error creating provider '{target_provider_name}': {e}", exc_info=True)
             raise LLMProviderError(f"Unexpected failure creating provider '{target_provider_name}': {e}") from e