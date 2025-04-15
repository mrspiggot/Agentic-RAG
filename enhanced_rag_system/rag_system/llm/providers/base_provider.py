# Defines ILLMProvider (Interface)
# File: rag_system/llm/providers/base_provider.py
# Instruction: Replace the entire content of this file.

import abc
from typing import List, Optional

# Use relative import to avoid circular dependencies
from ..interaction import ILLMInteraction

class LLMProviderError(Exception):
    """Custom exception for LLM Provider errors."""
    pass

class ILLMProvider(abc.ABC):
    """
    Interface defining the contract for an LLM provider.
    Handles provider-specific setup, model listing, and interaction object creation.
    """

    @abc.abstractmethod
    def get_provider_name(self) -> str:
        """
        Returns the unique name identifier for this provider (e.g., 'openai').
        Should be lowercase.
        """
        pass

    @abc.abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Returns a list of model name strings available through this provider
        with the current credentials/configuration.
        """
        pass

    @abc.abstractmethod
    def create_interaction(self, model_name: str, **kwargs) -> ILLMInteraction:
        """
        Factory method to create an interaction object for a specific model.

        Args:
            model_name: The name of the model to interact with (must be one listed by get_available_models).
            **kwargs: Additional configuration for the interaction (e.g., temperature, passed from Configuration).

        Returns:
            An instance conforming to the ILLMInteraction interface.

        Raises:
            ValueError: If the model_name is not supported by this provider.
            LLMProviderError: If authentication fails or the interaction object cannot be created.
        """
        pass