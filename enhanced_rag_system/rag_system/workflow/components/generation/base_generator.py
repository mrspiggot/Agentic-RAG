# Defines Generator (Interface)
# File: rag_system/workflow/components/generation/base_generator.py
# Instruction: Replace the entire content of this file.

import abc
import logging
from typing import Dict, Any, Tuple, Optional

# Use relative imports for data models and config
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration

logger = logging.getLogger(__name__)

class GeneratorError(Exception):
    """Custom exception for Generator errors."""
    pass

class Generator(abc.ABC):
    """
    Abstract Base Class (Interface) for answer generation strategies.

    Concrete implementations will provide different methods for generating
    an answer string based on the workflow state (e.g., RAG, fallback).
    """

    @abc.abstractmethod
    def generate(self, state: WorkflowState, config: Configuration) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an answer based on the current workflow state and configuration.

        Args:
            state: The current WorkflowState object, providing context (query,
                   relevant documents) and allowing updates (answer, generation_results).
            config: The application Configuration object.

        Returns:
            A tuple containing:
                1. The generated answer string.
                2. A dictionary containing metadata about the generation process
                   (e.g., model used, duration, whether it was a fallback).
            Implementations must handle updating the WorkflowState's 'answer' and
            'generation_results' attributes within this method.

        Raises:
            GeneratorError: If the generation process encounters a critical error.
        """
        pass

    def get_name(self) -> str:
        """
        Returns the identifiable name of the generator strategy.
        Defaults to the class name. Concrete classes can override if needed.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Default representation showing the generator name."""
        return f"<{self.get_name()}>"