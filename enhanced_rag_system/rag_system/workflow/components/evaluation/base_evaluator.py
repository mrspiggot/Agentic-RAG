# Defines Evaluator (Interface)
# File: rag_system/workflow/components/evaluation/base_evaluator.py
# Instruction: Replace the entire content of this file.

import abc
import logging
from typing import Dict, Any, Optional

# Use relative imports for data models and config
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration

logger = logging.getLogger(__name__)

class EvaluatorError(Exception):
    """Custom exception for Evaluator errors."""
    pass

class Evaluator(abc.ABC):
    """
    Abstract Base Class (Interface) for evaluation components.

    Concrete implementations will provide different methods for assessing
    aspects like document relevance, answer quality, or factual grounding.
    """

    @abc.abstractmethod
    def evaluate(self, state: WorkflowState, config: Configuration, **kwargs) -> Dict[str, Any]:
        """
        Perform an evaluation based on the current workflow state and config.

        Args:
            state: The current WorkflowState object, providing context (documents,
                   query, generated answer) and allowing updates (evaluation_results,
                   url_usage_tracking).
            config: The application Configuration object.
            **kwargs: Allows passing specific data if the evaluator doesn't need the
                      entire state (e.g., a single document for relevance grading).

        Returns:
            A dictionary containing the evaluation results (e.g., scores, reasoning).
            Implementations should handle updating the WorkflowState's
            evaluation_results and potentially url_usage_tracking within this method.

        Raises:
            EvaluatorError: If the evaluation process encounters a critical error.
        """
        pass

    def get_name(self) -> str:
        """
        Returns the identifiable name of the evaluator type.
        Defaults to the class name. Concrete classes can override if needed.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Default representation showing the evaluator name."""
        return f"<{self.get_name()}>"