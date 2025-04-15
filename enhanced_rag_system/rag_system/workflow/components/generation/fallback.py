# Contains FallbackGenerator class
# File: rag_system/workflow/components/generation/fallback.py
# Instruction: Create this file with the following content.

import logging
from typing import Dict, Any, Tuple, Optional

# --- Relative Imports ---
from .base_generator import Generator, GeneratorError
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration
from ....llm.interaction import ILLMInteraction # Imported but not used

logger = logging.getLogger(__name__)

class FallbackGenerator(Generator):
    """
    A simple generator that provides a predefined fallback answer when
    the main generation or evaluation process indicates a failure or
    inability to provide a satisfactory response.
    """

    DEFAULT_FALLBACK_MESSAGE = ("I cannot answer this question reliably based on the "
                                "available information or quality checks. Please "
                                "try rephrasing your question or asking about "
                                "a different topic.")

    def __init__(self, llm_interaction: Optional[ILLMInteraction] = None):
        """
        Initializes the FallbackGenerator.

        Args:
            llm_interaction: Not used by this generator, included for interface consistency.
        """
        # Does not require LLM interaction
        logger.debug("FallbackGenerator initialized.")


    def generate(self, state: WorkflowState, config: Configuration, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Generates the predefined fallback message.

        Args:
            state: The current WorkflowState.
            config: The application Configuration object.
            **kwargs: Not used by this generator.

        Returns:
            A tuple containing the fallback message string and associated metadata.
            Updates the state with the fallback answer and metadata.
        """
        node_name = "fallback_generate" # Approximate node name
        logger.warning("Executing FallbackGenerator.")
        state.add_log("WARNING", "Generating fallback answer due to previous failure or low evaluation score.", node=node_name)

        fallback_answer = self.DEFAULT_FALLBACK_MESSAGE
        metadata = {
            "generator_type": "fallback",
            "reason": "Previous step indicated failure or low quality.",
            "fallback_used": True, # Explicitly mark fallback usage
            # Include previous evaluation scores if available in state?
            "evaluation_results_at_fallback": state.evaluation_results,
        }

        # Update the state directly (although the engine node usually calls set_final_answer)
        # This ensures the state reflects the fallback even if used outside the main engine call pattern
        state.set_final_answer(fallback_answer, metadata)

        return fallback_answer, metadata