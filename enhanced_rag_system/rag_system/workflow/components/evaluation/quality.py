# File: rag_system/workflow/components/evaluation/quality.py
# Instruction: Create this file or replace its entire content.

import logging
from typing import Dict, Any, List, Optional, Type

# --- Core Dependencies (Direct Imports) ---
from pydantic import BaseModel, Field, ValidationError

# --- Relative Imports ---
from .base_evaluator import Evaluator, EvaluatorError
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration
from ....llm.interaction import ILLMInteraction, LLMInteractionError

logger = logging.getLogger(__name__)

# --- Pydantic Model for Quality Score ---
class QualityScore(BaseModel):
    """Schema for answer quality evaluation results."""
    completeness_score: int = Field(
        description="How completely the answer addresses the question on a scale of 1-5"
    )
    clarity_score: int = Field(
        description="Clarity and understandability of the answer on a scale of 1-5"
    )
    usefulness_score: int = Field(
        description="Practical usefulness of the answer for the user's likely need on a scale of 1-5"
    )
    overall_score: int = Field(
        description="Overall quality score considering all aspects on a scale of 1-5"
    )
    analysis: str = Field(
        description="Brief analysis of the answer's strengths and weaknesses based on the criteria."
    )

# --- Evaluator Class ---

class AnswerQualityEvaluator(Evaluator):
    """
    Evaluates the overall quality of a generated answer based on the original question.
    Uses an LLM to assess completeness, clarity, usefulness, and assign an overall score.
    """

    SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of an AI-generated answer to a user's question. Evaluate the answer based *only* on the provided User Question and the Answer, focusing on the following dimensions:

1.  **Completeness (1-5):** How thoroughly does the answer address all explicit and implicit aspects of the question? (1=Missing major parts, 5=Fully comprehensive)
2.  **Clarity (1-5):** How clear, concise, well-structured, and easy to understand is the answer? Is the language precise? (1=Confusing/Ambiguous, 5=Very clear and well-organized)
3.  **Usefulness (1-5):** How practically helpful and relevant is the answer to the user who asked the question? Does it provide actionable information or clear explanations? (1=Not useful, 5=Highly useful)
4.  **Overall (1-5):** Considering all the above, what is the overall quality score?

Provide a score (1-5) for each dimension and a brief analysis summarizing the answer's strengths and weaknesses against these criteria.
"""

    HUMAN_PROMPT_TEMPLATE = "User question:\n---\n{question}\n---\n\nGenerated Answer:\n---\n{generation}\n---"

    def __init__(self, llm_interaction: ILLMInteraction, quality_threshold: int = 3):
        """
        Initializes the AnswerQualityEvaluator.

        Args:
            llm_interaction: An initialized instance conforming to the ILLMInteraction interface.
            quality_threshold: The minimum overall_score (inclusive) for an answer to be
                               considered good quality. Defaults to 3.

        Raises:
            ValueError: If llm_interaction is not provided or threshold is invalid.
        """
        if not llm_interaction:
            raise ValueError("AnswerQualityEvaluator requires a valid ILLMInteraction instance.")
        if not isinstance(quality_threshold, int) or not (1 <= quality_threshold <= 5):
             logger.warning(f"Invalid quality_threshold ({quality_threshold}). Using default 3.")
             quality_threshold = 3

        self.llm_interaction = llm_interaction
        self.quality_threshold = quality_threshold
        # Pydantic schema used directly in evaluate method
        self._quality_schema = QualityScore
        logger.debug(f"AnswerQualityEvaluator initialized. Threshold: {self.quality_threshold}")

    def evaluate(self, state: WorkflowState, config: Configuration, **kwargs) -> Dict[str, Any]:
        """
        Evaluates the quality of the answer stored in the WorkflowState.

        Args:
            state: The current WorkflowState, containing the generated answer (`state.answer`)
                   and the original question (`state.original_question`).
            config: The application Configuration object.
            **kwargs: Not used by this evaluator.

        Returns:
            A dictionary containing scores for completeness, clarity, usefulness,
            overall quality, an analysis string, and 'is_good_quality' (bool).
            Updates state.evaluation_results["quality"].
            Returns default error values if evaluation fails.
        """
        node_name = "evaluate_quality" # Approximate node name
        generation = state.answer
        question = state.original_question

        if not generation:
             state.add_log("WARNING", "No generation (answer) found in state to evaluate quality.", node=node_name)
             result_dict = {"overall_score": 0, "analysis": "No answer generated.", "is_good_quality": False, "error": "Missing generation"}
             state.add_evaluation_result("quality", result_dict)
             return result_dict

        logger.info(f"Evaluating answer quality for answer (length {len(generation)})...")
        state.add_log("INFO", "Starting answer quality evaluation.", node=node_name)

        # Prepare prompt input
        prompt_input = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "human", "content": self.HUMAN_PROMPT_TEMPLATE.format(question=question, generation=generation)}
        ]

        result_dict: Dict[str, Any] = {}
        try:
            if not self._quality_schema: raise ImportError("QualityScore model not defined.")

            # Invoke LLM for structured output
            score_result: QualityScore = self.llm_interaction.invoke_structured_output(
                prompt=prompt_input,
                output_schema=self._quality_schema
            )

            overall_score = score_result.overall_score
            is_good_quality = overall_score >= self.quality_threshold

            logger.info(f"Answer quality evaluation result: Overall Score={overall_score}/5. Is Good Quality (Threshold {self.quality_threshold}): {is_good_quality}")
            logger.debug(f"Quality Analysis: {score_result.analysis}")
            state.add_log("DEBUG", f"Quality Score={overall_score}, Good Quality={is_good_quality}. Analysis: {score_result.analysis}", node=node_name)

            # Convert Pydantic model to dict for storing/returning
            result_dict = score_result.model_dump() # Use .model_dump() for Pydantic v2+
            result_dict["is_good_quality"] = is_good_quality


        except (LLMInteractionError, NotImplementedError, ValidationError) as e:
            logger.error(f"LLM quality evaluation failed: {e}", exc_info=False)
            state.add_log("ERROR", f"LLM quality evaluation failed: {e}", node=node_name)
            result_dict = {"overall_score": 0, "analysis": f"Evaluation Error: {e}", "is_good_quality": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during quality evaluation: {e}", exc_info=True)
            state.add_log("ERROR", f"Unexpected quality evaluation error: {e}", node=node_name)
            result_dict = {"overall_score": 0, "analysis": f"Unexpected Evaluation Error: {e}", "is_good_quality": False, "error": str(e)}

        # Update the evaluation results in the state
        state.add_evaluation_result("quality", result_dict)
        return result_dict