# File: rag_system/workflow/components/evaluation/factual.py
# Instruction: Replace the entire content of this file.
#              (Removed try/except for Pydantic import)

import logging
from typing import Dict, Any, List, Optional, Type

# --- Core Dependencies (Direct Imports) ---
from pydantic import BaseModel, Field, ValidationError # Import directly

# --- Relative Imports ---
from .base_evaluator import Evaluator, EvaluatorError
from ....data_models.document import Document
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration
from ....llm.interaction import ILLMInteraction, LLMInteractionError


logger = logging.getLogger(__name__)

# --- Pydantic Model for Factual Score ---
class FactualScore(BaseModel):
    """Schema for factual grounding evaluation results."""
    factual_score: int = Field(
        description="Factual grounding score on a scale of 1 (contains hallucinations) to 5 (fully grounded in facts)"
    )
    reasoning: str = Field(
        description="Explanation of which parts are well-grounded and which parts might not be supported by the documents"
    )

# --- Evaluator Class ---

class FactualGroundingEvaluator(Evaluator):
    """
    Evaluates the factual grounding of a generated answer against provided documents.
    Uses an LLM to assign a score from 1 (hallucinated) to 5 (fully grounded).
    """

    SYSTEM_PROMPT = """You are an expert evaluator assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Rate the factual grounding of the generation on a scale of 1-5:

1 - MOSTLY HALLUCINATED:
   * Contains multiple statements not supported by the documents
   * Makes specific claims that contradict the documents
   * Invents details, statistics, or quotes not present in any document

2 - PARTIALLY HALLUCINATED:
   * Contains some statements not supported by the documents
   * May include reasonable inferences but goes too far beyond what the documents state
   * Mixes factual information with unsupported claims

3 - SOMEWHAT GROUNDED:
   * Most high-level claims are supported by the documents
   * Contains some reasonable inferences that go slightly beyond the documents
   * May include minor details not explicitly in the documents

4 - WELL GROUNDED:
   * Almost all statements are supported by the documents
   * Makes appropriate inferences that stay close to the information provided
   * Any extrapolation is reasonable and clearly follows from the documents

5 - COMPLETELY GROUNDED:
   * Every statement is directly supported by information in the documents
   * Makes no claims beyond what the documents state
   * All details, examples, and explanations come directly from the documents

In your evaluation, identify which specific parts of the generation are well-grounded and which parts might not be fully supported by the documents.
"""

    HUMAN_PROMPT_TEMPLATE = "Set of facts:\n---\n{documents}\n---\n\nLLM generation:\n---\n{generation}\n---"

    def __init__(self, llm_interaction: ILLMInteraction, factual_threshold: int = 3):
        """
        Initializes the FactualGroundingEvaluator.

        Args:
            llm_interaction: An initialized instance conforming to the ILLMInteraction interface.
            factual_threshold: The minimum score (inclusive) for an answer to be considered
                               factually grounded enough. Defaults to 3.

        Raises:
            ValueError: If llm_interaction is not provided or threshold is invalid.
        """
        # Removed check for PYDANTIC_AVAILABLE
        if not llm_interaction:
            raise ValueError("FactualGroundingEvaluator requires a valid ILLMInteraction instance.")
        if not isinstance(factual_threshold, int) or not (1 <= factual_threshold <= 5):
             logger.warning(f"Invalid factual_threshold ({factual_threshold}). Using default 3.")
             factual_threshold = 3

        self.llm_interaction = llm_interaction
        self.factual_threshold = factual_threshold
        logger.debug(f"FactualGroundingEvaluator initialized. Threshold: {self.factual_threshold}")


    def _format_docs_for_eval(self, documents: List[Document]) -> str:
        """Helper method to format documents for the evaluation prompt context."""
        if not documents:
            return "No supporting documents provided."
        formatted_docs = []
        for i, doc in enumerate(documents):
             header = f"--- Document {i+1} (ID: {doc.id}, Source: {doc.metadata.get('source', 'unknown')}) ---"
             formatted_docs.append(f"{header}\n{doc.content}")
        return "\n\n".join(formatted_docs)


    def evaluate(self, state: WorkflowState, config: Configuration, **kwargs) -> Dict[str, Any]:
        """
        Evaluates the factual grounding of the answer stored in the WorkflowState.
        (Docstring remains the same)
        """
        node_name = "evaluate_factual"
        generation = state.answer
        context_docs = state.documents

        if not generation:
             state.add_log("WARNING", "No generation (answer) found in state to evaluate.", node=node_name)
             result_dict = { "factual_score": 0, "reasoning": "No answer generated to evaluate.", "is_factual": False, "error": "Missing generation" }
             state.add_evaluation_result("factual", result_dict)
             return result_dict

        if not context_docs:
            state.add_log("WARNING", "No context documents found in state for factual evaluation.", node=node_name)
            result_dict = { "factual_score": 1, "reasoning": "No context documents provided to evaluate grounding.", "is_factual": False, "error": "Missing context" }
            state.add_evaluation_result("factual", result_dict)
            return result_dict

        logger.info(f"Evaluating factual grounding for answer (length {len(generation)})...")
        state.add_log("INFO", "Starting factual grounding evaluation.", node=node_name)

        docs_text = self._format_docs_for_eval(context_docs)

        prompt_input = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "human", "content": self.HUMAN_PROMPT_TEMPLATE.format(documents=docs_text, generation=generation)}
        ]

        result_dict: Dict[str, Any] = {}
        try:
            # FactualScore class is imported directly at top now
            score_result: FactualScore = self.llm_interaction.invoke_structured_output(
                prompt=prompt_input,
                output_schema=FactualScore
            )

            score = score_result.factual_score
            reasoning = score_result.reasoning
            is_factual = score >= self.factual_threshold

            logger.info(f"Factual grounding evaluation result: Score={score}/5. Is Factual (Threshold {self.factual_threshold}): {is_factual}")
            logger.debug(f"Factual grounding reasoning: {reasoning}")
            state.add_log("DEBUG", f"Factual Score={score}, Is Factual={is_factual}. Reasoning: {reasoning}", node=node_name)

            result_dict = {
                "factual_score": score,
                "reasoning": reasoning,
                "is_factual": is_factual
            }

        except (LLMInteractionError, NotImplementedError, ValidationError) as e:
            logger.error(f"LLM factual evaluation failed: {e}", exc_info=False)
            state.add_log("ERROR", f"LLM factual evaluation failed: {e}", node=node_name)
            result_dict = {"factual_score": 0, "reasoning": f"Evaluation Error: {e}", "is_factual": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during factual evaluation: {e}", exc_info=True)
            state.add_log("ERROR", f"Unexpected factual evaluation error: {e}", node=node_name)
            result_dict = {"factual_score": 0, "reasoning": f"Unexpected Evaluation Error: {e}", "is_factual": False, "error": str(e)}

        state.add_evaluation_result("factual", result_dict)
        return result_dict