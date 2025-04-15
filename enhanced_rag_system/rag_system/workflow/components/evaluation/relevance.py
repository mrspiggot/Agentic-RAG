# Contains RelevanceEvaluator class
# File: rag_system/workflow/components/evaluation/relevance.py
# Instruction: Replace the entire content of this file.

import logging
from typing import Dict, Any, List, Optional

# Use relative imports
from .base_evaluator import Evaluator, EvaluatorError
from ....data_models.document import Document
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration
from ....llm.interaction import ILLMInteraction, LLMInteractionError

# Attempt safe Pydantic import
try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None # type: ignore
    Field = None # type: ignore
    ValidationError = None # type: ignore
    PYDANTIC_AVAILABLE = False


logger = logging.getLogger(__name__)

# --- Pydantic Model for Structured Output ---
# Define outside the class if needed elsewhere, or inside if only used here.
# Needs BaseModel imported or available globally. Check imports.
if PYDANTIC_AVAILABLE:
    class RelevanceScore(BaseModel):
        """Schema for relevance evaluation results, used for structured LLM output."""
        relevance_score: int = Field(
            description="Document relevance to the question on a scale of 1 (irrelevant) to 5 (highly relevant)"
        )
        reasoning: str = Field(
            description="Brief explanation for the relevance score based on the scoring guide provided."
        )
else:
    # Define a fallback if Pydantic isn't available, though it's a core dependency
    RelevanceScore = None


# --- Evaluator Class ---

class RelevanceEvaluator(Evaluator):
    """
    Evaluates the relevance of a single document to a given query using an LLM.
    Assigns a score from 1 to 5 based on provided criteria.
    """

    # System prompt defining the evaluation task and scale
    SYSTEM_PROMPT = """You are an expert evaluator assessing how relevant a retrieved document is to a user's question. Your task is to assign a relevance score from 1 to 5 and provide reasoning for your assessment based *only* on the provided Document and Question.

SCORING GUIDE:
1 - NOT RELEVANT: Contains none of the key concepts in the question or discusses entirely different subject matter.
2 - SLIGHTLY RELEVANT: Mentions some keywords from the question but lacks substantial information or contains only tangential references.
3. MODERATELY RELEVANT: Contains related concepts that partially address the question, provides useful context, or has some applicable insights but doesn't directly answer.
4 - VERY RELEVANT: Addresses major aspects of the question with substantial information, contains specific details, examples, or explanations that illuminate the question.
5 - HIGHLY RELEVANT: Comprehensively addresses the question (or a significant sub-part) with specific and detailed information, provides direct answers or strong supporting evidence/reasoning.

IMPORTANT:
- Evaluate based *solely* on the provided Document Content and the User Question.
- A document can be highly relevant (4-5) even if it only answers part of the question well.
- Consider direct and indirect relevance.
- Assign a score from 1 to 5 and provide concise reasoning referencing the document content and the scoring guide.
"""

    HUMAN_PROMPT_TEMPLATE = "Retrieved document content:\n---\n{document}\n---\n\nUser question: {question}"

    def __init__(self, llm_interaction: ILLMInteraction, relevance_threshold: int = 4):
        """
        Initializes the RelevanceEvaluator.

        Args:
            llm_interaction: An initialized instance conforming to the ILLMInteraction interface.
            relevance_threshold: The minimum score (inclusive) for a document to be
                                 considered relevant by default.

        Raises:
            ImportError: If Pydantic is not installed.
            ValueError: If llm_interaction is not provided or threshold is invalid.
        """
        if not PYDANTIC_AVAILABLE or not BaseModel or not Field or not RelevanceScore:
            raise ImportError("Pydantic package is required for RelevanceEvaluator structured output.")
        if not llm_interaction:
            raise ValueError("RelevanceEvaluator requires a valid ILLMInteraction instance.")
        if not isinstance(relevance_threshold, int) or not (1 <= relevance_threshold <= 5):
             logger.warning(f"Invalid relevance_threshold ({relevance_threshold}). Using default 4.")
             relevance_threshold = 4

        self.llm_interaction = llm_interaction
        self.relevance_threshold = relevance_threshold
        self.prompt = self._create_prompt_template()
        logger.debug(f"RelevanceEvaluator initialized. Threshold: {self.relevance_threshold}")

    def _create_prompt_template(self) -> Any:
        """Creates the ChatPromptTemplate if Langchain is available."""
        # This method could be used if we strictly depend on Langchain prompts
        # For now, we will format the string directly before passing to invoke_structured_output
        # if LANGCHAIN_CORE_AVAILABLE and ChatPromptTemplate:
        #     from langchain_core.prompts import ChatPromptTemplate
        #     return ChatPromptTemplate.from_messages([
        #         ("system", self.SYSTEM_PROMPT),
        #         ("human", self.HUMAN_PROMPT_TEMPLATE)
        #     ])
        # else:
        #     logger.warning("Langchain ChatPromptTemplate not available. Will format strings manually.")
        #     return None # Indicate manual formatting needed
        return None # Formatting will be done manually for now

    def evaluate(self, state: WorkflowState, config: Configuration, **kwargs) -> Dict[str, Any]:
        """
        Evaluates the relevance of a single document passed via kwargs.

        Args:
            state: The current WorkflowState (used primarily for the question and to update url usage).
            config: The application Configuration object (not directly used here but part of interface).
            **kwargs: Must contain 'document' (a Document object) to be evaluated.

        Returns:
            A dictionary containing 'relevance_score' (int), 'reasoning' (str),
            'is_relevant' (bool), and 'document_id' (str). Returns default error
            values if evaluation fails.

        Raises:
            ValueError: If 'document' is not provided in kwargs.
        """
        node_name = "evaluate_relevance" # Approximate node name
        document_to_evaluate: Optional[Document] = kwargs.get('document')
        if not isinstance(document_to_evaluate, Document):
             raise ValueError("RelevanceEvaluator requires 'document' (type Document) in kwargs.")

        question = state.original_question # Evaluate against the original question
        doc_id = document_to_evaluate.id
        doc_content = document_to_evaluate.content
        doc_url = document_to_evaluate.metadata.get('source', 'unknown_source')

        logger.info(f"Evaluating relevance for Doc ID: {doc_id} (Source: {doc_url})")
        state.add_log("INFO", f"Evaluating relevance for Doc ID: {doc_id}", node=node_name)


        # Manually format the prompt input for the LLM interaction
        # Assuming invoke_structured_output can handle a simple dictionary or a string prompt
        # For chat models, often a list of messages or a PromptValue is expected.
        # Let's create a basic prompt structure that might work, assuming interaction layer handles it.
        # This might need adjustment based on the specific ILLMInteraction implementation.
        prompt_input = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "human", "content": self.HUMAN_PROMPT_TEMPLATE.format(document=doc_content, question=question)}
        ]

        try:
            if not RelevanceScore: # Check if Pydantic model loaded
                 raise ImportError("RelevanceScore model not defined due to missing Pydantic.")

            # Invoke LLM for structured output
            result: RelevanceScore = self.llm_interaction.invoke_structured_output(
                prompt=prompt_input,
                output_schema=RelevanceScore
            )

            score = result.relevance_score
            reasoning = result.reasoning
            # Apply configured threshold to determine relevance
            is_relevant = score >= self.relevance_threshold

            logger.info(f"Doc ID {doc_id}: Relevance Score={score}/5. Is Relevant (Threshold {self.relevance_threshold}): {is_relevant}")
            logger.debug(f"Doc ID {doc_id}: Reasoning: {reasoning}")
            state.add_log("DEBUG", f"Doc ID {doc_id}: Score={score}, Relevant={is_relevant}. Reasoning: {reasoning}", node=node_name)

            # Update URL usage tracking in the state
            state.add_url_relevance_score(url=doc_url, score=score)

            return {
                "relevance_score": score,
                "reasoning": reasoning,
                "is_relevant": is_relevant,
                "document_id": doc_id
            }

        except (LLMInteractionError, NotImplementedError, ValidationError) as e:
            logger.error(f"LLM relevance evaluation failed for Doc ID {doc_id}: {e}", exc_info=False)
            state.add_log("ERROR", f"LLM evaluation failed for Doc ID {doc_id}: {e}", node=node_name)
        except Exception as e:
            logger.error(f"Unexpected error during relevance evaluation for Doc ID {doc_id}: {e}", exc_info=True)
            state.add_log("ERROR", f"Unexpected evaluation error for Doc ID {doc_id}: {e}", node=node_name)

        # Return default error dictionary if evaluation failed
        return {
            "relevance_score": 0, # Assign lowest score on error
            "reasoning": "Evaluation failed.",
            "is_relevant": False,
            "document_id": doc_id
        }

    # Note: filter_documents logic is better placed in the RAGWorkflowManager's
    # grading node function, which will call this evaluator's evaluate method
    # iteratively or concurrently for each document.