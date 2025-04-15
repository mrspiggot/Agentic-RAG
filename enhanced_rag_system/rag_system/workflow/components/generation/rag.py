# Contains RAGGenerator class
# File: rag_system/workflow/components/generation/rag.py
# Instruction: Replace the entire content of this file.

import logging
import time
from typing import List, Dict, Any, Optional, Tuple

# Use relative imports
from .base_generator import Generator, GeneratorError
from ....data_models.document import Document
from ....data_models.workflow_state import WorkflowState
from ....config.settings import Configuration
from ....llm.interaction import ILLMInteraction, LLMInteractionError

# Attempt safe LangChain import for default prompt template
try:
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    ChatPromptTemplate = None # type: ignore
    LANGCHAIN_CORE_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGGenerator(Generator):
    """
    Generates answers using Retrieval-Augmented Generation (RAG).
    Formats retrieved documents as context for an LLM.
    """

    # Default RAG prompt template instructing the LLM
    DEFAULT_RAG_PROMPT_TEMPLATE = """
You are an AI assistant providing helpful, detailed, and accurate responses based exclusively on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question based *only* on the information present in the CONTEXT above.
2. If the context does not contain the answer, state clearly that the answer is not available in the provided documents. Do not make up information or use external knowledge.
3. Structure your answer clearly. Use headings, bullet points, or numbered lists if appropriate based on the content.
4. Prioritize accuracy and faithfulness to the context. If quoting directly, cite the relevant Document number if possible (though document numbers may not be available in the context formatting).
5. If the context contains conflicting information, point out the discrepancy if relevant to the question.

YOUR ANSWER:
"""

    def __init__(self,
                 llm_interaction: ILLMInteraction,
                 prompt_template_str: Optional[str] = None):
        """
        Initializes the RAGGenerator.

        Args:
            llm_interaction: An initialized instance conforming to the ILLMInteraction interface.
            prompt_template_str: (Optional) A custom prompt template string. If None,
                                 uses DEFAULT_RAG_PROMPT_TEMPLATE. Must contain
                                 placeholders for '{context}' and '{question}'.

        Raises:
            ValueError: If llm_interaction is not provided or if the prompt template is invalid.
            ImportError: If Langchain is needed for the default prompt but not installed.
        """
        if not llm_interaction:
            raise ValueError("RAGGenerator requires a valid ILLMInteraction instance.")
        self.llm_interaction = llm_interaction

        template_str = prompt_template_str or self.DEFAULT_RAG_PROMPT_TEMPLATE

        # Validate template minimally
        if "{context}" not in template_str or "{question}" not in template_str:
            raise ValueError("Prompt template must include '{context}' and '{question}' placeholders.")

        # Store the template string - formatting happens during generation
        self.prompt_template_str = template_str
        logger.debug(f"RAGGenerator initialized with LLM Interaction: {type(llm_interaction).__name__}")


    def _format_docs(self, documents: List[Document]) -> str:
        """Helper method to format documents into a single context string."""
        if not documents:
            return "No context documents provided."
        # Include document ID and source if available in metadata for traceability
        formatted_docs = []
        for i, doc in enumerate(documents):
             source = doc.metadata.get('source', 'unknown')
             doc_id = doc.metadata.get('id', f'doc-{i+1}')
             header = f"--- Document {i+1} (ID: {doc_id}, Source: {source}) ---"
             formatted_docs.append(f"{header}\n{doc.content}")
        return "\n\n".join(formatted_docs)


    def generate(self, state: WorkflowState, config: Configuration) -> Tuple[str, Dict[str, Any]]:
        """
        Generates an answer using RAG based on the current state.

        Args:
            state: The current WorkflowState, containing the question and relevant documents.
            config: The application Configuration object (used for LLM settings like temperature).

        Returns:
            A tuple containing:
                - The generated answer string.
                - A dictionary with generation metadata.

        Raises:
            GeneratorError: If the generation process fails.
        """
        start_time = time.time()
        question = state.original_question # Use original question for generation
        documents = state.documents # Use the current (likely filtered) documents

        logger.info(f"Starting RAG generation for question '{question[:50]}...' using {len(documents)} documents.")

        if not documents:
            logger.warning("No documents provided in state for RAG generation. Returning default message.")
            answer = "Based on the provided context, I cannot answer the question as no relevant documents were found."
            metadata = {
                "generation_time_sec": time.time() - start_time,
                "documents_used_count": 0,
                "model_info": type(self.llm_interaction).__name__, # Basic info
                "error": "No documents provided",
                "is_fallback": True # Treat as a form of fallback
            }
            # Note: The calling node function is responsible for calling state.set_final_answer()
            return answer, metadata

        # Format documents into context
        context_str = self._format_docs(documents)

        # Prepare the final prompt string
        try:
             final_prompt = self.prompt_template_str.format(context=context_str, question=question)
        except KeyError as e:
             logger.error(f"Failed to format prompt template. Missing key: {e}")
             raise GeneratorError(f"Prompt template formatting error: Missing key {e}")

        # Get LLM settings from config
        temperature = config.get_llm_temperature()
        # Add other potential LLM params from config if needed (e.g., max_tokens)

        try:
            # Invoke the LLM via the interaction interface
            logger.debug(f"Invoking LLM with temperature={temperature}...")
            answer = self.llm_interaction.invoke(
                prompt=final_prompt,
                temperature=temperature # Pass temperature if invoke supports it in kwargs
                # Add other kwargs as needed based on ILLMInteraction implementation
            )
            logger.info(f"RAG generation successful. Answer length: {len(answer)}")

            generation_time = time.time() - start_time
            metadata = {
                "generation_time_sec": round(generation_time, 3),
                "documents_used_count": len(documents),
                "document_ids": [doc.id for doc in documents],
                "model_info": type(self.llm_interaction).__name__, # Or get more specific info if available
                "prompt_template_used": self.prompt_template_str[:100] + "...", # Log snippet
                "is_fallback": False
            }
            # Note: The calling node function is responsible for calling state.set_final_answer()
            return answer, metadata

        except LLMInteractionError as e:
            logger.error(f"LLM interaction failed during RAG generation: {e}", exc_info=True)
            raise GeneratorError(f"LLM interaction failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during RAG generation: {e}", exc_info=True)
            raise GeneratorError(f"Unexpected error during generation: {e}") from e