# Defines ILLMInteraction (Interface), LangchainLLMInteraction
# File: rag_system/llm/interaction.py
# Instruction: Replace the entire content of this file.

import abc
import logging
from typing import Any, Optional, List, Iterator, Type
from pydantic import BaseModel

# Attempt to import LangChain elements safely
try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.prompt_values import PromptValue
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    BaseChatModel = None # type: ignore
    AIMessageChunk = None # type: ignore
    PromptValue = None # type: ignore
    LANGCHAIN_CORE_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMInteractionError(Exception):
    """Custom exception for LLM interaction errors."""
    pass

class ILLMInteraction(abc.ABC):
    """
    Interface defining the contract for interacting with a specific LLM instance.
    This abstracts away the underlying LLM library (e.g., Langchain).
    """

    @abc.abstractmethod
    def invoke(self, prompt: Any, stop_sequences: Optional[List[str]] = None, **kwargs) -> str:
        """
        Invoke the LLM with a given prompt and return the text response.

        Args:
            prompt: The input prompt (can be string, Langchain PromptValue, etc.).
            stop_sequences: Optional list of sequences to stop generation at.
            **kwargs: Additional provider-specific arguments.

        Returns:
            The generated text content as a string.

        Raises:
            LLMInteractionError: If the LLM invocation fails.
        """
        pass

    @abc.abstractmethod
    def stream(self, prompt: Any, stop_sequences: Optional[List[str]] = None, **kwargs) -> Iterator[str]:
        """
        Stream the LLM response for a given prompt.

        Args:
            prompt: The input prompt.
            stop_sequences: Optional list of sequences to stop generation at.
            **kwargs: Additional provider-specific arguments.

        Yields:
            Chunks of the generated text content as strings.

        Raises:
            LLMInteractionError: If initiating the LLM stream fails.
        """
        pass

    @abc.abstractmethod
    def invoke_structured_output(self, prompt: Any, output_schema: Type[BaseModel], method: str = "auto", **kwargs) -> BaseModel:
        """
        Invoke the LLM expecting a structured output conforming to the Pydantic schema.

        Args:
            prompt: The input prompt.
            output_schema: The Pydantic BaseModel class defining the desired output structure.
            method: The preferred method for structured output (e.g., 'function_calling', 'json_mode', 'auto').
                    'auto' lets the implementation choose the best available method.
            **kwargs: Additional provider-specific arguments.

        Returns:
            An instance of the provided Pydantic BaseModel containing the structured output.

        Raises:
            LLMInteractionError: If the LLM invocation or parsing fails.
            NotImplementedError: If the underlying model/library doesn't support structured output well.
        """
        pass


class LangchainLLMInteraction(ILLMInteraction):
    """
    An implementation of ILLMInteraction that wraps a Langchain BaseChatModel.
    Acts as an Adapter between the Langchain API and our internal interface.
    """

    def __init__(self, chat_model: BaseChatModel):
        """
        Initialize with a Langchain chat model instance.

        Args:
            chat_model: An instance of a class derived from BaseChatModel (e.g., ChatOpenAI).

        Raises:
            ImportError: If langchain-core is not installed.
            TypeError: If chat_model is not a valid BaseChatModel instance.
        """
        if not LANGCHAIN_CORE_AVAILABLE or not BaseChatModel:
            raise ImportError("langchain-core is required to use LangchainLLMInteraction.")
        if not isinstance(chat_model, BaseChatModel):
            raise TypeError(f"Expected a Langchain BaseChatModel, got {type(chat_model)}")
        self.chat_model = chat_model
        logger.debug(f"LangchainLLMInteraction initialized with model: {type(chat_model)}")

    def invoke(self, prompt: Any, stop_sequences: Optional[List[str]] = None, **kwargs) -> str:
        """Invokes the underlying Langchain chat model."""
        try:
            # Langchain invoke usually takes 'stop' as the argument name
            response = self.chat_model.invoke(prompt, stop=stop_sequences, **kwargs)
            if hasattr(response, 'content'):
                return str(response.content) # AIMessage typically has content
            else:
                # Fallback if the response object structure is different
                return str(response)
        except Exception as e:
            logger.error(f"Langchain invoke failed: {e}", exc_info=True)
            raise LLMInteractionError(f"LLM invocation failed: {e}") from e

    def stream(self, prompt: Any, stop_sequences: Optional[List[str]] = None, **kwargs) -> Iterator[str]:
        """Streams the response from the underlying Langchain chat model."""
        try:
            stream_iterator = self.chat_model.stream(prompt, stop=stop_sequences, **kwargs)
            for chunk in stream_iterator:
                if AIMessageChunk and isinstance(chunk, AIMessageChunk) and hasattr(chunk, 'content'):
                    yield str(chunk.content)
                elif hasattr(chunk, 'content'): # Handle other potential chunk types
                    yield str(chunk.content)
                else:
                    # If chunk structure is unknown, yield its string representation
                    # This might or might not be useful depending on the model
                    logger.debug(f"Streaming unknown chunk type: {type(chunk)}")
                    yield str(chunk)
        except Exception as e:
            logger.error(f"Langchain stream failed: {e}", exc_info=True)
            raise LLMInteractionError(f"LLM stream failed: {e}") from e

    def invoke_structured_output(self, prompt: Any, output_schema: Type[BaseModel], method: str = "auto", **kwargs) -> BaseModel:
        """Invokes the Langchain model for structured output."""
        # Basic check for structured output capability
        if not hasattr(self.chat_model, 'with_structured_output'):
            logger.error(f"Model {type(self.chat_model)} may not support with_structured_output.")
            raise NotImplementedError(f"Model {type(self.chat_model)} does not support structured output via 'with_structured_output'.")

        try:
            # Prepare arguments for with_structured_output, handling 'auto' method
            structured_output_kwargs = {}
            if method != "auto":
                structured_output_kwargs['method'] = method

            structured_llm = self.chat_model.with_structured_output(
                schema=output_schema,
                **structured_output_kwargs,
                **kwargs # Pass other kwargs here too
            )
            result = structured_llm.invoke(prompt)

            if isinstance(result, output_schema):
                return result
            else:
                # This case might happen if parsing fails internally in Langchain
                logger.error(f"Structured output result type mismatch. Expected {output_schema}, got {type(result)}")
                raise LLMInteractionError(f"Structured output parsing failed or returned unexpected type: {type(result)}")

        except Exception as e:
            logger.error(f"Langchain structured output invoke failed: {e}", exc_info=True)
            # Check if it's a NotImplementedError from Langchain regarding the method
            if "NotImplementedError" in str(e) or "does not support" in str(e):
                 raise NotImplementedError(f"Model {type(self.chat_model)} or specified method '{method}' does not support structured output: {e}") from e
            raise LLMInteractionError(f"LLM structured output invocation failed: {e}") from e