# File: rag_system/workflow/components/factories.py
# Instruction: Replace the entire content of this file.
#              (Updated GeneratorFactory to create FallbackGenerator)

import logging
from typing import Optional

# --- Relative Imports ---
from ...config.settings import Configuration
from ...llm.interaction import ILLMInteraction
from ...corpus.corpus_manager import DocumentCorpus, CorpusManagerError

# Base Strategy Interfaces
from .retrieval.base_retriever import Retriever, RetrieverError
from .evaluation.base_evaluator import Evaluator, EvaluatorError
from .generation.base_generator import Generator, GeneratorError

# Concrete Strategy Implementations
# Retrieval
from .retrieval.semantic import SemanticRetriever
from .retrieval.web import WebSearchRetriever
# Placeholders:
# from .retrieval.keyword import KeywordRetriever
# from .retrieval.hybrid import HybridRetriever

# Evaluation
from .evaluation.relevance import RelevanceEvaluator
from .evaluation.factual import FactualGroundingEvaluator
from .evaluation.quality import AnswerQualityEvaluator

# Generation
from .generation.rag import RAGGenerator
from .generation.fallback import FallbackGenerator # <-- Uncommented/Imported

logger = logging.getLogger(__name__)

# --- Retriever Factory ---
# (Remains the same as response #103)
class RetrieverFactory:
    @staticmethod
    def create(
        strategy_name: str,
        config: Configuration,
        doc_corpus: DocumentCorpus,
        llm_interaction: Optional[ILLMInteraction] = None
    ) -> Retriever:
        strategy_lower = strategy_name.lower()
        logger.info(f"Creating retriever for strategy: '{strategy_lower}'")
        try:
            if strategy_lower == "semantic":
                vector_store = doc_corpus.get_vector_store()
                return SemanticRetriever(vector_store=vector_store)
            elif strategy_lower == "web":
                return WebSearchRetriever(config=config)
            elif strategy_lower == "keyword":
                if not llm_interaction: raise ValueError("KeywordRetriever requires an LLMInteraction instance.")
                # vector_store = doc_corpus.get_vector_store() # Get store if needed
                logger.warning("KeywordRetriever not fully implemented yet.")
                raise NotImplementedError("KeywordRetriever creation not implemented.")
            elif strategy_lower == "hybrid":
                if not llm_interaction: raise ValueError("HybridRetriever requires an LLMInteraction instance.")
                # vector_store = doc_corpus.get_vector_store() # Get store if needed
                logger.warning("HybridRetriever not fully implemented yet.")
                raise NotImplementedError("HybridRetriever creation not implemented.")
            else:
                logger.error(f"Unknown retriever strategy requested: '{strategy_name}'")
                raise ValueError(f"Unknown retriever strategy: {strategy_name}")
        except CorpusManagerError as e:
             logger.error(f"Failed to create retriever '{strategy_lower}': Corpus/VectorStore not ready. Error: {e}")
             raise
        except (ValueError, NotImplementedError, RetrieverError):
             raise
        except Exception as e:
             logger.error(f"Unexpected error creating retriever '{strategy_name}': {e}", exc_info=True)
             raise ValueError(f"Could not create retriever '{strategy_name}': {e}") from e


# --- Evaluator Factory ---
# (Remains the same as response #111)
class EvaluatorFactory:
    @staticmethod
    def create(
        evaluator_type: str,
        config: Configuration,
        llm_interaction: ILLMInteraction
    ) -> Evaluator:
        evaluator_lower = evaluator_type.lower()
        logger.info(f"Creating evaluator for type: '{evaluator_lower}'")
        if not llm_interaction: raise ValueError(f"Evaluator type '{evaluator_lower}' needs LLM.")
        try:
            if evaluator_lower == "relevance":
                threshold = config.get_relevance_threshold()
                return RelevanceEvaluator(llm_interaction=llm_interaction, relevance_threshold=threshold)
            elif evaluator_lower == "factual":
                 threshold = config.get_factual_threshold()
                 return FactualGroundingEvaluator(llm_interaction=llm_interaction, factual_threshold=threshold)
            elif evaluator_lower == "quality":
                 threshold = config.get_quality_threshold()
                 return AnswerQualityEvaluator(llm_interaction=llm_interaction, quality_threshold=threshold)
            else:
                logger.error(f"Unknown evaluator type requested: '{evaluator_type}'")
                raise ValueError(f"Unknown evaluator type: {evaluator_type}")
        except (ValueError, NotImplementedError, EvaluatorError):
             raise
        except Exception as e:
             logger.error(f"Unexpected error creating evaluator '{evaluator_type}': {e}", exc_info=True)
             raise ValueError(f"Could not create evaluator '{evaluator_type}': {e}") from e


# --- Generator Factory ---
# *** UPDATED ***
class GeneratorFactory:
    """
    Factory for creating different types of Generator instances.
    """
    @staticmethod
    def create(
        generator_type: str,
        config: Configuration,
        llm_interaction: ILLMInteraction # Main RAG needs LLM, Fallback doesn't
    ) -> Generator:
        """
        Creates a Generator instance based on the type name.

        Args:
            generator_type: The type of generator (e.g., "rag", "fallback").
            config: The application Configuration object.
            llm_interaction: An initialized LLMInteraction instance (required for some types).

        Returns:
            An instance conforming to the Generator interface.

        Raises:
            ValueError: If the generator_type is unknown or dependencies are missing.
        """
        generator_lower = generator_type.lower()
        logger.info(f"Creating generator for type: '{generator_lower}'")

        try:
            if generator_lower == "rag":
                if not llm_interaction: # RAG requires LLM
                    raise ValueError(f"Generator type '{generator_lower}' requires an LLMInteraction instance.")
                return RAGGenerator(llm_interaction=llm_interaction)

            elif generator_lower == "fallback":
                 # FallbackGenerator doesn't need llm_interaction passed to its init
                 return FallbackGenerator() # Instantiate the fallback generator

            else:
                logger.error(f"Unknown generator type requested: '{generator_type}'")
                raise ValueError(f"Unknown generator type: {generator_type}")

        except (ValueError, NotImplementedError, GeneratorError): # Added GeneratorError
             raise
        except Exception as e:
             logger.error(f"Unexpected error creating generator '{generator_type}': {e}", exc_info=True)
             raise ValueError(f"Could not create generator '{generator_type}': {e}") from e