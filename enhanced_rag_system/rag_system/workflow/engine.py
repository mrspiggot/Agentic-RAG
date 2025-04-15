# File: rag_system/workflow/engine.py
# Instruction: Replace the entire content of this file.
#              (Based on user snapshot from #123, adding missing _fallback_generate_node method)

import logging
import time
import sys
import traceback
import os
import tempfile
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Literal, Type
from datetime import datetime, timezone

# --- Core Dependencies (Direct Imports - Fail Fast if missing) ---
try:
    from pydantic import BaseModel, Field, ValidationError
    from langgraph.graph import StateGraph, START, END
    try: from langgraph.graph import CompiledGraph
    except ImportError: CompiledGraph = Any # Fallback type hint
    from langchain_core.prompts import ChatPromptTemplate
    Runnable = Any
except ImportError as e:
    logging.critical(f"FATAL ERROR: Failed to import core dependencies (pydantic, langgraph, langchain-core).")
    logging.critical(f"Ensure packages are installed correctly in your environment (.venv). Error: {e}", exc_info=True)
    raise

# --- Relative Imports ---
from ..config.settings import Configuration, ConfigurationError
from ..llm.factories import LLMProviderFactory
from ..llm.providers.base_provider import ILLMProvider, LLMProviderError
from ..llm.interaction import ILLMInteraction, LLMInteractionError
from ..corpus.corpus_manager import DocumentCorpus, CorpusManagerError
from ..data_models.workflow_state import WorkflowState, UrlUsageInfo
from ..data_models.result import Result, SourceInfo, LogEntry
from ..data_models.document import Document
from .components.factories import RetrieverFactory, EvaluatorFactory, GeneratorFactory
from .components.retrieval.base_retriever import Retriever, RetrieverError
from .components.evaluation.base_evaluator import Evaluator, EvaluatorError
from .components.evaluation.relevance import RelevanceEvaluator
from .components.evaluation.factual import FactualGroundingEvaluator
from .components.evaluation.quality import AnswerQualityEvaluator
from .components.generation.base_generator import Generator, GeneratorError
from .components.generation.rag import RAGGenerator
from .components.generation.fallback import FallbackGenerator
from .components.retrieval.web import WebSearchRetriever
from .components.retrieval.semantic import SemanticRetriever


logger = logging.getLogger(__name__)

class WorkflowError(Exception):
    """Custom exception for workflow execution errors."""
    pass

# --- Pydantic Model for Question Router ---
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(...,
        description="The data source to route the user's question to. Must be one of ['vectorstore', 'web_search']."
    )

class RAGWorkflowManager:
    """
    Orchestrates the RAG workflow using LangGraph.
    Includes initial routing, retrieval, grading, generation, and post-evaluation checks with fallback.
    """

    # (__init__ and _build_question_router copied directly from user snapshot #123)
    def __init__(self,
                 config: Configuration,
                 doc_corpus: DocumentCorpus,
                 llm_provider_factory: Optional[LLMProviderFactory] = None,
                 retriever_factory: Optional[RetrieverFactory] = None,
                 evaluator_factory: Optional[EvaluatorFactory] = None,
                 generator_factory: Optional[GeneratorFactory] = None):
        """ Initializes the RAGWorkflowManager. """
        self.config = config
        self.doc_corpus = doc_corpus
        self.llm_provider_factory = llm_provider_factory or LLMProviderFactory()
        self.retriever_factory = retriever_factory or RetrieverFactory()
        self.evaluator_factory = evaluator_factory or EvaluatorFactory()
        self.generator_factory = generator_factory or GeneratorFactory()
        self._compiled_graph: Optional[CompiledGraph] = None
        self._is_graph_built: bool = False
        self._llm_provider: Optional[ILLMProvider] = None
        self._llm_interaction: Optional[ILLMInteraction] = None
        self._router_interaction: Optional[ILLMInteraction] = None
        self._router_prompt: Optional[ChatPromptTemplate] = None
        self._router_schema: Optional[Type[BaseModel]] = None
        try:
            logger.info("Initializing LLM provider and interaction...")
            self._llm_provider = self.llm_provider_factory.create_provider(self.config)
            if self._llm_provider is None: raise LLMProviderError("LLM Provider could not be created.")
            main_model_name = self.config.get_llm_model_name()
            main_llm_kwargs = {'temperature': self.config.get_llm_temperature()}
            self._llm_interaction = self._llm_provider.create_interaction(main_model_name, **main_llm_kwargs)
            logger.info(f"Main LLM Interaction ({main_model_name}) ready.")
            router_model_name = self.config.get_llm_model_name()
            router_llm_kwargs = {'temperature': 0.0}
            self._router_interaction = self._llm_provider.create_interaction(router_model_name, **router_llm_kwargs)
            logger.info(f"Router LLM Interaction ({router_model_name}) ready.")
            logger.info("Building Question Router components...")
            self._build_question_router()
            logger.info("Question Router components built successfully.")
        except (LLMProviderError, LLMInteractionError, ConfigurationError, ImportError, ValueError) as e:
            logger.error(f"Fatal: Failed to initialize core LLM components or Router: {e}", exc_info=True)
            raise WorkflowError(f"Core component Initialization failed: {e}") from e
        except Exception as e:
             logger.error(f"Fatal: Unexpected error initializing components or Router: {e}", exc_info=True)
             raise WorkflowError(f"Unexpected Initialization error: {e}") from e

    def _build_question_router(self):
        logger.debug("Defining question router prompt and schema.")
        system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to: AI agents, LangChain, LangGraph, Prompt engineering, RAG architectures, LLM evaluation.
Use the vectorstore for specific questions about these topics.
Use web_search for: General knowledge questions, current events, or topics clearly outside the vectorstore scope (e.g., finance, sports, cooking).
Choose only 'vectorstore' or 'web_search' as the datasource."""
        self._router_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
        self._router_schema = RouteQuery
        logger.info("Question router prompt and schema prepared.")


    # --- Node Function Implementations ---
    # (_initialize_node, _retrieve_vectorstore_node, _web_search_node, _grade_documents_node, _generate_node, _evaluate_answer_node copied directly from user snapshot #123)
    def _initialize_node(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"Workflow initializing for question: '{state.original_question[:100]}...'")
        state.add_log("INFO", "Workflow started.", node="initialize")
        if not state.original_question: state.original_question = state.question
        state.relevance_threshold = self.config.get_relevance_threshold()
        state.current_retrieval_strategy = "unknown"
        return state

    def _retrieve_vectorstore_node(self, state: WorkflowState) -> WorkflowState:
        node_name = "retrieve_vectorstore"
        state.add_log("INFO", f"Starting document retrieval using 'semantic' strategy.", node=node_name)
        retriever: Optional[Retriever] = None
        retrieved_docs: List[Document] = []
        try:
            retriever = self.retriever_factory.create("semantic", self.config, self.doc_corpus)
            retrieved_docs = retriever.retrieve(state.question, state, self.config)
            state.add_retrieval_result(retriever.get_name(), state.question, retrieved_docs)
        except RetrieverError as e:
            logger.error(f"Vectorstore retrieval failed: {e}", exc_info=False); state.add_log("ERROR", f"Vectorstore retrieval failed: {e}", node=node_name); state.documents = []
        except Exception as e:
            logger.error(f"Unexpected error during vectorstore retrieval: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected vectorstore retrieval error: {e}", node=node_name); state.documents = []
        return state

    def _web_search_node(self, state: WorkflowState) -> WorkflowState:
        node_name = "retrieve_web_search"
        state.add_log("INFO", f"Starting document retrieval using 'web_search' strategy.", node=node_name)
        retriever: Optional[Retriever] = None
        retrieved_docs: List[Document] = []
        try:
            retriever = self.retriever_factory.create("web", self.config, self.doc_corpus)
            retrieved_docs = retriever.retrieve(state.original_question, state, self.config)
            state.add_retrieval_result(retriever.get_name(), state.original_question, retrieved_docs)
        except RetrieverError as e:
            logger.error(f"Web search retrieval failed: {e}", exc_info=False); state.add_log("ERROR", f"Web search retrieval failed: {e}", node=node_name); state.documents = []
        except Exception as e:
            logger.error(f"Unexpected error during web search retrieval: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected web search retrieval error: {e}", node=node_name); state.documents = []
        return state

    def _grade_documents_node(self, state: WorkflowState) -> WorkflowState:
        node_name = "grade_documents"
        documents_to_grade = state.documents
        if state.current_retrieval_strategy != "SemanticRetriever":
             state.add_log("INFO", f"Skipping grading for non-vectorstore strategy '{state.current_retrieval_strategy}'.", node=node_name)
             return state
        state.add_log("INFO", f"Starting document relevance grading for {len(documents_to_grade)} vectorstore documents.", node=node_name)
        relevant_docs: List[Document] = []
        if not documents_to_grade:
             state.add_log("WARNING", "No documents to grade.", node=node_name); state.documents = []; return state
        evaluator: Optional[Evaluator] = None
        try:
             if not self._llm_interaction: raise WorkflowError("LLM Interaction missing.")
             evaluator = self.evaluator_factory.create("relevance", self.config, self._llm_interaction)
             current_threshold = state.relevance_threshold
             if hasattr(evaluator, 'relevance_threshold'): evaluator.relevance_threshold = current_threshold
             for doc in documents_to_grade:
                 eval_result = evaluator.evaluate(state, self.config, document=doc)
                 threshold_used = getattr(evaluator, 'relevance_threshold', current_threshold)
                 is_relevant = eval_result.get("relevance_score", 0) >= threshold_used
                 log_level = "DEBUG"
                 if is_relevant:
                     relevant_docs.append(doc); state.add_log(log_level, f"Doc ID {doc.id} relevant (Score: {eval_result.get('relevance_score', 'N/A')} >= {threshold_used}). Keeping.", node=node_name)
                 else:
                      state.add_log(log_level, f"Doc ID {doc.id} irrelevant (Score: {eval_result.get('relevance_score', 'N/A')} < {threshold_used}). Discarding.", node=node_name)
             state.set_relevant_documents(relevant_docs, node_name=node_name)
        except EvaluatorError as e:
             logger.error(f"Grading failed: {e}", exc_info=False); state.add_log("ERROR", f"Grading failed: {e}", node=node_name); state.set_relevant_documents([], node_name=node_name)
        except Exception as e:
             logger.error(f"Unexpected error during grading: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected grading error: {e}", node=node_name); state.set_relevant_documents([], node_name=node_name)
        return state

    def _generate_node(self, state: WorkflowState) -> WorkflowState:
        node_name = "generate"
        docs_count = len(state.documents)
        source_type = "web search" if state.current_retrieval_strategy == "WebSearchRetriever" else "graded vectorstore"
        state.add_log("INFO", f"Starting answer generation using 'RAG' strategy with {docs_count} {source_type} documents.", node=node_name)
        generator: Optional[Generator] = None
        try:
            if not self._llm_interaction: raise WorkflowError("LLM Interaction missing.")
            generator = self.generator_factory.create("rag", self.config, self._llm_interaction)
            answer, metadata = generator.generate(state, self.config)
            state.set_final_answer(answer, metadata)
        except GeneratorError as e:
            logger.error(f"Generation failed: {e}", exc_info=False); state.add_log("ERROR", f"Generation failed: {e}", node=node_name); state.set_final_answer(f"Error: Generation failed - {e}", {"error": str(e)})
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected generation error: {e}", node=node_name); state.set_final_answer(f"Error: Unexpected generation error - {e}", {"error": str(e)})
        return state

    def _evaluate_answer_node(self, state: WorkflowState) -> WorkflowState:
         node_name = "evaluate_answer"
         state.add_log("INFO", "Starting post-generation answer evaluation.", node=node_name)
         if not state.answer or state.generation_results.get("error"):
              state.add_log("WARNING", "Skipping answer evaluation as generation failed or produced no answer.", node=node_name)
              state.evaluation_results["factual"] = {"status": "skipped", "reason": "No valid answer generated."}; state.evaluation_results["quality"] = {"status": "skipped", "reason": "No valid answer generated."}
              return state
         try:
             if not self._llm_interaction: raise WorkflowError("LLM Interaction missing.")
             logger.debug("Running factual grounding evaluation...")
             factual_evaluator = self.evaluator_factory.create("factual", self.config, self._llm_interaction)
             factual_evaluator.evaluate(state, self.config)
             logger.debug(f"Factual eval result added: {state.evaluation_results.get('factual')}")
             logger.debug("Running answer quality evaluation...")
             quality_evaluator = self.evaluator_factory.create("quality", self.config, self._llm_interaction)
             quality_evaluator.evaluate(state, self.config)
             logger.debug(f"Quality eval result added: {state.evaluation_results.get('quality')}")
             state.add_log("INFO", "Post-generation answer evaluation complete.", node=node_name)
         except EvaluatorError as e:
              logger.error(f"Post-generation evaluation failed: {e}", exc_info=False); state.add_log("ERROR", f"Post-generation evaluation failed: {e}", node=node_name)
              state.evaluation_results["factual"] = state.evaluation_results.get("factual", {"error": str(e)}); state.evaluation_results["quality"] = state.evaluation_results.get("quality", {"error": str(e)})
         except Exception as e:
              logger.error(f"Unexpected error during post-generation evaluation: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected evaluation error: {e}", node=node_name)
              state.evaluation_results["factual"] = state.evaluation_results.get("factual", {"error": str(e)}); state.evaluation_results["quality"] = state.evaluation_results.get("quality", {"error": str(e)})
         return state

    # *** ADDED MISSING NODE METHOD ***
    def _fallback_generate_node(self, state: WorkflowState) -> WorkflowState:
        """ Generates a predefined fallback answer if evaluation fails. """
        node_name = "fallback_generate"
        state.add_log("WARNING", "Executing fallback generation.", node=node_name)
        generator: Optional[Generator] = None
        try:
            # Factory create doesn't strictly need LLM interaction for fallback,
            # but pass main one for potential future fallback strategies that might use it.
            generator = self.generator_factory.create("fallback", self.config, self._llm_interaction)
            answer, metadata = generator.generate(state, self.config) # Pass state for context if needed by generator
            state.set_final_answer(answer, metadata)
        except GeneratorError as e:
             logger.error(f"Fallback generation failed: {e}", exc_info=False)
             state.add_log("ERROR", f"Fallback generation failed: {e}", node=node_name)
             state.set_final_answer("An error occurred during fallback generation.", {"error": str(e)})
        except Exception as e:
             logger.error(f"Unexpected error during fallback generation: {e}", exc_info=True)
             state.add_log("ERROR", f"Unexpected fallback generation error: {e}", node=node_name)
             state.set_final_answer(f"Error: Unexpected fallback error - {e}", {"error": str(e)})
        return state

    # --- Edge Functions ---
    # (_route_question_edge copied directly from user snapshot #123 - which correctly returns KEYS)
    def _route_question_edge(self, state: WorkflowState) -> Literal["vectorstore", "web_search"]:
        node_name = "route_question"
        state.add_log("INFO", f"Routing question: '{state.question[:100]}...'", node=node_name)
        interaction_to_use = self._router_interaction
        if not interaction_to_use or not self._router_prompt or not self._router_schema:
            logger.error("Question router components not initialized. Defaulting to vectorstore."); state.add_log("ERROR", "Question router not initialized, defaulting.", node=node_name); return "vectorstore"
        try:
            prompt_value = self._router_prompt.format_prompt(question=state.question)
            routing_result: RouteQuery = interaction_to_use.invoke_structured_output(prompt=prompt_value.to_messages() if hasattr(prompt_value, 'to_messages') else str(prompt_value), output_schema=self._router_schema)
            datasource = routing_result.datasource.lower()
            if datasource == "vectorstore": logger.info("Routing decision: 'vectorstore'"); state.add_log("INFO", "Routing decision: 'vectorstore'.", node=node_name); return "vectorstore"
            elif datasource == "web_search": logger.info("Routing decision: 'web_search'"); state.add_log("INFO", "Routing decision: 'web_search'.", node=node_name); return "web_search"
            else: logger.warning(f"Router returned invalid datasource '{datasource}'. Defaulting."); state.add_log("WARNING", f"Router invalid response '{datasource}', defaulting.", node=node_name); return "vectorstore"
        except (LLMInteractionError, NotImplementedError, ValidationError) as e:
             logger.error(f"Question routing LLM call failed: {e}", exc_info=False); state.add_log("ERROR", f"Question routing failed: {e}. Defaulting.", node=node_name); return "vectorstore"
        except Exception as e:
             logger.error(f"Unexpected error during question routing: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected routing error: {e}. Defaulting.", node=node_name); return "vectorstore"

    # (_decide_after_evaluation copied directly from user snapshot #123 - which correctly returns END/"needs_fallback")
    def _decide_after_evaluation(self, state: WorkflowState) -> Literal[END, "needs_fallback"]:
         node_name = "decide_after_evaluation"
         state.add_log("INFO", "Deciding next step after evaluation.", node=node_name)
         factual_results = state.evaluation_results.get("factual", {})
         quality_results = state.evaluation_results.get("quality", {})
         is_factual = factual_results.get("is_factual", True)
         is_good_quality = quality_results.get("is_good_quality", True)
         logger.info(f"Evaluation Scores: Factual={factual_results.get('factual_score', 'N/A')}, Quality={quality_results.get('overall_score', 'N/A')}")
         logger.info(f"Evaluation Check: Is Factual? {is_factual}, Is Good Quality? {is_good_quality}")
         state.add_log("INFO", f"Factual check: {is_factual}. Quality check: {is_good_quality}.", node=node_name)
         if is_factual and is_good_quality:
             logger.info("Decision: Answer quality is acceptable. Ending workflow."); state.add_log("INFO", "Decision: END (Acceptable)", node=node_name)
             return END
         else:
              logger.warning("Decision: Answer quality unacceptable. Routing to fallback generation."); state.add_log("WARNING", "Decision: Fallback Required", node=node_name)
              return "needs_fallback"


    # --- Graph Building ---
    # *** UPDATED to use correct node method name and wire in fallback ***
    def build_graph(self) -> Any: # Return Any or CompiledGraph
        """Builds and compiles the LangGraph StateGraph with routing and evaluation fallback."""
        if self._is_graph_built and self._compiled_graph:
            return self._compiled_graph
        logger.info("Building RAG workflow graph (with routing and evaluation fallback)...")
        workflow = StateGraph(WorkflowState)
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("retrieve_vectorstore", self._retrieve_vectorstore_node) # Corrected callable
        workflow.add_node("retrieve_web_search", self._web_search_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate_answer", self._evaluate_answer_node)
        workflow.add_node("fallback_generate", self._fallback_generate_node) # Add node definition
        workflow.add_edge(START, "initialize")
        workflow.add_conditional_edges(
             "initialize", self._route_question_edge,
             {"vectorstore": "retrieve_vectorstore", "web_search": "retrieve_web_search"}
        )
        workflow.add_edge("retrieve_vectorstore", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("retrieve_web_search", "generate")
        workflow.add_edge("generate", "evaluate_answer")
        workflow.add_conditional_edges(
             "evaluate_answer", self._decide_after_evaluation,
             {END: END, "needs_fallback": "fallback_generate"} # Map to new node
        )
        workflow.add_edge("fallback_generate", END) # Add edge from fallback to END
        try:
            self._compiled_graph = workflow.compile()
            self._is_graph_built = True
            logger.info("Workflow graph compiled successfully (with routing and evaluation fallback).")
            return self._compiled_graph
        except Exception as e:
             logger.error(f"Failed to compile workflow graph: {e}", exc_info=True)
             self._is_graph_built = False
             raise WorkflowError(f"Graph compilation failed: {e}") from e

    # --- Workflow Execution & Result Creation ---
    # (run_workflow and _create_result_from_state remain the same)
    def run_workflow(self, question: str, initial_source_urls: Optional[List[str]] = None) -> Result:
        start_run_time = time.time()
        logger.info(f"--- Starting Workflow Run for question: '{question[:100]}...' ---")
        compiled_graph = self.build_graph()
        initial_state_input = {
            "question": question, "original_question": question, "documents": [],
            "answer": None, "transform_attempts": 0,
            "relevance_threshold": self.config.get_relevance_threshold(),
            "current_retrieval_strategy": "unknown",
            "retrieval_history": [], "evaluation_results": {}, "generation_results": {},
            "url_usage_tracking": {}, "logs": []
        }
        if initial_source_urls:
             initial_url_tracking = {}
             current_time_utc = datetime.now(timezone.utc).isoformat()
             from ..data_models.workflow_state import UrlUsageInfo
             for url in initial_source_urls:
                  if url and isinstance(url, str): initial_url_tracking[url] = UrlUsageInfo(source_type='supplied', last_used_utc=current_time_utc)
             initial_state_input["url_usage_tracking"] = initial_url_tracking
             logger.info(f"Initialized url_usage_tracking with {len(initial_url_tracking)} supplied URLs.")

        final_state: Optional[WorkflowState] = None
        try:
            logger.info("Invoking workflow graph...")
            final_state_dict = compiled_graph.invoke(initial_state_input, config={"recursion_limit": 10})
            logger.info("Workflow graph invocation complete.")
            if final_state_dict and isinstance(final_state_dict, dict):
                 try: final_state = WorkflowState(**final_state_dict)
                 except TypeError as te:
                     logger.error(f"Mismatch between final state dict and WorkflowState dataclass: {te}")
                     logger.debug(f"Final state dict keys: {list(final_state_dict.keys())}")
                     raise WorkflowError(f"Could not reconstruct final WorkflowState: {te}")
            else:
                 logger.error(f"Workflow execution returned unexpected type: {type(final_state_dict)}")
                 raise WorkflowError("Workflow execution did not return a valid final state dictionary.")
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            logs_on_fail = [LogEntry(**log) for log in initial_state_input.get('logs', [])]
            error_result = Result(question=question, answer_text=f"Workflow execution failed: {e}", execution_metadata={"error": str(e)}, log_entries=logs_on_fail)
            return error_result

        if final_state is None:
             logger.error("Final state object is None after workflow invocation.")
             return Result(question=question, answer_text="Workflow failed: Final state object missing.", execution_metadata={"error": "Final state object missing"})
        result = self._create_result_from_state(final_state)
        end_run_time = time.time()
        result.execution_metadata["total_duration_sec"] = round(end_run_time - start_run_time, 3)
        logger.info(f"--- Workflow Run Completed in {result.execution_metadata['total_duration_sec']:.3f} seconds ---")
        return result

    def _create_result_from_state(self, final_state: WorkflowState) -> Result:
        from ..data_models.workflow_state import UrlUsageInfo
        from ..data_models.result import SourceInfo, LogEntry
        source_summary: List[SourceInfo] = []
        for url, usage_info in final_state.url_usage_tracking.items():
            final_relevance: Optional[float] = None
            if usage_info.relevance_scores:
                 try: final_relevance = round(sum(usage_info.relevance_scores) / len(usage_info.relevance_scores), 2)
                 except ZeroDivisionError: pass
            source_summary.append(SourceInfo(url=url, source_type=usage_info.source_type, final_relevance_metric=final_relevance, usage_count=usage_info.count))
        log_entries = [LogEntry(**log_dict) for log_dict in final_state.logs]
        exec_meta = {
             "original_question": final_state.original_question, "final_retrieval_strategy": final_state.current_retrieval_strategy,
             "transform_attempts": final_state.transform_attempts, "final_relevance_threshold": final_state.relevance_threshold,
             "generation_metadata": final_state.generation_results, "evaluation_results": final_state.evaluation_results,
             "docs_retrieved_count": len(final_state.retrieval_history[-1]['document_ids']) if final_state.retrieval_history else 0,
             "docs_passed_grading_count": len(final_state.documents)
        }
        return Result(question=final_state.original_question, answer_text=final_state.answer or "No answer was generated.", final_source_summary=source_summary, log_entries=log_entries, graph_diagram_data=None, execution_metadata=exec_meta)

    # --- Graph Visualization Method ---
    def get_graph_visualization_png(self) -> Optional[bytes]:
        """Generates a PNG visualization of the compiled workflow graph."""
        if not self._compiled_graph:
            logger.warning("Graph visualization requested, but graph is not compiled.")
            raise WorkflowError("Graph not compiled, cannot visualize.")
        logger.info("Attempting to generate workflow graph PNG...")
        # Direct import - will raise ImportError if missing AND --show-graph used
        try:
            import playwright
            import pyppeteer
        except ImportError as ie:
             logger.error(f"Missing visualization dependency: {ie}. Install playwright/pyppeteer.", exc_info=False)
             raise # Re-raise import error if hit here

        try:
            # Attempt to pass timeout (speculative)
            png_bytes = self._compiled_graph.get_graph(xray=True).draw_mermaid_png(timeout=30)
            logger.info(f"Successfully generated graph PNG data ({len(png_bytes)} bytes).")
            return png_bytes
        except Exception as e:
            logger.error(f"Failed to draw workflow graph: {e}", exc_info=True)
            if "timed out" in str(e).lower(): logger.error("Graph drawing timed out, possibly due to mermaid.ink service.")
            return None # Return None on other drawing errors