# File: rag_system/workflow/engine.py
# (Complete file content modified to grade web search results)

import logging
import time
import sys
import traceback
import os
import tempfile
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Literal, Type
from datetime import datetime, timezone

# --- Core Dependencies (Direct Imports - Rule #10) ---
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph.graph import CompiledGraph

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
        # Ensure ChatPromptTemplate is imported directly if it's core
        self._router_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
        self._router_schema = RouteQuery # RouteQuery uses Pydantic, also imported directly
        logger.info("Question router prompt and schema prepared.")


    # --- Node Function Implementations ---
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
            # Set strategy name before adding results
            state.current_retrieval_strategy = retriever.get_name()
            state.add_retrieval_result(state.current_retrieval_strategy, state.question, retrieved_docs)
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
            # Set strategy name before adding results
            state.current_retrieval_strategy = retriever.get_name()
            state.add_retrieval_result(state.current_retrieval_strategy, state.original_question, retrieved_docs)
        except RetrieverError as e:
            logger.error(f"Web search retrieval failed: {e}", exc_info=False); state.add_log("ERROR", f"Web search retrieval failed: {e}", node=node_name); state.documents = []
        except Exception as e:
            logger.error(f"Unexpected error during web search retrieval: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected web search retrieval error: {e}", node=node_name); state.documents = []
        return state

    # *** MODIFIED FUNCTION: _grade_documents_node ***
    def _grade_documents_node(self, state: WorkflowState) -> WorkflowState:
        node_name = "grade_documents"
        documents_to_grade = state.documents
        retrieval_source = state.current_retrieval_strategy # Get source strategy

        # --- REMOVED CONDITIONAL SKIP ---
        # No longer skip based on strategy; grade all retrieved documents.
        # --------------------------------

        state.add_log("INFO", f"Starting document relevance grading for {len(documents_to_grade)} documents from '{retrieval_source}'.", node=node_name)
        relevant_docs: List[Document] = []

        if not documents_to_grade:
             state.add_log("WARNING", "No documents to grade.", node=node_name)
             # Ensure state.documents is empty if none to grade
             state.documents = []
             return state

        evaluator: Optional[Evaluator] = None
        try:
             if not self._llm_interaction: raise WorkflowError("LLM Interaction missing for grading.")
             evaluator = self.evaluator_factory.create("relevance", self.config, self._llm_interaction)
             # Use the threshold currently set in the state
             current_threshold = state.relevance_threshold
             # Pass threshold to evaluator if it supports it (our RelevanceEvaluator does)
             if hasattr(evaluator, 'relevance_threshold'):
                 evaluator.relevance_threshold = current_threshold

             # Grade each document
             for doc in documents_to_grade:
                 # The evaluator's evaluate method now handles updating state.url_usage_tracking
                 eval_result = evaluator.evaluate(state, self.config, document=doc)

                 # Check if the document is relevant based on the threshold used by the evaluator
                 threshold_used = getattr(evaluator, 'relevance_threshold', current_threshold) # Get threshold used
                 is_relevant = eval_result.get("relevance_score", 0) >= threshold_used
                 log_level = "DEBUG" # Log individual grading results as DEBUG

                 if is_relevant:
                     relevant_docs.append(doc)
                     state.add_log(log_level, f"Doc ID {doc.id} relevant (Score: {eval_result.get('relevance_score', 'N/A')} >= {threshold_used}). Keeping.", node=node_name)
                 else:
                      state.add_log(log_level, f"Doc ID {doc.id} irrelevant (Score: {eval_result.get('relevance_score', 'N/A')} < {threshold_used}). Discarding.", node=node_name)

             # Update the main documents list in the state with only relevant ones
             state.set_relevant_documents(relevant_docs, node_name=node_name)

        except EvaluatorError as e:
             logger.error(f"Grading failed: {e}", exc_info=False)
             state.add_log("ERROR", f"Grading failed: {e}", node=node_name)
             # Discard all docs if grading fails? Or keep original? Let's discard.
             state.set_relevant_documents([], node_name=node_name)
        except Exception as e:
             logger.error(f"Unexpected error during grading: {e}", exc_info=True)
             state.add_log("ERROR", f"Unexpected grading error: {e}", node=node_name)
             state.set_relevant_documents([], node_name=node_name) # Discard on unexpected error

        return state

    def _generate_node(self, state: WorkflowState) -> WorkflowState:
        node_name = "generate"
        docs_count = len(state.documents)
        # Make source type description more general
        source_type = f"{docs_count} document(s) from '{state.current_retrieval_strategy}' after grading"
        state.add_log("INFO", f"Starting answer generation using 'RAG' strategy with {source_type}.", node=node_name)
        generator: Optional[Generator] = None
        try:
            if not self._llm_interaction: raise WorkflowError("LLM Interaction missing for generation.")
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
             if not self._llm_interaction: raise WorkflowError("LLM Interaction missing for evaluation.")
             logger.debug("Running factual grounding evaluation...")
             factual_evaluator = self.evaluator_factory.create("factual", self.config, self._llm_interaction)
             factual_evaluator.evaluate(state, self.config) # Updates state internally
             logger.debug(f"Factual eval result added: {state.evaluation_results.get('factual')}")
             logger.debug("Running answer quality evaluation...")
             quality_evaluator = self.evaluator_factory.create("quality", self.config, self._llm_interaction)
             quality_evaluator.evaluate(state, self.config) # Updates state internally
             logger.debug(f"Quality eval result added: {state.evaluation_results.get('quality')}")
             state.add_log("INFO", "Post-generation answer evaluation complete.", node=node_name)
         except EvaluatorError as e:
              logger.error(f"Post-generation evaluation failed: {e}", exc_info=False); state.add_log("ERROR", f"Post-generation evaluation failed: {e}", node=node_name)
              # Ensure results dicts exist before adding error keys
              if "factual" not in state.evaluation_results: state.evaluation_results["factual"] = {}
              if "quality" not in state.evaluation_results: state.evaluation_results["quality"] = {}
              state.evaluation_results["factual"]["error"] = str(e)
              state.evaluation_results["quality"]["error"] = str(e)
         except Exception as e:
              logger.error(f"Unexpected error during post-generation evaluation: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected evaluation error: {e}", node=node_name)
              if "factual" not in state.evaluation_results: state.evaluation_results["factual"] = {}
              if "quality" not in state.evaluation_results: state.evaluation_results["quality"] = {}
              state.evaluation_results["factual"]["error"] = str(e)
              state.evaluation_results["quality"]["error"] = str(e)
         return state

    def _fallback_generate_node(self, state: WorkflowState) -> WorkflowState:
        node_name = "fallback_generate"
        state.add_log("WARNING", "Executing fallback generation.", node=node_name)
        generator: Optional[Generator] = None
        try:
            # Pass llm_interaction even if fallback doesn't use it, for factory consistency
            generator = self.generator_factory.create("fallback", self.config, self._llm_interaction)
            answer, metadata = generator.generate(state, self.config)
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
    def _route_question_edge(self, state: WorkflowState) -> Literal["vectorstore", "web_search"]:
        node_name = "route_question"
        state.add_log("INFO", f"Routing question: '{state.question[:100]}...'", node=node_name)
        interaction_to_use = self._router_interaction
        if not interaction_to_use or not self._router_prompt or not self._router_schema:
            logger.error("Question router components not initialized. Defaulting to vectorstore."); state.add_log("ERROR", "Question router not initialized, defaulting.", node=node_name); return "vectorstore"
        try:
            prompt_value = self._router_prompt.format_prompt(question=state.question)
            prompt_input = prompt_value.to_messages() if hasattr(prompt_value, 'to_messages') else str(prompt_value)
            routing_result: RouteQuery = interaction_to_use.invoke_structured_output(prompt=prompt_input, output_schema=self._router_schema)
            datasource = routing_result.datasource.lower()
            if datasource == "vectorstore": logger.info("Routing decision: 'vectorstore'"); state.add_log("INFO", "Routing decision: 'vectorstore'.", node=node_name); return "vectorstore"
            elif datasource == "web_search": logger.info("Routing decision: 'web_search'"); state.add_log("INFO", "Routing decision: 'web_search'.", node=node_name); return "web_search"
            else: logger.warning(f"Router returned invalid datasource '{datasource}'. Defaulting."); state.add_log("WARNING", f"Router invalid response '{datasource}', defaulting.", node=node_name); return "vectorstore"
        except (LLMInteractionError, NotImplementedError, ValidationError) as e:
             logger.error(f"Question routing LLM call failed: {e}", exc_info=False); state.add_log("ERROR", f"Question routing failed: {e}. Defaulting.", node=node_name); return "vectorstore"
        except Exception as e:
             logger.error(f"Unexpected error during question routing: {e}", exc_info=True); state.add_log("ERROR", f"Unexpected routing error: {e}. Defaulting.", node=node_name); return "vectorstore"

    def _decide_after_evaluation(self, state: WorkflowState) -> Literal[END, "needs_fallback"]:
         node_name = "decide_after_evaluation"
         state.add_log("INFO", "Deciding next step after evaluation.", node=node_name)
         factual_results = state.evaluation_results.get("factual", {})
         quality_results = state.evaluation_results.get("quality", {})
         # Default to True if evaluation somehow failed or was skipped, to avoid unnecessary fallback
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
    # *** MODIFIED FUNCTION: build_graph ***
    def build_graph(self) -> CompiledGraph:
        """Builds and compiles the LangGraph StateGraph with routing and evaluation fallback."""
        if self._is_graph_built and self._compiled_graph:
            logger.debug("Returning cached compiled graph.")
            if self._compiled_graph is None:
                 logger.error("Graph marked as built but compiled graph is None. Rebuilding.")
                 self._is_graph_built = False
                 raise WorkflowError("Graph marked as built but compiled graph is None.")
            return self._compiled_graph

        logger.info("Building RAG workflow graph (with routing and evaluation fallback)...")
        # Ensure StateGraph is imported directly
        workflow = StateGraph(WorkflowState)
        # Define Nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("retrieve_vectorstore", self._retrieve_vectorstore_node)
        workflow.add_node("retrieve_web_search", self._web_search_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate_answer", self._evaluate_answer_node)
        workflow.add_node("fallback_generate", self._fallback_generate_node)

        # --- Define Edges ---
        workflow.add_edge(START, "initialize")
        # Route question after initialization
        workflow.add_conditional_edges(
             "initialize", self._route_question_edge,
             {"vectorstore": "retrieve_vectorstore", "web_search": "retrieve_web_search"}
        )
        # --- MODIFIED EDGES ---
        # BOTH retrieval methods now go to grading
        workflow.add_edge("retrieve_vectorstore", "grade_documents")
        workflow.add_edge("retrieve_web_search", "grade_documents") # Changed from "generate"
        # --------------------
        # After grading, proceed to generation
        workflow.add_edge("grade_documents", "generate")
        # After generation, evaluate the answer
        workflow.add_edge("generate", "evaluate_answer")
        # Decide final path after evaluation
        workflow.add_conditional_edges(
             "evaluate_answer", self._decide_after_evaluation,
             {END: END, "needs_fallback": "fallback_generate"}
        )
        # Fallback leads to the end
        workflow.add_edge("fallback_generate", END)

        # Compile the graph
        try:
            self._compiled_graph = workflow.compile()
            self._is_graph_built = True
            logger.info("Workflow graph compiled successfully (web search now routes to grading).")
            if self._compiled_graph is None:
                 raise WorkflowError("Graph compilation returned None unexpectedly.")
            return self._compiled_graph
        except Exception as e:
             logger.error(f"Failed to compile workflow graph: {e}", exc_info=True)
             self._is_graph_built = False
             self._compiled_graph = None
             raise WorkflowError(f"Graph compilation failed: {e}") from e

    # --- Graph Visualization Method (Using Graphviz/pygraphviz) ---
    def get_graph_visualization_png(self) -> Optional[bytes]:
        """
        Generates a PNG visualization of the compiled workflow graph using Graphviz/pygraphviz.
        Requires 'pygraphviz' Python package and system 'graphviz' library to be installed.
        """
        if not self._compiled_graph:
            # Attempt to build the graph if not already compiled, for visualization purposes
            try:
                logger.info("Graph not compiled for visualization request. Attempting build now.")
                self.build_graph()
                if not self._compiled_graph:
                     logger.warning("Graph compilation failed during visualization request.")
                     return None
            except Exception as e:
                logger.error(f"Failed to build graph for visualization: {e}", exc_info=False)
                return None

        logger.info("Attempting to generate workflow graph PNG using draw_png / Graphviz...")
        try:
            if self._compiled_graph is None:
                 logger.error("Cannot generate graph PNG: Compiled graph is None even after build attempt.")
                 return None

            # Call draw_png() - relies on the successfully installed pygraphviz
            png_bytes = self._compiled_graph.get_graph(xray=True).draw_png()

            if png_bytes:
                 logger.info(f"Successfully generated graph PNG data using draw_png ({len(png_bytes)} bytes).")
                 return png_bytes
            else:
                 logger.warning("draw_png() returned None or empty bytes without raising an exception.")
                 return None

        except ImportError as ie:
             logger.error(f"ImportError during Graphviz rendering: {ie}. Is 'pygraphviz' still installed?", exc_info=True)
             return None
        except FileNotFoundError as fnf_e:
             logger.error(f"Graphviz FileNotFoundError: {fnf_e}. Can't find 'dot' executable.", exc_info=True)
             return None
        except Exception as e:
             logger.error(f"Failed to draw workflow graph using draw_png: {e}", exc_info=True)
             return None


    # --- Workflow Execution & Result Creation ---
    def run_workflow(self, question: str, initial_source_urls: Optional[List[str]] = None) -> Result:
        start_run_time = time.time()
        logger.info(f"--- Starting Workflow Run for question: '{question[:100]}...' ---")
        graph_png_bytes: Optional[bytes] = None # Initialize graph data variable

        # Ensure graph is built before proceeding
        compiled_graph: Optional[CompiledGraph] = None
        try:
            compiled_graph = self.build_graph() # Will build or return cached graph
            if not compiled_graph:
                 raise WorkflowError("Graph compilation failed or returned None.")

            # Attempt to generate graph visualization AFTER compiling
            logger.info("Attempting to generate graph visualization PNG after compilation...")
            try:
                 graph_png_bytes = self.get_graph_visualization_png()
                 if graph_png_bytes: logger.info("Generated graph visualization PNG successfully.")
                 else: logger.warning("Graph visualization PNG generation returned None or failed.")
            except Exception as graph_err:
                  logger.error(f"Failed to generate graph visualization: {graph_err}", exc_info=False)

        except WorkflowError as e:
             logger.error(f"Cannot run workflow, graph setup failed: {e}", exc_info=True)
             return Result(question=question, answer_text=f"Workflow setup failed: {e}", execution_metadata={"error": str(e)}, graph_diagram_data=graph_png_bytes)
        except Exception as e:
            logger.error(f"Unexpected error during graph build: {e}", exc_info=True)
            return Result(question=question, answer_text=f"Unexpected workflow setup error: {e}", execution_metadata={"error": str(e)}, graph_diagram_data=graph_png_bytes)


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
                 try:
                     final_state = WorkflowState(**final_state_dict)
                 except TypeError as te:
                     logger.error(f"Mismatch between final state dict and WorkflowState dataclass: {te}. Keys: {list(final_state_dict.keys())}")
                     raise WorkflowError(f"Could not reconstruct final WorkflowState: {te}")
            else:
                 logger.error(f"Workflow execution returned unexpected type: {type(final_state_dict)}")
                 raise WorkflowError("Workflow execution did not return a valid final state dictionary.")

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            logs_on_fail = []
            if isinstance(initial_state_input.get('logs'), list):
                 logs_on_fail = [LogEntry(**log) for log in initial_state_input['logs'] if isinstance(log, dict)]
            error_result = Result(question=question, answer_text=f"Workflow execution failed: {e}", execution_metadata={"error": str(e)}, log_entries=logs_on_fail, graph_diagram_data=graph_png_bytes)
            return error_result

        if final_state is None:
             logger.error("Final state object is None after workflow invocation.")
             return Result(question=question, answer_text="Workflow failed: Final state object missing.", execution_metadata={"error": "Final state object missing"}, graph_diagram_data=graph_png_bytes)

        # Use the correctly scoped variable containing the PNG bytes (or None)
        result = self._create_result_from_state(final_state, graph_png_bytes)

        end_run_time = time.time()
        if isinstance(result.execution_metadata, dict):
            result.execution_metadata["total_duration_sec"] = round(end_run_time - start_run_time, 3)
        else:
             logger.warning("Execution metadata is not a dict, cannot add duration.")

        logger.info(f"--- Workflow Run Completed in {result.execution_metadata.get('total_duration_sec', 'N/A'):.3f} seconds ---")
        return result

    def _create_result_from_state(self, final_state: WorkflowState, graph_diagram_data: Optional[bytes] = None) -> Result:
        """Creates the final Result object from the WorkflowState."""
        from ..data_models.result import SourceInfo, LogEntry
        from ..data_models.workflow_state import UrlUsageInfo

        source_summary: List[SourceInfo] = []
        if isinstance(final_state.url_usage_tracking, dict):
            for url, usage_info in final_state.url_usage_tracking.items():
                # Handle potential dict representation if state passed through serialization
                if isinstance(usage_info, dict):
                     try: usage_info = UrlUsageInfo(**usage_info)
                     except Exception: logger.warning(f"Skipping invalid usage_info dict for URL '{url}': {usage_info}"); continue
                elif not isinstance(usage_info, UrlUsageInfo):
                     logger.warning(f"Skipping invalid usage_info for URL '{url}': Type {type(usage_info)}"); continue

                final_relevance: Optional[float] = None
                # Check scores list exists and is not empty
                if hasattr(usage_info, 'relevance_scores') and isinstance(usage_info.relevance_scores, list) and usage_info.relevance_scores:
                     try:
                         # Filter only numeric scores before averaging
                         numeric_scores = [s for s in usage_info.relevance_scores if isinstance(s, (int, float))]
                         if numeric_scores: final_relevance = round(sum(numeric_scores) / len(numeric_scores), 2)
                     except ZeroDivisionError: pass # Should not happen if numeric_scores check passed
                # If no scores, final_relevance remains None (will be displayed as N/A)

                source_summary.append(SourceInfo(
                    url=url,
                    source_type=getattr(usage_info, 'source_type', "unknown"),
                    final_relevance_metric=final_relevance, # Remains None if no scores
                    usage_count=getattr(usage_info, 'count', 0)
                ))
        else:
            logger.warning("Final state url_usage_tracking is not a dict.")

        log_entries = []
        if isinstance(final_state.logs, list):
            for log_item in final_state.logs:
                 if isinstance(log_item, dict):
                      try: log_entries.append(LogEntry(**log_item))
                      except Exception: logger.warning(f"Skipping invalid log entry dict: {log_item}")
                 elif isinstance(log_item, LogEntry): log_entries.append(log_item)
                 else: logger.warning(f"Skipping invalid log entry item: Type {type(log_item)}")
        else:
             logger.warning("Final state logs is not a list.")

        exec_meta = {
             "original_question": final_state.original_question,
             "final_retrieval_strategy": final_state.current_retrieval_strategy,
             "transform_attempts": final_state.transform_attempts,
             "final_relevance_threshold": final_state.relevance_threshold,
             "generation_metadata": final_state.generation_results if isinstance(final_state.generation_results, dict) else {},
             "evaluation_results": final_state.evaluation_results if isinstance(final_state.evaluation_results, dict) else {},
             "docs_retrieved_count": len(final_state.retrieval_history[-1]['document_ids']) if final_state.retrieval_history and isinstance(final_state.retrieval_history[-1].get('document_ids'), list) else 0,
             "docs_passed_grading_count": len(final_state.documents) if isinstance(final_state.documents, list) else 0,
             "error": str(final_state.generation_results.get("error")) if isinstance(final_state.generation_results, dict) and final_state.generation_results.get("error") else None
        }

        return Result(
            question=final_state.original_question,
            answer_text=final_state.answer or "No answer was generated.",
            final_source_summary=source_summary,
            log_entries=log_entries,
            graph_diagram_data=graph_diagram_data,
            execution_metadata=exec_meta
        )