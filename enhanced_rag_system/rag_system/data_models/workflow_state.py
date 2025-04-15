# Defines WorkflowState, UrlUsageInfo classes
# File: rag_system/data_models/workflow_state.py
# Instruction: Replace the entire content of this file.

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

# Use relative import assuming document.py is in the same directory
from .document import Document

logger = logging.getLogger(__name__)

@dataclass
class UrlUsageInfo:
    """Tracks usage and relevance scoring for a specific URL."""
    source_type: str  # e.g., 'supplied', 'web_search', 'vector_store'
    count: int = 0
    last_used_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    relevance_scores: List[int] = field(default_factory=list)

@dataclass
class WorkflowState:
    """
    Represents and manages the state data flowing through the RAG workflow graph.
    Designed to be mutable within LangGraph nodes via its methods.
    """
    question: str
    original_question: str
    # Documents currently being processed or considered relevant
    documents: List[Document] = field(default_factory=list)
    # Final generated answer
    answer: Optional[str] = None
    # Counter for query transformation attempts
    transform_attempts: int = 0
    # Current relevance score threshold being applied
    relevance_threshold: int = 4 # Default set here, can be overridden by Config
    # Name of the last retrieval strategy used
    current_retrieval_strategy: str = "semantic" # Starting default
    # History of retrieval attempts
    retrieval_history: List[Dict[str, Any]] = field(default_factory=list)
    # Stores results from evaluation steps (key: evaluator name, value: results dict)
    evaluation_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Stores metadata about the final generation step
    generation_results: Dict[str, Any] = field(default_factory=dict)
    # Tracks usage details for each source URL encountered
    url_usage_tracking: Dict[str, UrlUsageInfo] = field(default_factory=dict)
    # Optional list for logging specific workflow events tied to state
    logs: List[Dict[str, Any]] = field(default_factory=list)


    def __post_init__(self):
        """Initialize fields after basic dataclass setup."""
        if not self.original_question:
            self.original_question = self.question

    # --- State Update Methods ---

    def add_log(self, level: str, message: str, node: Optional[str] = None):
        """Adds a structured log entry to the state's log list."""
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        if node:
            entry["node"] = node
        self.logs.append(entry)
        # Optionally also log via standard logging
        log_func = getattr(logger, level.lower(), logger.info)
        log_prefix = f"[Node: {node}] " if node else ""
        log_func(f"{log_prefix}{message}")


    def add_retrieval_result(self, strategy: str, query: str, retrieved_docs: List[Document]):
        """Updates state after a retrieval step."""
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        self.current_retrieval_strategy = strategy
        self.documents = retrieved_docs # Assume this node sets the current docs
        history_entry = {
            "strategy": strategy,
            "timestamp_utc": timestamp_utc,
            "query": query,
            "document_count": len(retrieved_docs),
            "document_ids": [doc.id for doc in retrieved_docs]
        }
        self.retrieval_history.append(history_entry)
        self.add_log("info", f"Retrieved {len(retrieved_docs)} docs using '{strategy}'.", node=f"{strategy}_retrieval")

        # Update URL usage for retrieved docs
        for doc in retrieved_docs:
            url = doc.metadata.get('source', 'unknown_retrieved_source')
            source_type = doc.metadata.get('source_type', strategy) # Use strategy if type not in doc
            if url not in self.url_usage_tracking:
                self.url_usage_tracking[url] = UrlUsageInfo(source_type=source_type)
            self.url_usage_tracking[url].count += 1
            self.url_usage_tracking[url].last_used_utc = timestamp_utc
            # Reset relevance scores when retrieved? Or append? Let's reset for now.
            # self.url_usage_tracking[url].relevance_scores = []


    def add_url_relevance_score(self, url: str, score: int):
        """Adds a relevance score for a specific URL."""
        if not isinstance(url, str) or not url:
             self.add_log("warning", f"Attempted to add relevance score for invalid URL: {url}")
             return
        if url in self.url_usage_tracking:
            self.url_usage_tracking[url].relevance_scores.append(score)
        else:
            # Log a warning if the URL wasn't previously tracked (e.g., from retrieval)
            self.add_log("warning", f"Adding relevance score for untracked URL '{url}'. Creating entry.", node="grading")
            # Create a basic entry - source_type might be inaccurate here
            self.url_usage_tracking[url] = UrlUsageInfo(source_type="unknown_graded", relevance_scores=[score])


    def set_relevant_documents(self, relevant_docs: List[Document], node_name: str = "grading"):
        """Updates the documents list, typically after grading."""
        original_count = len(self.documents)
        self.documents = relevant_docs
        self.add_log("info", f"Filtered documents: {len(relevant_docs)} retained out of {original_count}.", node=node_name)


    def add_evaluation_result(self, evaluator_name: str, result_dict: Dict[str, Any]):
        """Adds results from an evaluation step."""
        self.evaluation_results[evaluator_name.lower()] = result_dict
        self.add_log("info", f"Added evaluation results for '{evaluator_name}'. Score(s): {result_dict.get('score', result_dict.get('overall_score', 'N/A'))}", node=f"{evaluator_name}_evaluation")


    def set_final_answer(self, answer_text: str, metadata: Dict[str, Any]):
        """Sets the final answer and associated generation metadata."""
        self.answer = answer_text
        self.generation_results = metadata
        self.add_log("info", f"Final answer generated. Length: {len(answer_text)}. Fallback: {metadata.get('is_fallback', False)}.", node="generation")


    def increment_transform_attempts(self) -> int:
        """Increments the query transform counter and returns the new count."""
        self.transform_attempts += 1
        self.add_log("info", f"Query transform attempt count incremented to {self.transform_attempts}.", node="transform_query")
        return self.transform_attempts


    def lower_relevance_threshold(self, min_threshold: int) -> int:
        """Lowers the relevance threshold, respecting the minimum, returns new threshold."""
        original_threshold = self.relevance_threshold
        new_threshold = max(min_threshold, self.relevance_threshold - 1)
        if new_threshold != original_threshold:
            self.relevance_threshold = new_threshold
            self.add_log("info", f"Relevance threshold lowered from {original_threshold} to {new_threshold}.", node="lower_threshold")
        else:
             self.add_log("info", f"Relevance threshold already at minimum ({min_threshold}).", node="lower_threshold")
        return self.relevance_threshold


    def update_question(self, new_question: str):
        """Updates the current question (used after transformation)."""
        self.question = new_question
        self.add_log("info", f"Current question updated after transformation.", node="transform_query")


    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"WorkflowState("
            f"question='{self.question[:50]}...', "
            f"original_question='{self.original_question[:50]}...', "
            f"docs_count={len(self.documents)}, "
            f"answer_len={len(self.answer) if self.answer else 0}, "
            f"attempts={self.transform_attempts}, "
            f"threshold={self.relevance_threshold}, "
            f"strategy='{self.current_retrieval_strategy}', "
            f"url_usage_count={len(self.url_usage_tracking)}"
            f")"
        )