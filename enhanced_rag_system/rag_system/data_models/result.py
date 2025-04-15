# Defines Result, SourceInfo, LogEntry classes
# File: rag_system/data_models/result.py
# Instruction: Replace the entire content of this file.

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class LogEntry:
    """Represents a single log message captured during workflow execution."""
    timestamp_utc: str
    level: str  # e.g., 'INFO', 'WARNING', 'ERROR'
    message: str
    node: Optional[str] = None # The workflow node where the log occurred

@dataclass
class SourceInfo:
    """Summarizes information about a source URL used in generating the answer."""
    url: str
    source_type: str # e.g., 'supplied', 'web_search', 'vector_store'
    # Aggregate relevance metric (e.g., max score, avg score)
    final_relevance_metric: Optional[float] = None
    usage_count: int = 0

@dataclass
class Result:
    """
    Data container for the final results of a RAG workflow execution,
    suitable for presentation by the UI layer.
    """
    question: str
    answer_text: str
    final_source_summary: List[SourceInfo] = field(default_factory=list)
    log_entries: List[LogEntry] = field(default_factory=list)
    # Data for rendering the graph (e.g., Mermaid string, image bytes, None)
    graph_diagram_data: Optional[Any] = None
    # Additional metadata about the execution (e.g., duration, model used, errors)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    # Raw final state for debugging? Optional.
    # final_workflow_state: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure essential fields have defaults if needed."""
        if not self.question:
            self.question = self.execution_metadata.get("original_question", "Unknown Question")
        if not self.answer_text:
             self.answer_text = "No answer generated."

    def __repr__(self) -> str:
        return (
            f"Result(question='{self.question[:50]}...', "
            f"answer_len={len(self.answer_text)}, "
            f"sources_count={len(self.final_source_summary)}, "
            f"logs_count={len(self.log_entries)})"
            )