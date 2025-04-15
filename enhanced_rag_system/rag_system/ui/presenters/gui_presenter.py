# Contains StreamlitGUIPresenter class
# File: rag_system/ui/presenters/gui_presenter.py
# Instruction: Create this file or replace its entire content.

import streamlit as st
import logging
import pandas as pd # For dataframes
from typing import Optional, Any, Dict

# Relative imports
from .base_presenter import OutputPresenter
from ...data_models.result import Result, SourceInfo, LogEntry

logger = logging.getLogger(__name__)

# Helper function to attempt mermaid rendering (optional)
def display_mermaid_graph(graph_data: Optional[Any]):
    """Attempts to render mermaid graph if data is present."""
    if graph_data:
        try:
            # Assuming graph_data is bytes for a PNG
            st.image(graph_data)
            logger.info("Mermaid graph displayed as image.")
        except Exception as e:
            logger.error(f"Failed to display graph image: {e}", exc_info=False)
            st.warning("Could not display graph image. Ensure dependencies are installed and rendering service is available.")
            # If graph_data could potentially be a mermaid string:
            # try:
            #     import streamlit.components.v1 as components
            #     components.html(f"<div class='mermaid'>{graph_data}</div>", height=600)
            #     # Requires mermaid JS loaded, potentially complex setup
            # except Exception as e_html:
            #     logger.error(f"Failed to render mermaid string: {e_html}")
            #     st.warning("Could not render Mermaid graph string.")
    else:
        st.info("No graph visualization data available.")


class StreamlitGUIPresenter(OutputPresenter):
    """Formats and displays results within a Streamlit interface using tabs."""

    def present(self, result: Result):
        """
        Renders the Result object into Streamlit tabs.

        Args:
            result: The Result object to present.
        """
        if not result:
            st.warning("No result object available to display.")
            return

        logger.info("Presenting results in Streamlit tabs.")
        tab_names = ["Answer", "Sources", "Execution Info", "Logs", "Workflow Graph"]
        tab_answer, tab_sources, tab_meta, tab_logs, tab_graph = st.tabs(tab_names)

        # --- Answer Tab ---
        with tab_answer:
            st.subheader("Generated Answer")
            st.markdown(result.answer_text) # Render answer as Markdown

        # --- Sources Tab ---
        with tab_sources:
            st.subheader("Sources Used / Relevance")
            if result.final_source_summary:
                # Convert SourceInfo objects to DataFrame for better display
                source_data = [
                    {
                        "URL": s.url,
                        "Type": s.source_type,
                        "Relevance": f"{s.final_relevance_metric:.1f}" if s.final_relevance_metric is not None else "N/A",
                        "Usage Count": s.usage_count
                    }
                    for s in result.final_source_summary
                ]
                df = pd.DataFrame(source_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No source information was tracked for this query.")

        # --- Execution Info Tab ---
        with tab_meta:
            st.subheader("Execution Metadata")
            if result.execution_metadata:
                st.json(result.execution_metadata, expanded=True) # Use json for nested dicts
            else:
                st.info("No execution metadata available.")

        # --- Logs Tab ---
        with tab_logs:
            st.subheader("Workflow Log Entries")
            if result.log_entries:
                 # Convert LogEntry objects to DataFrame
                 log_data = [
                     {
                         "Timestamp": log.timestamp_utc,
                         "Level": log.level,
                         "Node": log.node or "-",
                         "Message": log.message
                     }
                     for log in result.log_entries
                 ]
                 log_df = pd.DataFrame(log_data)
                 st.dataframe(log_df, use_container_width=True)
            else:
                 st.info("No detailed log entries captured in result.")

        # --- Graph Tab ---
        with tab_graph:
            st.subheader("Workflow Visualization")
            # Attempt to display graph - assumes result object might contain the data
            # The main script needs to call the manager's get_graph method and put data here
            display_mermaid_graph(result.graph_diagram_data)