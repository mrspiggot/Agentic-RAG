import streamlit as st
import logging
import pandas as pd # For dataframes
from typing import Optional, Any, Dict

# --- Remove streamlit-mermaid import ---

# Relative imports
from .base_presenter import OutputPresenter
from ...data_models.result import Result, SourceInfo, LogEntry

logger = logging.getLogger(__name__)

# *** REVERTED HELPER FUNCTION to use st.image ***
def display_mermaid_graph(graph_data: Optional[Any]):
    """Attempts to render graph PNG bytes using st.image."""
    if graph_data and isinstance(graph_data, bytes) and len(graph_data) > 0:
        try:
            st.subheader("Workflow Visualization")
            st.image(graph_data, caption="Workflow Execution Graph") # Use st.image
            logger.info("Workflow graph displayed as image.")
        except Exception as e:
            logger.error(f"Failed to display graph image using st.image: {e}", exc_info=False)
            st.warning("Could not display workflow graph image.")
    else:
        if graph_data is None:
            st.info("No graph visualization data was generated for this run.")
        else:
            logger.warning(f"Received invalid graph data (type: {type(graph_data)}, len: {len(graph_data) if isinstance(graph_data, bytes) else 'N/A'}). Cannot display.")
            st.warning("Graph visualization data is unavailable or invalid.")


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
        try:
            tab_answer, tab_sources, tab_meta, tab_logs, tab_graph = st.tabs(tab_names)
        except Exception as e:
             logger.error(f"Failed to create Streamlit tabs: {e}", exc_info=True)
             st.error(f"Error creating UI tabs: {e}")
             st.subheader("Generated Answer")
             st.markdown(result.answer_text)
             st.subheader("Sources Used")
             if result.final_source_summary: st.json(result.final_source_summary)
             return

        # --- Answer Tab ---
        with tab_answer:
            st.subheader("Generated Answer")
            st.markdown(result.answer_text)

        # --- Sources Tab ---
        with tab_sources:
            st.subheader("Sources Used / Relevance")
            if result.final_source_summary:
                try:
                    source_data = [
                        {
                            "URL": s.url,
                            "Type": s.source_type,
                            "Relevance": f"{s.final_relevance_metric:.1f}" if s.final_relevance_metric is not None else "N/A",
                            "Usage Count": s.usage_count
                        }
                        for s in result.final_source_summary if isinstance(s, SourceInfo)
                    ]
                    if source_data:
                        df = pd.DataFrame(source_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                         st.info("No valid source information available to display.")
                except Exception as e:
                     logger.error(f"Error creating sources dataframe: {e}", exc_info=True)
                     st.error("Could not display source information as a table.")
                     st.json(result.final_source_summary)
            else:
                st.info("No source information was tracked for this query.")

        # --- Execution Info Tab ---
        with tab_meta:
            st.subheader("Execution Metadata")
            if result.execution_metadata:
                st.json(result.execution_metadata, expanded=False)
            else:
                st.info("No execution metadata available.")

        # --- Logs Tab ---
        with tab_logs:
            st.subheader("Workflow Log Entries")
            if result.log_entries:
                 try:
                     log_data = [
                         {
                             "Timestamp": log.timestamp_utc,
                             "Level": log.level,
                             "Node": log.node or "-",
                             "Message": log.message
                         }
                         for log in result.log_entries if isinstance(log, LogEntry)
                     ]
                     if log_data:
                         log_df = pd.DataFrame(log_data)
                         st.dataframe(log_df, use_container_width=True)
                     else:
                          st.info("No valid log entries available to display.")
                 except Exception as e:
                      logger.error(f"Error creating logs dataframe: {e}", exc_info=True)
                      st.error("Could not display logs as a table.")
                      st.json(result.log_entries)
            else:
                 st.info("No detailed log entries captured in result.")

        # --- Graph Tab ---
        with tab_graph:
            # *** Reverted call to use st.image helper ***
            display_mermaid_graph(result.graph_diagram_data)