# Contains StreamlitGUIInterface class
# File: rag_system/ui/gui/interface.py
# Instruction: Create this file or replace its entire content.

import streamlit as st
import logging
from typing import Dict, Any

# Relative imports
from ..base_ui import UserInterface
from ...data_models.result import Result
from ..presenters.gui_presenter import StreamlitGUIPresenter # Import concrete presenter

logger = logging.getLogger(__name__)

class StreamlitGUIInterface(UserInterface):
    """Streamlit Graphical User Interface implementation."""

    def __init__(self):
        """Initializes the Streamlit interface with a StreamlitGUIPresenter."""
        self.presenter = StreamlitGUIPresenter()
        logger.debug("StreamlitGUIInterface initialized.")
        # Note: Most Streamlit state/widgets are managed directly in the main_gui.py script

    def get_configuration_input(self) -> Dict[str, Any]:
        """
        Configuration is typically handled via widgets directly in the main script's
        sidebar for Streamlit apps. Returns empty dict for now.
        """
        # This could potentially read from st.session_state if needed later
        return {}

    def get_question(self) -> str:
        """
        Retrieves the question from Streamlit's session state,
        assuming it's stored there by the main script's text_area widget.
        """
        return st.session_state.get("user_question", "")

    def display_result(self, result: Result):
        """
        Uses the StreamlitGUIPresenter to display results in the UI.
        The actual display logic using st elements happens in the presenter.
        """
        logger.debug(f"Displaying result for question: {result.question[:50]}...")
        self.presenter.present(result)

    def display_error(self, message: str):
        """Displays an error message using Streamlit's error element."""
        st.error(message, icon="🚨")

    def display_progress(self, message: str):
        """
        Displays a progress message using Streamlit's info or status element.
        Using st.info for simplicity here. Can be replaced with st.spinner context manager
        in the main script for operations that take time.
        """
        st.info(message, icon="⏳")