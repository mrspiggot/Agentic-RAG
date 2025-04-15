# File: main_gui.py
# Instruction: Replace the entire content of this file.
#              (Basic Streamlit structure, session state, simple workflow trigger)

import streamlit as st
import logging
from pathlib import Path
import sys
import os
import time # For showing messages

from rag_system.config.settings import Configuration
from rag_system.ui.factories import UserInterfaceFactory
from rag_system.ui.gui.interface import StreamlitGUIInterface
from rag_system.utils.logger import setup_logger
import pandas  # Needed by presenter
from dotenv import load_dotenv
from rag_system.core.application import RAGApplication
import tempfile

load_dotenv()

# --- Path Setup ---
# Add package root to path for imports
try:
    project_root_dir = Path(__file__).resolve().parent
    source_root = project_root_dir / "enhanced_rag_system"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
except Exception as e:
     # Fallback for potential issues finding relative paths in Streamlit context
     st.error(f"ERROR: Failed to setup Python path automatically: {e}. Please ensure the 'enhanced_rag_system' folder is accessible.")
     st.stop()

# --- Core Application Imports ---
# Encapsulate imports in a function to check dependencies clearly at startup


# Perform imports after check


# --- Initialize Logger ---
# Setup logger once
if 'logger_configured' not in st.session_state:
    setup_logger(level=logging.INFO) # Default to INFO level for GUI
    st.session_state.logger_configured = True
logger = logging.getLogger(__name__)


# --- Session State Initialization ---
# Initialize application and state variables if not already present
if 'app' not in st.session_state:
    logger.info("Initializing application in Streamlit session state...")
    try:
        # Point to .env file relative to this script's location (project root)
        config = Configuration(dotenv_path=project_root_dir / '.env')
        st.session_state.app = RAGApplication(config=config)
        st.session_state.corpus_ready = False
        st.session_state.last_result = None
        st.session_state.error_message = None
        logger.info("Application initialized and stored in session state.")
    except Exception as e:
        logger.error(f"Failed to initialize RAGApplication: {e}", exc_info=True)
        st.session_state.app = None
        st.session_state.error_message = f"Application Initialization Failed: {e}"

# --- Helper Function for Indexing ---
def trigger_corpus_initialization(app: RAGApplication, uploaded_file_object):
    """Handles file saving and triggers corpus initialization."""
    if not app or not app.doc_corpus:
         st.error("Application or Document Corpus not initialized.")
         return False

    file_path = None
    temp_dir = None
    if uploaded_file_object is not None:
        # Save uploaded file temporarily
        try:
            temp_dir = tempfile.TemporaryDirectory()
            file_path = Path(temp_dir.name) / uploaded_file_object.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file_object.getvalue())
            logger.info(f"Uploaded file saved temporarily to: {file_path}")
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            if temp_dir: temp_dir.cleanup() # Clean up temp dir on error
            return False

    # Trigger initialization
    success = False
    with st.spinner("Initializing document corpus... This may take several minutes."):
        try:
            app.initialize_corpus(url_file_path=str(file_path) if file_path else None)
            st.session_state.corpus_ready = True
            st.session_state.error_message = None # Clear previous errors
            success = True
        except Exception as e:
            st.session_state.corpus_ready = False
            st.session_state.error_message = f"Corpus Initialization Failed: {e}"
            logger.error(f"Corpus initialization failed: {e}", exc_info=True)

    # Clean up temp dir after use
    if temp_dir:
        try: temp_dir.cleanup()
        except Exception: pass

    return success


# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="Enhanced RAG System")
st.title("Enhanced RAG System Interface")

# Display initialization errors if any
if 'error_message' in st.session_state and st.session_state.error_message:
    st.error(st.session_state.error_message)

# Sidebar for Configuration and Indexing
with st.sidebar:
    st.header("Configuration & Indexing")

    uploaded_file = st.file_uploader(
        "Upload URL File (.txt, one URL per line)",
        type=['txt'],
        key="url_uploader", # Add a key for stability
        help="Upload a file with URLs to index. If no file is uploaded, default URLs from config will be used."
    )

    if st.button("Initialize/Build Index", key="build_index_button"):
        if 'app' in st.session_state and st.session_state.app:
             if trigger_corpus_initialization(st.session_state.app, uploaded_file):
                  st.success("Corpus Initialization Complete!")
             else:
                  # Error message already displayed by trigger function
                  pass
        else:
             st.error("Application not initialized.")

    st.divider()
    # Placeholder for future configuration options (LLM selection etc.)
    st.subheader("Workflow Settings (Future)")
    # llm_provider = st.selectbox("LLM Provider", ["openai", "anthropic", "google"], key="llm_provider_select")
    # llm_model = st.text_input("LLM Model Name", value=st.session_state.get('app_config', {}).get('llm_model', 'gpt-4o-mini'), key="llm_model_input")
    st.info("LLM selection coming soon.")


# Main Area for Q&A and Results
st.header("Ask a Question")

# Check if corpus is ready before allowing questions
corpus_status = st.session_state.get('corpus_ready', False)
if corpus_status:
    st.success("Corpus is ready. Ask your question below.")
else:
    st.warning("Corpus not initialized. Please upload a URL file (optional) and click 'Initialize/Build Index' in the sidebar.")

# Disable input/button if corpus is not ready
question = st.text_area(
    "Enter your question:",
    height=100,
    key="user_question",
    disabled=not corpus_status
)

if st.button("Answer", key="answer_button", disabled=not corpus_status):
    if not question:
        st.warning("Please enter a question.")
    elif 'app' in st.session_state and st.session_state.app:
        app = st.session_state.app
        ui = StreamlitGUIInterface() # Get presenter via interface instance
        with st.spinner("Processing your question..."):
            try:
                result = app.process_question(question)
                st.session_state.last_result = result # Store result for display
                st.session_state.error_message = None # Clear errors on success
            except Exception as e:
                 logger.error(f"Error processing question via GUI: {e}", exc_info=True)
                 st.session_state.error_message = f"Failed to process question: {e}"
                 st.session_state.last_result = None # Clear previous result on error
    else:
        st.error("Application not initialized correctly.")


# Display results if available in session state
if 'last_result' in st.session_state and st.session_state.last_result:
    st.divider()
    st.header("Results")
    # Use the presenter from the interface object to display
    # We need an interface instance here, let's create one
    # Ideally, presenter would be stateless and could be called directly
    # For now, create interface instance to access its presenter
    if 'gui_interface' not in st.session_state:
        st.session_state.gui_interface = StreamlitGUIInterface()
    st.session_state.gui_interface.display_result(st.session_state.last_result)

# Display errors if they occurred during processing
if 'error_message' in st.session_state and st.session_state.error_message:
    st.error(st.session_state.error_message)