# File: main_gui.py
# (Complete file content with corrected .env file path loading)

import streamlit as st
import logging
from pathlib import Path
import sys
import os
import time # For showing messages
from typing import List, Dict, Optional
import pandas # Needed by presenter for dataframes
from dotenv import load_dotenv # Make sure dotenv is imported
import tempfile

# --- Core Project Imports (Direct, per Rule #10) ---
# Assuming these imports are correct relative to the location of this script
# If rag_system is directly under enhanced_rag_system, these relative imports work.
# If rag_system is elsewhere, sys.path manipulation below handles it.
from rag_system.config.settings import Configuration
from rag_system.ui.gui.interface import StreamlitGUIInterface
from rag_system.utils.logger import setup_logger
from rag_system.core.application import RAGApplication
from rag_system.llm.providers.openai_provider import OpenAIProvider
from rag_system.llm.providers.anthropic_provider import AnthropicProvider
from rag_system.llm.providers.gemini_provider import GeminiProvider
from rag_system.llm.factories import LLMProviderFactory


# --- Determine Project Root and Add to Path ---
try:
    # Assuming main_gui.py is in enhanced_rag_system/
    # Path(__file__) is enhanced_rag_system/main_gui.py
    # .parent is enhanced_rag_system/
    # .parent.parent is Agentic-RAG/ (The actual project root where .env resides)
    script_dir = Path(__file__).resolve().parent
    project_root_dir = script_dir.parent # Go one level up to Agentic-RAG/
    env_path = project_root_dir / '.env' # Look for .env in Agentic-RAG/

    # Add source root (enhanced_rag_system/rag_system) to sys.path if needed
    # For imports like 'from rag_system.config...', Python needs to find 'rag_system'
    # Assuming 'rag_system' package is inside 'enhanced_rag_system' directory
    source_package_root = script_dir / "rag_system" # enhanced_rag_system/rag_system
    # We might need to add the *parent* of rag_system (i.e., enhanced_rag_system) to the path
    if str(script_dir) not in sys.path:
         sys.path.insert(0, str(script_dir)) # Add enhanced_rag_system/ to path

except Exception as path_e:
     st.exception(f"CRITICAL ERROR: Failed to setup Python path automatically: {path_e}. Ensure the script is run from the correct directory structure.")
     st.stop()

# --- Load Environment Variables ---
# Use the correctly calculated path to the .env file in the parent directory
load_success = load_dotenv(dotenv_path=env_path, override=True) # Use override=True just in case
# Add logging to confirm loading attempt and result
logger = logging.getLogger(__name__) # Get logger after path setup potentially fixes imports
if load_success:
    logger.info(f"Successfully loaded .env file from: {env_path}")
else:
    logger.warning(f"Could not find or load .env file from expected location: {env_path}. Relying on system environment variables.")
    # Optionally add st.warning here if .env is strictly required

# --- Initialize Logger (Ensure logger is initialized AFTER potential path setup) ---
if 'logger_configured' not in st.session_state:
    setup_logger(level=logging.INFO)
    st.session_state.logger_configured = True
# Re-get logger in case setup_logger was called before this point was reached
logger = logging.getLogger(__name__)

# --- Session State Initialization ---
# (Same as before, but now relies on load_dotenv having worked correctly)
if 'app' not in st.session_state:
    logger.info("Initializing RAGApplication in Streamlit session state...")
    try:
        # Configuration now correctly uses loaded env vars
        config = Configuration(dotenv_path=env_path) # Pass correct path if needed, though load_dotenv already ran
        st.session_state.app = RAGApplication(config=config) # Should find keys now
        st.session_state.app_config = config
        st.session_state.corpus_ready = False
        st.session_state.last_result = None
        st.session_state.error_message = None
        st.session_state.selected_provider = config.get_llm_provider_name()
        st.session_state.selected_model = config.get_llm_model_name()
        logger.info("Application initialized and stored in session state.")
    except Exception as init_e:
        # If init still fails (e.g., invalid key, network issue), display error
        logger.error(f"Failed to initialize RAGApplication: {init_e}", exc_info=True)
        st.session_state.app = None
        # Display the error prominently in the UI and stop
        st.error(f"Application Initialization Failed: {init_e}")
        st.stop()

# --- Helper Function for Indexing (Unchanged) ---
def trigger_corpus_initialization(app: RAGApplication, uploaded_file_object):
    # (Content is the same as previous version)
    if not app or not app.doc_corpus:
         st.error("Application or Document Corpus not initialized.")
         return False
    file_path = None
    temp_dir = None
    if uploaded_file_object is not None:
        try:
            temp_dir = tempfile.TemporaryDirectory()
            file_path = Path(temp_dir.name) / uploaded_file_object.name
            with open(file_path, "wb") as f: f.write(uploaded_file_object.getvalue())
            logger.info(f"Uploaded file saved temporarily to: {file_path}")
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}"); logger.error("File save error", exc_info=True)
            if temp_dir: temp_dir.cleanup()
            return False
    success = False
    with st.spinner("Initializing document corpus... This may take several minutes."):
        try:
            app.initialize_corpus(url_file_path=str(file_path) if file_path else None)
            st.session_state.corpus_ready = True
            st.session_state.error_message = None
            success = True
            logger.info("Corpus initialization successful.")
        except Exception as e:
            st.session_state.corpus_ready = False
            st.session_state.error_message = f"Corpus Initialization Failed: {e}"
            logger.error(f"Corpus initialization failed: {e}", exc_info=True)
    if temp_dir:
        try: temp_dir.cleanup()
        except Exception as e: logger.warning(f"Failed to cleanup temp dir: {e}")
    return success

# --- Corrected Helper Function to get models for provider (Unchanged from previous working version) ---
@st.cache_data
def get_models_for_provider(provider_name: str) -> List[str]:
    logger.info(f"Getting models for provider: {provider_name}")
    provider_name_lower = provider_name.lower()
    try:
        if provider_name_lower == "openai": return OpenAIProvider._COMMON_MODELS
        elif provider_name_lower == "anthropic": return AnthropicProvider._COMMON_MODELS
        elif provider_name_lower == "google": return GeminiProvider._COMMON_MODELS
        else:
            logger.error(f"Unknown provider name '{provider_name}' in get_models_for_provider.")
            return [f"Error: Unknown provider {provider_name}"]
    except AttributeError as e:
        logger.error(f"Could not find model list for provider '{provider_name}': {e}", exc_info=False)
        return [f"Error: Model list missing for {provider_name}"]
    except ImportError as e:
         logger.error(f"Failed to import provider class for {provider_name}: {e}", exc_info=False)
         return [f"Error: Missing provider code for {provider_name}"]
    except Exception as e:
        logger.error(f"Unexpected error getting models for provider '{provider_name}': {e}", exc_info=True)
        return [f"Error loading models for {provider_name}"]

# --- Callback Function for Provider Change (Unchanged) ---
def handle_provider_change():
    selected_provider = st.session_state.selected_provider
    logger.info(f"Provider changed to: {selected_provider}. Updating model list.")
    models = get_models_for_provider(selected_provider)
    if models and not models[0].startswith("Error"):
        st.session_state.selected_model = models[0]
        logger.info(f"Reset selected model in session state to: {models[0]}")
    else:
        logger.warning(f"Could not retrieve valid models for {selected_provider}. Model selection might not be accurate.")
        st.session_state.selected_model = ""


# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="Enhanced RAG System")
st.title("Enhanced RAG System Interface")

# Display initialization errors (more robust check now)
if 'app' not in st.session_state or not st.session_state.app:
     # This message persists if app init failed
     st.error("RAG Application failed to initialize. Please check logs and configuration (especially API keys).")
     st.stop() # Stop execution if app isn't loaded
elif 'error_message' in st.session_state and st.session_state.error_message:
     # Display non-fatal errors that might have occurred later
     st.error(st.session_state.error_message)

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration & Indexing")
    uploaded_file = st.file_uploader(
        "Upload URL File (.txt, one URL per line)", type=['txt'], key="url_uploader",
        help="Upload a file with URLs to index. If no file is uploaded, default URLs from config will be used."
    )
    if st.button("Initialize/Build Index", key="build_index_button"):
        if trigger_corpus_initialization(st.session_state.app, uploaded_file):
              st.success("Corpus Initialization Complete!")
              st.session_state.error_message = None
    st.divider()
    st.subheader("Workflow Settings")
    # LLM Provider Selection
    available_providers = ["openai", "anthropic", "google"]
    selected_provider = st.selectbox(
        "LLM Provider", available_providers,
        key="selected_provider", on_change=handle_provider_change,
        help="Select the LLM provider to use."
    )
    # LLM Model Selection
    models_list = get_models_for_provider(st.session_state.selected_provider)
    if models_list and not models_list[0].startswith("Error"):
        current_model = st.session_state.get("selected_model", "")
        try: current_model_index = models_list.index(current_model)
        except ValueError:
            logger.warning(f"Selected model '{current_model}' not found in list for {st.session_state.selected_provider}. Defaulting to first.")
            current_model_index = 0
            if models_list: st.session_state.selected_model = models_list[0]
            else: st.session_state.selected_model = ""
        selected_model_widget_value = st.selectbox(
            "LLM Model Name", options=models_list, index=current_model_index,
            key="selected_model",
            help=f"Select a model from the '{st.session_state.selected_provider}' provider."
        )
    else:
        st.warning(f"Could not load model list for {st.session_state.selected_provider}: {models_list[0] if models_list else 'List empty'}. Please enter model name manually.")
        selected_model_widget_value = st.text_input(
            "LLM Model Name (Enter Manually)", value=st.session_state.get("selected_model", ""),
            key="selected_model",
            help=f"Enter model name manually for '{st.session_state.selected_provider}'."
        )

# --- Main Area ---
st.header("Ask a Question")
corpus_status = st.session_state.get('corpus_ready', False)
if corpus_status: st.success("Corpus is ready. Ask your question below.")
else: st.warning("Corpus not initialized. Please build the index using the sidebar.")
question = st.text_area(
    "Enter your question:", height=100, key="user_question",
    disabled=not corpus_status
)
if st.button("Answer", key="answer_button", disabled=not corpus_status):
    if not question: st.warning("Please enter a question.")
    else:
        app: RAGApplication = st.session_state.app
        provider = st.session_state.selected_provider
        model = st.session_state.selected_model
        logger.info(f"Processing question with selected LLM: {provider}/{model}")
        with st.spinner(f"Processing question using {provider}/{model}..."):
            try:
                result = app.process_question(
                    question=question, selected_provider=provider, selected_model=model
                )
                st.session_state.last_result = result
                st.session_state.error_message = None
                logger.info("Question processing complete.")
            except Exception as e:
                 logger.error(f"Error processing question via GUI: {e}", exc_info=True)
                 st.session_state.error_message = f"Failed to process question: {e}"
                 st.session_state.last_result = None

# --- Display results area ---
st.divider()
st.header("Results")
if 'last_result' in st.session_state and st.session_state.last_result:
    if 'gui_interface' not in st.session_state: st.session_state.gui_interface = StreamlitGUIInterface()
    try: st.session_state.gui_interface.display_result(st.session_state.last_result)
    except Exception as display_e:
         logger.error(f"Error displaying results using presenter: {display_e}", exc_info=True)
         st.error(f"Error rendering results display: {display_e}")
# Display processing errors
if 'error_message' in st.session_state and st.session_state.error_message:
    # Avoid showing old errors if a new result exists
    if not ('last_result' in st.session_state and st.session_state.last_result):
        st.error(st.session_state.error_message)